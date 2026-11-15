"""Interchangeable embedding backends and chunk-record utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite, sqrt
import os
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence


BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_DEFAULT_MODEL_NAME = "text-embedding-3-small"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbeddingConfig:
    """Select a backend and configure its batching and model options."""

    provider: str = "huggingface"
    model_name: Optional[str] = None
    batch_size: int = 32
    device: Optional[str] = None
    dimensions: Optional[int] = None
    api_key_env_var: str = "OPENAI_API_KEY"


@dataclass(frozen=True)
class EmbeddingRunStats:
    """Measurements from the latest successful or failed embedding call."""

    model_name: str
    dimension: Optional[int]
    processing_time_seconds: float
    vector_count: int


class EmbeddingModel(ABC):
    """Common contract consumed by retrieval implementations."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Return one normalized vector per document, preserving order."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Return a normalized query vector in the same vector space."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the size of vectors produced by this model."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the configured model identifier."""

    def get_last_run_stats(self) -> Optional[EmbeddingRunStats]:
        """Return metrics for the most recent embedding operation, if any."""
        return None


class _SentenceTransformerBackend(Protocol):
    def encode(
        self,
        sentences: List[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        ...

    def get_sentence_embedding_dimension(self) -> Optional[int]:
        ...


class _OpenAIEmbeddingClient(Protocol):
    embeddings: Any


def _prepare_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Embedding inputs must be strings.")

    prepared = text.strip()
    if not prepared:
        raise ValueError("Embedding inputs must not be empty.")

    return prepared


def _validate_config(config: EmbeddingConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.dimensions is not None and config.dimensions <= 0:
        raise ValueError("dimensions must be positive.")


def _resolve_model_name(config: EmbeddingConfig, default: str) -> str:
    model_name = config.model_name if config.model_name is not None else default
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    return model_name


class _BaseEmbeddingModel(EmbeddingModel, ABC):
    """Shared input checks, normalization, shape checks, and timing."""

    def __init__(
        self,
        model_name: str,
        dimension: Optional[int],
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._last_run_stats: Optional[EmbeddingRunStats] = None

    @abstractmethod
    def _encode_documents(self, texts: List[str]) -> Any:
        ...

    @abstractmethod
    def _encode_query(self, query: str) -> Any:
        ...

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings.")

        prepared_texts = [_prepare_text(text) for text in texts]
        if not prepared_texts:
            self._record_run(0.0, 0)
            return []

        return self._run(
            lambda: self._encode_documents(prepared_texts),
            expected_count=len(prepared_texts),
        )

    def embed_query(self, query: str) -> List[float]:
        prepared_query = _prepare_text(query)
        vectors = self._run(
            lambda: [self._encode_query(prepared_query)],
            expected_count=1,
        )
        return vectors[0]

    def get_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                f"Embedding dimension for '{self._model_name}' is not known "
                "until it has produced a vector."
            )
        return self._dimension

    def get_model_name(self) -> str:
        return self._model_name

    def get_last_run_stats(self) -> Optional[EmbeddingRunStats]:
        return self._last_run_stats

    def _run(self, encode: Any, expected_count: int) -> List[List[float]]:
        started = perf_counter()
        vector_count = 0
        try:
            vectors = self._normalize_vectors(encode(), expected_count)
            vector_count = len(vectors)
            return vectors
        finally:
            self._record_run(perf_counter() - started, vector_count)

    def _normalize_vectors(self, encoded: Any, expected_count: int) -> List[List[float]]:
        try:
            rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
            if len(rows) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} vectors, received {len(rows)}."
                )

            vectors = []
            observed_dimension = self._dimension
            for row in rows:
                values = row.tolist() if hasattr(row, "tolist") else row
                vector = [float(value) for value in values]
                if not vector:
                    raise ValueError("Vectors must not be empty.")
                if observed_dimension is None:
                    observed_dimension = len(vector)
                if len(vector) != observed_dimension:
                    raise ValueError(
                        f"Expected vector dimension {observed_dimension}, "
                        f"received {len(vector)}."
                    )
                if not all(isfinite(value) for value in vector):
                    raise ValueError("Vectors must contain only finite values.")

                norm = sqrt(sum(value * value for value in vector))
                if not isfinite(norm) or norm == 0:
                    raise ValueError("Vectors must have a finite, non-zero norm.")
                vectors.append([value / norm for value in vector])

            if self._dimension is not None and observed_dimension != self._dimension:
                raise ValueError(
                    f"Expected vector dimension {self._dimension}, "
                    f"received {observed_dimension}."
                )
            if self._dimension is None:
                self._dimension = observed_dimension
            return vectors
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                f"Model '{self._model_name}' returned invalid embedding data: "
                f"{error}"
            ) from error

    def _record_run(self, elapsed: float, vector_count: int) -> None:
        self._last_run_stats = EmbeddingRunStats(
            model_name=self._model_name,
            dimension=self._dimension,
            processing_time_seconds=elapsed,
            vector_count=vector_count,
        )


class _SentenceTransformerModel(_BaseEmbeddingModel, ABC):
    DEFAULT_MODEL_NAME = ""
    QUERY_PREFIX = ""

    def __init__(
        self,
        config: EmbeddingConfig,
        backend: Optional[_SentenceTransformerBackend] = None,
    ) -> None:
        _validate_config(config)
        if config.dimensions is not None:
            raise ValueError(
                "The dimensions option is available only for OpenAI "
                "text-embedding-3 models."
            )
        model_name = _resolve_model_name(config, self.DEFAULT_MODEL_NAME)
        self._config = config
        self._backend = backend if backend is not None else self._load_backend()

        try:
            dimension = self._backend.get_sentence_embedding_dimension()
        except Exception as error:
            raise RuntimeError(
                f"Could not read the embedding dimension for '{model_name}'."
            ) from error
        if dimension is None or dimension <= 0:
            raise RuntimeError(
                f"Model '{model_name}' did not report a valid embedding dimension."
            )

        super().__init__(model_name, int(dimension))

    def _load_backend(self) -> _SentenceTransformerBackend:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Sentence Transformers is required for BGE and MiniLM "
                "embeddings. Install the project requirements first."
            ) from error

        try:
            return SentenceTransformer(
                self._model_name_for_loading(),
                device=self._config.device,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load embedding model "
                f"'{_resolve_model_name(self._config, self.DEFAULT_MODEL_NAME)}'."
            ) from error

    def _model_name_for_loading(self) -> str:
        return _resolve_model_name(self._config, self.DEFAULT_MODEL_NAME)

    def _encode_documents(self, texts: List[str]) -> List[Any]:
        vectors = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start:start + self._config.batch_size]
            vectors.extend(self._encode_batch(batch))
        return vectors

    def _encode_query(self, query: str) -> Any:
        return self._encode_batch([self.QUERY_PREFIX + query])[0]

    def _encode_batch(self, texts: List[str]) -> Any:
        try:
            return self._backend.encode(
                texts,
                batch_size=min(self._config.batch_size, len(texts)),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to encode a document batch with "
                f"'{self.get_model_name()}'."
            ) from error


class BGEEmbeddingModel(_SentenceTransformerModel):
    """BGE small English model, with its retrieval query instruction."""

    DEFAULT_MODEL_NAME = BGE_MODEL_NAME
    QUERY_PREFIX = BGE_QUERY_PREFIX


class MiniLMEmbeddingModel(_SentenceTransformerModel):
    """Hugging Face all-MiniLM-L6-v2 baseline without query prefixes."""

    DEFAULT_MODEL_NAME = MINILM_MODEL_NAME


class OpenAIEmbeddingModel(_BaseEmbeddingModel):
    """OpenAI embeddings backend using an environment-provided API key."""

    _KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        config: EmbeddingConfig,
        client: Optional[_OpenAIEmbeddingClient] = None,
    ) -> None:
        _validate_config(config)
        model_name = _resolve_model_name(config, OPENAI_DEFAULT_MODEL_NAME)
        if (
            not isinstance(config.api_key_env_var, str)
            or not config.api_key_env_var.strip()
        ):
            raise ValueError("api_key_env_var must be a non-empty string.")
        if config.dimensions is not None and not model_name.startswith(
            "text-embedding-3-"
        ):
            raise ValueError(
                "The dimensions option is supported only by OpenAI "
                "text-embedding-3 models."
            )

        self._config = config
        self._client = client if client is not None else self._load_client()
        dimension = config.dimensions or self._KNOWN_DIMENSIONS.get(model_name)
        super().__init__(model_name, dimension)

    def _load_client(self) -> _OpenAIEmbeddingClient:
        api_key = os.getenv(self._config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                "OpenAI API key is unavailable. Set the "
                f"{self._config.api_key_env_var} environment variable."
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "The OpenAI Python package is required for OpenAI embeddings. "
                "Install the project requirements first."
            ) from error

        try:
            return OpenAI(api_key=api_key)
        except Exception as error:
            raise RuntimeError("Could not initialize the OpenAI client.") from error

    def _encode_documents(self, texts: List[str]) -> List[Any]:
        vectors = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start:start + self._config.batch_size]
            vectors.extend(self._encode_batch(batch))
        return vectors

    def _encode_query(self, query: str) -> Any:
        return self._encode_batch([query])[0]

    def _encode_batch(self, texts: List[str]) -> List[Any]:
        arguments = {
            "model": self.get_model_name(),
            "input": texts,
        }
        if self._config.dimensions is not None:
            arguments["dimensions"] = self._config.dimensions

        try:
            response = self._client.embeddings.create(**arguments)
            data = list(response.data)
            if len(data) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} vectors, received {len(data)}."
                )
            data.sort(key=lambda item: item.index)
            if [item.index for item in data] != list(range(len(texts))):
                raise ValueError("OpenAI response contained invalid item indices.")
            return [item.embedding for item in data]
        except Exception as error:
            raise RuntimeError(
                f"OpenAI embedding request failed for "
                f"'{self.get_model_name()}'."
            ) from error


def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    """Create one of the configured backends without silent substitutions."""
    if not isinstance(config, EmbeddingConfig):
        raise TypeError("config must be an EmbeddingConfig.")

    provider = config.provider
    if provider == "bge":
        return BGEEmbeddingModel(config)
    if provider == "huggingface":
        return MiniLMEmbeddingModel(config)
    if provider == "openai":
        return OpenAIEmbeddingModel(config)
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. "
        "Choose 'bge', 'huggingface', or 'openai'."
    )


def embed_chunk_records(
    chunks: Sequence[Mapping[str, Any]],
    embedding_model: EmbeddingModel,
) -> List[Dict[str, Any]]:
    """Copy chunk records and attach vectors without dropping metadata."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of chunk mappings.")
    if not chunks:
        return []

    records = []
    texts = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError(f"Chunk at position {index} must be a mapping.")

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Chunk at position {index} must have a non-empty chunk_id."
            )

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk '{chunk_id}' must have non-empty text."
            )

        records.append(dict(chunk))
        texts.append(text)

    vectors = embedding_model.embed_documents(texts)
    if len(vectors) != len(records):
        raise RuntimeError(
            f"Embedding model returned {len(vectors)} vectors for "
            f"{len(records)} chunks."
        )

    dimension = embedding_model.get_dimension()
    model_name = embedding_model.get_model_name()
    for index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise RuntimeError(
                f"Embedding for chunk '{records[index]['chunk_id']}' "
                f"has dimension {len(vector)}; expected {dimension}."
            )
        records[index]["embedding"] = vector
        records[index]["retrieval_embedding_model"] = model_name
        records[index]["embedding_dimension"] = dimension

    return records
