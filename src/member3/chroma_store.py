"""ChromaDB-backed vector storage using the shared embedding interface."""

import hashlib
import json
import logging
import re
import uuid
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .embeddings import EmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_ROOT = PROJECT_ROOT / "indexes" / "chroma"
_DEFAULT_BATCH_SIZE = 256
_NORMALIZATION_TOLERANCE = 1e-4
_RECORD_METADATA_KEY = "_member3_record_json"
_CHUNKING_CONFIGURATION_FIELDS = {
    "fixed_size": {"chunk_size", "overlap"},
    "sentence_level": {"target_size", "overlap_sentences"},
    "semantic": {
        "target_size",
        "min_size",
        "similarity_threshold",
        "embedding_model",
        "overlap_sentences",
    },
}
_LOGGER = logging.getLogger(__name__)


def _load_chromadb() -> Any:
    try:
        import chromadb
    except ModuleNotFoundError as error:
        if error.name != "chromadb":
            raise
        raise ImportError(
            "ChromaDB is required for this vector store. Install the "
            "chromadb dependency from requirements.txt."
        ) from error
    return chromadb


def _stable_json(value: Any, description: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be JSON serializable.") from error


def _configuration_for(record: Mapping[str, Any]) -> Dict[str, Any]:
    fields = _CHUNKING_CONFIGURATION_FIELDS.get(record.get("strategy"), set())
    return {
        key: record[key]
        for key in fields
        if key in record
    }


def _metadata_for_modify(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key != "hnsw:space"
    }


def _corpus_fingerprint(
    records: Sequence[Mapping[str, Any]],
    vectors: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    ordered_records = sorted(
        zip(records, vectors),
        key=lambda item: item[0]["chunk_id"],
    )
    for record, vector in ordered_records:
        serialized = _stable_json(
            {
                **record,
                "embedding": vector.tolist(),
            },
            "Chunk records",
        )
        digest.update(serialized.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ChromaVectorStore:
    """Persistent cosine-similarity collection bound to one embedder."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_directory: Optional[Path] = None,
        collection_prefix: str = "member3",
        client: Optional[Any] = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._persist_directory = (
            Path(persist_directory) if persist_directory is not None else CHROMA_ROOT
        )
        if not isinstance(collection_prefix, str):
            raise TypeError("collection_prefix must be a string.")
        prefix = re.sub(r"[^a-z0-9]+", "-", collection_prefix.lower()).strip("-")
        if not prefix or len(prefix) > 24:
            raise ValueError(
                "collection_prefix must normalize to 1-24 lowercase "
                "letters, digits, dots, underscores or hyphens."
            )
        self._collection_prefix = prefix
        self._client = client
        self._collection: Optional[Any] = None
        self._index_info: Optional[Dict[str, Any]] = None
        self._retrieval_latency_seconds: Optional[float] = None

    def _get_client(self) -> Any:
        if self._client is None:
            chromadb = _load_chromadb()
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_directory)
            )
        return self._client

    def build_index(
        self,
        embedded_chunks: Sequence[Mapping[str, Any]],
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> Dict[str, Any]:
        if isinstance(embedded_chunks, (str, bytes)) or not isinstance(
            embedded_chunks, Sequence
        ):
            raise TypeError("embedded_chunks must be a sequence of chunk mappings.")
        if not embedded_chunks:
            raise ValueError("Cannot build a Chroma collection from empty input.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        model_name = self._embedding_model.get_model_name()
        dimension = self._embedding_model.get_dimension()
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        started = perf_counter()
        seen_ids = set()
        records: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        first_strategy: Optional[str] = None
        first_configuration_json: Optional[str] = None

        for position, input_record in enumerate(embedded_chunks):
            if not isinstance(input_record, Mapping):
                raise TypeError(
                    f"Chunk at position {position} must be a mapping."
                )
            record = deepcopy(dict(input_record))
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Chunk at position {position} must have a non-empty chunk_id."
                )
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in collection input: {chunk_id}")
            seen_ids.add(chunk_id)

            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk '{chunk_id}' must have non-empty text.")
            strategy = record.get("strategy")
            if not isinstance(strategy, str) or not strategy.strip():
                raise ValueError(f"Chunk '{chunk_id}' must have a strategy.")

            configuration_json = _stable_json(
                _configuration_for(record),
                "Chunking configuration",
            )
            if first_strategy is None:
                first_strategy = strategy
                first_configuration_json = configuration_json
            elif strategy != first_strategy:
                raise ValueError("One Chroma collection cannot mix chunking strategies.")
            elif configuration_json != first_configuration_json:
                raise ValueError(
                    "One Chroma collection cannot mix chunking configurations."
                )

            recorded_model = record.get("retrieval_embedding_model")
            if recorded_model != model_name:
                raise ValueError(
                    f"Chunk '{chunk_id}' has retrieval embedding model "
                    f"'{recorded_model}', expected '{model_name}'."
                )
            recorded_dimension = record.get("embedding_dimension")
            if (
                not isinstance(recorded_dimension, int)
                or isinstance(recorded_dimension, bool)
                or recorded_dimension != dimension
            ):
                raise ValueError(
                    f"Chunk '{chunk_id}' has embedding dimension "
                    f"'{recorded_dimension}', expected {dimension}."
                )

            vector = self._validated_vector(record, dimension)
            vectors.append(vector)
            record.pop("embedding", None)
            records.append(record)

        serialized_records = [
            _stable_json(record, "Chunk metadata")
            for record in records
        ]
        configuration = json.loads(first_configuration_json or "{}")
        corpus_fingerprint = _corpus_fingerprint(records, vectors)
        identity = {
            "retrieval_embedding_model": model_name,
            "embedding_dimension": dimension,
            "strategy": first_strategy,
            "chunking_configuration": configuration,
            "corpus_fingerprint": corpus_fingerprint,
        }
        index_id = hashlib.sha256(
            _stable_json(identity, "Collection identity").encode("utf-8")
        ).hexdigest()
        collection_name = f"{self._collection_prefix}-{index_id[:32]}"
        collection_metadata = {
            "index_id": index_id,
            "retrieval_embedding_model": model_name,
            "embedding_dimension": dimension,
            "strategy": first_strategy or "",
            "chunking_configuration_json": _stable_json(
                configuration,
                "Chunking configuration",
            ),
            "corpus_fingerprint": corpus_fingerprint,
            "hnsw:space": "cosine",
            "similarity_metric": "cosine",
            "build_status": "building",
            "expected_record_count": len(records),
        }

        client = self._get_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata,
            embedding_function=None,
        )
        cleanup_incomplete = False
        try:
            existing_metadata = collection.metadata or {}
            if existing_metadata.get("index_id") != index_id:
                raise ValueError(
                    f"Chroma collection '{collection_name}' has incompatible "
                    "identity."
                )
            existing_status = existing_metadata.get("build_status")
            if existing_status == "complete":
                raise ValueError(
                    f"Chroma collection '{collection_name}' is already complete. "
                    "Reopen it with load_index() instead of building it again."
                )
            if existing_status != "building":
                raise ValueError(
                    f"Chroma collection '{collection_name}' is not marked as "
                    "building; refusing to reuse it."
                )
            cleanup_incomplete = True

            if (
                existing_metadata.get("expected_record_count") != len(records)
                or collection.count() != 0
            ):
                client.delete_collection(name=collection_name)
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata=collection_metadata,
                    embedding_function=None,
                )
                existing_metadata = collection.metadata or {}
                if (
                    existing_metadata.get("index_id") != index_id
                    or existing_metadata.get("build_status") != "building"
                    or existing_metadata.get("expected_record_count") != len(records)
                    or collection.count() != 0
                ):
                    raise RuntimeError(
                        f"Could not prepare Chroma collection '{collection_name}' "
                        "for a clean build."
                    )

            for start in range(0, len(records), batch_size):
                record_batch = records[start:start + batch_size]
                vector_batch = vectors[start:start + batch_size]
                collection.add(
                    ids=[record["chunk_id"] for record in record_batch],
                    documents=[record["text"] for record in record_batch],
                    embeddings=[vector.tolist() for vector in vector_batch],
                    metadatas=[
                        {
                            _RECORD_METADATA_KEY: serialized_records[start + offset]
                        }
                        for offset, _ in enumerate(record_batch)
                    ],
                )

            actual_count = collection.count()
            if actual_count != len(records):
                raise RuntimeError(
                    f"Chroma collection '{collection_name}' contains "
                    f"{actual_count} records; expected {len(records)}."
                )

            completed_metadata = _metadata_for_modify(collection.metadata or {})
            completed_metadata.update(
                build_status="complete",
                expected_record_count=len(records),
            )
            collection.modify(metadata=completed_metadata)
        except Exception:
            if cleanup_incomplete:
                try:
                    incomplete_metadata = _metadata_for_modify(
                        collection.metadata or {}
                    )
                    incomplete_metadata.update(
                        build_status="building",
                        expected_record_count=len(records),
                    )
                    collection.modify(metadata=incomplete_metadata)
                except Exception:
                    _LOGGER.warning(
                        "Could not mark failed Chroma build as incomplete.",
                        extra={
                            "store_name": "ChromaVectorStore",
                            "collection_name": collection_name,
                        },
                        exc_info=True,
                    )
                try:
                    client.delete_collection(name=collection_name)
                except Exception:
                    _LOGGER.warning(
                        "Could not delete failed Chroma build; it remains marked "
                        "as incomplete when metadata updates are available.",
                        extra={
                            "store_name": "ChromaVectorStore",
                            "collection_name": collection_name,
                        },
                        exc_info=True,
                    )
            raise

        indexing_time = perf_counter() - started
        self._collection = collection
        self._index_info = {
            "collection_name": collection_name,
            "persist_directory": str(self._persist_directory),
            "index_id": index_id,
            **identity,
            "chunk_count": len(records),
            "similarity_metric": "cosine",
            "indexing_time_seconds": indexing_time,
            "retrieval_latency_seconds": None,
        }
        self._retrieval_latency_seconds = None
        return self.get_index_info()

    def save_index(self) -> Path:
        if self._collection is None or self._index_info is None:
            raise RuntimeError("Cannot save before build_index() or load_index().")
        client = self._get_client()
        get_settings = getattr(client, "get_settings", None)
        settings = get_settings() if callable(get_settings) else None
        if not getattr(settings, "is_persistent", False):
            raise RuntimeError(
                "save_index() requires a persistent Chroma client; the supplied "
                "client does not report persistent storage."
            )
        return self._persist_directory

    @classmethod
    def load_index(
        cls,
        persist_directory: Path,
        collection_name: str,
        embedding_model: EmbeddingModel,
        collection_prefix: str = "member3",
        client: Optional[Any] = None,
    ) -> "ChromaVectorStore":
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")

        store = cls(
            embedding_model,
            persist_directory=persist_directory,
            collection_prefix=collection_prefix,
            client=client,
        )
        try:
            collection = store._get_client().get_collection(
                name=collection_name,
                embedding_function=None,
            )
        except KeyError as error:
            raise FileNotFoundError(
                f"Chroma collection '{collection_name}' not found in "
                f"'{store._persist_directory}'."
            ) from error
        except Exception as error:
            chromadb = _load_chromadb()
            if isinstance(error, chromadb.errors.NotFoundError):
                raise FileNotFoundError(
                    f"Chroma collection '{collection_name}' was not found in "
                    f"'{store._persist_directory}'."
                ) from error
            raise

        metadata = collection.metadata or {}
        model_name = embedding_model.get_model_name()
        dimension = embedding_model.get_dimension()
        if metadata.get("retrieval_embedding_model") != model_name:
            raise ValueError(
                f"Collection uses '{metadata.get('retrieval_embedding_model')}', "
                f"not '{model_name}'."
            )
        if metadata.get("embedding_dimension") != dimension:
            raise ValueError(
                f"Collection dimension is {metadata.get('embedding_dimension')}; "
                f"query model dimension is {dimension}."
            )
        configuration_settings = getattr(collection, "configuration", {})
        hnsw_configuration = (
            configuration_settings.get("hnsw", {})
            if isinstance(configuration_settings, Mapping)
            else {}
        )
        if (
            metadata.get("similarity_metric") != "cosine"
            or hnsw_configuration.get("space") != "cosine"
        ):
            raise ValueError("Chroma collection must use cosine similarity.")
        if metadata.get("build_status") != "complete":
            raise ValueError(
                f"Chroma collection '{collection_name}' is not marked complete."
            )
        expected_count = metadata.get("expected_record_count")
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count <= 0
        ):
            raise ValueError(
                f"Chroma collection '{collection_name}' has an invalid expected "
                "record count."
            )
        actual_count = collection.count()
        if actual_count != expected_count:
            raise ValueError(
                f"Chroma collection '{collection_name}' contains {actual_count} "
                f"records; expected {expected_count}."
            )

        try:
            configuration = json.loads(
                metadata["chunking_configuration_json"]
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Chroma collection has invalid chunking configuration metadata."
            ) from error

        store._collection = collection
        store._index_info = {
            "collection_name": collection_name,
            "persist_directory": str(store._persist_directory),
            "index_id": metadata.get("index_id"),
            "retrieval_embedding_model": model_name,
            "embedding_dimension": dimension,
            "strategy": metadata.get("strategy"),
            "chunking_configuration": configuration,
            "corpus_fingerprint": metadata.get("corpus_fingerprint"),
            "chunk_count": actual_count,
            "similarity_metric": "cosine",
            "indexing_time_seconds": None,
            "retrieval_latency_seconds": None,
        }
        return store

    def search(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection is None or self._index_info is None:
            raise RuntimeError("Cannot search before build_index() or load_index().")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_model = embedding_model or self._embedding_model
        expected_model = self._index_info["retrieval_embedding_model"]
        expected_dimension = self._index_info["embedding_dimension"]
        if query_model.get_model_name() != expected_model:
            raise ValueError(
                f"Collection requires embedding model '{expected_model}', "
                f"not '{query_model.get_model_name()}'."
            )
        if query_model.get_dimension() != expected_dimension:
            raise ValueError(
                f"Collection requires dimension {expected_dimension}, "
                f"not {query_model.get_dimension()}."
            )

        started = perf_counter()
        query_id = uuid.uuid4().hex
        results: List[Dict[str, Any]] = []
        try:
            query_vector = np.asarray(
                query_model.embed_query(query),
                dtype=np.float32,
            )
            self._validate_query_vector(query_vector, expected_dimension)
            count = self._collection.count()
            if count == 0:
                return []

            response = self._collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            ids = response.get("ids", [[]])[0]
            documents = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0]
            for rank, (chunk_id, text, stored_metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            ):
                if not isinstance(stored_metadata, Mapping):
                    raise RuntimeError(
                        f"Chroma returned invalid metadata for chunk '{chunk_id}'."
                    )
                serialized_record = stored_metadata.get(_RECORD_METADATA_KEY)
                try:
                    record = json.loads(serialized_record)
                except (TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"Chroma returned invalid stored metadata for "
                        f"chunk '{chunk_id}'."
                    ) from error
                if not isinstance(record, dict) or record.get("chunk_id") != chunk_id:
                    raise RuntimeError(
                        f"Chroma ID and stored chunk ID differ for '{chunk_id}'."
                    )
                if not isinstance(distance, (int, float, np.number)) or not np.isfinite(distance):
                    raise RuntimeError(
                        f"Chroma returned an invalid cosine distance for '{chunk_id}'."
                    )
                similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
                results.append({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": deepcopy(record),
                    "similarity_score": similarity,
                })
            return results
        finally:
            self._retrieval_latency_seconds = perf_counter() - started
            self._index_info["retrieval_latency_seconds"] = (
                self._retrieval_latency_seconds
            )
            _LOGGER.info(
                "retrieval completed",
                extra={
                    "store_name": "ChromaVectorStore",
                    "query_id": query_id,
                    "chunk_ids": [result["chunk_id"] for result in results],
                    "similarity_scores": [
                        result["similarity_score"] for result in results
                    ],
                    "embedding_model": expected_model,
                    "top_k": top_k,
                    "elapsed_seconds": self._retrieval_latency_seconds,
                },
            )

    def get_index_info(self) -> Dict[str, Any]:
        if self._index_info is None:
            raise RuntimeError("Collection information is unavailable before indexing.")
        return deepcopy(self._index_info)

    @staticmethod
    def _validated_vector(
        record: Mapping[str, Any],
        dimension: int,
    ) -> np.ndarray:
        chunk_id = record.get("chunk_id", "unknown")
        try:
            vector = np.asarray(record.get("embedding"), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Chunk '{chunk_id}' has an invalid embedding vector."
            ) from error
        if vector.shape != (dimension,):
            raise ValueError(
                f"Chunk '{chunk_id}' has embedding shape {vector.shape}; "
                f"expected ({dimension},)."
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Chunk '{chunk_id}' embedding is not finite.")
        norm = float(np.linalg.norm(vector))
        if (
            not np.isfinite(norm)
            or abs(norm - 1.0) > _NORMALIZATION_TOLERANCE
        ):
            raise ValueError(
                f"Chunk '{chunk_id}' embedding must be L2-normalized."
            )
        return vector

    @staticmethod
    def _validate_query_vector(vector: np.ndarray, dimension: int) -> None:
        if vector.shape != (dimension,):
            raise ValueError(
                f"Query vector has shape {vector.shape}; expected ({dimension},)."
            )
        if not np.isfinite(vector).all():
            raise ValueError("Query vector contains NaN or infinity.")
        norm = float(np.linalg.norm(vector))
        if (
            not np.isfinite(norm)
            or norm == 0
            or abs(norm - 1.0) > _NORMALIZATION_TOLERANCE
        ):
            raise ValueError("Query vector must have a finite, unit L2 norm.")
