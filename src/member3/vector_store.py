"""FAISS-backed vector storage for embedded chunk records."""

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .embeddings import EmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = PROJECT_ROOT / "indexes"
_INDEX_FORMAT_VERSION = 1
_NORMALIZATION_TOLERANCE = 1e-4
_NON_CONFIGURATION_FIELDS = {
    "chunk_id",
    "text",
    "embedding",
    "source",
    "document_id",
    "doc_id",
    "filename",
    "page",
    "page_number",
    "start_index",
    "end_index",
    "sentence_start_index",
    "sentence_end_index",
    "retrieval_embedding_model",
    "embedding_dimension",
}


def _load_faiss() -> Any:
    try:
        import faiss
    except ModuleNotFoundError as error:
        if error.name != "faiss":
            raise
        raise ImportError(
            "FAISS is required for vector indexing. Install the declared "
            "faiss-cpu dependency in the active Python environment."
        ) from error
    except ImportError as error:
        raise RuntimeError(
            "FAISS is installed, but its native extension could not be loaded."
        ) from error
    return faiss


def _configuration_for(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in _NON_CONFIGURATION_FIELDS
    }


def _configuration_json(configuration: Mapping[str, Any]) -> str:
    try:
        return json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Chunking configuration must be JSON serializable.") from error


def _corpus_fingerprint(
    records: Sequence[Mapping[str, Any]],
    vectors: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    ordered_records = sorted(
        zip(records, vectors),
        key=lambda item: item[0]["chunk_id"],
    )
    try:
        for record, vector in ordered_records:
            fingerprint_record = dict(record)
            fingerprint_record["embedding"] = vector.tolist()
            serialized = json.dumps(
                fingerprint_record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            digest.update(serialized.encode("utf-8"))
            digest.update(b"\n")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Chunk records must be JSON serializable to identify the corpus."
        ) from error
    return digest.hexdigest()


class FaissVectorStore:
    """A normalized-vector FAISS index bound to one embedding model."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model
        self._index: Any = None
        self._records_by_faiss_id: Dict[int, Dict[str, Any]] = {}
        self._index_info: Optional[Dict[str, Any]] = None
        self._directory: Optional[Path] = None
        self._retrieval_latency_seconds: Optional[float] = None

    def build_index(
        self,
        embedded_chunks: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(embedded_chunks, (str, bytes)) or not isinstance(
            embedded_chunks, Sequence
        ):
            raise TypeError("embedded_chunks must be a sequence of chunk mappings.")
        if not embedded_chunks:
            raise ValueError("Cannot build a FAISS index from empty input.")

        model_name = self._embedding_model.get_model_name()
        dimension = self._embedding_model.get_dimension()
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        started = perf_counter()
        ids = set()
        vectors = []
        records_by_id: Dict[int, Dict[str, Any]] = {}
        first_strategy = None
        first_configuration = None

        for faiss_id, record in enumerate(embedded_chunks):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Chunk at position {faiss_id} must be a mapping."
                )

            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Chunk at position {faiss_id} must have a non-empty chunk_id."
                )
            if chunk_id in ids:
                raise ValueError(f"Duplicate chunk_id in index input: {chunk_id}")
            ids.add(chunk_id)

            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk '{chunk_id}' must have non-empty text.")

            strategy = record.get("strategy")
            if not isinstance(strategy, str) or not strategy.strip():
                raise ValueError(f"Chunk '{chunk_id}' must have a strategy.")

            configuration = _configuration_for(record)
            configuration_json = _configuration_json(configuration)
            if first_strategy is None:
                first_strategy = strategy
                first_configuration = configuration_json
            elif strategy != first_strategy:
                raise ValueError(
                    "One FAISS index cannot mix chunking strategies."
                )
            elif configuration_json != first_configuration:
                raise ValueError(
                    "One FAISS index cannot mix chunking configurations."
                )

            recorded_model = record.get("retrieval_embedding_model")
            if not isinstance(recorded_model, str) or not recorded_model.strip():
                raise ValueError(
                    f"Chunk '{chunk_id}' is missing retrieval embedding model "
                    "provenance."
                )
            if recorded_model != model_name:
                raise ValueError(
                    f"Chunk '{chunk_id}' was embedded with "
                    f"'{recorded_model}', not '{model_name}'."
                )
            recorded_dimension = record.get("embedding_dimension")
            if (
                not isinstance(recorded_dimension, int)
                or isinstance(recorded_dimension, bool)
            ):
                raise ValueError(
                    f"Chunk '{chunk_id}' is missing a valid embedding dimension."
                )
            if recorded_dimension != dimension:
                raise ValueError(
                    f"Chunk '{chunk_id}' records dimension {recorded_dimension}; "
                    f"expected {dimension}."
                )

            vector = self._validated_vector(record, dimension)
            vectors.append(vector)

            stored_record = deepcopy(record)
            stored_record.pop("embedding", None)
            stored_record["retrieval_embedding_model"] = model_name
            stored_record["embedding_dimension"] = dimension
            records_by_id[faiss_id] = stored_record

        corpus_fingerprint = _corpus_fingerprint(
            list(records_by_id.values()),
            vectors,
        )
        faiss = _load_faiss()
        matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
        vector_ids = np.arange(len(vectors), dtype=np.int64)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        index.add_with_ids(matrix, vector_ids)

        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
        identity = {
            "retrieval_embedding_model": model_name,
            "embedding_dimension": dimension,
            "strategy": first_strategy,
            "chunking_configuration": json.loads(first_configuration or "{}"),
            "corpus_fingerprint": corpus_fingerprint,
        }
        index_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        self._index = index
        self._records_by_faiss_id = records_by_id
        self._retrieval_latency_seconds = None
        self._index_info = {
            "index_id": index_id,
            **identity,
            "chunk_count": len(vectors),
            "similarity_metric": "cosine_similarity (normalized inner product)",
            "indexing_time_seconds": perf_counter() - started,
            "retrieval_latency_seconds": None,
        }
        self._directory = (
            INDEX_ROOT
            / model_slug
            / re.sub(r"[^A-Za-z0-9._-]+", "-", first_strategy or "unknown")
            / f"{dimension}-{index_id}"
        )
        return self.get_index_info()

    def save_index(self, directory: Optional[Path] = None) -> Path:
        if self._index is None or self._index_info is None:
            raise RuntimeError("Cannot save an index before build_index().")

        target = Path(directory) if directory is not None else self._directory
        if target is None:
            raise RuntimeError("The default FAISS index directory is unavailable.")

        faiss = _load_faiss()
        target.mkdir(parents=True, exist_ok=True)
        final_index = target / "index.faiss"
        final_metadata = target / "metadata.json"
        with tempfile.TemporaryDirectory(
            prefix=".faiss-staging-",
            dir=str(target),
        ) as staging_name:
            staging = Path(staging_name)
            staged_index = staging / "index.faiss"
            staged_metadata = staging / "metadata.json"
            faiss.write_index(self._index, str(staged_index))

            payload = {
                "format_version": _INDEX_FORMAT_VERSION,
                "index_info": self.get_index_info(),
                "records_by_faiss_id": {
                    str(faiss_id): record
                    for faiss_id, record in self._records_by_faiss_id.items()
                },
            }
            try:
                with staged_metadata.open("w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False, indent=2)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Index metadata must be JSON serializable to save."
                ) from error

            backups = {}
            installed = []
            try:
                for destination in (final_index, final_metadata):
                    if destination.exists():
                        backup = staging / f"{destination.name}.backup"
                        os.replace(str(destination), str(backup))
                        backups[destination] = backup

                for staged, destination in (
                    (staged_index, final_index),
                    (staged_metadata, final_metadata),
                ):
                    os.replace(str(staged), str(destination))
                    installed.append(destination)
            except OSError as error:
                rollback_errors = []
                for destination in reversed(installed):
                    try:
                        if destination.exists():
                            destination.unlink()
                    except OSError as rollback_error:
                        rollback_errors.append(rollback_error)
                for destination, backup in backups.items():
                    try:
                        if backup.exists():
                            os.replace(str(backup), str(destination))
                    except OSError as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    raise RuntimeError(
                        "Could not update the FAISS index pair, and rollback "
                        "could not fully restore the previous files."
                    ) from rollback_errors[0]
                raise RuntimeError(
                    "Could not update the FAISS index pair; previous files "
                    "were restored."
                ) from error

        self._directory = target
        return target

    @classmethod
    def load_index(
        cls,
        directory: Path,
        embedding_model: EmbeddingModel,
    ) -> "FaissVectorStore":
        source = Path(directory)
        index_path = source / "index.faiss"
        metadata_path = source / "metadata.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"FAISS index files were not found in '{source}'."
            )

        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Could not read FAISS metadata from '{metadata_path}'."
            ) from error

        if payload.get("format_version") != _INDEX_FORMAT_VERSION:
            raise ValueError("Unsupported FAISS index metadata format version.")

        info = payload.get("index_info")
        if not isinstance(info, dict):
            raise ValueError("FAISS metadata is missing index_info.")

        model_name = embedding_model.get_model_name()
        dimension = embedding_model.get_dimension()
        if info.get("retrieval_embedding_model") != model_name:
            raise ValueError(
                f"Index uses '{info.get('retrieval_embedding_model')}', "
                f"not '{model_name}'."
            )
        if info.get("embedding_dimension") != dimension:
            raise ValueError(
                f"Index dimension is {info.get('embedding_dimension')}; "
                f"query model dimension is {dimension}."
            )

        records_data = payload.get("records_by_faiss_id")
        if not isinstance(records_data, dict):
            raise ValueError("FAISS metadata is missing the ID-to-record mapping.")
        records_by_id = {}
        try:
            for faiss_id, record in records_data.items():
                numeric_id = int(faiss_id)
                if not isinstance(record, dict):
                    raise ValueError("FAISS metadata contains an invalid record.")
                records_by_id[numeric_id] = record
        except (TypeError, ValueError) as error:
            raise ValueError("FAISS metadata contains invalid numeric IDs.") from error

        faiss = _load_faiss()
        try:
            index = faiss.read_index(str(index_path))
        except Exception as error:
            raise RuntimeError(f"Could not load FAISS index '{index_path}'.") from error

        if not isinstance(index, faiss.IndexIDMap2):
            raise ValueError("Stored FAISS index must be an IndexIDMap2.")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                "Stored FAISS index must use the inner-product similarity metric."
            )
        base_index = faiss.downcast_index(index.index)
        if not isinstance(base_index, faiss.IndexFlatIP):
            raise ValueError("Stored FAISS index must wrap an IndexFlatIP.")

        if index.d != dimension:
            raise ValueError(
                f"Stored FAISS dimension is {index.d}; metadata says {dimension}."
            )
        if index.ntotal != len(records_by_id):
            raise ValueError(
                "FAISS index size does not match the stored metadata mapping."
            )

        stored_ids = set(faiss.vector_to_array(index.id_map).tolist())
        if stored_ids != set(records_by_id):
            raise ValueError("FAISS IDs do not match the metadata mapping.")

        store = cls(embedding_model)
        store._index = index
        store._records_by_faiss_id = records_by_id
        store._index_info = info
        store._directory = source
        store._retrieval_latency_seconds = info.get("retrieval_latency_seconds")
        return store

    def search(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> List[Dict[str, Any]]:
        if self._index is None or self._index_info is None:
            raise RuntimeError("Cannot search before an index has been built or loaded.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_model = embedding_model or self._embedding_model
        expected_model = self._index_info["retrieval_embedding_model"]
        expected_dimension = self._index_info["embedding_dimension"]
        if query_model.get_model_name() != expected_model:
            raise ValueError(
                f"Index requires embedding model '{expected_model}', "
                f"not '{query_model.get_model_name()}'."
            )
        if query_model.get_dimension() != expected_dimension:
            raise ValueError(
                f"Index requires dimension {expected_dimension}, "
                f"not {query_model.get_dimension()}."
            )

        started = perf_counter()
        try:
            query_vector = np.asarray(
                query_model.embed_query(query),
                dtype=np.float32,
            )
            if query_vector.shape != (expected_dimension,):
                raise ValueError(
                    f"Query vector has shape {query_vector.shape}; expected "
                    f"({expected_dimension},)."
                )
            if not np.isfinite(query_vector).all():
                raise ValueError("Query vector contains NaN or infinity.")

            norm = float(np.linalg.norm(query_vector))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Query vector must have a finite, non-zero norm.")
            if abs(norm - 1.0) > _NORMALIZATION_TOLERANCE:
                raise ValueError("Query vector must be L2-normalized.")

            scores, faiss_ids = self._index.search(
                np.ascontiguousarray(query_vector.reshape(1, -1)),
                min(top_k, self._index.ntotal),
            )
            results = []
            for rank, (score, faiss_id) in enumerate(
                zip(scores[0], faiss_ids[0]),
                start=1,
            ):
                if faiss_id < 0:
                    continue
                record = self._records_by_faiss_id[int(faiss_id)]
                results.append({
                    "rank": rank,
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "metadata": deepcopy(record),
                    "similarity_score": max(-1.0, min(1.0, float(score))),
                })
            return results
        finally:
            self._retrieval_latency_seconds = perf_counter() - started
            self._index_info["retrieval_latency_seconds"] = (
                self._retrieval_latency_seconds
            )

    def get_index_info(self) -> Dict[str, Any]:
        if self._index_info is None:
            raise RuntimeError("Index information is unavailable before indexing.")
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
        if not np.isfinite(norm) or abs(norm - 1.0) > _NORMALIZATION_TOLERANCE:
            raise ValueError(
                f"Chunk '{chunk_id}' embedding must be L2-normalized."
            )
        return vector
