"""Run a reproducible retrieval comparison over existing chunk JSON outputs."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .chroma_store import ChromaVectorStore
from .embeddings import (
    BGE_MODEL_NAME,
    MINILM_MODEL_NAME,
    OPENAI_DEFAULT_MODEL_NAME,
    EmbeddingConfig,
    EmbeddingModel,
    create_embedding_model,
    embed_chunk_records,
)
from .vector_store import FaissVectorStore, _load_faiss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_FILE = PROJECT_ROOT / "data" / "retrieval_experiment_queries.json"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "retrieval_experiments"
CHROMA_ROOT = PROJECT_ROOT / "indexes" / "chroma"
TOP_K = 5

_STRATEGY_ORDER = ("fixed_size", "sentence_level", "semantic")
_STRATEGY_INPUTS = {
    "fixed_size": ("chunks", "_fixed_*.json"),
    "sentence_level": ("sentence_chunks", "_sentence_*.json"),
    "semantic": ("semantic_chunks", "_semantic_*.json"),
}
_PARAMETER_FIELDS = {
    "fixed_size": ("chunk_size", "overlap"),
    "sentence_level": ("target_size", "overlap_sentences"),
    "semantic": (
        "target_size",
        "min_size",
        "similarity_threshold",
        "embedding_model",
        "overlap_sentences",
    ),
}
_MODEL_CONFIGS = {
    "bge": EmbeddingConfig(provider="bge", model_name=BGE_MODEL_NAME),
    "minilm": EmbeddingConfig(provider="huggingface", model_name=MINILM_MODEL_NAME),
}


@dataclass(frozen=True)
class ChunkDataset:
    strategy: str
    configuration: Dict[str, Any]
    files: Tuple[Path, ...]
    source_file_hashes: Dict[str, str]
    sources: Tuple[str, ...]
    chunk_count: int
    configuration_id: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _configuration_for(strategy: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    fields = _PARAMETER_FIELDS[strategy]
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(
            f"{strategy} chunk is missing recorded configuration fields: {missing}"
        )
    return {field: record[field] for field in fields}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_chunk_datasets(project_root: Path = PROJECT_ROOT) -> List[ChunkDataset]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for strategy in _STRATEGY_ORDER:
        folder_name, pattern = _STRATEGY_INPUTS[strategy]
        files = sorted((project_root / "data" / folder_name).glob(f"*{pattern}"))
        if not files:
            raise FileNotFoundError(
                f"No existing chunk JSON files found for strategy '{strategy}'."
            )

        for path in files:
            with path.open("r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list) or not records:
                raise ValueError(f"Chunk output '{path}' must contain a non-empty list.")

            configuration: Optional[Dict[str, Any]] = None
            sources = set()
            ids = set()
            for record in records:
                if not isinstance(record, dict) or record.get("strategy") != strategy:
                    raise ValueError(
                        f"Chunk output '{path}' contains an invalid {strategy} record."
                    )
                observed = _configuration_for(strategy, record)
                if configuration is None:
                    configuration = observed
                elif observed != configuration:
                    raise ValueError(
                        f"Chunk output '{path}' mixes chunking configurations."
                    )
                chunk_id = record.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(f"Chunk output '{path}' has an invalid chunk_id.")
                if chunk_id in ids:
                    raise ValueError(
                        f"Chunk output '{path}' contains duplicate ID '{chunk_id}'."
                    )
                ids.add(chunk_id)
                sources.add(record.get("source"))

            assert configuration is not None
            key = _canonical_json({"strategy": strategy, "configuration": configuration})
            state = grouped.setdefault(
                key,
                {
                    "strategy": strategy,
                    "configuration": configuration,
                    "files": [],
                    "file_hashes": {},
                    "sources": set(),
                    "chunk_count": 0,
                },
            )
            relative_path = str(path.relative_to(project_root))
            state["files"].append(path)
            state["file_hashes"][relative_path] = _file_sha256(path)
            state["sources"].update(source for source in sources if source)
            state["chunk_count"] += len(records)

    datasets = []
    for state in grouped.values():
        file_hashes = dict(sorted(state["file_hashes"].items()))
        identity = {
            "strategy": state["strategy"],
            "configuration": state["configuration"],
            "source_file_hashes": file_hashes,
        }
        config_digest = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()[:12]
        datasets.append(
            ChunkDataset(
                strategy=state["strategy"],
                configuration=state["configuration"],
                files=tuple(sorted(state["files"])),
                source_file_hashes=file_hashes,
                sources=tuple(sorted(state["sources"])),
                chunk_count=state["chunk_count"],
                configuration_id=f"{state['strategy']}-{config_digest}",
            )
        )

    return sorted(
        datasets,
        key=lambda item: (
            _STRATEGY_ORDER.index(item.strategy),
            item.configuration.get("chunk_size", item.configuration.get("target_size", 0)),
        ),
    )


def load_chunk_records(dataset: ChunkDataset) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for path in dataset.files:
        with path.open("r", encoding="utf-8") as file:
            file_records = json.load(file)
        for record in file_records:
            if record["chunk_id"] in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID '{record['chunk_id']}' across chunk files."
                )
            seen_ids.add(record["chunk_id"])
            records.append(record)
    if len(records) != dataset.chunk_count:
        raise ValueError(
            f"Loaded {len(records)} chunks; inventory recorded {dataset.chunk_count}."
        )
    return records


def load_queries(query_file: Path = QUERY_FILE) -> Dict[str, Any]:
    with query_file.open("r", encoding="utf-8") as file:
        query_set = json.load(file)
    queries = query_set.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("The query set must contain a non-empty queries array.")
    query_ids = [item.get("query_id") for item in queries]
    if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
        raise ValueError("Every query must have a non-empty query_id.")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Query IDs must be unique.")
    if any(not isinstance(item.get("query"), str) or not item["query"].strip() for item in queries):
        raise ValueError("Every query must contain non-empty query text.")
    return query_set


def make_retrieval_record(
    *,
    run_id: str,
    experiment_id: str,
    dataset: ChunkDataset,
    embedding_model: str,
    vector_store: str,
    query: Mapping[str, str],
    results: Sequence[Mapping[str, Any]],
    retrieval_latency_seconds: float,
    indexed_chunk_count: int,
    repeatability_status: str,
    repeat_latency_seconds: float,
    warnings: Sequence[str],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "chunking_strategy": dataset.strategy,
        "chunking_configuration": dataset.configuration,
        "embedding_model": embedding_model,
        "vector_store": vector_store,
        "query_id": query["query_id"],
        "retrieved_chunk_ids": [result["chunk_id"] for result in results],
        "similarity_scores": [float(result["similarity_score"]) for result in results],
        "top_k": TOP_K,
        "retrieval_latency_seconds": retrieval_latency_seconds,
        "indexed_chunk_count": indexed_chunk_count,
        "repeatability_status": repeatability_status,
        "repeat_latency_seconds": repeat_latency_seconds,
        "validation_status": "passed",
        "warnings": list(warnings),
    }


def _create_run_directory(output_root: Path) -> Tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        run_directory = output_root / f"run_{run_id}"
        try:
            run_directory.mkdir()
        except FileExistsError:
            continue
        return run_id, run_directory


def _write_json(path: Path, data: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _existing_index(
    output_root: Path,
    experiment_id: str,
    source_file_hashes: Mapping[str, str],
    embedding_model: str,
    vector_store: str,
) -> Optional[Dict[str, Any]]:
    summaries = sorted(
        output_root.glob("run_*/configuration_summary.json"),
        reverse=True,
    )
    for path in summaries:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for configuration in summary.get("configurations", []):
            if (
                configuration.get("status") == "success"
                and configuration.get("experiment_id") == experiment_id
                and configuration.get("source_file_hashes") == source_file_hashes
                and configuration.get("embedding_model") == embedding_model
                and configuration.get("vector_store") == vector_store
                and configuration.get("index_info")
            ):
                return configuration
    return None


def _query_results_are_repeatable(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> bool:
    if [item["chunk_id"] for item in first] != [
        item["chunk_id"] for item in second
    ]:
        return False
    return all(
        math.isclose(
            float(left["similarity_score"]),
            float(right["similarity_score"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        for left, right in zip(first, second)
    )


def _validate_results(
    results: Sequence[Mapping[str, Any]],
    expected_count: int,
    embedding_model: str,
) -> None:
    if len(results) != min(TOP_K, expected_count):
        raise RuntimeError(
            f"Expected {min(TOP_K, expected_count)} results, received {len(results)}."
        )
    required_keys = {"rank", "chunk_id", "text", "metadata", "similarity_score"}
    for rank, result in enumerate(results, start=1):
        if set(result) != required_keys or result["rank"] != rank:
            raise RuntimeError("The vector store returned an incompatible result shape.")
        if result["metadata"].get("retrieval_embedding_model") != embedding_model:
            raise RuntimeError("A retrieved chunk has mismatched embedding provenance.")
        score = float(result["similarity_score"])
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise RuntimeError("A retrieved similarity score is invalid.")


def _probe_faiss() -> Tuple[bool, Optional[str]]:
    try:
        _load_faiss()
    except Exception as error:
        cause = error.__cause__
        cause_message = f" Cause: {cause}" if cause is not None else ""
        return False, f"{type(error).__name__}: {error}.{cause_message}"
    return True, None


def _embedding_configurations(include_openai: bool) -> Tuple[Dict[str, EmbeddingConfig], List[Dict[str, str]]]:
    configurations = dict(_MODEL_CONFIGS)
    skipped = []
    if not os.environ.get("OPENAI_API_KEY"):
        skipped.append({
            "embedding_model": OPENAI_DEFAULT_MODEL_NAME,
            "status": "skipped",
            "reason": "OPENAI_API_KEY is not configured.",
        })
    elif importlib.util.find_spec("openai") is None:
        skipped.append({
            "embedding_model": OPENAI_DEFAULT_MODEL_NAME,
            "status": "skipped",
            "reason": "The OpenAI Python package is not installed.",
        })
    elif not include_openai:
        skipped.append({
            "embedding_model": OPENAI_DEFAULT_MODEL_NAME,
            "status": "skipped",
            "reason": "Explicit --include-openai consent is required before paid API calls.",
        })
    else:
        configurations["openai"] = EmbeddingConfig(
            provider="openai",
            model_name=OPENAI_DEFAULT_MODEL_NAME,
        )
    return configurations, skipped


def _run_store_configuration(
    *,
    run_id: str,
    model_alias: str,
    model: EmbeddingModel,
    dataset: ChunkDataset,
    vector_store: str,
    store_class: Any,
    queries: Sequence[Mapping[str, str]],
    output_root: Path,
    run_directory: Path,
    project_root: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records = load_chunk_records(dataset)
    model_name = model.get_model_name()
    experiment_id = f"{dataset.configuration_id}-{model_alias}-{vector_store.lower()}"
    previous = _existing_index(
        output_root,
        experiment_id,
        dataset.source_file_hashes,
        model_name,
        vector_store,
    )

    embedding_time = None
    persistence_status = "built_and_persisted"
    if previous:
        index_info_before = previous["index_info"]
        if vector_store == "ChromaDB":
            store = ChromaVectorStore.load_index(
                project_root / "indexes" / "chroma",
                index_info_before["collection_name"],
                model,
            )
        else:
            store = FaissVectorStore.load_index(
                Path(previous["index_path"]),
                model,
            )
        index_info = store.get_index_info()
        if (
            index_info.get("index_id") != index_info_before.get("index_id")
            or index_info.get("chunk_count") != dataset.chunk_count
        ):
            raise RuntimeError("Previously saved index does not match the current corpus.")
        persistence_status = "loaded_existing"
    else:
        embedded_records = embed_chunk_records(records, model)
        run_stats = model.get_last_run_stats()
        embedding_time = (
            run_stats.processing_time_seconds if run_stats is not None else None
        )
        if vector_store == "ChromaDB":
            prefix = f"exp-{run_id[-8:]}-{model_alias}"
            store = store_class(
                model,
                persist_directory=project_root / "indexes" / "chroma",
                collection_prefix=prefix,
            )
        else:
            store = store_class(model)
        store.build_index(embedded_records)
        save_path = store.save_index()
        index_info = store.get_index_info()

    retrieval_records = []
    config_warnings = []
    for query in queries:
        first_results = store.search(query["query"], top_k=TOP_K)
        first_latency = store.get_index_info()["retrieval_latency_seconds"]
        second_results = store.search(query["query"], top_k=TOP_K)
        second_latency = store.get_index_info()["retrieval_latency_seconds"]
        _validate_results(first_results, dataset.chunk_count, model_name)
        _validate_results(second_results, dataset.chunk_count, model_name)
        repeatable = _query_results_are_repeatable(first_results, second_results)
        warnings = [] if repeatable else [
            "Repeated retrieval returned different chunk IDs or scores."
        ]
        config_warnings.extend(warnings)
        retrieval_records.append(
            make_retrieval_record(
                run_id=run_id,
                experiment_id=experiment_id,
                dataset=dataset,
                embedding_model=model_name,
                vector_store=vector_store,
                query=query,
                results=first_results,
                retrieval_latency_seconds=float(first_latency),
                indexed_chunk_count=dataset.chunk_count,
                repeatability_status="passed" if repeatable else "warning",
                repeat_latency_seconds=float(second_latency),
                warnings=warnings,
            )
        )

    summary = {
        "experiment_id": experiment_id,
        "chunking_strategy": dataset.strategy,
        "chunking_configuration": dataset.configuration,
        "source_chunk_files": [
            str(path.relative_to(project_root)) for path in dataset.files
        ],
        "source_file_hashes": dataset.source_file_hashes,
        "source_documents": list(dataset.sources),
        "number_of_chunks": dataset.chunk_count,
        "embedding_model": model_name,
        "embedding_dimension": model.get_dimension(),
        "embedding_processing_time_seconds": embedding_time,
        "vector_store": vector_store,
        "index_build_time_seconds": (
            previous.get("index_build_time_seconds")
            if previous
            else index_info.get("indexing_time_seconds")
        ),
        "index_persistence_status": persistence_status,
        "index_info": index_info,
        "index_path": (
            str(save_path) if not previous and vector_store == "FAISS"
            else previous.get("index_path") if previous else None
        ),
        "collection_name": index_info.get("collection_name"),
        "retrieval_status": "success",
        "repeatability_status": (
            "passed" if not config_warnings else "warning"
        ),
        "warnings": sorted(set(config_warnings)),
        "status": "success",
    }
    return summary, retrieval_records


def _analyze_results(
    configurations: Sequence[Mapping[str, Any]],
    retrieval_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    records_by_experiment: Dict[str, List[Mapping[str, Any]]] = {}
    for record in retrieval_records:
        records_by_experiment.setdefault(record["experiment_id"], []).append(record)

    metrics = []
    successful = [item for item in configurations if item.get("status") == "success"]
    for configuration in successful:
        records = records_by_experiment.get(configuration["experiment_id"], [])
        latencies = [float(item["retrieval_latency_seconds"]) for item in records]
        scores = [
            score
            for item in records
            for score in item["similarity_scores"]
        ]
        metrics.append({
            "experiment_id": configuration["experiment_id"],
            "chunking_strategy": configuration["chunking_strategy"],
            "chunking_configuration": configuration["chunking_configuration"],
            "embedding_model": configuration["embedding_model"],
            "vector_store": configuration["vector_store"],
            "number_of_chunks": configuration["number_of_chunks"],
            "index_build_time_seconds": configuration["index_build_time_seconds"],
            "mean_retrieval_latency_seconds": statistics.mean(latencies) if latencies else None,
            "median_retrieval_latency_seconds": statistics.median(latencies) if latencies else None,
            "similarity_score_distribution": (
                {
                    "count": len(scores),
                    "min": min(scores),
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "max": max(scores),
                }
                if scores else None
            ),
        })

    overlap_comparisons = []
    for left_index, left in enumerate(successful):
        for right in successful[left_index + 1:]:
            if (
                left["chunking_strategy"] != right["chunking_strategy"]
                or left["chunking_configuration"]
                != right["chunking_configuration"]
                or left.get("source_file_hashes")
                != right.get("source_file_hashes")
            ):
                continue
            left_records = {
                item["query_id"]: set(item["retrieved_chunk_ids"])
                for item in records_by_experiment.get(left["experiment_id"], [])
            }
            right_records = {
                item["query_id"]: set(item["retrieved_chunk_ids"])
                for item in records_by_experiment.get(right["experiment_id"], [])
            }
            overlaps = []
            exact_ranked_matches = []
            for query_id in sorted(left_records.keys() & right_records.keys()):
                union = left_records[query_id] | right_records[query_id]
                intersection = left_records[query_id] & right_records[query_id]
                overlaps.append(len(intersection) / len(union) if union else 1.0)
                left_ranked = next(
                    item["retrieved_chunk_ids"]
                    for item in records_by_experiment[left["experiment_id"]]
                    if item["query_id"] == query_id
                )
                right_ranked = next(
                    item["retrieved_chunk_ids"]
                    for item in records_by_experiment[right["experiment_id"]]
                    if item["query_id"] == query_id
                )
                exact_ranked_matches.append(left_ranked == right_ranked)
            overlap_comparisons.append({
                "chunking_strategy": left["chunking_strategy"],
                "chunking_configuration": left["chunking_configuration"],
                "left_embedding_model": left["embedding_model"],
                "right_embedding_model": right["embedding_model"],
                "left_vector_store": left["vector_store"],
                "right_vector_store": right["vector_store"],
                "left_experiment_id": left["experiment_id"],
                "right_experiment_id": right["experiment_id"],
                "shared_query_count": len(overlaps),
                "mean_top_k_id_jaccard": statistics.mean(overlaps) if overlaps else None,
                "identical_ranked_list_fraction": (
                    sum(exact_ranked_matches) / len(exact_ranked_matches)
                    if exact_ranked_matches else None
                ),
            })

    return {
        "interpretation": (
            "This experiment validates retrieval behavior and reproducibility, "
            "not retrieval quality. No relevance labels or answer-generation "
            "outputs were available. Similarity scores are not quality metrics."
        ),
        "chunk_id_comparison_note": (
            "Top-k ID overlap is compared only when chunking configuration and "
            "source-file hashes match. IDs from different chunking configurations "
            "are configuration-specific; inspect retrieval_results.jsonl for "
            "their ranked IDs rather than interpreting cross-configuration "
            "overlap as a quality measure."
        ),
        "configuration_metrics": metrics,
        "retrieved_id_overlap_comparisons": overlap_comparisons,
    }


def run_experiment(
    *,
    project_root: Path = PROJECT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    query_file: Path = QUERY_FILE,
    include_openai: bool = False,
    model_factory: Callable[[EmbeddingConfig], EmbeddingModel] = create_embedding_model,
) -> Path:
    datasets = discover_chunk_datasets(project_root)
    source_sets = {dataset.sources for dataset in datasets}
    if len(source_sets) != 1:
        raise ValueError("Chunk configurations do not cover the same source documents.")
    queries = load_queries(query_file)
    run_id, run_directory = _create_run_directory(output_root)
    (run_directory / "queries.json").write_text(
        query_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    results_path = run_directory / "retrieval_results.jsonl"
    results_path.touch()

    model_configs, skipped_models = _embedding_configurations(include_openai)
    faiss_available, faiss_error = _probe_faiss()
    stores = [("ChromaDB", ChromaVectorStore)]
    skipped_stores = []
    if faiss_available:
        stores.append(("FAISS", FaissVectorStore))
    else:
        skipped_stores.append({
            "vector_store": "FAISS",
            "status": "skipped",
            "reason": faiss_error or "Native FAISS is unavailable.",
        })

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_set_id": queries["query_set_id"],
        "query_set_synthetic": bool(queries.get("synthetic")),
        "relevance_labels_available": bool(
            queries.get("relevance_labels_available", False)
        ),
        "query_count": len(queries["queries"]),
        "top_k": TOP_K,
        "source_documents": list(next(iter(source_sets))),
        "chunking_configuration_count": len(datasets),
        "embedding_models": [],
        "skipped_models": skipped_models,
        "skipped_vector_stores": skipped_stores,
        "configurations": [],
    }
    summary_path = run_directory / "configuration_summary.json"
    _write_json(summary_path, summary)

    with results_path.open("a", encoding="utf-8") as output_file:
        for model_alias, model_config in model_configs.items():
            try:
                model = model_factory(model_config)
            except Exception as error:
                summary["embedding_models"].append({
                    "alias": model_alias,
                    "model_name": model_config.model_name,
                    "status": "failed_to_load",
                    "error": f"{type(error).__name__}: {error}",
                })
                for dataset in datasets:
                    for vector_store, _ in stores:
                        summary["configurations"].append({
                            "experiment_id": (
                                f"{dataset.configuration_id}-{model_alias}-"
                                f"{vector_store.lower()}"
                            ),
                            "chunking_strategy": dataset.strategy,
                            "chunking_configuration": dataset.configuration,
                            "embedding_model": model_config.model_name,
                            "vector_store": vector_store,
                            "number_of_chunks": dataset.chunk_count,
                            "retrieval_status": "not_run",
                            "status": "failed",
                            "warnings": [f"Embedding model load failed: {error}"],
                        })
                _write_json(summary_path, summary)
                continue

            summary["embedding_models"].append({
                "alias": model_alias,
                "model_name": model.get_model_name(),
                "dimension": model.get_dimension(),
                "status": "loaded",
            })

            for vector_store, store_class in stores:
                for dataset in datasets:
                    try:
                        configuration_summary, records = _run_store_configuration(
                            run_id=run_id,
                            model_alias=model_alias,
                            model=model,
                            dataset=dataset,
                            vector_store=vector_store,
                            store_class=store_class,
                            queries=queries["queries"],
                            output_root=output_root,
                            run_directory=run_directory,
                            project_root=project_root,
                        )
                        summary["configurations"].append(configuration_summary)
                        for record in records:
                            output_file.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                        output_file.flush()
                    except Exception as error:
                        summary["configurations"].append({
                            "experiment_id": (
                                f"{dataset.configuration_id}-{model_alias}-"
                                f"{vector_store.lower()}"
                            ),
                            "chunking_strategy": dataset.strategy,
                            "chunking_configuration": dataset.configuration,
                            "source_chunk_files": [
                                str(path.relative_to(project_root))
                                for path in dataset.files
                            ],
                            "source_file_hashes": dataset.source_file_hashes,
                            "number_of_chunks": dataset.chunk_count,
                            "embedding_model": model.get_model_name(),
                            "embedding_dimension": model.get_dimension(),
                            "vector_store": vector_store,
                            "retrieval_status": "failed",
                            "index_persistence_status": "failed_or_unavailable",
                            "status": "failed",
                            "warnings": [f"{type(error).__name__}: {error}"],
                        })
                    _write_json(summary_path, summary)

    _write_json(
        run_directory / "analysis.json",
        _analyze_results(summary["configurations"], [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]),
    )
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["successful_configurations"] = sum(
        1 for item in summary["configurations"] if item.get("status") == "success"
    )
    summary["failed_configurations"] = sum(
        1 for item in summary["configurations"] if item.get("status") == "failed"
    )
    _write_json(summary_path, summary)
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare retrieval over existing Member 1/2 chunk outputs."
    )
    parser.add_argument(
        "--include-openai",
        action="store_true",
        help="Allow OpenAI embedding API calls; may incur charges.",
    )
    arguments = parser.parse_args()
    output_directory = run_experiment(include_openai=arguments.include_openai)
    print(f"Retrieval experiment outputs: {output_directory}")


if __name__ == "__main__":
    main()
