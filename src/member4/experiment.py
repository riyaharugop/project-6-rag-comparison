
"""Person 4 - Retrieval comparison experiments."""

import csv
import json
from pathlib import Path

from src.member3.embeddings import (
    EmbeddingConfig,
    create_embedding_model,
    embed_chunk_records,
)
from src.member3.vector_store import FaissVectorStore

from src.member4.evaluation import (
    context_recall,
    context_precision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation_questions.csv"
RESULTS_FILE = PROJECT_ROOT / "results" / "retrieval_metrics.csv"

TOP_K = 5

EXPERIMENT_CONFIGS = [
    ("fixed_size", 500, "bge"),
    ("fixed_size", 500, "huggingface"),
    ("sentence_level", 500, "bge"),
    ("sentence_level", 500, "huggingface"),
    ("semantic", 500, "bge"),
    ("semantic", 500, "huggingface"),
]

RESULT_FIELDS = [
    "question",
    "chunking_strategy",
    "chunk_size",
    "embedding_model",
    "vector_store",
    "top_k",
    "context_recall",
    "context_precision",
    "retrieval_latency_seconds",
]


def load_questions():
    """Load and validate evaluation questions."""

    with QUESTIONS_FILE.open("r", encoding="utf-8-sig") as file:
        questions = list(csv.DictReader(file))

    if not questions:
        raise ValueError("No evaluation questions found.")

    required_fields = {"question", "supporting_passage"}

    if not required_fields.issubset(questions[0].keys()):
        raise ValueError(
            "Evaluation CSV must contain question and supporting_passage columns."
        )

    for number, item in enumerate(questions, start=1):
        if not item["question"].strip():
            raise ValueError(f"Question {number} is empty.")

        if not item["supporting_passage"].strip():
            raise ValueError(
                f"Supporting passage for question {number} is empty."
            )

    return questions


def load_chunks(strategy, size):
    """Load chunks from all documents for one configuration."""

    folder_map = {
        "fixed_size": ("chunks", "fixed"),
        "sentence_level": ("sentence_chunks", "sentence"),
        "semantic": ("semantic_chunks", "semantic"),
    }

    if strategy not in folder_map:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    folder, suffix = folder_map[strategy]
    chunks_dir = PROJECT_ROOT / "data" / folder
    all_chunks = []

    for chunk_file in sorted(
        chunks_dir.glob(f"paper*_{suffix}_{size}.json")
    ):
        with chunk_file.open("r", encoding="utf-8") as file:
            all_chunks.extend(json.load(file))

    if not all_chunks:
        raise FileNotFoundError(
            f"No chunk files found for {strategy} with size {size}."
        )

    return all_chunks


def save_result(result):
    """Append one retrieval result to the CSV."""

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = (
        RESULTS_FILE.exists()
        and RESULTS_FILE.stat().st_size > 0
    )

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            field: result.get(field, "")
            for field in RESULT_FIELDS
        })


def run_retrieval_experiment(
    questions,
    strategy,
    size,
    embedding_provider,
):
    print("\n" + "=" * 60)
    print(
        f"Strategy: {strategy} | Size: {size} | "
        f"Embedding: {embedding_provider}"
    )
    print("=" * 60)

    chunks = load_chunks(strategy, size)
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = create_embedding_model(
        EmbeddingConfig(provider=embedding_provider)
    )

    print("Generating chunk embeddings...")
    embedded_chunks = embed_chunk_records(chunks, embedding_model)

    print("Building FAISS index...")
    vector_store = FaissVectorStore(embedding_model)
    vector_store.build_index(embedded_chunks)

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"\nQuestion {number}/{len(questions)}: {question}")

        retrieved_chunks = vector_store.search(
            question,
            top_k=TOP_K,
        )

        index_info = vector_store.get_index_info()
        retrieval_latency = index_info["retrieval_latency_seconds"]

        recall = context_recall(
            item["supporting_passage"],
            retrieved_chunks,
        )

        precision = context_precision(
            item["supporting_passage"],
            retrieved_chunks,
        )

        result = {
            "question": question,
            "chunking_strategy": strategy,
            "chunk_size": size,
            "embedding_model": embedding_model.get_model_name(),
            "vector_store": "FAISS",
            "top_k": TOP_K,
            "context_recall": recall,
            "context_precision": precision,
            "retrieval_latency_seconds": retrieval_latency,
        }

        save_result(result)

        print(f"Context recall: {recall:.4f}")
        print(f"Context precision: {precision:.4f}")
        print(f"Retrieval latency: {retrieval_latency:.4f} seconds")
        print("Result saved.")

    print(f"\nCompleted: {strategy} + {embedding_provider}")


def run_all_experiments(question_limit=20):
    """Run all six configurations, unless results already exist."""

    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        print("Retrieval experiment results already exist.")
        print(f"Existing results preserved: {RESULTS_FILE}")
        print("Not running again to avoid duplicate records.")
        return

    questions = load_questions()[:question_limit]

    if not questions:
        raise ValueError("No questions selected for the experiment.")

    print(f"Running retrieval comparisons on {len(questions)} questions.")
    print(f"Total configurations: {len(EXPERIMENT_CONFIGS)}")

    for strategy, size, embedding_provider in EXPERIMENT_CONFIGS:
        run_retrieval_experiment(
            questions,
            strategy,
            size,
            embedding_provider,
        )

    print("\nAll retrieval experiments completed.")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_all_experiments()
