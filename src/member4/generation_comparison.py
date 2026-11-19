
"""Person 4 - Small generation comparison across six RAG configurations."""

import csv
from pathlib import Path

from google.genai import errors

from src.member3.embeddings import (
    EmbeddingConfig,
    create_embedding_model,
    embed_chunk_records,
)
from src.member3.vector_store import FaissVectorStore

from src.member4.experiment import (
    load_questions,
    load_chunks,
    EXPERIMENT_CONFIGS,
)
from src.member4.generation import generate_answer, MODEL_NAME
from src.member4.evaluation import evaluate_response


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "results" / "generation_comparison_flash_lite.csv"

GENERATION_MODEL = MODEL_NAME
TOP_K = 5

RESULT_FIELDS = [
    "question",
    "chunking_strategy",
    "chunk_size",
    "embedding_model",
    "vector_store",
    "generation_model",
    "answer",
    "context_recall",
    "context_precision",
    "groundedness",
    "retrieval_latency_seconds",
    "generation_latency_seconds",
    "end_to_end_latency_seconds",
    "error",
]


def save_result(result):
    """Append one result using a consistent CSV structure."""

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = (
        RESULTS_FILE.exists()
        and RESULTS_FILE.stat().st_size > 0
    )

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            field: result.get(field, "")
            for field in RESULT_FIELDS
        })



def main():
    """Retry failed configurations while preserving successful results."""

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []

    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        with RESULTS_FILE.open("r", newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))

    def get_provider(model_name):
        name = str(model_name).lower()

        if "bge" in name:
            return "bge"

        if "minilm" in name or "huggingface" in name:
            return "huggingface"

        return name

    # Keep successful results and identify configurations needing retries.
    successful = set()

    for row in existing_rows:
        if not row.get("error", "").strip() and row.get("answer", "").strip():
            successful.add((
                row["chunking_strategy"],
                get_provider(row["embedding_model"]),
            ))

    question = load_questions()[0]

    print("Generation comparison using:", GENERATION_MODEL)
    print("Existing successful configurations:", len(successful))
    print("Retrying only missing or failed configurations.")

    for strategy, size, provider in EXPERIMENT_CONFIGS:
        config_key = (strategy, provider)

        if config_key in successful:
            print(f"\nSkipping successful configuration: {strategy} + {provider}")
            continue

        print("\n" + "=" * 60)
        print(f"Retrying: {strategy} + {provider}")
        print("=" * 60)

        try:
            chunks = load_chunks(strategy, size)

            embedding_model = create_embedding_model(
                EmbeddingConfig(provider=provider)
            )

            embedded_chunks = embed_chunk_records(
                chunks,
                embedding_model,
            )

            vector_store = FaissVectorStore(embedding_model)
            vector_store.build_index(embedded_chunks)

            retrieved_chunks = vector_store.search(
                question["question"],
                top_k=TOP_K,
            )

            retrieval_latency = vector_store.get_index_info()[
                "retrieval_latency_seconds"
            ]

            generation_result = generate_answer(
                question["question"],
                retrieved_chunks,
            )

            metrics = evaluate_response(
                question["supporting_passage"],
                retrieved_chunks,
                generation_result["answer"],
                retrieval_latency,
                generation_result["latency_seconds"],
            )

            result = {
                "question": question["question"],
                "chunking_strategy": strategy,
                "chunk_size": size,
                "embedding_model": embedding_model.get_model_name(),
                "vector_store": "FAISS",
                "generation_model": GENERATION_MODEL,
                "answer": generation_result["answer"],
                **metrics,
                "error": "",
            }

            print("\nGenerated answer:", result["answer"])
            print("Groundedness:", metrics["groundedness"])
            print("Generation latency:", metrics["generation_latency_seconds"])

        except Exception as error:
            print(f"Retry failed: {error}")

            result = {
                "question": question["question"],
                "chunking_strategy": strategy,
                "chunk_size": size,
                "embedding_model": provider,
                "vector_store": "FAISS",
                "generation_model": GENERATION_MODEL,
                "error": str(error),
            }

        # Replace the previous row for this configuration, not append a duplicate.
        existing_rows = [
            row for row in existing_rows
            if not (
                row.get("chunking_strategy") == strategy
                and get_provider(row.get("embedding_model", "")) == provider
            )
        ]

        existing_rows.append({
            field: result.get(field, "")
            for field in RESULT_FIELDS
        })

        temp_file = RESULTS_FILE.with_suffix(".tmp")

        with temp_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(existing_rows)

        temp_file.replace(RESULTS_FILE)

        if not result.get("error"):
            successful.add(config_key)
            print("Result saved successfully.")

    print("\nGENERATION COMPARISON SUMMARY")
    print("Successful configurations:", len(successful), "out of 6")
    print("Results:", RESULTS_FILE)


if __name__ == "__main__":
    main()
