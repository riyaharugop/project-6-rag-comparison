
import json
import csv
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIXED_FOLDER = PROJECT_ROOT / "data" / "chunks"
SENTENCE_FOLDER = PROJECT_ROOT / "data" / "sentence_chunks"
RESULTS_FOLDER = PROJECT_ROOT / "results"


def analyse_file(file_path):

    with file_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    lengths = [len(chunk["text"]) for chunk in chunks]

    if not lengths:
        return None

    strategy = (
        "fixed_size"
        if "fixed" in file_path.name
        else "sentence_level"
    )

    target_size = (
        chunks[0].get("chunk_size")
        or chunks[0].get("target_size")
    )

    overlap = (
        chunks[0].get("overlap")
        if strategy == "fixed_size"
        else chunks[0].get("overlap_sentences")
    )

    return {
        "document": chunks[0]["source"],
        "strategy": strategy,
        "target_size": target_size,
        "overlap": overlap,
        "total_chunks": len(chunks),
        "average_length": round(mean(lengths), 2),
        "minimum_length": min(lengths),
        "maximum_length": max(lengths),
        "total_characters": sum(lengths)
    }


def main():

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    files = (
        sorted(FIXED_FOLDER.glob("*.json"))
        + sorted(SENTENCE_FOLDER.glob("*.json"))
    )

    results = []

    for file_path in files:

        result = analyse_file(file_path)

        if result:
            results.append(result)

    if not results:
        print("No chunk data found!")
        return

    output_file = RESULTS_FOLDER / "chunk_statistics.csv"

    fieldnames = list(results[0].keys())

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== CHUNK ANALYSIS ==========")

    for strategy in ["fixed_size", "sentence_level"]:

        print(f"\nStrategy: {strategy}")

        for size in [500, 1000]:

            group = [
                row for row in results
                if row["strategy"] == strategy
                and row["target_size"] == size
            ]

            if not group:
                continue

            total = sum(
                row["total_chunks"] for row in group
            )

            avg_length = mean(
                row["average_length"] for row in group
            )

            print(f"\nTarget size: {size}")
            print(f"Total chunks: {total}")
            print(
                f"Average chunk length: "
                f"{avg_length:.2f}"
            )

    print("\nDetailed statistics saved to:")
    print(output_file)

    print("\nChunk analysis completed!")


if __name__ == "__main__":
    main()
