
import json
import csv
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIXED_FOLDER = PROJECT_ROOT / "data" / "chunks"
SENTENCE_FOLDER = PROJECT_ROOT / "data" / "sentence_chunks"
RESULTS_FOLDER = PROJECT_ROOT / "results"


def load_chunks(folder):

    all_chunks = []

    for file_path in sorted(folder.glob("*.json")):

        with file_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        for chunk in chunks:
            chunk["json_file"] = file_path.name

        all_chunks.extend(chunks)

    return all_chunks


def validate_chunks(chunks, strategy):

    errors = []
    warnings = []

    if not chunks:
        return {
            "strategy": strategy,
            "total_chunks": 0,
            "empty_chunks": 0,
            "duplicate_ids": 0,
            "oversized_chunks": 0,
            "overlap_errors": 0,
            "missing_fields": 0,
            "status": "FAILED"
        }, []

    required_fields = [
        "chunk_id",
        "source",
        "text",
        "strategy"
    ]

    seen_ids = set()
    empty_count = 0
    oversized_count = 0
    missing_count = 0
    duplicate_count = 0
    overlap_errors = 0

    # Group chunks by their source JSON file
    groups = defaultdict(list)

    for chunk in chunks:

        # Check required fields
        missing = [
            field for field in required_fields
            if field not in chunk
        ]

        if missing:
            missing_count += 1
            errors.append(
                f"Missing fields in {chunk.get('chunk_id', 'unknown')}: {missing}"
            )
            continue

        # Check duplicate IDs
        chunk_id = chunk["chunk_id"]

        if chunk_id in seen_ids:
            duplicate_count += 1
            errors.append(f"Duplicate ID: {chunk_id}")

        seen_ids.add(chunk_id)

        # Check empty chunks
        if not chunk["text"].strip():
            empty_count += 1
            errors.append(f"Empty chunk: {chunk_id}")

        # Check chunk length
        actual_length = len(chunk["text"])

        if strategy == "fixed_size":
            target = chunk["chunk_size"]
        else:
            target = chunk["target_size"]

        if actual_length > target:
            oversized_count += 1

            # Sentence-level chunks may exceed the target
            if strategy == "fixed_size":
                errors.append(
                    f"Fixed-size chunk exceeds target: {chunk_id}"
                )
            else:
                warnings.append(
                    f"Long sentence chunk exceeds target: {chunk_id}"
                )

        groups[chunk["json_file"]].append(chunk)

    # Check overlap between consecutive chunks
    for filename, group in groups.items():

        group.sort(
            key=lambda x: (
                x.get("start_index")
                if strategy == "fixed_size"
                else x.get("sentence_start_index")
            )
        )

        for previous, current in zip(group, group[1:]):

            if strategy == "fixed_size":

                actual_overlap = (
                    previous["end_index"]
                    - current["start_index"]
                )

                expected_overlap = previous["overlap"]

            else:

                actual_overlap = (
                    previous["sentence_end_index"]
                    - current["sentence_start_index"]
                )

                previous_sentence_count = (
                    previous["sentence_end_index"]
                    - previous["sentence_start_index"]
                )

                expected_overlap = min(
                    previous["overlap_sentences"],
                    max(0, previous_sentence_count - 1)
                )

            if actual_overlap != expected_overlap:
                overlap_errors += 1
                errors.append(
                    f"Overlap mismatch in {filename}: "
                    f"expected {expected_overlap}, "
                    f"found {actual_overlap}"
                )

    status = "PASSED" if not errors else "FAILED"

    report = {
        "strategy": strategy,
        "total_chunks": len(chunks),
        "empty_chunks": empty_count,
        "duplicate_ids": duplicate_count,
        "oversized_chunks": oversized_count,
        "overlap_errors": overlap_errors,
        "missing_fields": missing_count,
        "status": status
    }

    return report, errors + warnings


def main():

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    fixed_chunks = load_chunks(FIXED_FOLDER)
    sentence_chunks = load_chunks(SENTENCE_FOLDER)

    fixed_report, fixed_messages = validate_chunks(
        fixed_chunks, "fixed_size"
    )

    sentence_report, sentence_messages = validate_chunks(
        sentence_chunks, "sentence_level"
    )

    reports = [fixed_report, sentence_report]

    for report, messages in [
        (fixed_report, fixed_messages),
        (sentence_report, sentence_messages)
    ]:

        print("\n================================")
        print("Strategy:", report["strategy"])
        print("================================")

        for key, value in report.items():
            print(f"{key}: {value}")

        for message in messages[:10]:
            print("WARNING/ERROR:", message)

        if len(messages) > 10:
            print(
                f"... and {len(messages) - 10} more messages"
            )

    # Save validation report
    output_file = RESULTS_FOLDER / "validation_report.csv"

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=list(reports[0].keys())
        )

        writer.writeheader()
        writer.writerows(reports)

    print("\nValidation report saved:", output_file)

    if all(r["status"] == "PASSED" for r in reports):
        print("\nAll validation checks passed!")
    else:
        print("\nSome validation checks need attention.")


if __name__ == "__main__":
    main()
