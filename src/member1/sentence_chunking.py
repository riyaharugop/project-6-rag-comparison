
import json
import re
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output folders
INPUT_FOLDER = PROJECT_ROOT / "data" / "processed_docs"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "sentence_chunks"


def ensure_nltk_data():
    """Download required NLTK resources."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def is_structured_block(lines):
    """
    Identify glossary entries, tables and TOC blocks
    that NLTK may incorrectly treat as one sentence.
    """

    if len(lines) < 3:
        return False

    total = len(lines)

    # Detect table-of-contents entries
    dotted_lines = sum(
        bool(re.search(r"\.{3,}", line))
        for line in lines
    )

    # Detect glossary entries such as ACL, ARP, CCE
    acronym_lines = sum(
        bool(re.match(r"^[A-Z][A-Z0-9/&().-]{1,10}\s+", line))
        for line in lines
    )

    # Detect table-like rows
    short_lines = sum(len(line) <= 180 for line in lines)

    unfinished_lines = sum(
        not re.search(r'[.!?]["\')\]]?$', line)
        for line in lines
    )

    return (
        dotted_lines >= 2
        or acronym_lines / total >= 0.4
        or (
            short_lines / total >= 0.8
            and unfinished_lines / total >= 0.8
        )
    )


def prepare_units(text, target_size):
    """
    Separate normal sentences and structured content
    into meaningful units.
    """

    raw_sentences = sent_tokenize(text)
    units = []

    for sentence in raw_sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        lines = [
            line.strip()
            for line in sentence.splitlines()
            if line.strip()
        ]

        # Split oversized structured blocks into rows
        if (
            len(sentence) > target_size
            and is_structured_block(lines)
        ):
            units.extend(lines)

        else:
            units.append(sentence)

    return units


def sentence_chunk(text, target_size, overlap_sentences=1):
    """
    Group complete sentences or structured rows
    into chunks of approximately the target size.
    """

    if target_size <= 0:
        raise ValueError("Target size must be positive.")

    if overlap_sentences < 0:
        raise ValueError("Overlap cannot be negative.")

    sentences = prepare_units(text, target_size)

    chunks = []

    start = 0
    chunk_number = 1

    while start < len(sentences):

        current_sentences = []
        current_length = 0
        end = start

        while end < len(sentences):

            sentence = sentences[end]

            added_length = len(sentence)

            if current_sentences:
                added_length += 1

            if current_length + added_length > target_size:
                break

            current_sentences.append(sentence)
            current_length += added_length
            end += 1

        # Preserve a single long sentence or row
        if end == start:
            current_sentences.append(sentences[end])
            end += 1

        chunk_text = " ".join(current_sentences)

        chunks.append({
            "chunk_number": chunk_number,
            "text": chunk_text,
            "sentence_start_index": start,
            "sentence_end_index": end
        })

        # Preserve sentence overlap
        next_start = max(
            start + 1,
            end - overlap_sentences
        )

        start = next_start
        chunk_number += 1

    return chunks


def process_document(file_path, target_size, overlap_sentences):

    text = file_path.read_text(encoding="utf-8")

    chunks = sentence_chunk(
        text,
        target_size,
        overlap_sentences
    )

    output_data = []

    for chunk in chunks:

        output_data.append({
            "chunk_id": (
                f"{file_path.stem}_sentence_"
                f"{target_size}_{chunk['chunk_number']}"
            ),
            "source": file_path.name,
            "strategy": "sentence_level",
            "target_size": target_size,
            "overlap_sentences": overlap_sentences,
            "text": chunk["text"],
            "sentence_start_index": chunk["sentence_start_index"],
            "sentence_end_index": chunk["sentence_end_index"]
        })

    return output_data


def main():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    ensure_nltk_data()

    configurations = [
        (500, 1),
        (1000, 1)
    ]

    for file_path in sorted(INPUT_FOLDER.glob("*.txt")):

        print(f"\nProcessing: {file_path.name}")

        for target_size, overlap in configurations:

            chunks = process_document(
                file_path,
                target_size,
                overlap
            )

            output_file = OUTPUT_FOLDER / (
                f"{file_path.stem}_sentence_{target_size}.json"
            )

            with output_file.open("w", encoding="utf-8") as file:
                json.dump(
                    chunks,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                f"Target size: {target_size}, "
                f"Sentence overlap: {overlap}, "
                f"Total chunks: {len(chunks)}"
            )

    print("\nSentence-level chunking completed!")


if __name__ == "__main__":
    main()
