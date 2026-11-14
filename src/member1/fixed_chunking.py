
import json
from pathlib import Path


# Locate the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output folders
INPUT_FOLDER = PROJECT_ROOT / "data" / "processed_docs"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "chunks"


def fixed_size_chunk(text, chunk_size, overlap):
    """
    Divide text into fixed-size chunks with overlap.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks = []

    step_size = chunk_size - overlap

    start = 0
    chunk_number = 1

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk_text = text[start:end]

        chunks.append({
            "chunk_number": chunk_number,
            "text": chunk_text,
            "start_index": start,
            "end_index": end
        })

        start += step_size
        chunk_number += 1

    return chunks


def process_document(file_path, chunk_size, overlap):

    text = file_path.read_text(encoding="utf-8")

    chunks = fixed_size_chunk(text, chunk_size, overlap)

    output_data = []

    for chunk in chunks:

        output_data.append({
            "chunk_id": f"{file_path.stem}_fixed_{chunk_size}_{chunk['chunk_number']}",
            "source": file_path.name,
            "strategy": "fixed_size",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "text": chunk["text"],
            "start_index": chunk["start_index"],
            "end_index": chunk["end_index"]
        })

    return output_data


def main():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    configurations = [
        (500, 50),
        (1000, 100)
    ]

    for file_path in sorted(INPUT_FOLDER.glob("*.txt")):

        print(f"\nProcessing: {file_path.name}")

        for chunk_size, overlap in configurations:

            chunks = process_document(
                file_path,
                chunk_size,
                overlap
            )

            output_file = OUTPUT_FOLDER / (
                f"{file_path.stem}_fixed_{chunk_size}.json"
            )

            with output_file.open("w", encoding="utf-8") as file:
                json.dump(chunks, file, indent=4, ensure_ascii=False)

            print(
                f"Chunk size: {chunk_size}, "
                f"Overlap: {overlap}, "
                f"Total chunks: {len(chunks)}"
            )

    print("\nFixed-size chunking completed!")


if __name__ == "__main__":
    main()
