"""
Member 2 - Semantic chunking.

Boundary rule
-------------
1. Split each document into units with Member 1's own sentence logic
   (NLTK sent_tokenize + structured-block handling), so sentence boundaries are
   IDENTICAL to the sentence-level strategy.
2. Embed every unit with a small local sentence-transformer (all-MiniLM-L6-v2).
3. Walk through the units in order and start a NEW chunk when
      - cosine similarity between this unit and the previous one < THRESHOLD
        (and the current chunk already has >= MIN_SIZE characters), or
      - adding the unit would push the chunk past MAX_SIZE characters.
4. Optionally repeat the last OVERLAP_SENTENCES units at the start of the next chunk
   (default 1, same as Member 1's sentence chunker, for a fair comparison).

Sizes are in CHARACTERS (not tokens) because Member 1's fixed and sentence chunkers
use characters. MAX_SIZE values 500 and 1000 match Member 1's two configurations.

Output matches Member 1's JSON style, one file per document per size:
    data/semantic_chunks/<doc>_semantic_<MAX_SIZE>.json
plus a statistics file in the same columns as results/chunk_statistics.csv:
    results/semantic_chunk_statistics.csv

Run from the project root:
    python src/member2/semantic_chunking.py
    python src/member2/semantic_chunking.py --threshold 0.4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse Member 1's sentence/unit preparation so boundaries are identical.
from member1.sentence_chunking import ensure_nltk_data, prepare_units  # noqa: E402

INPUT_FOLDER = PROJECT_ROOT / "data" / "processed_docs"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "semantic_chunks"
STATS_FILE = PROJECT_ROOT / "results" / "semantic_chunk_statistics.csv"

# ----------------------- parameters (record in the report) -----------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = 0.45          # similarity below this = topic shift. Tune ONCE, then freeze.
MIN_SIZE_RATIO = 0.3      # min chunk size = 30% of MAX_SIZE (avoids tiny chunks)
OVERLAP_SENTENCES = 1     # same as Member 1's sentence chunker
MAX_SIZES = [500, 1000]   # same sizes as Member 1
# ---------------------------------------------------------------------------------


def load_embedder(model_name=EMBEDDING_MODEL):
    """Return a function: list[str] -> L2-normalised numpy array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts):
        return model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return embed


def adjacent_similarities(units, embed):
    """Cosine similarity between each unit and the next (vectors are normalised)."""
    if len(units) < 2:
        return np.array([])
    vectors = np.asarray(embed(units))
    return (vectors[:-1] * vectors[1:]).sum(axis=1)


def semantic_chunk(units, sims, max_size, min_size, threshold, overlap_sentences):
    """
    Group units into chunks. Returns a list of dicts with text and unit indexes.
    sims[i] is the similarity between units[i] and units[i + 1].
    """
    chunks = []
    start = 0

    while start < len(units):
        end = start + 1
        length = len(units[start])

        while end < len(units):
            added = len(units[end]) + 1  # +1 for the joining space

            too_big = length + added > max_size
            topic_shift = sims[end - 1] < threshold and length >= min_size

            if too_big or topic_shift:
                break

            length += added
            end += 1

        chunks.append({
            "text": " ".join(units[start:end]),
            "sentence_start_index": start,
            "sentence_end_index": end,
        })

        if end >= len(units):
            break

        # Same overlap rule as Member 1's sentence chunker (always moves forward)
        start = max(start + 1, end - overlap_sentences)

    # Merge a tiny last chunk into the previous one if it still fits
    if len(chunks) > 1 and len(chunks[-1]["text"]) < min_size:
        last, prev = chunks[-1], chunks[-2]
        if len(prev["text"]) + 1 + len(last["text"]) <= max_size:
            prev["text"] = prev["text"] + " " + last["text"]
            prev["sentence_end_index"] = last["sentence_end_index"]
            chunks.pop()

    return chunks


def process_document(file_path, max_size, threshold, overlap_sentences, embed):
    text = file_path.read_text(encoding="utf-8")

    units = prepare_units(text, max_size)
    sims = adjacent_similarities(units, embed)

    min_size = int(max_size * MIN_SIZE_RATIO)
    chunks = semantic_chunk(
        units, sims, max_size, min_size, threshold, overlap_sentences
    )

    output = []
    for number, chunk in enumerate(chunks, start=1):
        output.append({
            "chunk_id": f"{file_path.stem}_semantic_{max_size}_{number}",
            "source": file_path.name,
            "strategy": "semantic",
            "target_size": max_size,
            "min_size": min_size,
            "similarity_threshold": threshold,
            "embedding_model": EMBEDDING_MODEL,
            "overlap_sentences": overlap_sentences,
            "text": chunk["text"],
            "sentence_start_index": chunk["sentence_start_index"],
            "sentence_end_index": chunk["sentence_end_index"],
        })

    return output, sims


def run(embed, threshold=THRESHOLD, overlap_sentences=OVERLAP_SENTENCES,
        input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, stats_file=STATS_FILE,
        max_sizes=MAX_SIZES):
    output_folder.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_rows = []

    for file_path in sorted(input_folder.glob("*.txt")):
        print(f"\nProcessing: {file_path.name}")

        for max_size in max_sizes:
            chunks, sims = process_document(
                file_path, max_size, threshold, overlap_sentences, embed
            )

            out = output_folder / f"{file_path.stem}_semantic_{max_size}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=4, ensure_ascii=False)

            lengths = [len(c["text"]) for c in chunks]
            stats_rows.append({
                "document": file_path.name,
                "strategy": "semantic",
                "target_size": max_size,
                "overlap": overlap_sentences,
                "total_chunks": len(chunks),
                "average_length": round(sum(lengths) / len(lengths), 2),
                "minimum_length": min(lengths),
                "maximum_length": max(lengths),
                "total_characters": sum(lengths),
            })

            # Percentiles help you pick the threshold once (then freeze it)
            if len(sims) and max_size == max_sizes[0]:
                p10, p25, p50 = np.percentile(sims, [10, 25, 50])
                print(f"  adjacent similarity percentiles: "
                      f"10%={p10:.2f} 25%={p25:.2f} 50%={p50:.2f}")

            print(f"  max size {max_size}: {len(chunks)} chunks, "
                  f"avg {stats_rows[-1]['average_length']} chars")

    with stats_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"\nSaved statistics to {stats_file}")
    print("Semantic chunking completed!")


def main():
    parser = argparse.ArgumentParser(description="Semantic chunking (Member 2)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--overlap", type=int, default=OVERLAP_SENTENCES)
    args = parser.parse_args()

    ensure_nltk_data()
    run(load_embedder(), threshold=args.threshold, overlap_sentences=args.overlap)


if __name__ == "__main__":
    main()