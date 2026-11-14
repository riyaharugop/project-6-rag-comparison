"""
Show how fixed, sentence and semantic chunking split the SAME passage.
Run from the project root:
    python src/member2/compare_example.py paper3 "access control"
(arguments: document name, a phrase that appears in the text; optional size, default 500)
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
doc, phrase = sys.argv[1], sys.argv[2]
size = sys.argv[3] if len(sys.argv) > 3 else "500"

sources = {
    "FIXED-SIZE":     ROOT / "data" / "chunks"          / f"{doc}_fixed_{size}.json",
    "SENTENCE-LEVEL": ROOT / "data" / "sentence_chunks" / f"{doc}_sentence_{size}.json",
    "SEMANTIC":       ROOT / "data" / "semantic_chunks" / f"{doc}_semantic_{size}.json",
}

for name, path in sources.items():
    chunks = json.loads(path.read_text(encoding="utf-8"))
    hits = [c for c in chunks if phrase.lower() in c["text"].lower()]
    print("=" * 80)
    print(f"{name}: {len(hits)} chunk(s) contain '{phrase}'")
    for c in hits[:2]:
        print("-" * 80)
        print(f"[{c['chunk_id']}]  ({len(c['text'])} chars)")
        print(c["text"])
