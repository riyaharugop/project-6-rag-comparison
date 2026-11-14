# Semantic Chunking: Parameter Notes (Member 2)

Script: `src/member2/semantic_chunking.py`  |  Output: `data/semantic_chunks/`  |  Stats: `results/semantic_chunk_statistics.csv`

## Method
1. Each processed document is split into units using Member 1's `prepare_units` (NLTK `sent_tokenize` plus structured-block handling), so sentence boundaries are identical to the sentence-level strategy.
2. Every unit is embedded with `sentence-transformers/all-MiniLM-L6-v2` (L2-normalised).
3. Cosine similarity is computed between each pair of adjacent units.
4. A new chunk starts when
   - similarity between the next unit and the previous unit is below the threshold **and** the current chunk is at least `min_size` characters, or
   - adding the next unit would exceed `max_size` characters.
5. The last unit is repeated at the start of the next chunk (sentence overlap = 1), the same as Member 1's sentence chunker.
6. A tiny final chunk is merged into the previous one if the result still fits `max_size`.

## Parameters used (frozen for all experiments)
| Parameter | Value |
|---|---|
| Boundary model | sentence-transformers/all-MiniLM-L6-v2 |
| Similarity threshold | 0.25 |
| Max chunk size | 500 and 1000 characters (same as Member 1) |
| Min chunk size | 30% of max size (150 / 300 characters) |
| Overlap | 1 sentence/unit |
| Size unit | characters (matches Member 1's chunkers) |

Threshold choice: with 0.45 the median adjacent-sentence similarity (about 0.30 to 0.42) was above the threshold, so chunks were too small (average about 298 characters at size 500). Lowering it to 0.25, near the 25th percentile of adjacent similarities (0.17 to 0.26), gave chunk sizes closer to the other strategies. Only two values (0.45, 0.25) were tried; 0.25 was kept and not tuned further.

## Chunk statistics, all 5 documents combined
| Strategy | Target size | Total chunks | Avg length (chars) | Min | Max |
|---|---|---|---|---|---|
| Fixed-size | 500 | 2376 | 499.5 | 143 | 500 |
| Sentence-level | 500 | 3776 | 402.6 | 3 | 1279 |
| Semantic | 500 | 4484 | 351.0 | 19 | 1279 |
| Fixed-size | 1000 | 1189 | 998.1 | 357 | 1000 |
| Sentence-level | 1000 | 1533 | 867.9 | 3 | 1279 |
| Semantic | 1000 | 1879 | 738.9 | 46 | 1279 |

Observations (from the numbers above):
- Semantic chunking produced the most chunks and the shortest average chunk at both sizes, because it also ends chunks at topic shifts, not only at the size limit.
- Fixed-size chunks are almost exactly the target length; sentence and semantic chunks are shorter on average.
- The maximum of 1279 characters for sentence and semantic chunks comes from single units (one very long sentence or table row) that are kept whole rather than cut.
- Chunks under 100 characters at size 500: fixed 0.0%, sentence 0.5%, semantic 0.4%.

## Example comparison
Document: paper3.txt (NIST SP 800-61r3), size 500, passage containing "root cause analysis help improve".

- **Fixed-size split:** the chunk starts in the middle of a word ("ctively when an incident occurs...") and ends mid-text with page-header text, so the sentence "Lessons learned from incident response activities and root cause analysis help improve cybersecurity risk management and governance efforts..." is cut apart at an arbitrary position.
- **Sentence-level split:** the chunk starts at the beginning of the sentence "Lessons learned from incident response activities..." and keeps it whole, continuing to the end of the next sentence.
- **Semantic split:** almost the same as the sentence-level chunk. It also starts at the sentence beginning, but ends after "...recover from future incidents." without carrying an extra overlapping sentence.
- **Which kept context best:** sentence-level and semantic chunking both kept complete sentences, while fixed-size broke words and sentences. For this passage semantic chunking was nearly identical to sentence-level, so no clear advantage was seen; any benefit of semantic chunking would have to show up in the retrieval metrics.
- **Shared problem:** a running page header ("NIST SP 800-61r3 Incident Response Recommendations and Considerations for Cyber Risk Management") appears in the middle of the sentence in all three strategies, because the preprocessing step does not remove repeated headers.

## Limitations
- The threshold is hand-picked from the similarity percentiles, not optimised against retrieval results.
- Similarity between adjacent short or fragmentary units (headings, table rows, references) is noisy, so boundaries there can be unreliable.
- Sentence splitting inherits NLTK's errors on abbreviations, URLs and tables, and the cover pages, page headers and references in the processed text are not removed, so some chunks contain boilerplate.
- Every unit is embedded, so chunking is slower than fixed-size or sentence chunking.
- Sizes are in characters, not tokens, so they are not directly comparable to model token limits.
- Results apply only to this 5-document corpus and these settings.