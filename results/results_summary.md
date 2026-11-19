# Results and Discussion

## 1. Experimental Overview

The experiment compared three chunking strategies (fixed-size, sentence-level, and semantic) with two embedding models (BGE and all-MiniLM-L6-v2). All six configurations used FAISS as the vector database and retrieved the top five chunks for each query.

A total of 20 evaluation questions were tested across each configuration, resulting in 120 retrieval evaluations.

## 2. Context Recall

Fixed-size chunking with BGE achieved the highest average context recall of 0.8039. This indicates that it retrieved the largest proportion of the reference evidence among the tested configurations.

Sentence-level chunking with BGE recorded the lowest recall of 0.7477.

## 3. Context Precision

Semantic chunking with all-MiniLM-L6-v2 achieved the highest average context precision of 0.66. This suggests that its retrieved results contained the greatest proportion of relevant chunks according to the project's evaluation method.

Fixed-size chunking with all-MiniLM-L6-v2 recorded the lowest precision of 0.53.

## 4. Retrieval Latency

Fixed-size chunking with all-MiniLM-L6-v2 achieved the lowest average retrieval latency of 0.0190 seconds.

Sentence-level chunking with BGE recorded the highest average latency of 0.0341 seconds. Overall, retrieval latency remained low across all configurations.

## 5. Overall Comparison

No single configuration achieved the highest score across all evaluation metrics.

Semantic chunking with all-MiniLM-L6-v2 offered the best balance of context precision, recall, and retrieval speed. Fixed-size chunking with BGE was preferable when maximizing context recall was the primary objective.

Therefore, semantic chunking with all-MiniLM-L6-v2 was selected as the recommended configuration based on the retrieval results.

## 6. Limitations

- Generation evaluation was limited because the Gemini API free-tier daily request quota was exhausted.
- A successful end-to-end generation pilot was completed, but generation performance was not compared across all six configurations.
- Context recall, context precision, and groundedness were implemented using custom lexical-overlap evaluation methods rather than a dedicated RAG evaluation framework.
- The experiments used a fixed top-k value of five and a chunk size of 500 for the six core configurations.
