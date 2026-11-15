# Project 6: Comparative Analysis of Chunking Strategies and Embedding Models on RAG Retrieval

## 📌 Project Overview

This project focuses on studying how different text chunking strategies and embedding models affect the retrieval performance and answer generation quality of Retrieval-Augmented Generation (RAG) systems.

RAG systems retrieve relevant information from a knowledge base before passing it to a Large Language Model (LLM) to generate answers.

The quality of the generated answers depends significantly on how documents are divided into chunks and how those chunks are represented using embeddings.

Our project aims to experimentally compare different chunking strategies and embedding models to identify effective combinations for technical documentation.

## 🎯 Objectives

- Implement and compare different text chunking strategies.
- Evaluate fixed-size, sentence-level and semantic chunking.
- Experiment with multiple embedding models, including BGE, OpenAI and Hugging Face open-source models.
- Implement vector-based retrieval using FAISS and ChromaDB, subject to the final experiment scope.
- Evaluate retrieval and answer generation using:
  - Context Recall
  - Context Precision
  - Groundedness
  - Latency
- Analyse performance trade-offs and develop a decision matrix.
- Recommend suitable chunking and embedding combinations for technical documentation.

## ⚙️ Proposed Methodology

The project follows this pipeline:

1. **Document Collection:** Collect technical documents and prepare a knowledge base.
2. **Preprocessing:** Extract and clean text from the collected documents.
3. **Chunking:** Divide documents using fixed-size, sentence-level and semantic chunking.
4. **Embedding Generation:** Convert text chunks into numerical vector representations using different embedding models.
5. **Vector Storage:** Store embeddings in vector databases such as FAISS and ChromaDB.
6. **Retrieval:** Retrieve the most relevant chunks based on user queries.
7. **Answer Generation:** Pass retrieved context to an LLM to generate answers.
8. **Evaluation:** Compare configurations using Context Recall, Context Precision, Groundedness and Latency.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | VS Code, Jupyter Notebook |
| Document Processing | PyPDF |
| Text Processing | NLTK / spaCy |
| Chunking | Fixed-size, Sentence-level, Semantic |
| Embedding Models | BGE, OpenAI, Hugging Face |
| Vector Stores | FAISS, ChromaDB |
| RAG Pipeline | Python / LangChain (if required) |
| Evaluation | RAGAS / Custom evaluation |
| Data Analysis | Pandas, Matplotlib |
| Version Control | Git, GitHub |

### Member 3 embedding backends

The shared interface in `src/member3/embeddings.py` supports `bge`,
`huggingface`, and `openai` providers. Defaults are `BAAI/bge-small-en-v1.5`,
`sentence-transformers/all-MiniLM-L6-v2`, and `text-embedding-3-small`.
Each provider can be selected with `EmbeddingConfig`; `model_name` and
`batch_size` can be overridden. BGE adds its retrieval instruction to queries,
not document passages. All adapters return unit-normalized vectors and expose
the latest call's model, dimension, elapsed time, and vector count through
`get_last_run_stats()`.

OpenAI embeddings require the `OPENAI_API_KEY` environment variable. The key is
read at runtime; it should not be stored in source files.

### Member 3 FAISS vector store

`src/member3/vector_store.py` builds cosine-similarity indexes with normalized
vectors and FAISS inner product. Indexes are separated by embedding model,
chunking strategy/configuration, dimension, and a deterministic corpus
fingerprint. Embedded records retain retrieval model and dimension provenance,
separate from any semantic boundary model metadata. `FaissVectorStore` keeps
chunk metadata in a JSON sidecar and saves indexes under the ignored `indexes/`
directory by default. Pass the same embedding model to the store when loading
or searching an index; searches default to five results.

### Member 3 ChromaDB vector store

`src/member3/chroma_store.py` provides a persistent ChromaDB backend using the
same embedded chunk records and `EmbeddingModel` interface as FAISS. Document
and query vectors are supplied explicitly, so ChromaDB does not select or apply
its default embedding function. Collections are isolated by model, dimension,
strategy/configuration, and corpus fingerprint. Nested chunk metadata is stored
as JSON and restored in search results. ChromaDB cosine distances are converted
to cosine similarity scores. Persistent collections are stored under the
ignored `indexes/chroma/` directory by default; use `get_index_info()` to get
the collection name and `load_index()` to reopen it. Collection builds remain
marked `building` until the expected record count is verified; `load_index()`
rejects incomplete or count-mismatched collections. `save_index()` requires a
Chroma client whose settings report persistent storage. Both vector stores emit
one INFO retrieval record per search with a generated query ID, store name,
model, top-k, ranked chunk IDs and scores, and elapsed time; raw query text is
not logged.

### Member 3 retrieval experiment

Run `python -m src.member3.run_retrieval_experiment` from the repository root.
The runner reuses the existing five-document JSON chunk outputs and does not
regenerate chunks. It compares all six recorded chunk configurations: fixed
500/50 and 1000/100, sentence-level 500/1 and 1000/1, and semantic 500/150/0.25/1
and 1000/300/0.25/1 (target/minimum size, similarity threshold, overlap
sentences). Semantic chunk records retain Member 2's boundary model metadata.

The five-query set at `data/retrieval_experiment_queries.json` is synthetic,
curated from the source-document topics, and has no relevance labels. BGE and
MiniLM models are loaded once per run and reused across configurations. ChromaDB
is used when available; FAISS results are included only when its native
extension loads normally. OpenAI API calls are skipped unless credentials and
the package are present and `--include-openai` is explicitly supplied; those
calls may incur charges.

Each run gets a unique directory under `results/retrieval_experiments/` so
previous results are not overwritten. It contains:

- `queries.json`: the exact query set used.
- `retrieval_results.jsonl`: top-5 IDs, cosine scores, latency, repeatability,
  configuration, and validation status for each query/configuration.
- `configuration_summary.json`: source file hashes, actual chunk parameters,
  indexed chunk counts, model dimensions, embedding/index timings, and
  persistence/load and retrieval statuses.
- `analysis.json`: latency and score distributions plus pairwise top-k ID
  overlap only among results with matching chunking configuration and source
  hashes (chunk IDs from different chunking configurations are not comparable).
  It explicitly treats these as retrieval-behavior observations, not quality
  measurements.

Runs reuse a prior successful index only when its configuration and input-file
hashes match. Repeated searches check result reproducibility. No Context Recall,
Context Precision, Groundedness, or answer-quality metric is calculated because
there are no relevance labels or generation outputs.

## 📁 Project Structure

```text
project-6-rag-comparison/
│
├── data/
│   ├── raw_docs/
│   └── processed_docs/
│
├── src/
│   ├── member1/
│   ├── member2/
│   ├── member3/
│   └── member4/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 👥 Team Structure

The project is divided among four members:

| Member | Responsibility |
|---|---|
| Member 1 | Dataset collection, preprocessing, fixed-size and sentence-level chunking |
| Member 2 | Semantic chunking implementation |
| Member 3 | Embedding models, vector storage and retrieval |
| Member 4 | Answer generation, evaluation, integration and result analysis |

All members will contribute to testing, integration, documentation and final project preparation.

## 📊 Expected Outcomes

- A functional RAG experimentation pipeline.
- Implementation of multiple chunking strategies.
- Comparative evaluation of different embedding models.
- Performance measurements using the selected evaluation metrics.
- Graphical comparisons of experimental results.
- A decision matrix recommending suitable configurations for technical documentation.
- A guidelines report describing findings and trade-offs.

## 📅 Project Duration

The original project guidelines specify a 10–12 week mini-project. The team is currently following an accelerated implementation schedule for its initial prototype.

## 📚 Project Type

Fundamental / Empirical Study

**Course:** Prompt Engineering and Optimization for Generative AI  
**Course Code:** UE24AM343AB5  
**Stream:** B.Tech CSE – Artificial Intelligence and Machine Learning

---

*Note: Final experimental configurations, datasets, model versions and results will be documented as implementation progresses.*
