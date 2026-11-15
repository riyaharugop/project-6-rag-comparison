import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.member3.embeddings import EmbeddingModel
from src.member3.vector_store import FaissVectorStore


class FakeIndexFlatIP:
    metric_type = 0

    def __init__(self, dimension):
        self.d = dimension


class FakeIndexFlatL2:
    metric_type = 1

    def __init__(self, dimension):
        self.d = dimension


class FakeIndexIDMap2:
    def __init__(self, index):
        self.index = index
        self.d = index.d
        self.metric_type = index.metric_type
        self.ntotal = 0
        self.id_map = []

    def add_with_ids(self, vectors, ids):
        self.ntotal = len(ids)
        self.id_map = list(ids)

    def search(self, query, top_k):
        return np.array([[1.0]]), np.array([[0]], dtype=np.int64)


class FakeFaissModule:
    METRIC_INNER_PRODUCT = 0
    IndexFlatIP = FakeIndexFlatIP
    IndexFlatL2 = FakeIndexFlatL2
    IndexIDMap2 = FakeIndexIDMap2

    def __init__(self, index_to_read=None):
        self.index_to_read = index_to_read

    def write_index(self, index, path):
        Path(path).write_bytes(b"fake-faiss-index")

    def read_index(self, path):
        return self.index_to_read

    @staticmethod
    def downcast_index(index):
        return index

    @staticmethod
    def vector_to_array(values):
        return values


class DeterministicEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.vectors = {
            "search query": [1.0, 0.0, 0.0],
            "passage-0": [1.0, 0.0, 0.0],
            "passage-1": [0.8, 0.6, 0.0],
            "passage-2": [0.0, 1.0, 0.0],
            "passage-3": [-0.8, 0.6, 0.0],
            "passage-4": [-1.0, 0.0, 0.0],
            "passage-5": [0.0, 0.0, 1.0],
        }

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, query):
        return self.vectors[query]

    def get_dimension(self):
        return 3

    def get_model_name(self):
        return self.model_name


def make_chunks(strategy="fixed_size", count=6):
    chunks = []
    for number in range(count):
        chunk = {
            "chunk_id": f"paper1_{strategy}_{number}",
            "source": "paper1.txt",
            "strategy": strategy,
            "text": f"passage-{number}",
            "custom_metadata": {"section": "test"},
        }
        if strategy == "fixed_size":
            chunk.update({
                "chunk_size": 500,
                "overlap": 50,
                "start_index": number * 450,
                "end_index": number * 450 + 500,
            })
        elif strategy == "sentence_level":
            chunk.update({
                "target_size": 500,
                "overlap_sentences": 1,
                "sentence_start_index": number * 4,
                "sentence_end_index": number * 4 + 5,
            })
        elif strategy == "semantic":
            chunk.update({
                "target_size": 500,
                "min_size": 150,
                "similarity_threshold": 0.25,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "overlap_sentences": 1,
                "sentence_start_index": number * 4,
                "sentence_end_index": number * 4 + 5,
            })
        chunks.append(chunk)
    return chunks


class FaissVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.embedding_model = DeterministicEmbeddingModel()

    def test_build_and_search_returns_top_five_with_metadata(self):
        chunks = make_chunks()
        from src.member3.embeddings import embed_chunk_records

        embedded = embed_chunk_records(chunks, self.embedding_model)
        store = FaissVectorStore(self.embedding_model)
        info = store.build_index(embedded)

        results = store.search("search query")

        self.assertEqual(len(results), 5)
        self.assertEqual([result["rank"] for result in results], [1, 2, 3, 4, 5])
        self.assertEqual(results[0]["chunk_id"], chunks[0]["chunk_id"])
        self.assertEqual(results[0]["text"], chunks[0]["text"])
        self.assertEqual(
            results[0]["metadata"]["retrieval_embedding_model"],
            self.embedding_model.get_model_name(),
        )
        self.assertEqual(results[0]["metadata"]["custom_metadata"], {"section": "test"})
        self.assertEqual(info["similarity_metric"], "cosine_similarity (normalized inner product)")
        self.assertGreaterEqual(info["indexing_time_seconds"], 0.0)
        self.assertGreaterEqual(
            store.get_index_info()["retrieval_latency_seconds"],
            0.0,
        )
        scores = [result["similarity_score"] for result in results]
        self.assertTrue(all(-1.0 <= score <= 1.0 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_logs_required_fields_without_query_text(self):
        from src.member3.embeddings import embed_chunk_records

        with patch(
            "src.member3.vector_store._load_faiss",
            return_value=FakeFaissModule(),
        ):
            store = FaissVectorStore(self.embedding_model)
            store.build_index(
                embed_chunk_records(make_chunks(count=2), self.embedding_model)
            )

        with self.assertLogs("src.member3.vector_store", level="INFO") as captured:
            results = store.search("search query", top_k=1)

        self.assertEqual(len(captured.records), 1)
        log_record = captured.records[0]
        self.assertEqual(log_record.store_name, "FaissVectorStore")
        self.assertTrue(log_record.query_id)
        self.assertNotIn("search query", log_record.getMessage())
        self.assertEqual(
            log_record.chunk_ids,
            [result["chunk_id"] for result in results],
        )
        self.assertEqual(
            log_record.similarity_scores,
            [result["similarity_score"] for result in results],
        )
        self.assertEqual(
            log_record.embedding_model,
            self.embedding_model.get_model_name(),
        )
        self.assertEqual(log_record.top_k, 1)
        self.assertGreaterEqual(log_record.elapsed_seconds, 0.0)

    def test_search_logs_empty_results(self):
        from src.member3.embeddings import embed_chunk_records

        class EmptySearchIndex(FakeIndexIDMap2):
            def search(self, query, top_k):
                return np.array([[-np.inf]]), np.array([[-1]], dtype=np.int64)

        with patch(
            "src.member3.vector_store._load_faiss",
            return_value=FakeFaissModule(),
        ):
            store = FaissVectorStore(self.embedding_model)
            store.build_index(
                embed_chunk_records(make_chunks(count=1), self.embedding_model)
            )
        empty_index = EmptySearchIndex(FakeIndexFlatIP(3))
        empty_index.ntotal = 1
        store._index = empty_index

        with self.assertLogs("src.member3.vector_store", level="INFO") as captured:
            results = store.search("search query")

        self.assertEqual(results, [])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].chunk_ids, [])
        self.assertEqual(captured.records[0].similarity_scores, [])

    def test_all_chunking_strategies_and_configurations_are_supported(self):
        from src.member3.embeddings import embed_chunk_records

        for strategy in ("fixed_size", "sentence_level", "semantic"):
            with self.subTest(strategy=strategy):
                chunks = make_chunks(strategy, count=2)
                embedded = embed_chunk_records(chunks, self.embedding_model)
                store = FaissVectorStore(self.embedding_model)
                info = store.build_index(embedded)

                self.assertEqual(info["strategy"], strategy)
                self.assertEqual(info["chunk_count"], 2)
                self.assertEqual(
                    store.search("search query", top_k=2)[0]["metadata"]["strategy"],
                    strategy,
                )
                if strategy == "semantic":
                    metadata = store.search("search query", top_k=1)[0]["metadata"]
                    self.assertEqual(
                        metadata["embedding_model"],
                        "sentence-transformers/all-MiniLM-L6-v2",
                    )
                    self.assertEqual(
                        metadata["retrieval_embedding_model"],
                        self.embedding_model.get_model_name(),
                    )

    def test_indexes_are_separated_by_model_and_configuration(self):
        from src.member3.embeddings import embed_chunk_records

        faiss_stub = FakeFaissModule()
        with patch("src.member3.vector_store._load_faiss", return_value=faiss_stub):
            first = FaissVectorStore(self.embedding_model)
            first_info = first.build_index(
                embed_chunk_records(make_chunks("semantic", 2), self.embedding_model)
            )

            second_model = DeterministicEmbeddingModel(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            second = FaissVectorStore(second_model)
            second_info = second.build_index(
                embed_chunk_records(make_chunks("semantic", 2), second_model)
            )

            other_corpus = make_chunks("semantic", 2)
            other_corpus[0]["source"] = "paper2.txt"
            third = FaissVectorStore(self.embedding_model)
            third_info = third.build_index(
                embed_chunk_records(other_corpus, self.embedding_model)
            )

            other_configuration = make_chunks("semantic", 2)
            other_configuration[0]["similarity_threshold"] = 0.3
            other_configuration[1]["similarity_threshold"] = 0.3
            fourth = FaissVectorStore(self.embedding_model)
            fourth_info = fourth.build_index(
                embed_chunk_records(other_configuration, self.embedding_model)
            )

        self.assertNotEqual(first_info["index_id"], second_info["index_id"])
        self.assertNotEqual(first_info["index_id"], third_info["index_id"])
        self.assertNotEqual(first_info["index_id"], fourth_info["index_id"])
        self.assertNotEqual(first_info["corpus_fingerprint"], third_info["corpus_fingerprint"])
        self.assertNotEqual(first._directory, third._directory)

    def test_bge_vectors_cannot_be_mislabeled_as_minilm(self):
        from src.member3.embeddings import embed_chunk_records

        bge = DeterministicEmbeddingModel("BAAI/bge-small-en-v1.5")
        minilm = DeterministicEmbeddingModel(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedded = embed_chunk_records(make_chunks("semantic", 1), bge)

        self.assertEqual(embedded[0]["embedding_dimension"], minilm.get_dimension())
        with self.assertRaisesRegex(ValueError, "was embedded with"):
            FaissVectorStore(minilm).build_index(embedded)

    def test_build_rejects_missing_embedding_provenance(self):
        from src.member3.embeddings import embed_chunk_records

        embedded = embed_chunk_records(
            make_chunks(count=1),
            self.embedding_model,
        )
        del embedded[0]["retrieval_embedding_model"]
        with self.assertRaisesRegex(ValueError, "missing retrieval embedding model"):
            FaissVectorStore(self.embedding_model).build_index(embedded)

        embedded = embed_chunk_records(
            make_chunks(count=1),
            self.embedding_model,
        )
        del embedded[0]["embedding_dimension"]
        with self.assertRaisesRegex(ValueError, "missing a valid embedding dimension"):
            FaissVectorStore(self.embedding_model).build_index(embedded)

        embedded = embed_chunk_records(
            make_chunks(count=1),
            self.embedding_model,
        )
        embedded[0]["embedding_dimension"] = 384
        with self.assertRaisesRegex(ValueError, "records dimension 384"):
            FaissVectorStore(self.embedding_model).build_index(embedded)

    def test_load_rejects_incompatible_faiss_metric_and_type(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "index.faiss").write_bytes(b"placeholder")
            metadata = {
                "format_version": 1,
                "index_info": {
                    "retrieval_embedding_model": self.embedding_model.get_model_name(),
                    "embedding_dimension": self.embedding_model.get_dimension(),
                },
                "records_by_faiss_id": {"0": {"chunk_id": "chunk-0"}},
            }
            (directory / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            invalid_indexes = (
                (
                    FakeIndexIDMap2(FakeIndexFlatL2(3)),
                    "inner-product similarity metric",
                ),
                (
                    FakeIndexFlatIP(3),
                    "must be an IndexIDMap2",
                ),
            )
            for invalid_index, message in invalid_indexes:
                with self.subTest(message=message):
                    fake_faiss = FakeFaissModule(invalid_index)
                    with patch(
                        "src.member3.vector_store._load_faiss",
                        return_value=fake_faiss,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            FaissVectorStore.load_index(
                                directory,
                                self.embedding_model,
                            )

    def test_save_serialization_failure_preserves_existing_index_pair(self):
        store = FaissVectorStore(self.embedding_model)
        store._index = object()
        store._index_info = {
            "retrieval_embedding_model": self.embedding_model.get_model_name(),
            "embedding_dimension": self.embedding_model.get_dimension(),
            "chunking_configuration": {"chunk_size": 500},
        }
        store._records_by_faiss_id = {
            0: {"chunk_id": "chunk-0", "text": "text"}
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            index_path = directory / "index.faiss"
            metadata_path = directory / "metadata.json"

            with patch(
                "src.member3.vector_store._load_faiss",
                return_value=FakeFaissModule(),
            ):
                store.save_index(directory)
                previous_index = index_path.read_bytes()
                previous_metadata = metadata_path.read_bytes()
                store._records_by_faiss_id[0]["not_json_serializable"] = object()
                with self.assertRaisesRegex(ValueError, "JSON serializable"):
                    store.save_index(directory)

            self.assertEqual(index_path.read_bytes(), previous_index)
            self.assertEqual(metadata_path.read_bytes(), previous_metadata)

    def test_failed_file_replacement_restores_existing_index_pair(self):
        store = FaissVectorStore(self.embedding_model)
        store._index = object()
        store._index_info = {
            "retrieval_embedding_model": self.embedding_model.get_model_name(),
            "embedding_dimension": self.embedding_model.get_dimension(),
            "chunking_configuration": {"chunk_size": 500},
        }
        store._records_by_faiss_id = {
            0: {"chunk_id": "chunk-0", "text": "original text"}
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            index_path = directory / "index.faiss"
            metadata_path = directory / "metadata.json"
            with patch(
                "src.member3.vector_store._load_faiss",
                return_value=FakeFaissModule(),
            ):
                store.save_index(directory)
                previous_index = index_path.read_bytes()
                previous_metadata = metadata_path.read_bytes()

                store._records_by_faiss_id[0]["text"] = "updated text"
                real_replace = os.replace
                replace_calls = 0

                def fail_metadata_install(source, destination):
                    nonlocal replace_calls
                    replace_calls += 1
                    if replace_calls == 4:
                        raise OSError("simulated metadata replacement failure")
                    return real_replace(source, destination)

                with patch(
                    "src.member3.vector_store.os.replace",
                    side_effect=fail_metadata_install,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "previous files were restored",
                    ):
                        store.save_index(directory)

            self.assertEqual(index_path.read_bytes(), previous_index)
            self.assertEqual(metadata_path.read_bytes(), previous_metadata)

    def test_get_index_info_returns_deep_copy(self):
        from src.member3.embeddings import embed_chunk_records

        with patch(
            "src.member3.vector_store._load_faiss",
            return_value=FakeFaissModule(),
        ):
            store = FaissVectorStore(self.embedding_model)
            store.build_index(
                embed_chunk_records(
                    make_chunks("semantic", 1),
                    self.embedding_model,
                )
            )

        returned_info = store.get_index_info()
        returned_info["chunking_configuration"]["similarity_threshold"] = 0.99
        self.assertEqual(
            store.get_index_info()["chunking_configuration"]["similarity_threshold"],
            0.25,
        )

    def test_build_index_copies_nested_metadata_from_input_records(self):
        from src.member3.embeddings import embed_chunk_records

        chunks = make_chunks(count=1)
        chunks[0]["custom_metadata"] = {
            "labels": ["alpha"],
            "details": {"scores": [1, 2]},
        }
        embedded = embed_chunk_records(chunks, self.embedding_model)

        with patch(
            "src.member3.vector_store._load_faiss",
            return_value=FakeFaissModule(),
        ):
            store = FaissVectorStore(self.embedding_model)
            store.build_index(embedded)

            embedded[0]["custom_metadata"]["labels"].append("caller-change")
            embedded[0]["custom_metadata"]["details"]["scores"][0] = 99

            stored_metadata = store._records_by_faiss_id[0]["custom_metadata"]
            self.assertEqual(stored_metadata["labels"], ["alpha"])
            self.assertEqual(stored_metadata["details"]["scores"], [1, 2])

            with tempfile.TemporaryDirectory() as temporary_directory:
                store.save_index(Path(temporary_directory))
                sidecar = json.loads(
                    (Path(temporary_directory) / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )

        saved_metadata = sidecar["records_by_faiss_id"]["0"]["custom_metadata"]
        self.assertEqual(saved_metadata["labels"], ["alpha"])
        self.assertEqual(saved_metadata["details"]["scores"], [1, 2])

    def test_search_results_do_not_expose_nested_internal_metadata(self):
        from src.member3.embeddings import embed_chunk_records

        chunks = make_chunks(count=1)
        chunks[0]["custom_metadata"] = {
            "labels": ["alpha"],
            "details": {"scores": [1, 2]},
        }
        embedded = embed_chunk_records(chunks, self.embedding_model)

        with patch(
            "src.member3.vector_store._load_faiss",
            return_value=FakeFaissModule(),
        ):
            store = FaissVectorStore(self.embedding_model)
            store.build_index(embedded)
            first_result = store.search("search query")
            first_result[0]["metadata"]["custom_metadata"]["labels"].append(
                "caller-change"
            )
            first_result[0]["metadata"]["custom_metadata"]["details"]["scores"][0] = 99

            second_result = store.search("search query")

        metadata = second_result[0]["metadata"]["custom_metadata"]
        self.assertEqual(metadata["labels"], ["alpha"])
        self.assertEqual(metadata["details"]["scores"], [1, 2])

    def test_save_and_reload_preserves_search_results(self):
        from src.member3.embeddings import embed_chunk_records

        embedded = embed_chunk_records(make_chunks(), self.embedding_model)
        store = FaissVectorStore(self.embedding_model)
        store.build_index(embedded)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = store.save_index(Path(temporary_directory) / "saved-index")
            loaded = FaissVectorStore.load_index(path, self.embedding_model)

        results = loaded.search("search query")
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["chunk_id"], "paper1_fixed_size_0")
        self.assertEqual(
            results[0]["metadata"]["retrieval_embedding_model"],
            self.embedding_model.get_model_name(),
        )

    def test_incompatible_query_embedding_model_is_rejected(self):
        from src.member3.embeddings import embed_chunk_records

        embedded = embed_chunk_records(make_chunks(count=2), self.embedding_model)
        store = FaissVectorStore(self.embedding_model)
        store.build_index(embedded)
        incompatible_model = DeterministicEmbeddingModel("another-model")

        with self.assertRaisesRegex(ValueError, "requires embedding model"):
            store.search("search query", embedding_model=incompatible_model)

    def test_empty_input_duplicate_ids_and_mixed_configuration_are_rejected(self):
        store = FaissVectorStore(self.embedding_model)
        with self.assertRaisesRegex(ValueError, "empty input"):
            store.build_index([])

        from src.member3.embeddings import embed_chunk_records

        duplicate_chunks = make_chunks(count=2)
        duplicate_chunks[1]["chunk_id"] = duplicate_chunks[0]["chunk_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
            store.build_index(
                embed_chunk_records(duplicate_chunks, self.embedding_model)
            )

        mixed_chunks = make_chunks(count=2)
        mixed_chunks[1]["overlap"] = 25
        with self.assertRaisesRegex(ValueError, "chunking configurations"):
            store.build_index(
                embed_chunk_records(mixed_chunks, self.embedding_model)
            )

    def test_dimension_and_normalization_mismatches_are_rejected(self):
        from src.member3.embeddings import embed_chunk_records

        store = FaissVectorStore(self.embedding_model)
        chunks = embed_chunk_records(make_chunks(count=1), self.embedding_model)
        chunks[0]["embedding"] = [1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "embedding shape"):
            store.build_index(chunks)

        chunks[0]["embedding"] = [2.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "L2-normalized"):
            store.build_index(chunks)

    def test_missing_index_and_invalid_top_k_are_reported(self):
        store = FaissVectorStore(self.embedding_model)
        with self.assertRaisesRegex(RuntimeError, "before an index"):
            store.search("search query")
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            FaissVectorStore.load_index(
                Path("missing-index"),
                self.embedding_model,
            )

        from src.member3.embeddings import embed_chunk_records

        store.build_index(
            embed_chunk_records(make_chunks(count=1), self.embedding_model)
        )
        with self.assertRaisesRegex(ValueError, "top_k"):
            store.search("search query", top_k=0)


if __name__ == "__main__":
    unittest.main()
