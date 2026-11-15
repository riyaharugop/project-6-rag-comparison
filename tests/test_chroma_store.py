import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.member3.chroma_store import ChromaVectorStore
from src.member3.embeddings import EmbeddingModel, embed_chunk_records


_RECORD_KEY = "_member3_record_json"


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
        }
        self.queries = []

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, query):
        self.queries.append(query)
        return self.vectors[query]

    def get_dimension(self):
        return 3

    def get_model_name(self):
        return self.model_name


class FakeCollection:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self.configuration = {
            "hnsw": {"space": metadata.get("hnsw:space")}
        }
        self.records = {}
        self.add_calls = 0
        self.fail_on_add_call = None

    def count(self):
        return len(self.records)

    def modify(self, metadata):
        self.metadata = metadata

    def add(self, ids, documents, embeddings, metadatas):
        if self.metadata.get("build_status") != "building":
            raise AssertionError("Records must be added while the build is incomplete")
        if not isinstance(self.metadata.get("expected_record_count"), int):
            raise AssertionError("The expected record count must be persisted")
        self.add_calls += 1
        if self.add_calls == self.fail_on_add_call:
            self.fail_on_add_call = None
            raise RuntimeError("simulated later batch failure")
        for chunk_id, text, vector, metadata in zip(
            ids,
            documents,
            embeddings,
            metadatas,
        ):
            if chunk_id in self.records:
                raise ValueError("duplicate ID")
            self.records[chunk_id] = {
                "text": text,
                "embedding": vector,
                "metadata": metadata,
            }

    def query(self, query_embeddings, n_results, include):
        if not {"documents", "metadatas", "distances"}.issubset(include):
            raise AssertionError("Search must request documents, metadata and distances")
        query = query_embeddings[0]
        ranked = sorted(
            (
                (
                    1.0 - float(np.dot(query, item["embedding"])),
                    chunk_id,
                    item,
                )
                for chunk_id, item in self.records.items()
            ),
            key=lambda result: result[0],
        )[:n_results]
        return {
            "ids": [[chunk_id for _, chunk_id, _ in ranked]],
            "documents": [[item["text"] for _, _, item in ranked]],
            "metadatas": [[item["metadata"] for _, _, item in ranked]],
            "distances": [[distance for distance, _, _ in ranked]],
        }


class FakeChromaClient:
    def __init__(self, collections=None):
        self.collections = collections if collections is not None else {}
        self.fail_next_collection_on_add_call = None
        self.deleted_collections = []

    def get_or_create_collection(self, name, metadata, embedding_function):
        if embedding_function is not None:
            raise AssertionError("Chroma default embedding function must be disabled")
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, metadata)
            self.collections[name].fail_on_add_call = (
                self.fail_next_collection_on_add_call
            )
            self.fail_next_collection_on_add_call = None
        return self.collections[name]

    def delete_collection(self, name):
        del self.collections[name]
        self.deleted_collections.append(name)

    def get_collection(self, name, embedding_function):
        if embedding_function is not None:
            raise AssertionError("Chroma default embedding function must be disabled")
        try:
            return self.collections[name]
        except KeyError as error:
            raise FileNotFoundError(f"Chroma collection {name} not found") from error


def make_chunks(count=5, strategy="fixed_size"):
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
            chunk.update({"chunk_size": 500, "overlap": 50})
        elif strategy == "sentence_level":
            chunk.update({"target_size": 500, "overlap_sentences": 1})
        elif strategy == "semantic":
            chunk.update({
                "target_size": 500,
                "min_size": 150,
                "similarity_threshold": 0.25,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "overlap_sentences": 1,
            })
        chunks.append(chunk)
    return chunks


class ChromaVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.embedding_model = DeterministicEmbeddingModel()
        self.client = FakeChromaClient()

    def _build(self, chunks=None, batch_size=256):
        embedded = embed_chunk_records(
            chunks if chunks is not None else make_chunks(),
            self.embedding_model,
        )
        store = ChromaVectorStore(
            self.embedding_model,
            persist_directory=Path("test-chroma"),
            client=self.client,
        )
        info = store.build_index(embedded, batch_size=batch_size)
        return store, info, embedded

    def test_build_search_preserves_ids_metadata_and_similarity_order(self):
        store, info, _ = self._build(batch_size=2)

        results = store.search("search query", top_k=5)

        self.assertEqual(len(results), 5)
        self.assertEqual([result["rank"] for result in results], [1, 2, 3, 4, 5])
        self.assertEqual(results[0]["chunk_id"], "paper1_fixed_size_0")
        self.assertEqual(results[0]["text"], "passage-0")
        self.assertEqual(
            results[0]["metadata"]["retrieval_embedding_model"],
            self.embedding_model.get_model_name(),
        )
        self.assertEqual(results[0]["metadata"]["custom_metadata"], {"section": "test"})
        scores = [result["similarity_score"] for result in results]
        self.assertTrue(all(-1.0 <= score <= 1.0 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(info["similarity_metric"], "cosine")
        self.assertEqual(store._collection.metadata["build_status"], "complete")
        self.assertEqual(store._collection.metadata["expected_record_count"], 5)
        self.assertGreaterEqual(
            store.get_index_info()["retrieval_latency_seconds"],
            0.0,
        )
        self.assertEqual(self.embedding_model.queries, ["search query"])

    def test_build_accepts_different_per_chunk_metadata(self):
        chunks = make_chunks(count=2)
        chunks[0]["custom_metadata"] = {"tags": ["first"], "source_group": "a"}
        chunks[1]["custom_metadata"] = {"tags": ["second"], "source_group": "b"}

        store, info, _ = self._build(chunks)

        self.assertEqual(info["chunk_count"], 2)
        self.assertEqual(
            [result["metadata"]["custom_metadata"] for result in store.search("search query", top_k=2)],
            [chunks[0]["custom_metadata"], chunks[1]["custom_metadata"]],
        )

    def test_later_batch_failure_cleans_up_and_allows_retry(self):
        client = FakeChromaClient()
        client.fail_next_collection_on_add_call = 2
        store = ChromaVectorStore(self.embedding_model, client=client)
        records = embed_chunk_records(make_chunks(count=3), self.embedding_model)

        with self.assertRaisesRegex(RuntimeError, "simulated later batch failure"):
            store.build_index(records, batch_size=1)

        self.assertEqual(len(client.deleted_collections), 1)
        self.assertEqual(client.collections, {})

        info = store.build_index(records, batch_size=1)

        self.assertEqual(info["chunk_count"], 3)
        self.assertEqual(store._collection.metadata["build_status"], "complete")
        self.assertEqual(len(store.search("search query", top_k=3)), 3)

    def test_cleanup_failure_preserves_original_error_and_incomplete_marker(self):
        class CleanupFailureClient(FakeChromaClient):
            fail_delete = True

            def delete_collection(self, name):
                if self.fail_delete:
                    raise OSError("simulated cleanup failure")
                super().delete_collection(name)

        client = CleanupFailureClient()
        client.fail_next_collection_on_add_call = 2
        store = ChromaVectorStore(self.embedding_model, client=client)
        records = embed_chunk_records(make_chunks(count=3), self.embedding_model)

        with self.assertLogs("src.member3.chroma_store", level="WARNING"):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated later batch failure",
            ):
                store.build_index(records, batch_size=1)

        collection_name, collection = next(iter(client.collections.items()))
        self.assertEqual(collection.metadata["build_status"], "building")
        with self.assertRaisesRegex(ValueError, "not marked complete"):
            ChromaVectorStore.load_index(
                Path("test-chroma"),
                collection_name,
                self.embedding_model,
                client=client,
            )

        client.fail_delete = False
        info = store.build_index(records, batch_size=1)
        self.assertEqual(info["chunk_count"], 3)
        self.assertEqual(store._collection.metadata["build_status"], "complete")

    def test_load_rejects_building_collection(self):
        store, info, _ = self._build()
        store._collection.modify(
            metadata={**store._collection.metadata, "build_status": "building"}
        )

        with self.assertRaisesRegex(ValueError, "not marked complete"):
            ChromaVectorStore.load_index(
                Path("test-chroma"),
                info["collection_name"],
                self.embedding_model,
                client=self.client,
            )

    def test_load_rejects_completed_collection_with_wrong_count(self):
        store, info, _ = self._build()
        store._collection.modify(
            metadata={
                **store._collection.metadata,
                "expected_record_count": info["chunk_count"] + 1,
            }
        )

        with self.assertRaisesRegex(ValueError, "expected 6"):
            ChromaVectorStore.load_index(
                Path("test-chroma"),
                info["collection_name"],
                self.embedding_model,
                client=self.client,
            )

    def test_build_does_not_overwrite_completed_collection(self):
        store, info, records = self._build()
        original_count = store._collection.count()

        with self.assertRaisesRegex(ValueError, "already complete"):
            store.build_index(records)

        self.assertEqual(store._collection.count(), original_count)
        self.assertEqual(store._collection.metadata["build_status"], "complete")
        self.assertEqual(info["chunk_count"], original_count)

    def test_build_rejects_bad_provenance_vectors_duplicates_and_mixed_config(self):
        store = ChromaVectorStore(
            self.embedding_model,
            client=self.client,
        )
        records = embed_chunk_records(make_chunks(count=1), self.embedding_model)

        invalid = [dict(records[0])]
        invalid[0]["retrieval_embedding_model"] = "another-model"
        with self.assertRaisesRegex(ValueError, "retrieval embedding model"):
            store.build_index(invalid)

        invalid = [dict(records[0])]
        invalid[0]["embedding_dimension"] = 4
        with self.assertRaisesRegex(ValueError, "embedding dimension"):
            store.build_index(invalid)

        invalid = [dict(records[0])]
        invalid[0]["embedding"] = [float("nan"), 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "not finite"):
            store.build_index(invalid)

        duplicate = [dict(records[0]), dict(records[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
            store.build_index(duplicate)

        mixed = embed_chunk_records(make_chunks(count=2), self.embedding_model)
        mixed[1]["overlap"] = 25
        with self.assertRaisesRegex(ValueError, "chunking configurations"):
            store.build_index(mixed)

    def test_collections_are_separated_by_model_corpus_and_configuration(self):
        store_one, info_one, _ = self._build()

        model_two = DeterministicEmbeddingModel("another-model")
        store_two = ChromaVectorStore(model_two, client=self.client)
        info_two = store_two.build_index(
            embed_chunk_records(make_chunks(), model_two)
        )

        other_corpus = make_chunks()
        other_corpus[0]["source"] = "paper2.txt"
        store_three = ChromaVectorStore(self.embedding_model, client=self.client)
        info_three = store_three.build_index(
            embed_chunk_records(other_corpus, self.embedding_model)
        )

        other_configuration = make_chunks()
        for chunk in other_configuration:
            chunk["overlap"] = 25
        store_four = ChromaVectorStore(self.embedding_model, client=self.client)
        info_four = store_four.build_index(
            embed_chunk_records(other_configuration, self.embedding_model)
        )

        self.assertEqual(len({
            info_one["collection_name"],
            info_two["collection_name"],
            info_three["collection_name"],
            info_four["collection_name"],
        }), 4)
        self.assertIsNot(store_one._collection, store_three._collection)

    def test_semantic_boundary_model_is_preserved(self):
        store, _, _ = self._build(make_chunks(count=1, strategy="semantic"))
        metadata = store.search("search query", top_k=1)[0]["metadata"]

        self.assertEqual(
            metadata["embedding_model"],
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(
            metadata["retrieval_embedding_model"],
            self.embedding_model.get_model_name(),
        )

    def test_nested_metadata_is_independent_from_inputs_and_results(self):
        chunks = make_chunks(count=1)
        chunks[0]["custom_metadata"] = {
            "labels": ["alpha"],
            "details": {"scores": [1, 2]},
        }
        store, _, embedded = self._build(chunks)
        collection = store._collection

        chunks[0]["custom_metadata"]["labels"].append("original-source-change")
        chunks[0]["custom_metadata"]["details"]["scores"].append(3)
        embedded[0]["custom_metadata"]["labels"].append("caller-change")
        embedded[0]["custom_metadata"]["details"]["scores"][0] = 99
        first_result = store.search("search query")
        first_result[0]["metadata"]["custom_metadata"]["labels"].append(
            "result-change"
        )
        first_result[0]["metadata"]["custom_metadata"]["details"]["scores"][1] = 88
        second_result = store.search("search query")

        expected = {
            "labels": ["alpha"],
            "details": {"scores": [1, 2]},
        }
        self.assertEqual(
            second_result[0]["metadata"]["custom_metadata"],
            expected,
        )
        stored_record = json.loads(
            collection.records["paper1_fixed_size_0"]["metadata"][_RECORD_KEY]
        )
        self.assertEqual(stored_record["custom_metadata"], expected)

    def test_save_rejects_nonpersistent_client(self):
        store, info, _ = self._build()
        with self.assertRaisesRegex(RuntimeError, "requires a persistent Chroma client"):
            store.save_index()

        loaded = ChromaVectorStore.load_index(
            Path("test-chroma"),
            info["collection_name"],
            self.embedding_model,
            client=self.client,
        )
        self.assertEqual(loaded.get_index_info()["index_id"], info["index_id"])

    def test_search_logs_required_fields_and_does_not_log_query_text(self):
        store, _, _ = self._build()

        with self.assertLogs("src.member3.chroma_store", level="INFO") as captured:
            results = store.search("search query", top_k=2)

        self.assertEqual(len(captured.records), 1)
        log_record = captured.records[0]
        self.assertEqual(log_record.store_name, "ChromaVectorStore")
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
        self.assertEqual(log_record.top_k, 2)
        self.assertGreaterEqual(log_record.elapsed_seconds, 0.0)

    def test_search_logs_empty_results(self):
        store, _, _ = self._build()
        store._collection.records.clear()

        with self.assertLogs("src.member3.chroma_store", level="INFO") as captured:
            results = store.search("search query")

        self.assertEqual(results, [])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].chunk_ids, [])
        self.assertEqual(captured.records[0].similarity_scores, [])

    def test_missing_collection_and_incompatible_query_model_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            ChromaVectorStore.load_index(
                Path("missing"),
                "member3-missing",
                self.embedding_model,
                client=self.client,
            )

        store, _, _ = self._build()
        other_model = DeterministicEmbeddingModel("another-model")
        with self.assertRaisesRegex(ValueError, "requires embedding model"):
            store.search("search query", embedding_model=other_model)

    def test_invalid_empty_input_and_top_k_are_rejected(self):
        store = ChromaVectorStore(self.embedding_model, client=self.client)
        with self.assertRaisesRegex(ValueError, "empty input"):
            store.build_index([])
        with self.assertRaisesRegex(RuntimeError, "before build_index"):
            store.search("search query")

        built, _, _ = self._build()
        with self.assertRaisesRegex(ValueError, "top_k"):
            built.search("search query", top_k=0)


try:
    import chromadb as _chromadb
except ImportError:
    _chromadb = None


@unittest.skipIf(_chromadb is None, "chromadb package is not installed")
class ChromaPersistenceIntegrationTests(unittest.TestCase):
    def test_real_persistent_collection_can_be_reopened(self):
        model = DeterministicEmbeddingModel()
        with tempfile.TemporaryDirectory() as temporary_directory:
            client = _chromadb.PersistentClient(path=temporary_directory)
            reopen_client = None
            try:
                store = ChromaVectorStore(
                    model,
                    persist_directory=Path(temporary_directory),
                    client=client,
                )
                records = embed_chunk_records(
                    make_chunks(count=2),
                    model,
                )
                info = store.build_index(records)
                self.assertEqual(store.save_index(), Path(temporary_directory))
                reopen_client = _chromadb.PersistentClient(
                    path=temporary_directory
                )
                reopened = ChromaVectorStore.load_index(
                    Path(temporary_directory),
                    info["collection_name"],
                    model,
                    client=reopen_client,
                )

                self.assertEqual(
                    reopened.search("search query", top_k=2)[0]["chunk_id"],
                    "paper1_fixed_size_0",
                )
            finally:
                if reopen_client is not None:
                    reopen_client.close()
                client.close()
