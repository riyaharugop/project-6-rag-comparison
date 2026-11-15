import unittest
from pathlib import Path

from src.member3.run_retrieval_experiment import (
    ChunkDataset,
    _analyze_results,
    discover_chunk_datasets,
    load_chunk_records,
    load_queries,
    make_retrieval_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RetrievalExperimentTests(unittest.TestCase):
    def test_existing_chunk_outputs_cover_all_six_configurations(self):
        datasets = discover_chunk_datasets(PROJECT_ROOT)

        self.assertEqual(len(datasets), 6)
        self.assertEqual(
            {dataset.strategy for dataset in datasets},
            {"fixed_size", "sentence_level", "semantic"},
        )
        self.assertTrue(all(len(dataset.sources) == 5 for dataset in datasets))
        for dataset in datasets:
            records = load_chunk_records(dataset)
            self.assertEqual(len(records), dataset.chunk_count)
            self.assertTrue(all(record["strategy"] == dataset.strategy for record in records))

        fixed_500 = next(
            dataset
            for dataset in datasets
            if dataset.strategy == "fixed_size"
            and dataset.configuration["chunk_size"] == 500
        )
        semantic_500 = next(
            dataset
            for dataset in datasets
            if dataset.strategy == "semantic"
            and dataset.configuration["target_size"] == 500
        )
        self.assertEqual(fixed_500.configuration["overlap"], 50)
        self.assertEqual(semantic_500.configuration["similarity_threshold"], 0.25)
        self.assertEqual(semantic_500.configuration["min_size"], 150)

    def test_query_set_is_synthetic_and_unlabeled(self):
        query_set = load_queries(PROJECT_ROOT / "data" / "retrieval_experiment_queries.json")

        self.assertTrue(query_set["synthetic"])
        self.assertFalse(query_set["relevance_labels_available"])
        self.assertEqual(len(query_set["queries"]), 5)
        self.assertEqual(
            len({query["query_id"] for query in query_set["queries"]}),
            5,
        )

    def test_retrieval_record_contains_required_comparison_fields(self):
        dataset = ChunkDataset(
            strategy="fixed_size",
            configuration={"chunk_size": 500, "overlap": 50},
            files=(),
            source_file_hashes={},
            sources=("paper1.txt",),
            chunk_count=1,
            configuration_id="fixed-test",
        )
        result = {
            "chunk_id": "paper1_fixed_500_1",
            "similarity_score": 0.9,
        }

        record = make_retrieval_record(
            run_id="run-test",
            experiment_id="experiment-test",
            dataset=dataset,
            embedding_model="BAAI/bge-small-en-v1.5",
            vector_store="ChromaDB",
            query={"query_id": "q01", "query": "not stored in result"},
            results=[result],
            retrieval_latency_seconds=0.01,
            indexed_chunk_count=1,
            repeatability_status="passed",
            repeat_latency_seconds=0.009,
            warnings=[],
        )

        self.assertEqual(
            record["retrieved_chunk_ids"],
            ["paper1_fixed_500_1"],
        )
        self.assertEqual(record["similarity_scores"], [0.9])
        self.assertEqual(record["top_k"], 5)
        self.assertEqual(record["query_id"], "q01")
        self.assertNotIn("query", record)
        self.assertTrue({
            "experiment_id",
            "chunking_strategy",
            "chunking_configuration",
            "embedding_model",
            "vector_store",
            "retrieval_latency_seconds",
            "indexed_chunk_count",
            "warnings",
            "validation_status",
        }.issubset(record))

    def test_analysis_compares_ids_only_within_the_same_chunk_output(self):
        config_one = {
            "experiment_id": "one",
            "status": "success",
            "embedding_model": "model",
            "chunking_strategy": "fixed_size",
            "chunking_configuration": {"chunk_size": 500},
            "vector_store": "ChromaDB",
            "number_of_chunks": 10,
            "index_build_time_seconds": 0.1,
            "source_file_hashes": {"data/chunks/paper1.json": "same"},
        }
        config_two = {
            **config_one,
            "experiment_id": "two",
            "vector_store": "FAISS",
        }
        config_three = {
            **config_one,
            "experiment_id": "three",
            "chunking_configuration": {"chunk_size": 1000},
        }
        config_four = {
            **config_one,
            "experiment_id": "four",
            "embedding_model": "other-model",
        }
        rows = [
            {
                "experiment_id": "one",
                "query_id": "q01",
                "retrieved_chunk_ids": ["a", "b"],
                "similarity_scores": [0.9, 0.8],
                "retrieval_latency_seconds": 0.01,
            },
            {
                "experiment_id": "two",
                "query_id": "q01",
                "retrieved_chunk_ids": ["b", "c"],
                "similarity_scores": [0.7, 0.6],
                "retrieval_latency_seconds": 0.02,
            },
            {
                "experiment_id": "three",
                "query_id": "q01",
                "retrieved_chunk_ids": ["x", "y"],
                "similarity_scores": [0.8, 0.7],
                "retrieval_latency_seconds": 0.03,
            },
            {
                "experiment_id": "four",
                "query_id": "q01",
                "retrieved_chunk_ids": ["a", "c"],
                "similarity_scores": [0.8, 0.7],
                "retrieval_latency_seconds": 0.04,
            },
        ]

        analysis = _analyze_results(
            [config_one, config_two, config_three, config_four],
            rows,
        )

        self.assertIn("not retrieval quality", analysis["interpretation"])
        self.assertIn("configuration-specific", analysis["chunk_id_comparison_note"])
        comparisons = analysis["retrieved_id_overlap_comparisons"]
        self.assertEqual(len(comparisons), 3)
        self.assertTrue(
            all(item["chunking_configuration"] == {"chunk_size": 500} for item in comparisons)
        )
        store_pair = next(
            item
            for item in comparisons
            if {item["left_vector_store"], item["right_vector_store"]}
            == {"ChromaDB", "FAISS"}
            and item["left_embedding_model"] == item["right_embedding_model"]
        )
        self.assertEqual(store_pair["mean_top_k_id_jaccard"], 1 / 3)
        self.assertEqual(store_pair["shared_query_count"], 1)


if __name__ == "__main__":
    unittest.main()
