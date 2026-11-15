import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.member3.embeddings import (
    BGEEmbeddingModel,
    BGE_MODEL_NAME,
    BGE_QUERY_PREFIX,
    EmbeddingConfig,
    EmbeddingModel,
    MiniLMEmbeddingModel,
    MINILM_MODEL_NAME,
    OPENAI_DEFAULT_MODEL_NAME,
    OpenAIEmbeddingModel,
    create_embedding_model,
    embed_chunk_records,
)


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []
        self.fail_encoding = False
        self.vectors = None

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(
        self,
        sentences,
        *,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        if self.fail_encoding:
            raise ValueError("backend failure")
        if self.vectors is not None:
            return self.vectors
        return [[float(len(text)), 3.0, 4.0] for text in sentences]


class FakeOpenAIClient:
    def __init__(self):
        self.calls = []
        self.fail_encoding = False
        self.embeddings = self

    def create(self, **arguments):
        self.calls.append(arguments)
        if self.fail_encoding:
            raise ValueError("API failure")

        data = [
            SimpleNamespace(
                index=index,
                embedding=[float(len(text)), 3.0, 4.0],
            )
            for index, text in enumerate(arguments["input"])
        ]
        return SimpleNamespace(data=list(reversed(data)))


class FakeEmbeddingModel(EmbeddingModel):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, query):
        return [float(len(query)), 1.0]

    def get_dimension(self):
        return 2

    def get_model_name(self):
        return "fake-model"


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeSentenceTransformer()
        self.model = MiniLMEmbeddingModel(
            EmbeddingConfig(
                provider="huggingface",
                model_name="test/model",
                batch_size=2,
            ),
            backend=self.backend,
        )

    def test_all_three_backends_embed_the_same_texts(self):
        sample_texts = ["first sample", "second sample"]
        bge_backend = FakeSentenceTransformer()
        minilm_backend = FakeSentenceTransformer()
        openai_client = FakeOpenAIClient()

        models = [
            BGEEmbeddingModel(
                EmbeddingConfig(provider="bge", batch_size=2),
                backend=bge_backend,
            ),
            MiniLMEmbeddingModel(
                EmbeddingConfig(provider="huggingface", batch_size=2),
                backend=minilm_backend,
            ),
            OpenAIEmbeddingModel(
                EmbeddingConfig(
                    provider="openai",
                    dimensions=3,
                    batch_size=2,
                ),
                client=openai_client,
            ),
        ]

        outputs = [model.embed_documents(sample_texts) for model in models]

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        for model, vectors in zip(models, outputs):
            self.assertEqual(len(vectors), len(sample_texts))
            self.assertTrue(all(len(vector) == model.get_dimension() for vector in vectors))
            self.assertTrue(
                all(
                    abs(sum(value * value for value in vector) - 1.0) < 1e-9
                    for vector in vectors
                )
            )
            stats = model.get_last_run_stats()
            self.assertEqual(stats.model_name, model.get_model_name())
            self.assertEqual(stats.dimension, model.get_dimension())
            self.assertEqual(stats.vector_count, len(sample_texts))
            self.assertGreaterEqual(stats.processing_time_seconds, 0.0)

        self.assertEqual(models[0].get_model_name(), BGE_MODEL_NAME)
        self.assertEqual(models[1].get_model_name(), MINILM_MODEL_NAME)
        self.assertEqual(models[2].get_model_name(), OPENAI_DEFAULT_MODEL_NAME)

    def test_bge_prefixes_queries_but_not_documents(self):
        self.model = BGEEmbeddingModel(
            EmbeddingConfig(provider="bge", batch_size=2),
            backend=self.backend,
        )

        self.model.embed_documents(["passage"])
        self.model.embed_query("question")

        self.assertEqual(self.backend.calls[0]["sentences"], ["passage"])
        self.assertEqual(
            self.backend.calls[1]["sentences"],
            [BGE_QUERY_PREFIX + "question"],
        )

    def test_document_batches_are_bounded_and_keep_order(self):
        embeddings = self.model.embed_documents(
            [" a ", "bb", "ccc", "dddd", "eeeee"]
        )

        self.assertEqual(
            [len(call["sentences"]) for call in self.backend.calls],
            [2, 2, 1],
        )
        self.assertEqual(
            [call["batch_size"] for call in self.backend.calls],
            [2, 2, 1],
        )
        self.assertTrue(
            all(
                abs(sum(value * value for value in vector) - 1.0) < 1e-9
                for vector in embeddings
            )
        )

    def test_openai_batches_and_restores_response_order(self):
        client = FakeOpenAIClient()
        model = OpenAIEmbeddingModel(
            EmbeddingConfig(
                provider="openai",
                model_name="test-embedding-model",
                batch_size=2,
            ),
            client=client,
        )

        vectors = model.embed_documents(["a", "bb", "ccc"])

        self.assertEqual([len(call["input"]) for call in client.calls], [2, 1])
        self.assertEqual([call["model"] for call in client.calls], ["test-embedding-model"] * 2)
        self.assertEqual([len(vector) for vector in vectors], [3, 3, 3])
        self.assertEqual(model.get_last_run_stats().vector_count, 3)

    def test_query_uses_shared_text_preparation(self):
        self.model.embed_query("  query \n")
        self.assertEqual(self.backend.calls[0]["sentences"], ["query"])

    def test_empty_documents_return_without_calling_backend(self):
        self.assertEqual(self.model.embed_documents([]), [])
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.model.get_last_run_stats().vector_count, 0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(TypeError):
            self.model.embed_documents(None)
        with self.assertRaises(TypeError):
            self.model.embed_documents("not a collection")
        with self.assertRaises(ValueError):
            self.model.embed_documents(["  "])
        with self.assertRaises(ValueError):
            self.model.embed_query("")

    def test_inconsistent_or_invalid_vectors_are_rejected(self):
        self.backend.vectors = [[0.0, 0.0, 0.0]]
        with self.assertRaisesRegex(RuntimeError, "non-zero norm"):
            self.model.embed_documents(["text"])

        self.backend.vectors = [[1.0, 2.0]]
        with self.assertRaisesRegex(RuntimeError, "dimension"):
            self.model.embed_documents(["text"])

    def test_model_errors_are_reported_with_context(self):
        self.backend.fail_encoding = True
        with self.assertRaisesRegex(RuntimeError, "test/model") as raised:
            self.model.embed_documents(["text"])
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_openai_errors_do_not_fall_back_to_local_models(self):
        client = FakeOpenAIClient()
        client.fail_encoding = True
        model = OpenAIEmbeddingModel(
            EmbeddingConfig(provider="openai", dimensions=3),
            client=client,
        )
        with self.assertRaisesRegex(RuntimeError, "OpenAI embedding request failed"):
            model.embed_documents(["text"])

    def test_missing_openai_key_does_not_block_local_backend(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                OpenAIEmbeddingModel(
                    EmbeddingConfig(provider="openai", dimensions=3)
                )

            local = MiniLMEmbeddingModel(
                EmbeddingConfig(provider="huggingface"),
                backend=FakeSentenceTransformer(),
            )
            self.assertEqual(len(local.embed_documents(["text"])[0]), 3)
            bge = BGEEmbeddingModel(
                EmbeddingConfig(provider="bge"),
                backend=FakeSentenceTransformer(),
            )
            self.assertEqual(len(bge.embed_documents(["text"])[0]), 3)

    def test_unknown_provider_is_not_substituted(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding provider"):
            create_embedding_model(EmbeddingConfig(provider="unknown"))

    def test_factory_selects_requested_provider(self):
        providers = (
            ("bge", "BGEEmbeddingModel"),
            ("huggingface", "MiniLMEmbeddingModel"),
            ("openai", "OpenAIEmbeddingModel"),
        )
        for provider, class_name in providers:
            with self.subTest(provider=provider):
                target = f"src.member3.embeddings.{class_name}"
                with patch(target) as adapter:
                    model = create_embedding_model(
                        EmbeddingConfig(provider=provider)
                    )
                self.assertIs(model, adapter.return_value)
                adapter.assert_called_once()

    def test_chunk_records_keep_ids_and_all_metadata(self):
        chunk = {
            "chunk_id": "paper1_fixed_500_1",
            "source": "paper1.txt",
            "strategy": "fixed_size",
            "chunk_size": 500,
            "overlap": 50,
            "text": "chunk text",
            "start_index": 0,
            "end_index": 10,
            "embedding_model": "semantic-boundary-model",
            "custom_metadata": {"section": "intro"},
        }

        embedded = embed_chunk_records([chunk], FakeEmbeddingModel())

        self.assertEqual(embedded[0]["chunk_id"], chunk["chunk_id"])
        self.assertEqual(embedded[0]["custom_metadata"], chunk["custom_metadata"])
        self.assertEqual(embedded[0]["embedding"], [10.0, 1.0])
        self.assertEqual(
            embedded[0]["embedding_model"],
            "semantic-boundary-model",
        )
        self.assertEqual(embedded[0]["retrieval_embedding_model"], "fake-model")
        self.assertEqual(embedded[0]["embedding_dimension"], 2)
        self.assertNotIn("embedding", chunk)

    def test_sentence_chunk_schema_is_supported(self):
        chunk = {
            "chunk_id": "paper1_sentence_500_1",
            "source": "paper1.txt",
            "strategy": "sentence_level",
            "target_size": 500,
            "overlap_sentences": 1,
            "text": "sentence chunk",
            "sentence_start_index": 0,
            "sentence_end_index": 3,
        }

        embedded = embed_chunk_records([chunk], FakeEmbeddingModel())

        self.assertEqual(embedded[0]["chunk_id"], chunk["chunk_id"])
        self.assertEqual(embedded[0]["sentence_start_index"], 0)
        self.assertEqual(embedded[0]["embedding"], [14.0, 1.0])

    def test_invalid_chunk_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk_id"):
            embed_chunk_records([{"text": "missing id"}], FakeEmbeddingModel())
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            embed_chunk_records(
                [{"chunk_id": "empty", "text": " "}],
                FakeEmbeddingModel(),
            )
        with self.assertRaises(TypeError):
            embed_chunk_records(None, FakeEmbeddingModel())


if __name__ == "__main__":
    unittest.main()
