"""Tests for embedding extraction."""

import numpy as np
import pytest


class TestFinBERTEmbedder:
    """Test cases for FinBERT embedder.

    Note: These tests require the FinBERT model to be downloaded.
    Skip if running in CI without GPU/model access.
    """

    @pytest.fixture
    def embedder(self):
        """Create embedder instance.

        Returns None if model can't be loaded (e.g., in CI).
        """
        try:
            from src.nlp.embeddings import FinBERTEmbedder

            return FinBERTEmbedder()
        except Exception:
            pytest.skip("FinBERT model not available")

    def test_embedding_shape(self, embedder) -> None:
        """Test embedding has correct shape."""
        text = "What is the revenue outlook for next quarter?"
        embedding = embedder.embed(text)

        assert embedding.shape == (768,)  # FinBERT hidden size
        assert embedding.dtype == np.float32

    def test_batch_embedding(self, embedder) -> None:
        """Test batch embedding."""
        texts = [
            "What is the margin outlook?",
            "How is competition affecting sales?",
            "What is the guidance for next year?",
        ]

        embeddings = embedder.embed_batch(texts)

        assert embeddings.shape == (3, 768)

    def test_cosine_similarity_range(self, embedder) -> None:
        """Test cosine similarity is in valid range."""
        text1 = "What is the revenue growth?"
        text2 = "Revenue increased by 10% year over year."

        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)

        similarity = embedder.compute_cosine_similarity(emb1, emb2)

        assert -1.0 <= similarity <= 1.0

    def test_similar_texts_high_similarity(self, embedder) -> None:
        """Test that similar texts have high similarity."""
        text1 = "What is the margin performance?"
        text2 = "How are margins performing?"

        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)

        similarity = embedder.compute_cosine_similarity(emb1, emb2)

        # Similar questions should have reasonably high similarity
        assert similarity > 0.5

    def test_compute_alignment_baseline(self, embedder) -> None:
        """Test alignment baseline computation."""
        question = "What is the revenue outlook?"
        answer = "Revenue growth is expected to continue at double digits."

        similarity, q_emb, a_emb = embedder.compute_alignment_baseline(question, answer)

        assert -1.0 <= similarity <= 1.0
        assert q_emb.shape == (768,)
        assert a_emb.shape == (768,)


class TestCosineSimularity:
    """Test cosine similarity without model."""

    def test_identical_vectors(self) -> None:
        """Test identical vectors have similarity 1."""
        from src.nlp.embeddings import FinBERTEmbedder

        # Use static method logic
        vec = np.array([1.0, 2.0, 3.0])

        norm = np.linalg.norm(vec)
        similarity = np.dot(vec, vec) / (norm * norm)

        assert np.isclose(similarity, 1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test orthogonal vectors have similarity 0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        assert np.isclose(similarity, 0.0)
