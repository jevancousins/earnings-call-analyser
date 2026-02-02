"""FinBERT embedding extraction for questions and answers."""

from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import get_settings


class FinBERTEmbedder:
    """Extract embeddings using ProsusAI/FinBERT.

    FinBERT is specifically trained on financial text and performs
    better than general-purpose models for finance domain tasks.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: Literal["cpu", "cuda", "mps"] | None = None,
        max_length: int = 512,
    ) -> None:
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model name. Defaults to ProsusAI/finbert.
            device: Device to run model on. Defaults to config setting.
            max_length: Maximum token length for inputs.
        """
        settings = get_settings()
        self.model_name = model_name or settings.finbert_model
        self.device = device or settings.model_device
        self.max_length = max_length

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Embedding dimension
        self.embedding_dim = self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model output
        outputs = self.model(**inputs)

        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        return cls_embedding.cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model output
            outputs = self.model(**inputs)

            # Use [CLS] token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def embed_mean_pooling(self, text: str) -> np.ndarray:
        """Get embedding using mean pooling over all tokens.

        This can capture more information than just [CLS] token.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Get attention mask for proper mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state

        # Mask padding tokens
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Sum token embeddings and divide by mask sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask

        return mean_embedding.squeeze().cpu().numpy()

    def compute_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def compute_alignment_baseline(
        self, question: str, answer: str
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute baseline alignment score using cosine similarity.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Tuple of (similarity_score, question_embedding, answer_embedding)
        """
        q_embedding = self.embed(question)
        a_embedding = self.embed(answer)

        similarity = self.compute_cosine_similarity(q_embedding, a_embedding)

        return similarity, q_embedding, a_embedding

    def batch_compute_alignment(
        self, questions: list[str], answers: list[str], batch_size: int = 32
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        """Compute alignment scores for multiple Q&A pairs.

        Args:
            questions: List of questions
            answers: List of answers
            batch_size: Batch size for embedding extraction

        Returns:
            Tuple of (similarity_scores, question_embeddings, answer_embeddings)
        """
        if len(questions) != len(answers):
            raise ValueError("Questions and answers must have same length")

        q_embeddings = self.embed_batch(questions, batch_size)
        a_embeddings = self.embed_batch(answers, batch_size)

        # Compute cosine similarities
        similarities = []
        for q_emb, a_emb in zip(q_embeddings, a_embeddings):
            sim = self.compute_cosine_similarity(q_emb, a_emb)
            similarities.append(sim)

        return similarities, q_embeddings, a_embeddings


class SentenceTransformerEmbedder:
    """Alternative embedder using sentence-transformers.

    Useful for comparison with FinBERT.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Literal["cpu", "cuda", "mps"] | None = None,
    ) -> None:
        """Initialize sentence transformer embedder.

        Args:
            model_name: Model name from sentence-transformers
            device: Device to run on
        """
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.device = device or settings.model_device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for multiple texts."""
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
