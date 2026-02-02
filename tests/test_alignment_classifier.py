"""Tests for alignment classifier model."""

import numpy as np
import pytest
import torch

from src.model.alignment_classifier import (
    AlignmentClassifier,
    AlignmentLabel,
    AlignmentLoss,
    AlignmentOutput,
)


class TestAlignmentClassifier:
    """Test cases for AlignmentClassifier."""

    @pytest.fixture
    def model(self) -> AlignmentClassifier:
        """Create model instance."""
        return AlignmentClassifier(
            embedding_dim=768,
            hidden_dim=256,
            num_classes=3,
            dropout=0.1,
        )

    def test_model_forward(self, model: AlignmentClassifier) -> None:
        """Test forward pass."""
        batch_size = 4
        q_emb = torch.randn(batch_size, 768)
        a_emb = torch.randn(batch_size, 768)

        output = model(q_emb, a_emb)

        assert isinstance(output, AlignmentOutput)
        assert output.logits.shape == (batch_size, 3)
        assert output.alignment_score.shape == (batch_size,)
        assert output.question_projected.shape == (batch_size, 256)
        assert output.answer_projected.shape == (batch_size, 256)

    def test_model_predict(self, model: AlignmentClassifier) -> None:
        """Test prediction output."""
        batch_size = 4
        q_emb = torch.randn(batch_size, 768)
        a_emb = torch.randn(batch_size, 768)

        labels, scores, probs = model.predict(q_emb, a_emb)

        assert labels.shape == (batch_size,)
        assert scores.shape == (batch_size,)
        assert probs.shape == (batch_size, 3)

        # Scores should be in [0, 1]
        assert (scores >= 0).all() and (scores <= 1).all()

        # Labels should be valid class indices
        assert (labels >= 0).all() and (labels < 3).all()

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))

    def test_alignment_score_range(self, model: AlignmentClassifier) -> None:
        """Test alignment scores are in valid range."""
        q_emb = torch.randn(10, 768)
        a_emb = torch.randn(10, 768)

        output = model(q_emb, a_emb)

        assert (output.alignment_score >= 0).all()
        assert (output.alignment_score <= 1).all()

    def test_model_deterministic_eval(self, model: AlignmentClassifier) -> None:
        """Test model is deterministic in eval mode."""
        model.eval()
        q_emb = torch.randn(4, 768)
        a_emb = torch.randn(4, 768)

        output1 = model(q_emb, a_emb)
        output2 = model(q_emb, a_emb)

        assert torch.allclose(output1.logits, output2.logits)
        assert torch.allclose(output1.alignment_score, output2.alignment_score)


class TestAlignmentLoss:
    """Test cases for AlignmentLoss."""

    @pytest.fixture
    def loss_fn(self) -> AlignmentLoss:
        """Create loss function."""
        return AlignmentLoss(
            classification_weight=1.0,
            regression_weight=0.5,
            contrastive_weight=0.3,
        )

    def test_loss_computation(self, loss_fn: AlignmentLoss) -> None:
        """Test loss computation."""
        batch_size = 4

        output = AlignmentOutput(
            logits=torch.randn(batch_size, 3),
            alignment_score=torch.sigmoid(torch.randn(batch_size)),
            question_projected=torch.randn(batch_size, 256),
            answer_projected=torch.randn(batch_size, 256),
        )

        labels = torch.randint(0, 3, (batch_size,))
        target_scores = torch.rand(batch_size)
        is_matched = torch.randint(0, 2, (batch_size,)).float()

        loss, components = loss_fn(output, labels, target_scores, is_matched)

        assert loss.shape == ()
        assert loss.item() >= 0

        assert "classification" in components
        assert "regression" in components
        assert "contrastive" in components
        assert "total" in components

    def test_loss_gradients(self, loss_fn: AlignmentLoss) -> None:
        """Test that loss allows gradient computation."""
        model = AlignmentClassifier()
        q_emb = torch.randn(4, 768, requires_grad=True)
        a_emb = torch.randn(4, 768, requires_grad=True)

        output = model(q_emb, a_emb)

        labels = torch.randint(0, 3, (4,))
        target_scores = torch.rand(4)

        loss, _ = loss_fn(output, labels, target_scores)
        loss.backward()

        # Check gradients exist
        assert q_emb.grad is not None
        assert a_emb.grad is not None


class TestAlignmentLabel:
    """Test AlignmentLabel enum."""

    def test_label_values(self) -> None:
        """Test label enum values."""
        assert AlignmentLabel.ALIGNED.value == 0
        assert AlignmentLabel.PARTIALLY_ALIGNED.value == 1
        assert AlignmentLabel.EVASIVE.value == 2

    def test_label_count(self) -> None:
        """Test number of labels."""
        assert len(AlignmentLabel) == 3
