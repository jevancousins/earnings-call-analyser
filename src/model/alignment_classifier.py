"""PyTorch model for Q&A alignment classification using contrastive learning."""

from enum import Enum
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLabel(Enum):
    """Alignment classification labels."""

    ALIGNED = 0
    PARTIALLY_ALIGNED = 1
    EVASIVE = 2


class AlignmentOutput(NamedTuple):
    """Output from the alignment classifier."""

    logits: torch.Tensor  # (batch_size, num_classes)
    alignment_score: torch.Tensor  # (batch_size,) values in [0, 1]
    question_projected: torch.Tensor  # (batch_size, hidden_dim)
    answer_projected: torch.Tensor  # (batch_size, hidden_dim)


class AlignmentClassifier(nn.Module):
    """Contrastive learning model for Q&A alignment classification.

    This model projects question and answer embeddings into a shared space
    and learns to classify their alignment level.

    Architecture:
    - Separate projection heads for questions and answers
    - Interaction layer combining projected embeddings
    - Classification head for 3-class output (aligned, partial, evasive)
    - Regression head for continuous alignment score

    Training uses a combination of:
    - Cross-entropy loss for classification
    - MSE loss for alignment score regression
    - Contrastive loss to push matched pairs together and mismatched apart
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the alignment classifier.

        Args:
            embedding_dim: Input embedding dimension from FinBERT (768)
            hidden_dim: Hidden layer dimension
            num_classes: Number of alignment classes (3)
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Question projection head
        self.question_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Answer projection head
        self.answer_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Interaction layer
        # Takes concatenation, element-wise difference, and element-wise product
        # Input: [q_proj; a_proj; q_proj - a_proj; q_proj * a_proj] = 4 * hidden_dim
        interaction_dim = hidden_dim * 4

        self.interaction = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head: aligned / partially_aligned / evasive
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Alignment score head (0-1 regression)
        self.alignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        question_embeddings: torch.Tensor,
        answer_embeddings: torch.Tensor,
    ) -> AlignmentOutput:
        """Forward pass.

        Args:
            question_embeddings: Question embeddings (batch_size, embedding_dim)
            answer_embeddings: Answer embeddings (batch_size, embedding_dim)

        Returns:
            AlignmentOutput with logits, scores, and projected embeddings
        """
        # Project to shared space
        q_proj = self.question_proj(question_embeddings)
        a_proj = self.answer_proj(answer_embeddings)

        # Compute interaction features
        concat = torch.cat([q_proj, a_proj], dim=-1)
        diff = q_proj - a_proj
        product = q_proj * a_proj
        interaction_input = torch.cat([concat, diff, product], dim=-1)

        # Get interaction representation
        interaction_output = self.interaction(interaction_input)

        # Classification
        logits = self.classifier(interaction_output)

        # Alignment score
        alignment_score = self.alignment_head(interaction_output).squeeze(-1)

        return AlignmentOutput(
            logits=logits,
            alignment_score=alignment_score,
            question_projected=q_proj,
            answer_projected=a_proj,
        )

    def predict(
        self,
        question_embeddings: torch.Tensor,
        answer_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get predictions without computing gradients.

        Args:
            question_embeddings: Question embeddings
            answer_embeddings: Answer embeddings

        Returns:
            Tuple of (predicted_labels, alignment_scores, probabilities)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(question_embeddings, answer_embeddings)
            probs = F.softmax(output.logits, dim=-1)
            labels = torch.argmax(probs, dim=-1)
            return labels, output.alignment_score, probs


class AlignmentLoss(nn.Module):
    """Combined loss function for alignment classification.

    Combines:
    - Cross-entropy for classification
    - MSE for alignment score regression
    - Contrastive loss for embedding space learning
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 0.5,
        contrastive_weight: float = 0.3,
        margin: float = 0.5,
        temperature: float = 0.07,
    ) -> None:
        """Initialize the loss function.

        Args:
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
            contrastive_weight: Weight for contrastive loss
            margin: Margin for contrastive loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.temperature = temperature

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        output: AlignmentOutput,
        labels: torch.Tensor,
        target_scores: torch.Tensor,
        is_matched: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            output: AlignmentOutput from model forward pass
            labels: Ground truth labels (batch_size,)
            target_scores: Ground truth alignment scores (batch_size,)
            is_matched: Binary tensor indicating matched vs mismatched pairs

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Classification loss
        cls_loss = self.ce_loss(output.logits, labels)

        # Regression loss
        reg_loss = self.mse_loss(output.alignment_score, target_scores)

        # Contrastive loss (if we have matched/mismatched pairs)
        if is_matched is not None and self.contrastive_weight > 0:
            contrastive_loss = self._compute_contrastive_loss(
                output.question_projected,
                output.answer_projected,
                is_matched,
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=output.logits.device)

        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss
            + self.regression_weight * reg_loss
            + self.contrastive_weight * contrastive_loss
        )

        loss_components = {
            "classification": cls_loss.item(),
            "regression": reg_loss.item(),
            "contrastive": contrastive_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_components

    def _compute_contrastive_loss(
        self,
        q_proj: torch.Tensor,
        a_proj: torch.Tensor,
        is_matched: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Pushes matched Q&A pairs together and mismatched pairs apart.

        Args:
            q_proj: Projected question embeddings
            a_proj: Projected answer embeddings
            is_matched: Binary labels (1 for matched, 0 for mismatched)

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        q_norm = F.normalize(q_proj, p=2, dim=-1)
        a_norm = F.normalize(a_proj, p=2, dim=-1)

        # Cosine similarity
        similarity = torch.sum(q_norm * a_norm, dim=-1)

        # Contrastive loss: matched pairs should have high similarity,
        # mismatched pairs should have low similarity
        matched_mask = is_matched.float()
        mismatched_mask = 1 - matched_mask

        # For matched: loss = 1 - similarity
        matched_loss = matched_mask * (1 - similarity)

        # For mismatched: loss = max(0, similarity - margin)
        mismatched_loss = mismatched_mask * F.relu(similarity - self.margin)

        # Average over batch
        loss = (matched_loss + mismatched_loss).mean()

        return loss


class SimCSELoss(nn.Module):
    """SimCSE-style contrastive loss for self-supervised learning.

    Based on "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
    """

    def __init__(self, temperature: float = 0.05) -> None:
        """Initialize SimCSE loss.

        Args:
            temperature: Temperature scaling factor
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SimCSE loss.

        Args:
            embeddings1: First set of embeddings (batch_size, dim)
            embeddings2: Second set of embeddings (batch_size, dim)

        Returns:
            Loss value
        """
        # Normalize
        z1 = F.normalize(embeddings1, p=2, dim=-1)
        z2 = F.normalize(embeddings2, p=2, dim=-1)

        batch_size = z1.size(0)

        # Similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z1.device)

        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
