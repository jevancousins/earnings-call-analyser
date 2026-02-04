"""Training utilities for the alignment classifier."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import get_settings
from src.model.alignment_classifier import (
    AlignmentClassifier,
    AlignmentLabel,
    AlignmentLoss,
)

logger = logging.getLogger(__name__)


@dataclass
class QADataPoint:
    """A single Q&A data point for training."""

    question_embedding: np.ndarray
    answer_embedding: np.ndarray
    label: int  # 0=aligned, 1=partial, 2=evasive
    alignment_score: float  # 0-1 continuous score
    is_matched: bool = True  # True if original pair, False if shuffled


class QADataset(Dataset):
    """PyTorch dataset for Q&A alignment training."""

    def __init__(
        self,
        data_points: list[QADataPoint],
        create_hard_negatives: bool = True,
        negative_ratio: float = 0.3,
    ) -> None:
        """Initialize dataset.

        Args:
            data_points: List of QADataPoint objects
            create_hard_negatives: Whether to create hard negative pairs
            negative_ratio: Ratio of negative to positive samples
        """
        self.data_points = data_points
        self.create_hard_negatives = create_hard_negatives
        self.negative_ratio = negative_ratio

        # Create indices for hard negative sampling
        self._question_embeddings = np.stack(
            [dp.question_embedding for dp in data_points]
        )
        self._answer_embeddings = np.stack([dp.answer_embedding for dp in data_points])

    def __len__(self) -> int:
        return len(self.data_points)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dp = self.data_points[idx]

        return {
            "question_embedding": torch.tensor(dp.question_embedding, dtype=torch.float32),
            "answer_embedding": torch.tensor(dp.answer_embedding, dtype=torch.float32),
            "label": torch.tensor(dp.label, dtype=torch.long),
            "alignment_score": torch.tensor(dp.alignment_score, dtype=torch.float32),
            "is_matched": torch.tensor(dp.is_matched, dtype=torch.float32),
        }

    def create_negative_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a batch of hard negative pairs.

        Hard negatives: questions paired with wrong answers from same transcript.

        Args:
            batch_size: Number of negative pairs to create

        Returns:
            Tuple of (q_emb, a_emb, labels, scores, is_matched)
        """
        indices = np.random.choice(len(self.data_points), batch_size, replace=False)

        # Shuffle answer indices to create mismatches
        shuffled_indices = np.random.permutation(indices)

        q_emb = torch.tensor(self._question_embeddings[indices], dtype=torch.float32)
        a_emb = torch.tensor(
            self._answer_embeddings[shuffled_indices], dtype=torch.float32
        )

        # Mismatched pairs are labeled as evasive with low alignment score
        labels = torch.full((batch_size,), AlignmentLabel.EVASIVE.value, dtype=torch.long)
        scores = torch.full((batch_size,), 0.1, dtype=torch.float32)
        is_matched = torch.zeros(batch_size, dtype=torch.float32)

        return q_emb, a_emb, labels, scores, is_matched


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Optimisation
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_epochs: int = 10
    batch_size: int = 32

    # Loss weights
    classification_weight: float = 1.0
    regression_weight: float = 0.5
    contrastive_weight: float = 0.3

    # Scheduler
    scheduler: Literal["cosine", "onecycle"] = "cosine"

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True


class AlignmentTrainer:
    """Trainer for the alignment classifier."""

    def __init__(
        self,
        model: AlignmentClassifier,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: AlignmentClassifier instance
            config: Training configuration
            device: Device to train on
        """
        settings = get_settings()
        self.device = device or settings.model_device
        self.config = config or TrainingConfig()

        self.model = model.to(self.device)
        self.loss_fn = AlignmentLoss(
            classification_weight=self.config.classification_weight,
            regression_weight=self.config.regression_weight,
            contrastive_weight=self.config.contrastive_weight,
        )

        # Setup optimiser
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, Any]] = []

    def _setup_scheduler(self, num_training_steps: int) -> Any:
        """Setup learning rate scheduler."""
        if self.config.scheduler == "onecycle":
            return OneCycleLR(
                self.optimiser,
                max_lr=self.config.learning_rate,
                total_steps=num_training_steps,
                pct_start=0.1,
            )
        else:  # cosine
            return CosineAnnealingLR(
                self.optimiser,
                T_max=num_training_steps,
                eta_min=self.config.learning_rate / 10,
            )

    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: Any | None = None,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            scheduler: Learning rate scheduler

        Returns:
            Dict of average loss components
        """
        self.model.train()
        total_losses: dict[str, float] = {
            "classification": 0.0,
            "regression": 0.0,
            "contrastive": 0.0,
            "total": 0.0,
        }
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in progress_bar:
            # Move to device
            q_emb = batch["question_embedding"].to(self.device)
            a_emb = batch["answer_embedding"].to(self.device)
            labels = batch["label"].to(self.device)
            scores = batch["alignment_score"].to(self.device)
            is_matched = batch["is_matched"].to(self.device)

            # Forward pass
            output = self.model(q_emb, a_emb)

            # Compute loss
            loss, loss_components = self.loss_fn(output, labels, scores, is_matched)

            # Backward pass
            self.optimiser.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimiser.step()

            if scheduler is not None:
                scheduler.step()

            # Accumulate losses
            for key, value in loss_components.items():
                total_losses[key] += value
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss_components['total']:.4f}",
                cls=f"{loss_components['classification']:.4f}",
            )

        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dict of metrics including loss and accuracy
        """
        self.model.eval()
        total_losses: dict[str, float] = {
            "classification": 0.0,
            "regression": 0.0,
            "contrastive": 0.0,
            "total": 0.0,
        }
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            q_emb = batch["question_embedding"].to(self.device)
            a_emb = batch["answer_embedding"].to(self.device)
            labels = batch["label"].to(self.device)
            scores = batch["alignment_score"].to(self.device)
            is_matched = batch["is_matched"].to(self.device)

            output = self.model(q_emb, a_emb)
            _, loss_components = self.loss_fn(output, labels, scores, is_matched)

            for key, value in loss_components.items():
                total_losses[key] += value

            # Classification accuracy
            preds = torch.argmax(output.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        num_batches = len(val_loader)
        metrics = {key: value / num_batches for key, value in total_losses.items()}
        metrics["accuracy"] = correct / total if total > 0 else 0.0

        # Per-class accuracy
        for label in AlignmentLabel:
            mask = np.array(all_labels) == label.value
            if mask.sum() > 0:
                class_acc = (
                    (np.array(all_preds)[mask] == label.value).sum() / mask.sum()
                )
                metrics[f"accuracy_{label.name.lower()}"] = float(class_acc)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training history
        """
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = self._setup_scheduler(num_training_steps)

        # Create checkpoint directory
        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, scheduler)
            logger.info(f"Epoch {epoch + 1} - Train loss: {train_metrics['total']:.4f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                logger.info(
                    f"Epoch {epoch + 1} - Val loss: {val_metrics['total']:.4f}, "
                    f"Val acc: {val_metrics['accuracy']:.4f}"
                )

                # Early stopping check
                if val_metrics["total"] < self.best_loss - self.config.min_delta:
                    self.best_loss = val_metrics["total"]
                    self.patience_counter = 0

                    # Save best model
                    if self.config.save_best_only:
                        self.save_checkpoint(
                            checkpoint_path / "best_model.pt", val_metrics
                        )
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                val_metrics = {}

            # Record history
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

        return {
            "history": self.training_history,
            "best_loss": self.best_loss,
            "final_epoch": self.current_epoch + 1,
        }

    def save_checkpoint(self, path: Path | str, metrics: dict[str, float]) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            metrics: Metrics to include in checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "epoch": self.current_epoch,
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint


def create_labels_from_returns(
    returns: list[float],
    high_threshold: float = 0.03,
    low_threshold: float = -0.03,
) -> tuple[list[int], list[float]]:
    """Create alignment labels from stock returns.

    Uses 30-day forward returns as proxy for answer quality.
    High returns = likely aligned answers
    Low returns = likely evasive answers

    Args:
        returns: List of 30-day forward returns
        high_threshold: Return above this = aligned
        low_threshold: Return below this = evasive

    Returns:
        Tuple of (labels, normalized_scores)
    """
    labels = []
    scores = []

    for ret in returns:
        if ret >= high_threshold:
            labels.append(AlignmentLabel.ALIGNED.value)
            # Map return to score (higher return = higher score)
            score = min(1.0, 0.7 + (ret - high_threshold) * 2)
        elif ret <= low_threshold:
            labels.append(AlignmentLabel.EVASIVE.value)
            score = max(0.0, 0.3 + (ret - low_threshold) * 2)
        else:
            labels.append(AlignmentLabel.PARTIALLY_ALIGNED.value)
            # Linear interpolation in the middle range
            score = 0.3 + (ret - low_threshold) / (high_threshold - low_threshold) * 0.4

        scores.append(score)

    return labels, scores
