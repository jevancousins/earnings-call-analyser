"""Question type classifier for earnings call Q&A."""

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn


class QuestionCategory(Enum):
    """Categories of analyst questions in earnings calls."""

    REVENUE = "revenue"  # Revenue, sales, top-line
    MARGINS = "margins"  # Gross margin, operating margin, profitability
    GUIDANCE = "guidance"  # Forward-looking, outlook, expectations
    COMPETITION = "competition"  # Competitive dynamics, market share
    MACRO = "macro"  # Macroeconomic conditions, interest rates
    CAPITAL = "capital"  # CapEx, buybacks, dividends, M&A
    OPERATIONS = "operations"  # Supply chain, manufacturing, efficiency
    PRODUCT = "product"  # Product launches, features, roadmap
    REGULATORY = "regulatory"  # Regulations, compliance, legal
    OTHER = "other"


@dataclass
class QuestionClassification:
    """Result of question classification."""

    category: QuestionCategory
    confidence: float
    keywords_matched: list[str]


class QuestionClassifier:
    """Classify analyst questions by topic category.

    Uses a combination of keyword matching and optional neural classification.
    """

    # Keywords for each category
    CATEGORY_KEYWORDS: dict[QuestionCategory, list[str]] = {
        QuestionCategory.REVENUE: [
            "revenue",
            "sales",
            "top line",
            "top-line",
            "bookings",
            "orders",
            "backlog",
            "growth rate",
            "volume",
            "pricing",
            "arpu",
            "arr",
            "mrr",
            "subscription",
            "recurring",
        ],
        QuestionCategory.MARGINS: [
            "margin",
            "gross margin",
            "operating margin",
            "ebitda",
            "profitability",
            "profit",
            "cost structure",
            "opex",
            "expenses",
            "leverage",
            "efficiency",
            "cost savings",
            "cost reduction",
        ],
        QuestionCategory.GUIDANCE: [
            "guidance",
            "outlook",
            "forecast",
            "expect",
            "anticipate",
            "looking ahead",
            "next quarter",
            "next year",
            "fy24",
            "fy25",
            "2024",
            "2025",
            "target",
            "goal",
            "projection",
            "estimate",
        ],
        QuestionCategory.COMPETITION: [
            "competition",
            "competitor",
            "competitive",
            "market share",
            "win rate",
            "versus",
            "compared to",
            "differentiation",
            "moat",
            "advantage",
        ],
        QuestionCategory.MACRO: [
            "macro",
            "economy",
            "economic",
            "recession",
            "inflation",
            "interest rate",
            "fed",
            "tariff",
            "trade",
            "geopolitical",
            "china",
            "europe",
            "demand environment",
            "spending environment",
        ],
        QuestionCategory.CAPITAL: [
            "capital allocation",
            "capex",
            "capital expenditure",
            "buyback",
            "repurchase",
            "dividend",
            "acquisition",
            "m&a",
            "merger",
            "deal",
            "balance sheet",
            "debt",
            "cash",
            "liquidity",
        ],
        QuestionCategory.OPERATIONS: [
            "supply chain",
            "manufacturing",
            "production",
            "capacity",
            "inventory",
            "logistics",
            "operational",
            "headcount",
            "hiring",
            "layoff",
            "restructuring",
        ],
        QuestionCategory.PRODUCT: [
            "product",
            "launch",
            "roadmap",
            "feature",
            "innovation",
            "r&d",
            "research",
            "development",
            "pipeline",
            "new offering",
            "release",
        ],
        QuestionCategory.REGULATORY: [
            "regulatory",
            "regulation",
            "compliance",
            "legal",
            "lawsuit",
            "antitrust",
            "ftc",
            "sec",
            "investigation",
            "settlement",
        ],
    }

    def __init__(self, use_neural: bool = False) -> None:
        """Initialize the classifier.

        Args:
            use_neural: Whether to use neural classification (requires trained model)
        """
        self.use_neural = use_neural
        self._neural_model: nn.Module | None = None

        # Compile keyword patterns for efficiency
        self._patterns: dict[QuestionCategory, re.Pattern] = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            pattern = r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
            self._patterns[category] = re.compile(pattern, re.IGNORECASE)

    def classify(self, question: str) -> QuestionClassification:
        """Classify a single question.

        Args:
            question: Question text

        Returns:
            QuestionClassification with category and confidence
        """
        question_lower = question.lower()

        # Score each category based on keyword matches
        scores: dict[QuestionCategory, float] = {}
        matched_keywords: dict[QuestionCategory, list[str]] = {}

        for category, pattern in self._patterns.items():
            matches = pattern.findall(question_lower)
            scores[category] = len(matches)
            matched_keywords[category] = list(set(matches))

        # Get top category
        if all(s == 0 for s in scores.values()):
            return QuestionClassification(
                category=QuestionCategory.OTHER, confidence=0.5, keywords_matched=[]
            )

        top_category = max(scores, key=lambda k: scores[k])
        total_matches = sum(scores.values())
        confidence = scores[top_category] / total_matches if total_matches > 0 else 0.0

        return QuestionClassification(
            category=top_category,
            confidence=min(confidence, 1.0),
            keywords_matched=matched_keywords[top_category],
        )

    def classify_batch(self, questions: list[str]) -> list[QuestionClassification]:
        """Classify multiple questions.

        Args:
            questions: List of question texts

        Returns:
            List of QuestionClassification objects
        """
        return [self.classify(q) for q in questions]

    def get_category_distribution(
        self, questions: list[str]
    ) -> dict[QuestionCategory, int]:
        """Get distribution of categories across questions.

        Args:
            questions: List of question texts

        Returns:
            Dict mapping category to count
        """
        classifications = self.classify_batch(questions)
        distribution: dict[QuestionCategory, int] = {cat: 0 for cat in QuestionCategory}

        for clf in classifications:
            distribution[clf.category] += 1

        return distribution


class NeuralQuestionClassifier(nn.Module):
    """Neural network for question classification.

    Can be trained on labeled data for improved accuracy.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the neural classifier.

        Args:
            embedding_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of question categories
            dropout: Dropout rate
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.classifier(embeddings)

    def predict(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with probabilities.

        Args:
            embeddings: Input embeddings

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(embeddings)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            return preds, probs
