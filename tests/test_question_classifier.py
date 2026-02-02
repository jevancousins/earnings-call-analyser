"""Tests for question classifier."""

import pytest

from src.nlp.question_classifier import (
    QuestionCategory,
    QuestionClassification,
    QuestionClassifier,
)


class TestQuestionClassifier:
    """Test cases for QuestionClassifier."""

    @pytest.fixture
    def classifier(self) -> QuestionClassifier:
        """Create classifier instance."""
        return QuestionClassifier()

    def test_classify_margin_question(self, classifier: QuestionClassifier) -> None:
        """Test margin question classification."""
        question = "Can you talk about gross margin expansion and profitability trends?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.MARGINS
        assert result.confidence > 0

    def test_classify_guidance_question(self, classifier: QuestionClassifier) -> None:
        """Test guidance question classification."""
        question = "What is your outlook for revenue guidance next quarter?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.GUIDANCE
        assert "guidance" in result.keywords_matched or "outlook" in result.keywords_matched

    def test_classify_competition_question(self, classifier: QuestionClassifier) -> None:
        """Test competition question classification."""
        question = "How is the competitive landscape affecting your market share?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.COMPETITION

    def test_classify_macro_question(self, classifier: QuestionClassifier) -> None:
        """Test macro question classification."""
        question = "How are macroeconomic conditions and inflation affecting demand?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.MACRO

    def test_classify_revenue_question(self, classifier: QuestionClassifier) -> None:
        """Test revenue question classification."""
        question = "Can you break down revenue growth by segment?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.REVENUE

    def test_classify_capital_question(self, classifier: QuestionClassifier) -> None:
        """Test capital allocation question classification."""
        question = "What are your plans for share buybacks and capital allocation?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.CAPITAL

    def test_classify_unknown_question(self, classifier: QuestionClassifier) -> None:
        """Test question with no clear category."""
        question = "Can you elaborate on that point?"

        result = classifier.classify(question)

        assert result.category == QuestionCategory.OTHER
        assert result.confidence == 0.5

    def test_classify_batch(self, classifier: QuestionClassifier) -> None:
        """Test batch classification."""
        questions = [
            "What is the margin outlook?",
            "How is competition affecting you?",
            "What is your guidance for next year?",
        ]

        results = classifier.classify_batch(questions)

        assert len(results) == 3
        assert all(isinstance(r, QuestionClassification) for r in results)

    def test_category_distribution(self, classifier: QuestionClassifier) -> None:
        """Test category distribution calculation."""
        questions = [
            "What about margins?",
            "What about revenue?",
            "What about margins again?",
        ]

        distribution = classifier.get_category_distribution(questions)

        assert isinstance(distribution, dict)
        assert all(cat in distribution for cat in QuestionCategory)
        assert sum(distribution.values()) == len(questions)


class TestQuestionCategory:
    """Test QuestionCategory enum."""

    def test_all_categories(self) -> None:
        """Test all expected categories exist."""
        expected = [
            "REVENUE",
            "MARGINS",
            "GUIDANCE",
            "COMPETITION",
            "MACRO",
            "CAPITAL",
            "OPERATIONS",
            "PRODUCT",
            "REGULATORY",
            "OTHER",
        ]

        for cat in expected:
            assert hasattr(QuestionCategory, cat)

    def test_category_values(self) -> None:
        """Test category values are lowercase strings."""
        for cat in QuestionCategory:
            assert cat.value == cat.name.lower()
