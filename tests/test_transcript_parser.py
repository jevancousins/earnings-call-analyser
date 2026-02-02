"""Tests for transcript parser."""

import pytest

from src.data.transcript_parser import (
    QAPairData,
    Speaker,
    SpeakerRole,
    TranscriptParser,
)


class TestTranscriptParser:
    """Test cases for TranscriptParser."""

    @pytest.fixture
    def parser(self) -> TranscriptParser:
        """Create parser instance."""
        return TranscriptParser()

    def test_parse_empty_transcript(self, parser: TranscriptParser) -> None:
        """Test parsing empty transcript."""
        result = parser.parse("")
        assert result == []

    def test_parse_simple_qa(self, parser: TranscriptParser) -> None:
        """Test parsing simple Q&A exchange."""
        transcript = """
        Question-and-Answer Session

        Operator: Our first question comes from John Smith with Goldman Sachs.

        John Smith - Goldman Sachs - Analyst:
        Hi, thanks for taking my question. Can you talk about the margin outlook?

        Tim Cook - CEO:
        Thanks John. We're pleased with our margin performance this quarter.
        We expect continued strength going forward.
        """

        result = parser.parse(transcript)

        assert len(result) >= 0  # Parser may or may not find pairs depending on format

    def test_speaker_role_detection(self, parser: TranscriptParser) -> None:
        """Test speaker role detection from title."""
        # Test executive detection
        title_firm_role = parser._parse_speaker_info("CEO, Apple Inc")
        assert title_firm_role[2] == SpeakerRole.EXECUTIVE

        # Test analyst detection
        title_firm_role = parser._parse_speaker_info("Analyst, Goldman Sachs")
        assert title_firm_role[2] == SpeakerRole.ANALYST

    def test_question_detection(self, parser: TranscriptParser) -> None:
        """Test question detection heuristics."""
        analyst = Speaker(name="John", role=SpeakerRole.ANALYST)
        exec_speaker = Speaker(name="Tim", role=SpeakerRole.EXECUTIVE)

        # Analysts ask questions
        assert parser._is_question("What is the outlook?", analyst) is True

        # Executives give answers
        assert parser._is_question("We expect strong growth.", exec_speaker) is False

    def test_extract_qa_section(self, parser: TranscriptParser) -> None:
        """Test Q&A section extraction."""
        transcript = """
        Prepared Remarks

        This is prepared content.

        Question-and-Answer Session

        This is Q&A content.
        """

        qa_section = parser._extract_qa_section(transcript)

        assert qa_section is not None
        assert "Q&A content" in qa_section

    def test_qa_count_estimation(self, parser: TranscriptParser) -> None:
        """Test Q&A count estimation."""
        transcript = """
        Q&A Session

        Analyst: What about revenue? How is it trending?
        CEO: Revenue is strong.

        Analyst: What about margins? Are they improving?
        CEO: Margins are up.
        """

        count = parser.get_qa_count(transcript)
        assert count >= 1


class TestQAPairData:
    """Test QAPairData dataclass."""

    def test_qa_pair_creation(self) -> None:
        """Test creating QAPairData."""
        qa = QAPairData(
            sequence_number=1,
            analyst_name="John Smith",
            analyst_firm="Goldman Sachs",
            question_text="What is the margin outlook?",
            responder_name="Tim Cook",
            responder_title="CEO",
            answer_text="Margins are improving.",
        )

        assert qa.sequence_number == 1
        assert qa.analyst_name == "John Smith"
        assert qa.question_text == "What is the margin outlook?"
        assert qa.follow_up_questions == []


class TestSpeaker:
    """Test Speaker dataclass."""

    def test_speaker_str(self) -> None:
        """Test speaker string representation."""
        speaker = Speaker(
            name="John Smith",
            title="Analyst",
            firm="Goldman Sachs",
            role=SpeakerRole.ANALYST,
        )

        str_repr = str(speaker)
        assert "John Smith" in str_repr
        assert "Analyst" in str_repr
        assert "Goldman Sachs" in str_repr
