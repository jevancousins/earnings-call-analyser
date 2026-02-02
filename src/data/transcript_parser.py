"""Parser for extracting Q&A pairs from earnings call transcripts."""

import re
from dataclasses import dataclass, field
from enum import Enum


class SpeakerRole(Enum):
    """Role of a speaker in the earnings call."""

    OPERATOR = "operator"
    EXECUTIVE = "executive"
    ANALYST = "analyst"
    UNKNOWN = "unknown"


@dataclass
class Speaker:
    """Speaker information."""

    name: str
    title: str | None = None
    firm: str | None = None
    role: SpeakerRole = SpeakerRole.UNKNOWN

    def __str__(self) -> str:
        parts = [self.name]
        if self.title:
            parts.append(f"({self.title})")
        if self.firm:
            parts.append(f"- {self.firm}")
        return " ".join(parts)


@dataclass
class SpeechSegment:
    """A single speech segment from a speaker."""

    speaker: Speaker
    text: str
    is_question: bool = False


@dataclass
class QAPairData:
    """Extracted Q&A pair from transcript."""

    sequence_number: int
    analyst_name: str
    analyst_firm: str | None
    question_text: str
    responder_name: str
    responder_title: str | None
    answer_text: str
    follow_up_questions: list[str] = field(default_factory=list)
    follow_up_answers: list[str] = field(default_factory=list)


class TranscriptParser:
    """Parser for earnings call transcripts.

    Handles various transcript formats and extracts structured Q&A pairs.
    """

    # Common executive titles
    EXECUTIVE_TITLES = {
        "ceo",
        "chief executive officer",
        "cfo",
        "chief financial officer",
        "coo",
        "chief operating officer",
        "cto",
        "chief technology officer",
        "president",
        "chairman",
        "vice president",
        "vp",
        "evp",
        "svp",
        "executive vice president",
        "senior vice president",
        "director",
        "head of",
        "treasurer",
        "controller",
        "general counsel",
    }

    # Analyst firm indicators
    ANALYST_FIRM_INDICATORS = [
        "capital",
        "securities",
        "partners",
        "research",
        "advisors",
        "management",
        "investments",
        "bank",
        "group",
        "llc",
        "lp",
        "l.p.",
        "& co",
        "morgan",
        "goldman",
        "jpmorgan",
        "citi",
        "barclays",
        "credit suisse",
        "ubs",
        "deutsche",
        "wells fargo",
        "bofa",
        "jefferies",
        "piper",
        "wedbush",
        "needham",
        "cowen",
        "baird",
        "stifel",
        "oppenheimer",
        "bernstein",
        "evercore",
        "wolfe",
        "keybanc",
    ]

    # Q&A section markers
    QA_SECTION_MARKERS = [
        r"question[s]?\s*(?:and|&)\s*answer[s]?",
        r"q\s*&\s*a",
        r"q&a\s+session",
        r"operator instructions",
        r"we will now take questions",
        r"we'll now take questions",
        r"open the floor for questions",
        r"open it up for questions",
        r"begin the question",
        r"first question",
    ]

    def __init__(self) -> None:
        """Initialize the parser."""
        self._qa_section_pattern = re.compile(
            "|".join(self.QA_SECTION_MARKERS), re.IGNORECASE
        )

    def parse(self, transcript: str) -> list[QAPairData]:
        """Parse a transcript and extract Q&A pairs.

        Args:
            transcript: Raw transcript text

        Returns:
            List of QAPairData objects
        """
        if not transcript or not transcript.strip():
            return []

        # Find Q&A section
        qa_section = self._extract_qa_section(transcript)
        if not qa_section:
            # Try parsing entire transcript if no clear section markers
            qa_section = transcript

        # Parse into speech segments
        segments = self._parse_segments(qa_section)

        # Group into Q&A pairs
        qa_pairs = self._group_qa_pairs(segments)

        return qa_pairs

    def _extract_qa_section(self, transcript: str) -> str | None:
        """Extract the Q&A section from the transcript.

        Args:
            transcript: Full transcript text

        Returns:
            Q&A section text or None
        """
        match = self._qa_section_pattern.search(transcript)
        if match:
            return transcript[match.start() :]

        return None

    def _parse_segments(self, text: str) -> list[SpeechSegment]:
        """Parse text into speech segments.

        Args:
            text: Text to parse

        Returns:
            List of SpeechSegment objects
        """
        segments: list[SpeechSegment] = []

        # Pattern to match speaker headers
        # Common formats:
        # "John Smith - CEO"
        # "John Smith, Analyst, Goldman Sachs"
        # "[John Smith] (Analyst)"
        # "John Smith:"
        speaker_patterns = [
            # "Name - Title" or "Name - Company"
            r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*[-–—]\s*(?P<info>[^:\n]+)(?::|$)",
            # "Name, Title" or "Name, Firm"
            r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*,\s*(?P<info>[^:\n]+)(?::|$)",
            # [Name] or (Name)
            r"[\[\(](?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})[\]\)]\s*(?P<info>[^:\n]*)(?::|$)",
            # Simple "Name:"
            r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}):\s*(?P<info>)",
        ]

        combined_pattern = "|".join(f"(?:{p})" for p in speaker_patterns)

        # Split text by speaker headers
        parts = re.split(f"({combined_pattern})", text, flags=re.MULTILINE)

        current_speaker: Speaker | None = None
        current_text_parts: list[str] = []

        for part in parts:
            if not part or not part.strip():
                continue

            # Try to match as speaker header
            speaker = self._parse_speaker_header(part)

            if speaker:
                # Save previous segment
                if current_speaker and current_text_parts:
                    text_content = " ".join(current_text_parts).strip()
                    if text_content:
                        is_question = self._is_question(text_content, current_speaker)
                        segments.append(
                            SpeechSegment(
                                speaker=current_speaker,
                                text=text_content,
                                is_question=is_question,
                            )
                        )
                current_speaker = speaker
                current_text_parts = []
            else:
                # Add to current speaker's text
                current_text_parts.append(part)

        # Save final segment
        if current_speaker and current_text_parts:
            text_content = " ".join(current_text_parts).strip()
            if text_content:
                is_question = self._is_question(text_content, current_speaker)
                segments.append(
                    SpeechSegment(
                        speaker=current_speaker,
                        text=text_content,
                        is_question=is_question,
                    )
                )

        return segments

    def _parse_speaker_header(self, text: str) -> Speaker | None:
        """Parse a potential speaker header.

        Args:
            text: Text that might be a speaker header

        Returns:
            Speaker object or None
        """
        text = text.strip()

        # Skip if too long (not a header)
        if len(text) > 200:
            return None

        # Skip operator lines
        if text.lower().startswith("operator"):
            return Speaker(name="Operator", role=SpeakerRole.OPERATOR)

        # Try different patterns
        patterns = [
            r"^(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*[-–—]\s*(?P<info>.+?)(?::|$)",
            r"^(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*,\s*(?P<info>.+?)(?::|$)",
            r"^(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}):?\s*$",
        ]

        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                name = match.group("name").strip()
                info = match.group("info").strip() if "info" in match.groupdict() else ""

                title, firm, role = self._parse_speaker_info(info)

                return Speaker(name=name, title=title, firm=firm, role=role)

        return None

    def _parse_speaker_info(
        self, info: str
    ) -> tuple[str | None, str | None, SpeakerRole]:
        """Parse speaker info string to extract title, firm, and role.

        Args:
            info: Info string like "CEO" or "Analyst, Goldman Sachs"

        Returns:
            Tuple of (title, firm, role)
        """
        if not info:
            return None, None, SpeakerRole.UNKNOWN

        info_lower = info.lower()

        # Check for executive titles
        for title in self.EXECUTIVE_TITLES:
            if title in info_lower:
                return info, None, SpeakerRole.EXECUTIVE

        # Check for analyst firm indicators
        for indicator in self.ANALYST_FIRM_INDICATORS:
            if indicator in info_lower:
                # Try to split title and firm
                parts = re.split(r",\s*", info, maxsplit=1)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip(), SpeakerRole.ANALYST
                return "Analyst", info, SpeakerRole.ANALYST

        # Check for explicit "Analyst" role
        if "analyst" in info_lower:
            parts = re.split(r",\s*", info, maxsplit=1)
            if len(parts) == 2:
                return "Analyst", parts[1].strip(), SpeakerRole.ANALYST
            return "Analyst", None, SpeakerRole.ANALYST

        return info if info else None, None, SpeakerRole.UNKNOWN

    def _is_question(self, text: str, speaker: Speaker) -> bool:
        """Determine if a speech segment is a question.

        Args:
            text: Speech text
            speaker: Speaker who said it

        Returns:
            True if this appears to be a question
        """
        # Operator segments are neither questions nor answers
        if speaker.role == SpeakerRole.OPERATOR:
            return False

        # Analysts typically ask questions
        if speaker.role == SpeakerRole.ANALYST:
            return True

        # Executives typically give answers
        if speaker.role == SpeakerRole.EXECUTIVE:
            return False

        # Heuristics for unknown roles
        question_indicators = [
            "?",
            "could you",
            "can you",
            "would you",
            "will you",
            "how do",
            "how does",
            "how will",
            "what is",
            "what are",
            "what was",
            "why is",
            "why did",
            "when will",
            "where is",
            "is there",
            "are there",
            "do you",
            "does the",
            "i was wondering",
            "i wanted to ask",
            "my question",
            "follow-up",
            "follow up",
        ]

        text_lower = text.lower()
        return any(ind in text_lower for ind in question_indicators)

    def _group_qa_pairs(self, segments: list[SpeechSegment]) -> list[QAPairData]:
        """Group speech segments into Q&A pairs.

        Args:
            segments: List of parsed speech segments

        Returns:
            List of QAPairData objects
        """
        qa_pairs: list[QAPairData] = []
        sequence = 0

        i = 0
        while i < len(segments):
            segment = segments[i]

            # Skip operator segments
            if segment.speaker.role == SpeakerRole.OPERATOR:
                i += 1
                continue

            # Look for question
            if segment.is_question:
                question_segment = segment
                answer_segments: list[SpeechSegment] = []
                follow_up_questions: list[str] = []
                follow_up_answers: list[str] = []

                # Collect answer segments
                j = i + 1
                while j < len(segments):
                    next_seg = segments[j]

                    # Skip operator
                    if next_seg.speaker.role == SpeakerRole.OPERATOR:
                        j += 1
                        continue

                    # If another question from same analyst, it's a follow-up
                    if (
                        next_seg.is_question
                        and next_seg.speaker.name == question_segment.speaker.name
                    ):
                        follow_up_questions.append(next_seg.text)
                        j += 1
                        # Look for answer to follow-up
                        if j < len(segments) and not segments[j].is_question:
                            follow_up_answers.append(segments[j].text)
                            j += 1
                        continue

                    # If question from different analyst, we're done
                    if next_seg.is_question:
                        break

                    # Executive answer
                    if next_seg.speaker.role == SpeakerRole.EXECUTIVE:
                        answer_segments.append(next_seg)

                    j += 1

                # Create Q&A pair if we have both question and answer
                if answer_segments:
                    sequence += 1
                    combined_answer = " ".join(seg.text for seg in answer_segments)
                    primary_responder = answer_segments[0].speaker

                    qa_pairs.append(
                        QAPairData(
                            sequence_number=sequence,
                            analyst_name=question_segment.speaker.name,
                            analyst_firm=question_segment.speaker.firm,
                            question_text=question_segment.text,
                            responder_name=primary_responder.name,
                            responder_title=primary_responder.title,
                            answer_text=combined_answer,
                            follow_up_questions=follow_up_questions,
                            follow_up_answers=follow_up_answers,
                        )
                    )

                i = j
            else:
                i += 1

        return qa_pairs

    def get_qa_count(self, transcript: str) -> int:
        """Quick count of Q&A pairs without full parsing.

        Args:
            transcript: Raw transcript text

        Returns:
            Estimated number of Q&A pairs
        """
        # Simple heuristic: count question marks in Q&A section
        qa_section = self._extract_qa_section(transcript)
        if not qa_section:
            qa_section = transcript

        # Count patterns that look like analyst questions
        question_pattern = r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\s*[-,].*?:.*?\?)"
        matches = re.findall(question_pattern, qa_section)
        return len(matches) if matches else qa_section.count("?") // 2
