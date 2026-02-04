"""Analysis endpoints for processing transcripts."""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.session import get_db_dependency

router = APIRouter()


class TranscriptInput(BaseModel):
    """Input for transcript analysis."""

    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL"])
    transcript: str | None = Field(
        None, description="Raw transcript text (if not fetching from API)"
    )
    fiscal_year: int | None = Field(None, description="Fiscal year")
    fiscal_quarter: int | None = Field(
        None, ge=1, le=4, description="Fiscal quarter (1-4)"
    )
    fetch_from_api: bool = Field(
        True, description="Whether to fetch transcript from API sources"
    )


class ManualTranscriptInput(BaseModel):
    """Input for submitting a manual transcript."""

    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL"])
    transcript: str = Field(..., description="Raw transcript text")
    fiscal_year: int = Field(..., description="Fiscal year")
    fiscal_quarter: int = Field(..., ge=1, le=4, description="Fiscal quarter (1-4)")


class QAPairResult(BaseModel):
    """Single Q&A pair analysis result."""

    sequence_number: int
    analyst_name: str
    analyst_firm: str | None
    question_text: str
    question_category: str
    responder_name: str
    responder_title: str | None
    answer_text: str
    alignment_score: float = Field(..., ge=0, le=1)
    alignment_label: str
    cosine_similarity: float
    model_confidence: float


class AnalysisResult(BaseModel):
    """Complete transcript analysis result."""

    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    call_date: datetime
    overall_alignment_score: float
    total_qa_pairs: int
    qa_pairs: list[QAPairResult]
    category_distribution: dict[str, int]
    responder_scores: dict[str, float]
    data_source: str | None = None  # Which provider supplied the transcript


class AnalysisStatus(BaseModel):
    """Status of an analysis job."""

    job_id: str
    status: str
    progress: float
    message: str | None = None
    result: AnalysisResult | None = None


# In-memory job storage (use Redis in production)
_analysis_jobs: dict[str, AnalysisStatus] = {}


@router.post("/analyse", response_model=AnalysisStatus)
async def analyse_transcript(
    input_data: TranscriptInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_dependency),
) -> AnalysisStatus:
    """Analyse an earnings call transcript.

    This endpoint initiates analysis of a transcript. For large transcripts,
    processing happens in the background.

    Args:
        input_data: Transcript input data
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Analysis status with job ID for tracking
    """
    import uuid

    job_id = str(uuid.uuid4())

    # Create initial status
    status = AnalysisStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Analysis queued",
    )
    _analysis_jobs[job_id] = status

    # Add background task
    background_tasks.add_task(
        _run_analysis,
        job_id,
        input_data,
        db,
    )

    return status


@router.post("/analyse/manual", response_model=AnalysisStatus)
async def analyse_manual_transcript(
    input_data: ManualTranscriptInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_dependency),
) -> AnalysisStatus:
    """Analyse a manually submitted transcript.

    Use this endpoint when you have transcript text from a source
    not supported by the API (e.g., pasted from investor relations site).

    Args:
        input_data: Manual transcript input data
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Analysis status with job ID for tracking
    """
    # Convert to standard TranscriptInput
    standard_input = TranscriptInput(
        ticker=input_data.ticker,
        transcript=input_data.transcript,
        fiscal_year=input_data.fiscal_year,
        fiscal_quarter=input_data.fiscal_quarter,
        fetch_from_api=False,
    )
    return await analyse_transcript(standard_input, background_tasks, db)


@router.get("/providers", response_model=dict)
async def get_available_providers() -> dict:
    """Get list of available transcript providers.

    Returns information about which transcript data sources are
    configured and available for use.

    Returns:
        Dict with provider status information
    """
    from src.data.transcript_provider import TranscriptProvider, TranscriptSource

    async with TranscriptProvider() as provider:
        available = provider.available_providers()

    return {
        "available_providers": [p.value for p in available],
        "all_providers": [p.value for p in TranscriptSource if p != TranscriptSource.MANUAL],
        "primary_provider": available[0].value if available else None,
    }


@router.get("/analyse/{job_id}", response_model=AnalysisStatus)
async def get_analysis_status(job_id: str) -> AnalysisStatus:
    """Get status of an analysis job.

    Args:
        job_id: Analysis job ID

    Returns:
        Current analysis status
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _analysis_jobs[job_id]


async def _run_analysis(
    job_id: str,
    input_data: TranscriptInput,
    db: Session,
) -> None:
    """Run transcript analysis in background.

    Args:
        job_id: Job ID for status updates
        input_data: Input data
        db: Database session
    """
    try:
        _analysis_jobs[job_id].status = "processing"
        _analysis_jobs[job_id].progress = 0.1
        _analysis_jobs[job_id].message = "Fetching transcript..."

        # Fetch transcript if needed
        data_source: str | None = None
        if input_data.fetch_from_api and not input_data.transcript:
            from src.data.transcript_provider import TranscriptProvider

            async with TranscriptProvider() as provider:
                result = await provider.get_transcript_with_fallback(
                    input_data.ticker,
                    input_data.fiscal_year or datetime.now().year,
                    input_data.fiscal_quarter or ((datetime.now().month - 1) // 3 + 1),
                    manual_transcript=input_data.transcript,
                )

                if not result:
                    raise ValueError(
                        f"No transcript found for {input_data.ticker}. "
                        f"Available providers: {provider.available_providers()}"
                    )

                raw_transcript = result.transcript.raw_content
                fiscal_year = result.transcript.fiscal_year
                fiscal_quarter = result.transcript.fiscal_quarter
                call_date = result.transcript.call_date
                data_source = result.source.value
                if result.from_cache:
                    data_source = f"{data_source} (cached)"
        else:
            if not input_data.transcript:
                raise ValueError("Transcript text required when not fetching from API")
            raw_transcript = input_data.transcript
            fiscal_year = input_data.fiscal_year or datetime.now().year
            fiscal_quarter = input_data.fiscal_quarter or 1
            call_date = datetime.now()
            data_source = "manual"

        _analysis_jobs[job_id].progress = 0.3
        _analysis_jobs[job_id].message = "Parsing Q&A pairs..."

        # Parse transcript
        from src.data.transcript_parser import TranscriptParser

        parser = TranscriptParser()
        qa_pairs = parser.parse(raw_transcript)

        if not qa_pairs:
            raise ValueError("No Q&A pairs found in transcript")

        _analysis_jobs[job_id].progress = 0.5
        _analysis_jobs[job_id].message = "Computing embeddings..."

        # Get embeddings
        from src.nlp.embeddings import FinBERTEmbedder

        embedder = FinBERTEmbedder()

        questions = [qa.question_text for qa in qa_pairs]
        answers = [qa.answer_text for qa in qa_pairs]

        similarities, q_embeddings, a_embeddings = embedder.batch_compute_alignment(
            questions, answers
        )

        _analysis_jobs[job_id].progress = 0.7
        _analysis_jobs[job_id].message = "Classifying questions..."

        # Classify questions
        from src.nlp.question_classifier import QuestionClassifier

        classifier = QuestionClassifier()
        classifications = classifier.classify_batch(questions)

        _analysis_jobs[job_id].progress = 0.9
        _analysis_jobs[job_id].message = "Computing alignment scores..."

        # Build results
        qa_results = []
        responder_scores: dict[str, list[float]] = {}

        for i, qa in enumerate(qa_pairs):
            # Use cosine similarity as baseline alignment score
            alignment_score = float(max(0, min(1, (similarities[i] + 1) / 2)))

            # Determine label based on score
            if alignment_score >= 0.7:
                label = "aligned"
            elif alignment_score >= 0.4:
                label = "partially_aligned"
            else:
                label = "evasive"

            qa_result = QAPairResult(
                sequence_number=qa.sequence_number,
                analyst_name=qa.analyst_name,
                analyst_firm=qa.analyst_firm,
                question_text=qa.question_text,
                question_category=classifications[i].category.value,
                responder_name=qa.responder_name,
                responder_title=qa.responder_title,
                answer_text=qa.answer_text,
                alignment_score=alignment_score,
                alignment_label=label,
                cosine_similarity=float(similarities[i]),
                model_confidence=classifications[i].confidence,
            )
            qa_results.append(qa_result)

            # Track responder scores
            if qa.responder_name:
                if qa.responder_name not in responder_scores:
                    responder_scores[qa.responder_name] = []
                responder_scores[qa.responder_name].append(alignment_score)

        # Compute category distribution
        category_dist = classifier.get_category_distribution(questions)
        category_distribution = {cat.value: count for cat, count in category_dist.items()}

        # Average responder scores
        avg_responder_scores = {
            name: sum(scores) / len(scores)
            for name, scores in responder_scores.items()
        }

        # Overall alignment score
        overall_score = sum(r.alignment_score for r in qa_results) / len(qa_results)

        result = AnalysisResult(
            ticker=input_data.ticker,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            call_date=call_date,
            overall_alignment_score=overall_score,
            total_qa_pairs=len(qa_results),
            qa_pairs=qa_results,
            category_distribution=category_distribution,
            responder_scores=avg_responder_scores,
            data_source=data_source,
        )

        _analysis_jobs[job_id].status = "completed"
        _analysis_jobs[job_id].progress = 1.0
        _analysis_jobs[job_id].message = "Analysis complete"
        _analysis_jobs[job_id].result = result

    except Exception as e:
        _analysis_jobs[job_id].status = "failed"
        _analysis_jobs[job_id].message = str(e)
