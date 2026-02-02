"""Company-related API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Company, EarningsCall, QAPair
from src.db.session import get_db_dependency

router = APIRouter()


class CompanyInfo(BaseModel):
    """Company information response."""

    ticker: str
    name: str
    sector: str | None
    industry: str | None
    market_cap: float | None
    total_calls_analyzed: int
    earliest_call: datetime | None
    latest_call: datetime | None


class QuarterlyAlignment(BaseModel):
    """Alignment data for a single quarter."""

    fiscal_year: int
    fiscal_quarter: int
    quarter_label: str
    call_date: datetime
    overall_alignment_score: float
    total_qa_pairs: int
    price_at_call: float | None
    price_30d_after: float | None
    return_30d: float | None


class CompanyAlignmentHistory(BaseModel):
    """Historical alignment data for a company."""

    ticker: str
    name: str
    quarterly_data: list[QuarterlyAlignment]
    average_alignment: float
    alignment_trend: str  # "improving", "declining", "stable"


class QAPairSummary(BaseModel):
    """Summary of a Q&A pair for listing."""

    id: int
    sequence_number: int
    analyst_name: str
    question_preview: str
    question_category: str | None
    responder_name: str | None
    alignment_score: float | None
    alignment_label: str | None


@router.get("", response_model=list[CompanyInfo])
async def list_companies(
    sector: str | None = Query(None, description="Filter by sector"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    db: Session = Depends(get_db_dependency),
) -> list[CompanyInfo]:
    """List all tracked companies.

    Args:
        sector: Optional sector filter
        limit: Maximum number of results
        db: Database session

    Returns:
        List of company information
    """
    query = select(Company)

    if sector:
        query = query.where(Company.sector == sector)

    query = query.limit(limit)
    result = db.execute(query)
    companies = result.scalars().all()

    company_infos = []
    for company in companies:
        calls = company.earnings_calls
        company_infos.append(
            CompanyInfo(
                ticker=company.ticker,
                name=company.name,
                sector=company.sector,
                industry=company.industry,
                market_cap=company.market_cap,
                total_calls_analyzed=len(calls),
                earliest_call=min((c.call_date for c in calls), default=None),
                latest_call=max((c.call_date for c in calls), default=None),
            )
        )

    return company_infos


@router.get("/{ticker}", response_model=CompanyInfo)
async def get_company(
    ticker: str,
    db: Session = Depends(get_db_dependency),
) -> CompanyInfo:
    """Get information for a specific company.

    Args:
        ticker: Stock ticker symbol
        db: Database session

    Returns:
        Company information
    """
    query = select(Company).where(Company.ticker == ticker.upper())
    result = db.execute(query)
    company = result.scalar_one_or_none()

    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    calls = company.earnings_calls
    return CompanyInfo(
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        industry=company.industry,
        market_cap=company.market_cap,
        total_calls_analyzed=len(calls),
        earliest_call=min((c.call_date for c in calls), default=None),
        latest_call=max((c.call_date for c in calls), default=None),
    )


@router.get("/{ticker}/alignment", response_model=CompanyAlignmentHistory)
async def get_company_alignment(
    ticker: str,
    start_year: int | None = Query(None, description="Start year filter"),
    end_year: int | None = Query(None, description="End year filter"),
    db: Session = Depends(get_db_dependency),
) -> CompanyAlignmentHistory:
    """Get historical alignment scores for a company.

    Args:
        ticker: Stock ticker symbol
        start_year: Optional start year filter
        end_year: Optional end year filter
        db: Database session

    Returns:
        Historical alignment data
    """
    query = select(Company).where(Company.ticker == ticker.upper())
    result = db.execute(query)
    company = result.scalar_one_or_none()

    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    # Get earnings calls
    calls_query = select(EarningsCall).where(EarningsCall.company_id == company.id)

    if start_year:
        calls_query = calls_query.where(EarningsCall.fiscal_year >= start_year)
    if end_year:
        calls_query = calls_query.where(EarningsCall.fiscal_year <= end_year)

    calls_query = calls_query.order_by(
        EarningsCall.fiscal_year, EarningsCall.fiscal_quarter
    )

    result = db.execute(calls_query)
    calls = result.scalars().all()

    quarterly_data = []
    for call in calls:
        quarterly_data.append(
            QuarterlyAlignment(
                fiscal_year=call.fiscal_year,
                fiscal_quarter=call.fiscal_quarter,
                quarter_label=call.quarter_label,
                call_date=call.call_date,
                overall_alignment_score=call.overall_alignment_score or 0.0,
                total_qa_pairs=call.total_qa_pairs,
                price_at_call=call.price_at_call,
                price_30d_after=call.price_30d_after,
                return_30d=call.return_30d,
            )
        )

    # Calculate average and trend
    scores = [q.overall_alignment_score for q in quarterly_data]
    avg_alignment = sum(scores) / len(scores) if scores else 0.0

    # Simple trend detection
    if len(scores) >= 4:
        first_half = sum(scores[: len(scores) // 2]) / (len(scores) // 2)
        second_half = sum(scores[len(scores) // 2 :]) / (len(scores) - len(scores) // 2)

        if second_half > first_half + 0.05:
            trend = "improving"
        elif second_half < first_half - 0.05:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    return CompanyAlignmentHistory(
        ticker=company.ticker,
        name=company.name,
        quarterly_data=quarterly_data,
        average_alignment=avg_alignment,
        alignment_trend=trend,
    )


@router.get("/{ticker}/calls/{year}/{quarter}/qa-pairs", response_model=list[QAPairSummary])
async def get_qa_pairs(
    ticker: str,
    year: int,
    quarter: int,
    category: str | None = Query(None, description="Filter by question category"),
    min_score: float | None = Query(None, ge=0, le=1, description="Minimum alignment score"),
    max_score: float | None = Query(None, ge=0, le=1, description="Maximum alignment score"),
    db: Session = Depends(get_db_dependency),
) -> list[QAPairSummary]:
    """Get Q&A pairs for a specific earnings call.

    Args:
        ticker: Stock ticker symbol
        year: Fiscal year
        quarter: Fiscal quarter
        category: Optional question category filter
        min_score: Optional minimum alignment score
        max_score: Optional maximum alignment score
        db: Database session

    Returns:
        List of Q&A pair summaries
    """
    # Find company
    company_query = select(Company).where(Company.ticker == ticker.upper())
    company = db.execute(company_query).scalar_one_or_none()

    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    # Find earnings call
    call_query = select(EarningsCall).where(
        EarningsCall.company_id == company.id,
        EarningsCall.fiscal_year == year,
        EarningsCall.fiscal_quarter == quarter,
    )
    call = db.execute(call_query).scalar_one_or_none()

    if not call:
        raise HTTPException(
            status_code=404,
            detail=f"No earnings call found for {ticker} Q{quarter} {year}",
        )

    # Get Q&A pairs
    qa_query = select(QAPair).where(QAPair.earnings_call_id == call.id)

    if category:
        qa_query = qa_query.where(QAPair.question_category == category)
    if min_score is not None:
        qa_query = qa_query.where(QAPair.alignment_score >= min_score)
    if max_score is not None:
        qa_query = qa_query.where(QAPair.alignment_score <= max_score)

    qa_query = qa_query.order_by(QAPair.sequence_number)

    result = db.execute(qa_query)
    qa_pairs = result.scalars().all()

    return [
        QAPairSummary(
            id=qa.id,
            sequence_number=qa.sequence_number,
            analyst_name=qa.analyst_name or "Unknown",
            question_preview=qa.question_text[:200] + "..."
            if len(qa.question_text) > 200
            else qa.question_text,
            question_category=qa.question_category,
            responder_name=qa.responder_name,
            alignment_score=qa.alignment_score,
            alignment_label=qa.alignment_label,
        )
        for qa in qa_pairs
    ]


@router.post("/{ticker}/sync")
async def sync_company_data(
    ticker: str,
    years: list[int] | None = None,
    db: Session = Depends(get_db_dependency),
) -> dict[str, str]:
    """Sync transcript data for a company from FMP API.

    Args:
        ticker: Stock ticker symbol
        years: Optional list of years to sync
        db: Database session

    Returns:
        Sync status message
    """
    # This would trigger a background job to fetch and process transcripts
    # For now, return a placeholder
    return {
        "status": "queued",
        "message": f"Data sync queued for {ticker}",
        "ticker": ticker,
    }
