"""Alignment comparison and ranking endpoints."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Company, EarningsCall
from src.db.session import get_db_dependency

router = APIRouter()


class CompanyAlignmentRank(BaseModel):
    """Company ranking by alignment score."""

    rank: int
    ticker: str
    name: str
    sector: str | None
    average_alignment: float
    latest_alignment: float | None
    total_calls: int
    alignment_trend: str


class SectorComparison(BaseModel):
    """Sector-level alignment comparison."""

    sector: str
    company_count: int
    average_alignment: float
    top_company: str
    bottom_company: str
    companies: list[CompanyAlignmentRank]


class CompareResult(BaseModel):
    """Result of comparing multiple companies."""

    companies: list[CompanyAlignmentRank]
    sector_averages: dict[str, float]


@router.get("/rankings", response_model=list[CompanyAlignmentRank])
async def get_alignment_rankings(
    sector: str | None = Query(None, description="Filter by sector"),
    min_calls: int = Query(4, ge=1, description="Minimum calls to include"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    sort_by: str = Query("average", enum=["average", "latest", "trend"]),
    db: Session = Depends(get_db_dependency),
) -> list[CompanyAlignmentRank]:
    """Get companies ranked by alignment score.

    Args:
        sector: Optional sector filter
        min_calls: Minimum number of calls required
        limit: Maximum results
        sort_by: Ranking criteria
        db: Database session

    Returns:
        List of companies ranked by alignment
    """
    # Subquery for call counts and average alignment
    call_stats = (
        select(
            EarningsCall.company_id,
            func.count(EarningsCall.id).label("call_count"),
            func.avg(EarningsCall.overall_alignment_score).label("avg_alignment"),
        )
        .where(EarningsCall.overall_alignment_score.isnot(None))
        .group_by(EarningsCall.company_id)
        .having(func.count(EarningsCall.id) >= min_calls)
        .subquery()
    )

    # Main query joining with companies
    query = (
        select(
            Company,
            call_stats.c.call_count,
            call_stats.c.avg_alignment,
        )
        .join(call_stats, Company.id == call_stats.c.company_id)
    )

    if sector:
        query = query.where(Company.sector == sector)

    # Sort
    if sort_by == "latest":
        query = query.order_by(call_stats.c.avg_alignment.desc())
    else:  # average
        query = query.order_by(call_stats.c.avg_alignment.desc())

    query = query.limit(limit)

    result = db.execute(query)
    rows = result.all()

    rankings = []
    for rank, (company, call_count, avg_alignment) in enumerate(rows, 1):
        # Get latest call
        latest_call = (
            db.execute(
                select(EarningsCall)
                .where(EarningsCall.company_id == company.id)
                .order_by(EarningsCall.call_date.desc())
                .limit(1)
            )
            .scalar_one_or_none()
        )

        # Calculate trend (simplified)
        calls = (
            db.execute(
                select(EarningsCall)
                .where(EarningsCall.company_id == company.id)
                .order_by(EarningsCall.call_date)
            )
            .scalars()
            .all()
        )

        scores = [c.overall_alignment_score for c in calls if c.overall_alignment_score]
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

        rankings.append(
            CompanyAlignmentRank(
                rank=rank,
                ticker=company.ticker,
                name=company.name,
                sector=company.sector,
                average_alignment=float(avg_alignment) if avg_alignment else 0.0,
                latest_alignment=latest_call.overall_alignment_score if latest_call else None,
                total_calls=call_count,
                alignment_trend=trend,
            )
        )

    return rankings


@router.get("/sectors", response_model=list[SectorComparison])
async def get_sector_comparison(
    min_companies: int = Query(3, ge=1, description="Minimum companies per sector"),
    db: Session = Depends(get_db_dependency),
) -> list[SectorComparison]:
    """Compare alignment scores across sectors.

    Args:
        min_companies: Minimum companies required per sector
        db: Database session

    Returns:
        List of sector comparisons
    """
    # Get all companies with alignment data
    companies_query = (
        select(
            Company,
            func.avg(EarningsCall.overall_alignment_score).label("avg_alignment"),
        )
        .join(EarningsCall)
        .where(EarningsCall.overall_alignment_score.isnot(None))
        .group_by(Company.id)
    )

    result = db.execute(companies_query)
    company_data = result.all()

    # Group by sector
    sectors: dict[str, list[tuple]] = {}
    for company, avg_alignment in company_data:
        sector = company.sector or "Unknown"
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append((company, avg_alignment))

    # Build sector comparisons
    comparisons = []
    for sector, companies in sectors.items():
        if len(companies) < min_companies:
            continue

        # Sort by alignment
        sorted_companies = sorted(companies, key=lambda x: x[1] or 0, reverse=True)

        sector_avg = sum(c[1] or 0 for c in companies) / len(companies)

        company_ranks = []
        for rank, (company, avg_alignment) in enumerate(sorted_companies, 1):
            company_ranks.append(
                CompanyAlignmentRank(
                    rank=rank,
                    ticker=company.ticker,
                    name=company.name,
                    sector=company.sector,
                    average_alignment=float(avg_alignment) if avg_alignment else 0.0,
                    latest_alignment=None,
                    total_calls=0,
                    alignment_trend="unknown",
                )
            )

        comparisons.append(
            SectorComparison(
                sector=sector,
                company_count=len(companies),
                average_alignment=sector_avg,
                top_company=sorted_companies[0][0].ticker,
                bottom_company=sorted_companies[-1][0].ticker,
                companies=company_ranks,
            )
        )

    # Sort sectors by average alignment
    comparisons.sort(key=lambda x: x.average_alignment, reverse=True)

    return comparisons


@router.post("/compare", response_model=CompareResult)
async def compare_companies(
    tickers: list[str] = Query(..., description="List of tickers to compare"),
    db: Session = Depends(get_db_dependency),
) -> CompareResult:
    """Compare alignment scores for specific companies.

    Args:
        tickers: List of stock ticker symbols
        db: Database session

    Returns:
        Comparison results
    """
    companies_data = []
    sector_totals: dict[str, list[float]] = {}

    for ticker in tickers:
        # Get company
        company = db.execute(
            select(Company).where(Company.ticker == ticker.upper())
        ).scalar_one_or_none()

        if not company:
            continue

        # Get alignment stats
        calls = (
            db.execute(
                select(EarningsCall)
                .where(EarningsCall.company_id == company.id)
                .order_by(EarningsCall.call_date)
            )
            .scalars()
            .all()
        )

        scores = [c.overall_alignment_score for c in calls if c.overall_alignment_score]

        if not scores:
            continue

        avg_alignment = sum(scores) / len(scores)
        latest = calls[-1].overall_alignment_score if calls else None

        # Track sector averages
        sector = company.sector or "Unknown"
        if sector not in sector_totals:
            sector_totals[sector] = []
        sector_totals[sector].append(avg_alignment)

        # Calculate trend
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

        companies_data.append(
            (
                avg_alignment,
                CompanyAlignmentRank(
                    rank=0,  # Will be set after sorting
                    ticker=company.ticker,
                    name=company.name,
                    sector=company.sector,
                    average_alignment=avg_alignment,
                    latest_alignment=latest,
                    total_calls=len(calls),
                    alignment_trend=trend,
                ),
            )
        )

    # Sort and assign ranks
    companies_data.sort(key=lambda x: x[0], reverse=True)
    ranked_companies = []
    for rank, (_, company_rank) in enumerate(companies_data, 1):
        company_rank.rank = rank
        ranked_companies.append(company_rank)

    # Calculate sector averages
    sector_averages = {
        sector: sum(scores) / len(scores) for sector, scores in sector_totals.items()
    }

    return CompareResult(
        companies=ranked_companies,
        sector_averages=sector_averages,
    )
