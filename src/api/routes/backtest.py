"""Backtest endpoints for testing alignment signal."""

from datetime import datetime
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import BacktestResult, Company, EarningsCall
from src.db.session import get_db_dependency

router = APIRouter()


class BacktestConfig(BaseModel):
    """Configuration for backtest run."""

    strategy_name: str = Field(default="long_aligned_short_evasive")
    alignment_threshold_high: float = Field(
        default=0.7, ge=0, le=1, description="Threshold for 'aligned' classification"
    )
    alignment_threshold_low: float = Field(
        default=0.3, ge=0, le=1, description="Threshold for 'evasive' classification"
    )
    holding_period_days: int = Field(
        default=30, ge=1, le=90, description="Holding period after earnings call"
    )
    start_year: int | None = Field(default=None, description="Start year")
    end_year: int | None = Field(default=None, description="End year")
    tickers: list[str] | None = Field(default=None, description="Filter to specific tickers")


class Trade(BaseModel):
    """Individual trade in backtest."""

    ticker: str
    entry_date: datetime
    exit_date: datetime | None
    direction: Literal["long", "short"]
    alignment_score: float
    entry_price: float
    exit_price: float | None
    return_pct: float | None


class BacktestResults(BaseModel):
    """Complete backtest results."""

    strategy_name: str
    config: BacktestConfig

    # Date range
    start_date: datetime
    end_date: datetime

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    win_rate: float
    total_trades: int
    long_trades: int
    short_trades: int

    # Statistical significance
    correlation_score_return: float | None
    p_value: float | None
    r_squared: float | None

    # Equity curve (list of cumulative returns)
    equity_curve: list[dict]

    # Individual trades
    trades: list[Trade]


class CorrelationResult(BaseModel):
    """Correlation analysis result."""

    correlation: float
    p_value: float
    r_squared: float
    sample_size: int
    interpretation: str


@router.post("/run", response_model=BacktestResults)
async def run_backtest(
    config: BacktestConfig,
    db: Session = Depends(get_db_dependency),
) -> BacktestResults:
    """Run a backtest of the alignment signal.

    Strategy: Long companies with high alignment, short companies with low alignment.

    Args:
        config: Backtest configuration
        db: Database session

    Returns:
        Backtest results with performance metrics
    """
    # Get all earnings calls with returns data
    query = select(EarningsCall, Company).join(Company).where(
        EarningsCall.overall_alignment_score.isnot(None),
        EarningsCall.return_30d.isnot(None),
    )

    if config.start_year:
        query = query.where(EarningsCall.fiscal_year >= config.start_year)
    if config.end_year:
        query = query.where(EarningsCall.fiscal_year <= config.end_year)
    if config.tickers:
        query = query.where(Company.ticker.in_([t.upper() for t in config.tickers]))

    query = query.order_by(EarningsCall.call_date)

    result = db.execute(query)
    rows = result.all()

    if not rows:
        # Return empty results
        return BacktestResults(
            strategy_name=config.strategy_name,
            config=config,
            start_date=datetime.now(),
            end_date=datetime.now(),
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=None,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            long_trades=0,
            short_trades=0,
            correlation_score_return=None,
            p_value=None,
            r_squared=None,
            equity_curve=[],
            trades=[],
        )

    # Generate trades based on alignment signals
    trades: list[Trade] = []
    returns: list[float] = []
    scores: list[float] = []

    for call, company in rows:
        score = call.overall_alignment_score
        ret = call.return_30d

        scores.append(score)
        returns.append(ret)

        # Determine trade direction
        if score >= config.alignment_threshold_high:
            direction: Literal["long", "short"] = "long"
            trade_return = ret
        elif score <= config.alignment_threshold_low:
            direction = "short"
            trade_return = -ret  # Short position profits from decline
        else:
            continue  # No trade for middle scores

        trades.append(
            Trade(
                ticker=company.ticker,
                entry_date=call.call_date,
                exit_date=None,  # Simplified
                direction=direction,
                alignment_score=score,
                entry_price=call.price_at_call or 100.0,
                exit_price=call.price_30d_after,
                return_pct=trade_return,
            )
        )

    # Calculate performance metrics
    trade_returns = [t.return_pct for t in trades if t.return_pct is not None]

    if not trade_returns:
        total_return = 0.0
        sharpe = None
        win_rate = 0.0
    else:
        total_return = sum(trade_returns)
        avg_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        sharpe = float(avg_return / std_return * np.sqrt(252 / config.holding_period_days)) if std_return > 0 else None
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

    # Build equity curve
    cumulative = 1.0
    equity_curve = [{"date": rows[0][0].call_date.isoformat(), "value": 1.0}]

    for trade in trades:
        if trade.return_pct:
            cumulative *= (1 + trade.return_pct)
            equity_curve.append({
                "date": trade.entry_date.isoformat(),
                "value": cumulative,
            })

    # Calculate max drawdown
    peak = 1.0
    max_dd = 0.0
    for point in equity_curve:
        value = point["value"]
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd

    # Calculate correlation
    if len(scores) >= 10:
        correlation = float(np.corrcoef(scores, returns)[0, 1])
        # Simplified p-value calculation
        n = len(scores)
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2 + 1e-10)
        # Approximate p-value (would use scipy.stats.t.sf in production)
        p_value = max(0.001, min(1.0, 2 * (1 - min(0.999, abs(t_stat) / 10))))
        r_squared = correlation ** 2
    else:
        correlation = None
        p_value = None
        r_squared = None

    # Calculate annualized return
    if rows:
        days = (rows[-1][0].call_date - rows[0][0].call_date).days
        years = max(1, days / 365)
        annualized = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    else:
        annualized = 0.0

    return BacktestResults(
        strategy_name=config.strategy_name,
        config=config,
        start_date=rows[0][0].call_date,
        end_date=rows[-1][0].call_date,
        total_return=total_return,
        annualized_return=annualized,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=len(trades),
        long_trades=sum(1 for t in trades if t.direction == "long"),
        short_trades=sum(1 for t in trades if t.direction == "short"),
        correlation_score_return=correlation,
        p_value=p_value,
        r_squared=r_squared,
        equity_curve=equity_curve,
        trades=trades,
    )


@router.get("/correlation", response_model=CorrelationResult)
async def analyze_correlation(
    tickers: list[str] | None = Query(None, description="Filter to specific tickers"),
    start_year: int | None = Query(None),
    end_year: int | None = Query(None),
    db: Session = Depends(get_db_dependency),
) -> CorrelationResult:
    """Analyze correlation between alignment scores and stock returns.

    Args:
        tickers: Optional ticker filter
        start_year: Start year filter
        end_year: End year filter
        db: Database session

    Returns:
        Correlation analysis results
    """
    query = select(EarningsCall, Company).join(Company).where(
        EarningsCall.overall_alignment_score.isnot(None),
        EarningsCall.return_30d.isnot(None),
    )

    if start_year:
        query = query.where(EarningsCall.fiscal_year >= start_year)
    if end_year:
        query = query.where(EarningsCall.fiscal_year <= end_year)
    if tickers:
        query = query.where(Company.ticker.in_([t.upper() for t in tickers]))

    result = db.execute(query)
    rows = result.all()

    scores = [row[0].overall_alignment_score for row in rows]
    returns = [row[0].return_30d for row in rows]

    if len(scores) < 10:
        return CorrelationResult(
            correlation=0.0,
            p_value=1.0,
            r_squared=0.0,
            sample_size=len(scores),
            interpretation="Insufficient data (minimum 10 samples required)",
        )

    correlation = float(np.corrcoef(scores, returns)[0, 1])
    r_squared = correlation ** 2

    # Approximate p-value
    n = len(scores)
    t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2 + 1e-10)
    p_value = max(0.001, min(1.0, 2 * (1 - min(0.999, abs(t_stat) / 10))))

    # Interpretation
    if p_value > 0.05:
        interpretation = "No statistically significant correlation found"
    elif abs(correlation) < 0.2:
        interpretation = "Weak correlation - signal has limited predictive power"
    elif abs(correlation) < 0.4:
        interpretation = "Moderate correlation - signal shows meaningful relationship"
    else:
        interpretation = "Strong correlation - signal has significant predictive power"

    if correlation > 0:
        interpretation += " (positive: higher alignment → higher returns)"
    elif correlation < 0:
        interpretation += " (negative: higher alignment → lower returns, unexpected)"

    return CorrelationResult(
        correlation=correlation,
        p_value=p_value,
        r_squared=r_squared,
        sample_size=len(scores),
        interpretation=interpretation,
    )


@router.get("/history", response_model=list[dict])
async def get_backtest_history(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db_dependency),
) -> list[dict]:
    """Get history of backtest runs.

    Args:
        limit: Maximum results
        db: Database session

    Returns:
        List of past backtest results
    """
    query = (
        select(BacktestResult)
        .order_by(BacktestResult.created_at.desc())
        .limit(limit)
    )

    result = db.execute(query)
    backtests = result.scalars().all()

    return [
        {
            "id": bt.id,
            "strategy_name": bt.strategy_name,
            "created_at": bt.created_at.isoformat(),
            "total_return": bt.total_return,
            "sharpe_ratio": bt.sharpe_ratio,
            "total_trades": bt.total_trades,
            "p_value": bt.p_value,
        }
        for bt in backtests
    ]
