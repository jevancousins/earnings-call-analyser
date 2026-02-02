"""SQLAlchemy ORM models for the earnings call analyzer."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Company(Base):
    """Company information."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    earnings_calls: Mapped[list["EarningsCall"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Company(ticker={self.ticker}, name={self.name})>"


class EarningsCall(Base):
    """Earnings call transcript metadata."""

    __tablename__ = "earnings_calls"
    __table_args__ = (
        UniqueConstraint("company_id", "fiscal_year", "fiscal_quarter", name="uq_company_quarter"),
        Index("ix_earnings_calls_date", "call_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)

    # Fiscal period
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-4

    # Call details
    call_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    transcript_source: Mapped[str] = mapped_column(String(50))  # 'fmp' or 'sec'
    raw_transcript: Mapped[Optional[str]] = mapped_column(Text)

    # Computed metrics
    overall_alignment_score: Mapped[Optional[float]] = mapped_column(Float)
    total_qa_pairs: Mapped[int] = mapped_column(Integer, default=0)

    # Stock price data (for backtesting)
    price_at_call: Mapped[Optional[float]] = mapped_column(Float)
    price_30d_after: Mapped[Optional[float]] = mapped_column(Float)
    return_30d: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    company: Mapped["Company"] = relationship(back_populates="earnings_calls")
    qa_pairs: Mapped[list["QAPair"]] = relationship(
        back_populates="earnings_call", cascade="all, delete-orphan"
    )

    @property
    def quarter_label(self) -> str:
        """Return formatted quarter label like 'Q1 2024'."""
        return f"Q{self.fiscal_quarter} {self.fiscal_year}"

    def __repr__(self) -> str:
        return f"<EarningsCall(company_id={self.company_id}, quarter={self.quarter_label})>"


class QAPair(Base):
    """Individual Q&A pair from an earnings call."""

    __tablename__ = "qa_pairs"
    __table_args__ = (Index("ix_qa_pairs_alignment", "alignment_score"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    earnings_call_id: Mapped[int] = mapped_column(ForeignKey("earnings_calls.id"), nullable=False)

    # Sequence in the call
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Question details
    analyst_name: Mapped[Optional[str]] = mapped_column(String(255))
    analyst_firm: Mapped[Optional[str]] = mapped_column(String(255))
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_category: Mapped[Optional[str]] = mapped_column(
        String(50)
    )  # margins, guidance, competition, macro, etc.

    # Answer details
    responder_name: Mapped[Optional[str]] = mapped_column(String(255))
    responder_title: Mapped[Optional[str]] = mapped_column(String(100))  # CEO, CFO, etc.
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Embeddings (stored as JSON arrays for simplicity)
    question_embedding: Mapped[Optional[list]] = mapped_column(JSON)
    answer_embedding: Mapped[Optional[list]] = mapped_column(JSON)

    # Alignment metrics
    alignment_score: Mapped[Optional[float]] = mapped_column(Float)  # 0-1 score
    alignment_label: Mapped[Optional[str]] = mapped_column(
        String(20)
    )  # aligned, partially_aligned, evasive
    model_confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Cosine similarity baseline
    cosine_similarity: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    earnings_call: Mapped["EarningsCall"] = relationship(back_populates="qa_pairs")

    def __repr__(self) -> str:
        return f"<QAPair(id={self.id}, score={self.alignment_score})>"


class BacktestResult(Base):
    """Backtest results for a strategy."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Strategy parameters
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    alignment_threshold_high: Mapped[float] = mapped_column(Float, default=0.7)
    alignment_threshold_low: Mapped[float] = mapped_column(Float, default=0.3)
    holding_period_days: Mapped[int] = mapped_column(Integer, default=30)

    # Date range
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Performance metrics
    total_return: Mapped[float] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)

    # Statistical significance
    correlation_score_return: Mapped[Optional[float]] = mapped_column(Float)
    p_value: Mapped[Optional[float]] = mapped_column(Float)

    # Detailed results stored as JSON
    equity_curve: Mapped[Optional[list]] = mapped_column(JSON)
    trade_log: Mapped[Optional[list]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<BacktestResult(strategy={self.strategy_name}, sharpe={self.sharpe_ratio})>"
