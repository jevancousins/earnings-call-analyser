"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys - Transcript Providers
    earningscall_api_key: str = ""  # Primary - free tier: AAPL, MSFT only
    finnhub_api_key: str = ""  # Requires paid subscription for transcripts
    alphavantage_api_key: str = ""
    fmp_api_key: str = ""  # Legacy - deprecated for new users Aug 2025

    # Transcript Provider Configuration
    # Order determines fallback priority: first available provider is tried first
    transcript_providers: list[str] = ["earningscall", "finnhub", "alphavantage", "fmp"]

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/earnings_analyser"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Model Settings
    model_device: Literal["cpu", "cuda", "mps"] = "cpu"
    finbert_model: str = "ProsusAI/finbert"
    batch_size: int = 32

    # Dashboard
    streamlit_port: int = 8501

    # Rate Limiting
    fmp_rate_limit_per_day: int = 250


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
