"""Unified transcript provider with multi-source fallback and caching."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol

from src.config import get_settings
from src.data.fmp_client import TranscriptData

logger = logging.getLogger(__name__)


class TranscriptSource(str, Enum):
    """Available transcript data sources."""

    EARNINGSCALL = "earningscall"  # Primary - free tier for AAPL/MSFT
    FINNHUB = "finnhub"  # Requires paid subscription
    ALPHAVANTAGE = "alphavantage"
    FMP = "fmp"  # Legacy - deprecated Aug 2025
    MANUAL = "manual"


@dataclass
class CachedTranscript:
    """Cached transcript with metadata."""

    transcript: TranscriptData
    source: TranscriptSource
    fetched_at: datetime
    cache_key: str


class TranscriptClient(Protocol):
    """Protocol for transcript client implementations."""

    async def __aenter__(self) -> "TranscriptClient":
        ...

    async def __aexit__(self, *args: Any) -> None:
        ...

    async def get_transcript(
        self, ticker: str, year: int, quarter: int
    ) -> TranscriptData | None:
        ...


class TranscriptCache:
    """Simple in-memory cache for transcripts."""

    def __init__(self, ttl_hours: int = 24) -> None:
        self._cache: dict[str, CachedTranscript] = {}
        self._ttl = timedelta(hours=ttl_hours)

    @staticmethod
    def _make_key(ticker: str, year: int, quarter: int) -> str:
        """Generate cache key for a transcript."""
        return hashlib.md5(
            f"{ticker.upper()}:{year}:Q{quarter}".encode()
        ).hexdigest()

    def get(self, ticker: str, year: int, quarter: int) -> CachedTranscript | None:
        """Get cached transcript if available and not expired."""
        key = self._make_key(ticker, year, quarter)
        cached = self._cache.get(key)

        if cached is None:
            return None

        if datetime.now() - cached.fetched_at > self._ttl:
            # Expired
            del self._cache[key]
            return None

        return cached

    def set(
        self,
        transcript: TranscriptData,
        source: TranscriptSource,
    ) -> None:
        """Cache a transcript."""
        key = self._make_key(
            transcript.ticker, transcript.fiscal_year, transcript.fiscal_quarter
        )
        self._cache[key] = CachedTranscript(
            transcript=transcript,
            source=source,
            fetched_at=datetime.now(),
            cache_key=key,
        )

    def clear(self) -> None:
        """Clear all cached transcripts."""
        self._cache.clear()


@dataclass
class TranscriptResult:
    """Result from transcript provider including source metadata."""

    transcript: TranscriptData
    source: TranscriptSource
    from_cache: bool = False


class TranscriptProvider:
    """Unified interface for fetching transcripts from multiple sources.

    Tries each provider in priority order (configurable) until one succeeds.
    Includes caching to reduce API calls.

    Default priority: Finnhub -> Alpha Vantage -> FMP (legacy)

    Usage:
        async with TranscriptProvider() as provider:
            result = await provider.get_transcript("AAPL", 2025, 1)
            if result:
                print(f"Got transcript from {result.source}")
    """

    def __init__(
        self,
        providers: list[TranscriptSource] | None = None,
        cache_ttl_hours: int = 24,
    ) -> None:
        """Initialise the transcript provider.

        Args:
            providers: List of providers to try in order. If None, uses config.
            cache_ttl_hours: How long to cache transcripts (default 24 hours)
        """
        settings = get_settings()

        if providers is None:
            # Parse from config, defaulting to sensible order
            provider_names = settings.transcript_providers
            providers = []
            for name in provider_names:
                try:
                    providers.append(TranscriptSource(name.lower()))
                except ValueError:
                    logger.warning(f"Unknown transcript provider: {name}")

        if not providers:
            providers = [
                TranscriptSource.EARNINGSCALL,
                TranscriptSource.FINNHUB,
                TranscriptSource.ALPHAVANTAGE,
                TranscriptSource.FMP,
            ]

        self._provider_order = providers
        self._cache = TranscriptCache(ttl_hours=cache_ttl_hours)
        self._clients: dict[TranscriptSource, Any] = {}
        self._settings = settings

    async def __aenter__(self) -> "TranscriptProvider":
        """Enter async context and initialise available clients."""
        await self._init_clients()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context and close all clients."""
        for source, client in self._clients.items():
            try:
                await client.__aexit__(*args)
            except Exception as e:
                logger.warning(f"Error closing {source} client: {e}")
        self._clients.clear()

    async def _init_clients(self) -> None:
        """Initialise clients for configured providers."""
        for source in self._provider_order:
            try:
                client = await self._create_client(source)
                if client:
                    self._clients[source] = client
            except ValueError as e:
                # Missing API key - skip this provider
                logger.info(f"Skipping {source}: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialise {source} client: {e}")

    async def _create_client(self, source: TranscriptSource) -> Any:
        """Create and initialise a client for the given source."""
        if source == TranscriptSource.EARNINGSCALL:
            # EarningsCall works without API key for AAPL/MSFT (free tier)
            from src.data.earningscall_client import EarningsCallClient

            client = EarningsCallClient()
            await client.__aenter__()
            return client

        elif source == TranscriptSource.FINNHUB:
            if not self._settings.finnhub_api_key:
                raise ValueError("FINNHUB_API_KEY not configured")
            from src.data.finnhub_client import FinnhubClient

            client = FinnhubClient()
            await client.__aenter__()
            return client

        elif source == TranscriptSource.ALPHAVANTAGE:
            if not self._settings.alphavantage_api_key:
                raise ValueError("ALPHAVANTAGE_API_KEY not configured")
            from src.data.alphavantage_client import AlphaVantageClient

            client = AlphaVantageClient()
            await client.__aenter__()
            return client

        elif source == TranscriptSource.FMP:
            if not self._settings.fmp_api_key:
                raise ValueError("FMP_API_KEY not configured")
            from src.data.fmp_client import FMPClient

            client = FMPClient()
            await client.__aenter__()
            return client

        return None

    async def get_transcript(
        self,
        ticker: str,
        year: int,
        quarter: int,
        skip_cache: bool = False,
    ) -> TranscriptResult | None:
        """Fetch transcript, trying each provider in order.

        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter (1-4)
            skip_cache: If True, bypass cache and fetch fresh

        Returns:
            TranscriptResult with transcript and source, or None if not found
        """
        ticker = ticker.upper()

        # Check cache first
        if not skip_cache:
            cached = self._cache.get(ticker, year, quarter)
            if cached:
                logger.debug(f"Cache hit for {ticker} Q{quarter} {year}")
                return TranscriptResult(
                    transcript=cached.transcript,
                    source=cached.source,
                    from_cache=True,
                )

        # Try each provider in order
        for source in self._provider_order:
            client = self._clients.get(source)
            if not client:
                continue

            try:
                logger.debug(f"Trying {source} for {ticker} Q{quarter} {year}")
                transcript = await client.get_transcript(ticker, year, quarter)

                if transcript:
                    # Cache the result
                    self._cache.set(transcript, source)
                    logger.info(
                        f"Successfully fetched {ticker} Q{quarter} {year} from {source}"
                    )
                    return TranscriptResult(
                        transcript=transcript,
                        source=source,
                        from_cache=False,
                    )

            except Exception as e:
                logger.warning(f"Error fetching from {source}: {e}")
                continue

        logger.warning(f"No transcript found for {ticker} Q{quarter} {year}")
        return None

    async def get_transcript_with_fallback(
        self,
        ticker: str,
        year: int,
        quarter: int,
        manual_transcript: str | None = None,
    ) -> TranscriptResult | None:
        """Fetch transcript with optional manual fallback.

        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter (1-4)
            manual_transcript: Optional manually-provided transcript text

        Returns:
            TranscriptResult with transcript and source, or None if not found
        """
        # Try API sources first
        result = await self.get_transcript(ticker, year, quarter)
        if result:
            return result

        # Fall back to manual transcript if provided
        if manual_transcript:
            from datetime import datetime

            transcript = TranscriptData(
                ticker=ticker.upper(),
                fiscal_year=year,
                fiscal_quarter=quarter,
                call_date=datetime.now(),
                raw_content=manual_transcript,
            )
            return TranscriptResult(
                transcript=transcript,
                source=TranscriptSource.MANUAL,
                from_cache=False,
            )

        return None

    def available_providers(self) -> list[TranscriptSource]:
        """Get list of currently available (initialised) providers."""
        return list(self._clients.keys())

    def clear_cache(self) -> None:
        """Clear the transcript cache."""
        self._cache.clear()


# Synchronous wrapper for convenience
def fetch_transcript_sync(
    ticker: str,
    year: int,
    quarter: int,
    providers: list[TranscriptSource] | None = None,
) -> TranscriptResult | None:
    """Synchronous helper to fetch a transcript using the provider.

    Args:
        ticker: Stock ticker symbol
        year: Fiscal year
        quarter: Fiscal quarter
        providers: Optional list of providers to try

    Returns:
        TranscriptResult if found
    """

    async def _fetch() -> TranscriptResult | None:
        async with TranscriptProvider(providers=providers) as provider:
            return await provider.get_transcript(ticker, year, quarter)

    return asyncio.run(_fetch())
