"""Finnhub API client for earnings call transcripts."""

import asyncio
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.fmp_client import TranscriptData


class FinnhubRateLimiter:
    """Rate limiter for Finnhub API (60 calls per minute)."""

    def __init__(self, calls_per_minute: int = 60) -> None:
        self.calls_per_minute = calls_per_minute
        self.calls: list[float] = []

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]

        if len(self.calls) >= self.calls_per_minute:
            # Wait until the oldest call expires
            sleep_time = 60 - (now - self.calls[0]) + 0.1
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.calls = self.calls[1:]

        self.calls.append(time.time())


class FinnhubClient:
    """Client for Finnhub API - earnings call transcripts.

    Finnhub provides earnings call transcripts for US, UK, European,
    Australian, and Canadian companies. Transcripts are typically
    available 2-4 hours after the earnings call (US) or next day (intl).

    API Documentation: https://finnhub.io/docs/api/earnings-call-transcripts-api
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the Finnhub client.

        Args:
            api_key: Finnhub API key. If not provided, reads from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.finnhub_api_key
        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Set FINNHUB_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = FinnhubRateLimiter()

    async def __aenter__(self) -> "FinnhubClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client."""
        if self._client is None:
            raise RuntimeError("Client not initialised. Use async context manager.")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make authenticated GET request with retry logic and rate limiting.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        await self._rate_limiter.acquire()

        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["token"] = self.api_key

        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def get_transcript(
        self, ticker: str, year: int, quarter: int
    ) -> TranscriptData | None:
        """Fetch earnings call transcript for a specific quarter.

        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter (1-4)

        Returns:
            TranscriptData if found, None otherwise
        """
        from datetime import datetime

        try:
            # Finnhub uses symbol, not ticker in some cases
            # The API returns transcript data for the specified symbol
            data = await self._get(
                "stock/transcripts",
                {"symbol": ticker.upper(), "year": year, "quarter": quarter},
            )

            if not data or not data.get("transcript"):
                return None

            # Parse the transcript content from Finnhub format
            # Finnhub returns: {"symbol": "AAPL", "quarter": 1, "year": 2024,
            #                   "transcript": [{"name": "...", "speech": ["..."]}]}
            transcript_parts = data.get("transcript", [])
            content_parts = []

            for part in transcript_parts:
                name = part.get("name", "Unknown")
                speeches = part.get("speech", [])
                for speech in speeches:
                    content_parts.append(f"{name}: {speech}")

            raw_content = "\n\n".join(content_parts)

            if not raw_content:
                return None

            # Estimate call date based on quarter
            # Q1: Feb-Mar, Q2: Apr-May, Q3: Jul-Aug, Q4: Oct-Nov
            quarter_months = {1: 2, 2: 5, 3: 8, 4: 11}
            call_date = datetime(year, quarter_months.get(quarter, 1), 15)

            return TranscriptData(
                ticker=ticker.upper(),
                fiscal_year=year,
                fiscal_quarter=quarter,
                call_date=call_date,
                raw_content=raw_content,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_transcript_list(self, ticker: str) -> list[dict[str, Any]]:
        """Get list of available transcripts for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of available transcript metadata (year, quarter)
        """
        try:
            # Finnhub provides a list endpoint
            data = await self._get("stock/transcripts/list", {"symbol": ticker.upper()})
            return data if isinstance(data, list) else []
        except httpx.HTTPStatusError:
            return []

    async def get_all_transcripts(
        self, ticker: str, years: list[int] | None = None
    ) -> list[TranscriptData]:
        """Fetch all available transcripts for a ticker.

        Args:
            ticker: Stock ticker symbol
            years: Optional list of years to fetch. If None, fetches all available.

        Returns:
            List of TranscriptData objects
        """
        from datetime import datetime

        all_transcripts: list[TranscriptData] = []

        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 3, current_year + 1))

        for year in years:
            for quarter in [1, 2, 3, 4]:
                transcript = await self.get_transcript(ticker, year, quarter)
                if transcript:
                    all_transcripts.append(transcript)
                # Small delay between requests
                await asyncio.sleep(0.2)

        return sorted(
            all_transcripts, key=lambda x: (x.fiscal_year, x.fiscal_quarter)
        )


# Synchronous wrapper for convenience
def fetch_finnhub_transcript_sync(
    ticker: str, year: int, quarter: int, api_key: str | None = None
) -> TranscriptData | None:
    """Synchronous helper to fetch a single transcript from Finnhub.

    Args:
        ticker: Stock ticker symbol
        year: Fiscal year
        quarter: Fiscal quarter
        api_key: Optional API key

    Returns:
        TranscriptData if found
    """

    async def _fetch() -> TranscriptData | None:
        async with FinnhubClient(api_key=api_key) as client:
            return await client.get_transcript(ticker, year, quarter)

    return asyncio.run(_fetch())
