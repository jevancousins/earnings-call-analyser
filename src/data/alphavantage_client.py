"""Alpha Vantage API client for earnings call transcripts."""

import asyncio
import time
from datetime import date, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.fmp_client import TranscriptData


class AlphaVantageDailyLimiter:
    """Rate limiter for Alpha Vantage API (25 calls per day on free tier)."""

    def __init__(self, calls_per_day: int = 25) -> None:
        self.calls_per_day = calls_per_day
        self.calls: list[datetime] = []
        self._today: date | None = None

    def _reset_if_new_day(self) -> None:
        """Reset call counter if it's a new day."""
        today = date.today()
        if self._today != today:
            self.calls = []
            self._today = today

    def remaining_calls(self) -> int:
        """Get number of remaining API calls for today."""
        self._reset_if_new_day()
        return max(0, self.calls_per_day - len(self.calls))

    def can_make_call(self) -> bool:
        """Check if we can make another API call today."""
        self._reset_if_new_day()
        return len(self.calls) < self.calls_per_day

    def record_call(self) -> None:
        """Record that an API call was made."""
        self._reset_if_new_day()
        self.calls.append(datetime.now())

    async def acquire(self) -> None:
        """Acquire permission to make an API call.

        Raises:
            RuntimeError: If daily limit exceeded
        """
        self._reset_if_new_day()
        if not self.can_make_call():
            raise RuntimeError(
                f"Alpha Vantage daily API limit ({self.calls_per_day} calls) exceeded. "
                "Try again tomorrow or use a different data source."
            )
        self.record_call()
        # Small delay between calls to be polite
        await asyncio.sleep(0.5)


class AlphaVantageClient:
    """Client for Alpha Vantage API - earnings call transcripts.

    Alpha Vantage provides earnings call transcripts but has a stricter
    rate limit (25 calls/day on free tier). Use as a backup source.

    API Documentation: https://www.alphavantage.co/documentation/
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key. If not provided, reads from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.alphavantage_api_key
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key required. Set ALPHAVANTAGE_API_KEY "
                "environment variable or pass api_key parameter."
            )
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = AlphaVantageDailyLimiter()

    async def __aenter__(self) -> "AlphaVantageClient":
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

    def remaining_calls_today(self) -> int:
        """Get number of remaining API calls for today."""
        return self._rate_limiter.remaining_calls()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _get(self, params: dict[str, Any]) -> Any:
        """Make authenticated GET request with retry logic and rate limiting.

        Args:
            params: Query parameters

        Returns:
            JSON response data
        """
        await self._rate_limiter.acquire()

        params["apikey"] = self.api_key
        response = await self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Check for Alpha Vantage API error messages
        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        if "Note" in data and "API call frequency" in data.get("Note", ""):
            raise RuntimeError("Alpha Vantage rate limit exceeded")

        return data

    async def get_transcript(
        self, ticker: str, year: int, quarter: int
    ) -> TranscriptData | None:
        """Fetch earnings call transcript for a specific quarter.

        Note: Alpha Vantage transcript API may require premium subscription.
        This implementation uses the EARNINGS_TRANSCRIPT function.

        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter (1-4)

        Returns:
            TranscriptData if found, None otherwise
        """
        try:
            # Alpha Vantage uses function=EARNINGS_TRANSCRIPT
            # with symbol, year, and quarter parameters
            data = await self._get(
                {
                    "function": "EARNINGS_TRANSCRIPT",
                    "symbol": ticker.upper(),
                    "year": year,
                    "quarter": quarter,
                }
            )

            if not data or "transcript" not in data:
                return None

            # Parse transcript content
            # Alpha Vantage format may vary; adapt as needed
            transcript_content = data.get("transcript", "")
            if isinstance(transcript_content, list):
                # If it's a list of speech segments
                parts = []
                for segment in transcript_content:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    parts.append(f"{speaker}: {text}")
                transcript_content = "\n\n".join(parts)

            if not transcript_content:
                return None

            # Parse date if available
            date_str = data.get("date", "")
            if date_str:
                try:
                    call_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    quarter_months = {1: 2, 2: 5, 3: 8, 4: 11}
                    call_date = datetime(year, quarter_months.get(quarter, 1), 15)
            else:
                quarter_months = {1: 2, 2: 5, 3: 8, 4: 11}
                call_date = datetime(year, quarter_months.get(quarter, 1), 15)

            return TranscriptData(
                ticker=ticker.upper(),
                fiscal_year=year,
                fiscal_quarter=quarter,
                call_date=call_date,
                raw_content=transcript_content,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except (ValueError, RuntimeError):
            # API errors or rate limiting - return None to try next provider
            return None

    async def get_all_transcripts(
        self, ticker: str, years: list[int] | None = None
    ) -> list[TranscriptData]:
        """Fetch all available transcripts for a ticker.

        Warning: This can consume many API calls. Use sparingly on free tier.

        Args:
            ticker: Stock ticker symbol
            years: Optional list of years to fetch. If None, fetches last 2 years.

        Returns:
            List of TranscriptData objects
        """
        all_transcripts: list[TranscriptData] = []

        if years is None:
            current_year = datetime.now().year
            # Only fetch 2 years due to rate limits
            years = [current_year, current_year - 1]

        for year in years:
            for quarter in [1, 2, 3, 4]:
                # Check if we have calls remaining
                if self._rate_limiter.remaining_calls() <= 2:
                    # Reserve some calls for other operations
                    break

                transcript = await self.get_transcript(ticker, year, quarter)
                if transcript:
                    all_transcripts.append(transcript)

        return sorted(
            all_transcripts, key=lambda x: (x.fiscal_year, x.fiscal_quarter)
        )


# Synchronous wrapper for convenience
def fetch_alphavantage_transcript_sync(
    ticker: str, year: int, quarter: int, api_key: str | None = None
) -> TranscriptData | None:
    """Synchronous helper to fetch a single transcript from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol
        year: Fiscal year
        quarter: Fiscal quarter
        api_key: Optional API key

    Returns:
        TranscriptData if found
    """

    async def _fetch() -> TranscriptData | None:
        async with AlphaVantageClient(api_key=api_key) as client:
            return await client.get_transcript(ticker, year, quarter)

    return asyncio.run(_fetch())
