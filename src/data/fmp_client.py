"""Financial Modeling Prep API client for earnings call transcripts."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings


@dataclass
class TranscriptData:
    """Parsed transcript data from FMP API."""

    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    call_date: datetime
    raw_content: str


class FMPClient:
    """Client for Financial Modeling Prep API."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the FMP client.

        Args:
            api_key: FMP API key. If not provided, reads from settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.fmp_api_key
        if not self.api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "FMPClient":
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
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make authenticated GET request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["apikey"] = self.api_key

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
        try:
            data = await self._get(f"earning_call_transcript/{ticker}", {"year": year})

            if not data:
                return None

            # FMP returns list of transcripts, find the matching quarter
            for transcript in data:
                if transcript.get("quarter") == quarter:
                    call_date_str = transcript.get("date", "")
                    call_date = (
                        datetime.fromisoformat(call_date_str.replace("Z", "+00:00"))
                        if call_date_str
                        else datetime(year, quarter * 3, 15)
                    )

                    return TranscriptData(
                        ticker=ticker,
                        fiscal_year=year,
                        fiscal_quarter=quarter,
                        call_date=call_date,
                        raw_content=transcript.get("content", ""),
                    )

            return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

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
        try:
            # FMP requires year parameter, so we need to query each year
            if years is None:
                current_year = datetime.now().year
                years = list(range(current_year - 3, current_year + 1))

            all_transcripts: list[TranscriptData] = []

            for year in years:
                data = await self._get(f"earning_call_transcript/{ticker}", {"year": year})

                if not data or isinstance(data, dict):
                    continue

                for transcript in data:
                    quarter = transcript.get("quarter")
                    if quarter:
                        call_date_str = transcript.get("date", "")
                        try:
                            call_date = datetime.fromisoformat(
                                call_date_str.replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            call_date = datetime(year, quarter * 3, 15)

                        all_transcripts.append(
                            TranscriptData(
                                ticker=ticker,
                                fiscal_year=year,
                                fiscal_quarter=quarter,
                                call_date=call_date,
                                raw_content=transcript.get("content", ""),
                            )
                        )

                # Rate limiting - small delay between year requests
                await asyncio.sleep(0.5)

            return sorted(
                all_transcripts, key=lambda x: (x.fiscal_year, x.fiscal_quarter)
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            raise

    async def get_company_profile(self, ticker: str) -> dict[str, Any] | None:
        """Fetch company profile information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company profile dict or None
        """
        try:
            data = await self._get(f"profile/{ticker}")
            return data[0] if data else None
        except httpx.HTTPStatusError:
            return None

    async def batch_get_transcripts(
        self, tickers: list[str], years: list[int] | None = None
    ) -> dict[str, list[TranscriptData]]:
        """Fetch transcripts for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            years: Optional list of years to fetch

        Returns:
            Dict mapping ticker to list of transcripts
        """
        results: dict[str, list[TranscriptData]] = {}

        for ticker in tickers:
            transcripts = await self.get_all_transcripts(ticker, years)
            results[ticker] = transcripts
            # Rate limiting between companies
            await asyncio.sleep(1.0)

        return results


# Synchronous wrapper for convenience
def fetch_transcript_sync(
    ticker: str, year: int, quarter: int, api_key: str | None = None
) -> TranscriptData | None:
    """Synchronous helper to fetch a single transcript.

    Args:
        ticker: Stock ticker symbol
        year: Fiscal year
        quarter: Fiscal quarter

    Returns:
        TranscriptData if found
    """

    async def _fetch() -> TranscriptData | None:
        async with FMPClient(api_key=api_key) as client:
            return await client.get_transcript(ticker, year, quarter)

    return asyncio.run(_fetch())
