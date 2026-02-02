"""EarningsCall.biz API client for earnings call transcripts.

This is the primary transcript source. Free tier includes Apple and Microsoft.
For 5,000+ companies, sign up at https://earningscall.biz for an API key.
"""

import asyncio
from datetime import datetime
from typing import Any

from src.config import get_settings
from src.data.fmp_client import TranscriptData


class EarningsCallClient:
    """Client for EarningsCall.biz API - earnings call transcripts.

    Uses the official earningscall Python library.
    Free tier: Apple (AAPL) and Microsoft (MSFT) only.
    Paid tier: 5,000+ companies.

    API Documentation: https://earningscall.biz/api-guide
    Python Library: https://github.com/EarningsCall/earningscall-python
    """

    # Companies available on free tier
    FREE_TIER_COMPANIES = {"AAPL", "MSFT"}

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the EarningsCall client.

        Args:
            api_key: EarningsCall API key. If not provided, reads from settings.
                     Without a key, only AAPL and MSFT are available.
        """
        settings = get_settings()
        self.api_key = api_key or settings.earningscall_api_key

        # Set up the library
        try:
            import earningscall

            if self.api_key:
                earningscall.api_key = self.api_key
            self._earningscall = earningscall
        except ImportError:
            raise ImportError(
                "earningscall library not installed. "
                "Install with: pip install earningscall"
            )

    async def __aenter__(self) -> "EarningsCallClient":
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        pass  # No cleanup needed for this library

    def _is_free_tier_company(self, ticker: str) -> bool:
        """Check if ticker is available on free tier."""
        return ticker.upper() in self.FREE_TIER_COMPANIES

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
        ticker = ticker.upper()

        # Check access
        if not self.api_key and not self._is_free_tier_company(ticker):
            # No API key and not a free tier company
            return None

        try:
            # Run sync library call in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._fetch_transcript_sync,
                ticker,
                year,
                quarter,
            )
            return result
        except Exception as e:
            # Log but don't raise - allow fallback to other providers
            import logging

            logging.getLogger(__name__).warning(
                f"EarningsCall error for {ticker} Q{quarter} {year}: {e}"
            )
            return None

    def _fetch_transcript_sync(
        self, ticker: str, year: int, quarter: int
    ) -> TranscriptData | None:
        """Synchronous transcript fetch (called in executor)."""
        try:
            from earningscall import get_company

            company = get_company(ticker)
            if not company:
                return None

            # Get transcript with level=4 for Q&A separation
            transcript = company.get_transcript(year=year, quarter=quarter, level=1)

            if not transcript or not transcript.text:
                return None

            # Try to get event date
            call_date = datetime(year, quarter * 3, 15)  # Default estimate
            try:
                events = list(company.events())
                for event in events:
                    if event.year == year and event.quarter == quarter:
                        if hasattr(event, "conference_date") and event.conference_date:
                            call_date = event.conference_date
                        break
            except Exception:
                pass  # Use default date

            return TranscriptData(
                ticker=ticker,
                fiscal_year=year,
                fiscal_quarter=quarter,
                call_date=call_date,
                raw_content=transcript.text,
            )

        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Transcript fetch failed: {e}")
            return None

    async def get_transcript_list(self, ticker: str) -> list[dict[str, Any]]:
        """Get list of available transcripts for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of available transcript metadata (year, quarter)
        """
        ticker = ticker.upper()

        if not self.api_key and not self._is_free_tier_company(ticker):
            return []

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._get_events_sync, ticker
            )
            return result
        except Exception:
            return []

    def _get_events_sync(self, ticker: str) -> list[dict[str, Any]]:
        """Get available events synchronously."""
        try:
            from earningscall import get_company

            company = get_company(ticker)
            if not company:
                return []

            events = []
            for event in company.events():
                events.append(
                    {
                        "year": event.year,
                        "quarter": event.quarter,
                        "date": (
                            event.conference_date.isoformat()
                            if hasattr(event, "conference_date") and event.conference_date
                            else None
                        ),
                    }
                )
            return events
        except Exception:
            return []

    async def get_all_transcripts(
        self, ticker: str, years: list[int] | None = None
    ) -> list[TranscriptData]:
        """Fetch all available transcripts for a ticker.

        Args:
            ticker: Stock ticker symbol
            years: Optional list of years to fetch. If None, fetches last 3 years.

        Returns:
            List of TranscriptData objects
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 2, current_year + 1))

        all_transcripts: list[TranscriptData] = []

        for year in years:
            for quarter in [1, 2, 3, 4]:
                transcript = await self.get_transcript(ticker, year, quarter)
                if transcript:
                    all_transcripts.append(transcript)

        return sorted(
            all_transcripts, key=lambda x: (x.fiscal_year, x.fiscal_quarter)
        )


# Synchronous wrapper for convenience
def fetch_earningscall_transcript_sync(
    ticker: str, year: int, quarter: int, api_key: str | None = None
) -> TranscriptData | None:
    """Synchronous helper to fetch a single transcript from EarningsCall.

    Args:
        ticker: Stock ticker symbol
        year: Fiscal year
        quarter: Fiscal quarter
        api_key: Optional API key

    Returns:
        TranscriptData if found
    """

    async def _fetch() -> TranscriptData | None:
        async with EarningsCallClient(api_key=api_key) as client:
            return await client.get_transcript(ticker, year, quarter)

    return asyncio.run(_fetch())
