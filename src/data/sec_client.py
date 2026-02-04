"""SEC EDGAR API client for 8-K filings as backup transcript source."""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class SECFiling:
    """SEC filing metadata."""

    ticker: str
    cik: str
    accession_number: str
    filing_date: datetime
    form_type: str
    filing_url: str


@dataclass
class SECTranscriptData:
    """Parsed transcript data from SEC EDGAR."""

    ticker: str
    cik: str
    filing_date: datetime
    raw_content: str
    accession_number: str


class SECEdgarClient:
    """Client for SEC EDGAR API."""

    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/submissions"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    # Required user agent per SEC guidelines
    USER_AGENT = "EarningsCallAnalyser research@example.com"

    def __init__(self) -> None:
        """Initialize SEC EDGAR client."""
        self._client: httpx.AsyncClient | None = None
        self._cik_cache: dict[str, str] = {}

    async def __aenter__(self) -> "SECEdgarClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.USER_AGENT},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _get(self, url: str) -> Any:
        """Make GET request with retry logic."""
        response = await self.client.get(url)
        response.raise_for_status()
        return response.json()

    async def _get_text(self, url: str) -> str:
        """Make GET request returning text."""
        response = await self.client.get(url)
        response.raise_for_status()
        return response.text

    async def get_cik(self, ticker: str) -> str | None:
        """Get CIK (Central Index Key) for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK string (zero-padded to 10 digits) or None
        """
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        try:
            data = await self._get(self.COMPANY_TICKERS_URL)

            # Search for ticker in the response
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik

            return None

        except httpx.HTTPStatusError:
            return None

    async def get_company_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[SECFiling]:
        """Get recent filings for a company.

        Args:
            ticker: Stock ticker symbol
            form_types: Filter by form types (e.g., ['8-K', '10-Q'])
            limit: Maximum number of filings to return

        Returns:
            List of SECFiling objects
        """
        cik = await self.get_cik(ticker)
        if not cik:
            return []

        try:
            url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            data = await self._get(url)

            filings = []
            recent = data.get("filings", {}).get("recent", {})

            accession_numbers = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            form_types_list = recent.get("form", [])
            primary_docs = recent.get("primaryDocument", [])

            for i, acc_num in enumerate(accession_numbers[:limit]):
                form_type = form_types_list[i] if i < len(form_types_list) else ""

                if form_types and form_type not in form_types:
                    continue

                filing_date_str = filing_dates[i] if i < len(filing_dates) else ""
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""

                try:
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                except (ValueError, TypeError):
                    continue

                # Construct filing URL
                acc_num_nodash = acc_num.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{acc_num_nodash}/{primary_doc}"
                )

                filings.append(
                    SECFiling(
                        ticker=ticker,
                        cik=cik,
                        accession_number=acc_num,
                        filing_date=filing_date,
                        form_type=form_type,
                        filing_url=filing_url,
                    )
                )

            return filings

        except httpx.HTTPStatusError:
            return []

    async def get_8k_earnings_filings(
        self, ticker: str, start_date: datetime | None = None
    ) -> list[SECFiling]:
        """Get 8-K filings that may contain earnings information.

        Args:
            ticker: Stock ticker symbol
            start_date: Only return filings after this date

        Returns:
            List of 8-K filings
        """
        filings = await self.get_company_filings(ticker, form_types=["8-K"])

        if start_date:
            filings = [f for f in filings if f.filing_date >= start_date]

        return filings

    async def get_filing_content(self, filing: SECFiling) -> str | None:
        """Download and extract content from a filing.

        Args:
            filing: SECFiling object

        Returns:
            Filing content as text
        """
        try:
            # Rate limiting - SEC requires 10 req/sec max
            await asyncio.sleep(0.1)

            content = await self._get_text(filing.filing_url)

            # Basic HTML stripping for readability
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", content)
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            return text.strip()

        except httpx.HTTPStatusError:
            return None

    async def search_transcripts_in_filings(
        self, ticker: str, years: list[int] | None = None
    ) -> list[SECTranscriptData]:
        """Search 8-K filings for earnings call transcripts.

        Note: SEC doesn't require transcripts in 8-Ks, so coverage is limited.

        Args:
            ticker: Stock ticker symbol
            years: Years to search

        Returns:
            List of transcripts found
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 3, current_year + 1))

        start_date = datetime(min(years), 1, 1)
        filings = await self.get_8k_earnings_filings(ticker, start_date)

        transcripts = []
        transcript_keywords = [
            "earnings call",
            "conference call",
            "transcript",
            "earnings conference",
        ]

        for filing in filings:
            content = await self.get_filing_content(filing)

            if not content:
                continue

            # Check if this 8-K contains transcript content
            content_lower = content.lower()
            if any(kw in content_lower for kw in transcript_keywords):
                # Additional check for Q&A section indicators
                if "question" in content_lower and "answer" in content_lower:
                    transcripts.append(
                        SECTranscriptData(
                            ticker=ticker,
                            cik=filing.cik,
                            filing_date=filing.filing_date,
                            raw_content=content,
                            accession_number=filing.accession_number,
                        )
                    )

        return transcripts


# Synchronous wrapper
def get_cik_sync(ticker: str) -> str | None:
    """Get CIK synchronously."""

    async def _fetch() -> str | None:
        async with SECEdgarClient() as client:
            return await client.get_cik(ticker)

    return asyncio.run(_fetch())
