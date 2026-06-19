"""SEC EDGAR REST API client — resolves tickers to CIKs, lists 10-K/10-Q filings,
and downloads primary documents into data/raw/.

WHAT: SECEdgarLoader wraps the 3-hop SEC EDGAR flow:
    1. ticker -> CIK              (company_tickers.json)
    2. CIK -> filing list         (data.sec.gov/submissions/CIK{cik}.json)
    3. filing -> primary document (www.sec.gov/Archives/edgar/data/...)
WHY: SEC has no single "give me AAPL's 10-K for 2023" endpoint — three
separate REST calls with different response shapes are needed. Wrapping that
here keeps the complexity out of scripts/ and later pipeline phases, which
only need DownloadedFiling objects + files on disk.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.ingestion.models import DownloadedFiling, FilingForm, FilingRef
from src.utils.logging import get_logger

logger = get_logger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"


def _is_retryable(exc: BaseException) -> bool:
    """True for transient errors worth retrying: HTTP 429 (rate limit) and 5xx.

    WHY: SEC EDGAR rate-limits aggressively under load; a 429 almost always
    succeeds after backoff. Other 4xx errors mean a bad request — retrying
    won't fix those.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


class _RateLimiter:
    """Gate ensuring no more than `rate_per_sec` calls/sec.

    LEARN: SEC EDGAR ToS caps requests at 10/sec; we default to 8/sec to
    leave headroom for clock drift and retries.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self._min_interval = 1.0 / rate_per_sec
        self._last_call = 0.0

    def wait(self) -> None:
        """Block until enough time has passed since the previous call."""
        elapsed = time.monotonic() - self._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class SECEdgarLoader:
    """Downloads 10-K/10-Q filings from SEC EDGAR for given tickers and years."""

    def __init__(
        self,
        user_agent: str,
        raw_data_dir: str | Path,
        rate_limit: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Set up the HTTP client (with required User-Agent header) and rate limiter.

        Args:
            user_agent: value for the User-Agent header — SEC EDGAR ToS
                requires "Name email@example.com" so requests are traceable.
            raw_data_dir: root directory filings are saved under
                (data/raw/{ticker}/{form}_{fiscal_year}/...).
            rate_limit: max requests/sec (SEC caps at 10).
            client: optional pre-built httpx.Client — lets tests inject a
                MockTransport without hitting the network.
        """
        if not user_agent or "@" not in user_agent:
            # LEARN: fail loudly here rather than getting a cryptic 403 from
            # SEC later — a missing/invalid User-Agent is the #1 EDGAR gotcha.
            raise ValueError(
                "SEC_USER_AGENT must be set to 'Your Name your@email.com' "
                "(see .env.example) — SEC EDGAR rejects requests without it."
            )

        self._raw_data_dir = Path(raw_data_dir)
        self._client = client or httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0)
        self._rate_limiter = _RateLimiter(rate_limit)
        self._cik_map: dict[str, str] | None = None

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        """GET with rate limiting + retry on 429/5xx. Raises on other HTTP errors."""
        self._rate_limiter.wait()
        logger.debug(f"GET {url}")
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def resolve_cik(self, ticker: str) -> str:
        """Map a ticker symbol to its 10-digit zero-padded CIK.

        WHY: SEC's submissions API is keyed by CIK, not ticker. The
        ticker->CIK map is ~9k entries and rarely changes, so fetch it once
        and cache for the lifetime of this loader.
        """
        if self._cik_map is None:
            response = self._get(_TICKERS_URL)
            data = response.json()
            # LEARN: company_tickers.json is a dict-of-dicts keyed by index
            # strings ("0", "1", ...), not a list — build a ticker->cik10 map.
            self._cik_map = {
                entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()
            }

        cik = self._cik_map.get(ticker.upper())
        if cik is None:
            raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")
        return cik

    def list_filings(
        self,
        ticker: str,
        forms: list[FilingForm],
        years: list[int],
    ) -> list[FilingRef]:
        """List 10-K/10-Q filings for a ticker, filtered to `forms` and `years`.

        WHY: SEC's submissions API returns *all* filings (every form type,
        every year) as parallel arrays. We filter to what the pipeline needs,
        using report_date (not filing_date) so a 10-K covering FY2023 — often
        filed in early 2024 — is grouped under 2023.
        """
        cik = self.resolve_cik(ticker)
        response = self._get(_SUBMISSIONS_URL.format(cik=cik))
        recent = response.json()["filings"]["recent"]

        # LEARN: `filings.recent` covers roughly the last ~1000 filings per
        # company — sufficient for 2021-2024. Older filings live in the
        # paginated `filings.files[]` index, not handled here.
        refs: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            report_date = date.fromisoformat(recent["reportDate"][i])
            if report_date.year not in years:
                continue
            refs.append(
                FilingRef(
                    ticker=ticker.upper(),
                    cik=cik,
                    form=form,
                    filing_date=date.fromisoformat(recent["filingDate"][i]),
                    report_date=report_date,
                    accession_no=recent["accessionNumber"][i],
                    primary_doc=recent["primaryDocument"][i],
                )
            )
        return refs

    def download_filing(self, ref: FilingRef) -> DownloadedFiling:
        """Download one filing's primary document to data/raw/, write metadata.json.

        WHY idempotent: re-running download_filings.py shouldn't re-fetch
        documents already on disk — SEC rate limits are precious and a filed
        document never changes after filing.
        """
        ticker_dir = self._raw_data_dir / ref.ticker / f"{ref.form}_{ref.fiscal_year}"
        ticker_dir.mkdir(parents=True, exist_ok=True)

        doc_ext = Path(ref.primary_doc).suffix or ".htm"
        raw_path = ticker_dir / f"primary{doc_ext}"
        metadata_path = ticker_dir / "metadata.json"

        url = _ARCHIVES_URL.format(
            cik_int=int(ref.cik), accession=ref.accession_no_compact, doc=ref.primary_doc
        )

        if raw_path.exists():
            logger.info(f"[skip] {ref.ticker} {ref.form} {ref.fiscal_year} — already on disk")
            downloaded = DownloadedFiling.from_ref(
                ref,
                download_url=url,
                raw_path=raw_path,
                metadata_path=metadata_path,
                size_bytes=raw_path.stat().st_size,
                skipped=True,
            )
        else:
            response = self._get(url)
            raw_path.write_bytes(response.content)
            logger.info(
                f"[get]  {ref.ticker} {ref.form} {ref.fiscal_year} "
                f"-> {raw_path} ({len(response.content):,} bytes)"
            )
            downloaded = DownloadedFiling.from_ref(
                ref,
                download_url=url,
                raw_path=raw_path,
                metadata_path=metadata_path,
                size_bytes=len(response.content),
                skipped=False,
            )

        metadata_path.write_text(downloaded.model_dump_json(indent=2))
        return downloaded

    def download_all(
        self,
        tickers: list[str],
        forms: list[FilingForm],
        years: list[int],
    ) -> list[DownloadedFiling]:
        """List + download all matching filings for every ticker.

        WHY: single entry point scripts/download_filings.py calls — owns the
        overall progress reporting so the CLI script stays thin.
        """
        all_refs: list[FilingRef] = []
        for ticker in tickers:
            try:
                refs = self.list_filings(ticker, forms, years)
            except ValueError as exc:
                logger.warning(f"{ticker}: {exc}")
                continue
            logger.info(f"{ticker}: {len(refs)} filings match {forms} x {years}")
            all_refs.extend(refs)

        downloaded: list[DownloadedFiling] = []
        for ref in tqdm(all_refs, desc="Downloading filings", unit="filing"):
            downloaded.append(self.download_filing(ref))
        return downloaded
