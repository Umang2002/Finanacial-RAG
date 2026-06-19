# Phase 1 — Ingestion

## Summary
Pulls 10-K/10-Q filings for configured tickers/years off SEC EDGAR (free public API), saves raw `.htm` + `metadata.json` to `data/raw/`, then parses raw HTML into clean structured JSON (full text, Item-numbered sections, raw tables) saved to `data/processed/`. This is the entry point of the pipeline — everything downstream (chunking, indexing, retrieval) reads from `data/processed/*/parsed.json`.

## Files Changed / Added
- `src/ingestion/models.py` — new — pydantic models: `FilingRef`, `DownloadedFiling`, `ParsedSection`, `ParsedFiling`
- `src/ingestion/sec_loader.py` — modified (stub → full) — `SECEdgarLoader`: ticker→CIK resolution, filing listing, rate-limited+retried download
- `src/ingestion/html_parser.py` — modified (stub → full) — `HTMLFilingParser`: inline-XBRL HTML → `ParsedFiling`
- `src/ingestion/metadata_extractor.py` — modified (stub → full) — `extract_metadata()`: pulls `dei:` XBRL fields (company name, CIK, fiscal year, period end)
- `src/ingestion/pdf_parser.py` — modified (stub → explicit `NotImplementedError`) — deferred, EDGAR primary docs are HTML not PDF
- `scripts/download_filings.py` — modified (stub → full) — CLI wrapping `SECEdgarLoader`
- `scripts/parse_filings.py` — new — CLI wrapping `HTMLFilingParser`, scans `data/raw/` for metadata.json + primary doc pairs
- `configs/base.yaml` — modified — `ingestion:` section (tickers, filing_types, years, raw/processed dirs)
- `tests/unit/test_sec_loader.py` — new
- `tests/unit/test_html_parser.py` — new
- `tests/unit/test_metadata_extractor.py` — new
- `notebooks/01_data_exploration.ipynb` — modified — exploratory notebook confirming EDGAR filings are inline-XBRL (not plain HTML/PDF), used to derive parsing heuristics below

## Key Design Decisions
- **Inline-XBRL, not plain HTML**: confirmed via notebook that content lives in `<span name="dei:...">`-style tags, not `<p>`/`<h1-4>`. `soup.get_text()` walks all text nodes regardless of tag type, so full-text extraction works; section-splitting can't rely on heading tags and instead regexes for `Item N.` boundaries.
- **PDF parser deferred**: `parse_pdf()` raises `NotImplementedError` — no PDF in the real ingestion path (10-K/10-Q primaries are `.htm` since ~2002). Avoids speculative code per CLAUDE.md.
- **fiscal_year grouped by `report_date`, not `filing_date`**: a 10-K covering FY2023 is often filed in early 2024 — using filing_date would misfile it.
- **ToC stripped via gap-heuristic**: SEC's table of contents lists every `Item` back-to-back with only a title + page number (~10-40 chars) between matches; real section bodies are hundreds-thousands of chars apart. First inter-match gap ≥200 chars marks ToC→content boundary.
- **No dedup by `item_label`**: 10-Qs reuse Item numbers across Part I/Part II (Item 1 = "Financial Statements" in Part I, "Legal Proceedings" in Part II). Every regex match becomes its own `ParsedSection` in document order — chunking only needs label+text, not uniqueness.
- **`raw_tables` kept as raw HTML strings, not flattened**: table row/col structure is needed intact by Phase 2's `table_serializer.py` — flattening here would destroy it before it's used.
- **Idempotent downloads**: `download_filing()` skips re-fetching if `primary.*` already exists on disk — SEC rate limit (8 req/s, capped at 10 by ToS) is precious and filed documents never change.
- **Retry policy**: tenacity-based exponential backoff, retries only HTTP 429 + 5xx + transport errors (not other 4xx — those mean a bad request, retrying won't help).

## Execution Flow
1. `python scripts/download_filings.py --ticker AAPL --years 2023` (or no args → uses `configs/base.yaml` defaults)
2. `load_config()` reads `configs/base.yaml`, resolves `${oc.env:SEC_USER_AGENT}` from `.env`
3. `SECEdgarLoader.download_all(tickers, forms, years)`:
   a. `resolve_cik(ticker)` — fetches+caches `company_tickers.json`, maps ticker→10-digit CIK
   b. `list_filings(...)` — GETs `data.sec.gov/submissions/CIK{cik}.json`, filters `filings.recent` arrays by form+report_date year, builds `FilingRef` per match
   c. `download_filing(ref)` per ref — GETs primary doc from EDGAR archives URL, writes `data/raw/{ticker}/{form}_{year}/primary.htm` + `metadata.json` (skips if already on disk)
4. CLI prints rich summary table (ticker/form/year/size/status)
5. `python scripts/parse_filings.py --ticker AAPL --years 2023`
6. `_find_raw_filings()` scans `data/raw/` for `metadata.json` + `primary.*` pairs, rebuilds `FilingRef` from disk JSON
7. `HTMLFilingParser.parse(primary_path, ref)`:
   a. BeautifulSoup parse with `lxml`, decompose `<script>/<style>/<head>`
   b. extract `raw_tables` (stringified `<table>` tags) before any text flattening
   c. `full_text = soup.get_text()` (whitespace-collapsed)
   d. `_split_sections(full_text)` — regex `Item N.` matches, strip ToC via gap heuristic, slice into `ParsedSection` list
   e. `extract_metadata(soup, fallback_period_end=ref.report_date)` — `dei:` tag lookups for company name/CIK/fiscal year/period end
   f. assembles `ParsedFiling`
8. writes `data/processed/{ticker}/{form}_{year}/parsed.json` (skips if cached), CLI prints summary table (sections/tables count)

## Data Contract (Input → Output)
- Input: none (external) — SEC EDGAR public REST API, no auth, requires `User-Agent: Name email@example.com` header
- Intermediate: `data/raw/{ticker}/{form}_{year}/primary.htm` + `metadata.json` (`DownloadedFiling` JSON)
- Output (consumed by Phase 2): `data/processed/{ticker}/{form}_{year}/parsed.json` — `ParsedFiling` model: `full_text`, `sections: list[ParsedSection]`, `raw_tables: list[str]` (raw HTML)

## Tests
- `tests/unit/test_sec_loader.py` — CIK resolution, filing listing/filtering, download idempotency, retry-on-429, mocked `httpx.Client` (no real network calls)
- `tests/unit/test_html_parser.py` — section splitting incl. ToC-gap heuristic, duplicate Item labels across Part I/II, table extraction
- `tests/unit/test_metadata_extractor.py` — `dei:` tag extraction, fallback to `report_date` when `period_end_date` text is empty
- Gap: no test against a full real downloaded filing (fixtures use synthetic HTML snippets) — acceptable, real-filing shape confirmed manually via the notebook instead

## Config Keys Used
- `configs/base.yaml` → `ingestion.sec_user_agent` (from env), `ingestion.tickers`, `ingestion.filing_types`, `ingestion.years`, `ingestion.raw_data_dir`, `ingestion.processed_data_dir`

## Open Items / Deferred
- `src/ingestion/pdf_parser.py` — stub raises `NotImplementedError`, revisit only if an exhibit/filing type shows up that's PDF-only
- `filings.recent` only covers ~last 1000 filings per company — older filings (paginated `filings.files[]`) not handled; fine for 2021-2024 target range
