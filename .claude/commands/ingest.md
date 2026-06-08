Run the full data ingestion pipeline for a given ticker and year range.

Steps:
1. Check that Qdrant is running (`docker compose ps`)
2. Run `python scripts/download_filings.py --ticker $TICKER --years $YEARS`
3. Parse all PDFs in `data/raw/$TICKER/` using `src/ingestion/pdf_parser.py`
4. Extract metadata for each document
5. Report: number of documents downloaded, pages parsed, tables found

Arguments: TICKER (e.g. AAPL), YEARS (e.g. 2022 2023)
