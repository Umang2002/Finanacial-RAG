Run the document processing and chunking pipeline.

Steps:
1. Load processed documents from `data/processed/`
2. Run the chunking strategy specified in `configs/base.yaml` (default: recursive)
3. Show a sample of 3 chunks with their metadata
4. Report: total chunks created, avg chunk size, chunks with tables

To compare chunking strategies, pass strategy=semantic or strategy=sentence_window.
