# Financial RAG — Hybrid Search System for SEC Filings

End-to-end Hybrid Search RAG system for SEC financial document intelligence (10-K/10-Q filings).

## Quick Start

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Set up environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 3. Configure secrets
cp .env.example .env   # Fill in API keys

# 4. Download filings
python scripts/download_filings.py --ticker AAPL --years 2022 2023

# 5. Build hybrid index
python scripts/build_index.py --config configs/base.yaml

# 6. Run evaluation
python scripts/run_eval.py --config configs/base.yaml
```

## Architecture

7-phase pipeline: Ingestion → Processing → Indexing → Query → Retrieval → Generation → Evaluation

See `.claude/CLAUDE.md` for full architecture notes and dev rules.
