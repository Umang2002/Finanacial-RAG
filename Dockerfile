# Backend image for HF Spaces (Docker SDK) — serves the FastAPI app only.
# WHY a separate image instead of running scripts/*.py here: Spaces just needs
# /health + /query + /eval/summary up; ingestion/indexing already ran locally
# and data/index/ is baked into the build context below.
FROM python:3.11-slim

WORKDIR /app

# WHY apt build deps: pymupdf/docling have native extensions; slim base lacks a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# WHY CPU wheel index for torch+torchvision together: default PyPI torch ships
# CUDA builds (~2GB extra, unused on HF's free CPU tier); pinning torchvision
# here too avoids pip later pulling a mismatched GPU torchvision as some other
# package's transitive dep, which breaks transformers' torchvision::nms op lookup.
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY data/index/ data/index/
COPY data/eval/ data/eval/

# HF Spaces Docker SDK routes traffic to this port by default.
ENV PORT=7860
EXPOSE 7860

# WHY no .env copied: GROQ_API_KEY / SEC_USER_AGENT come from HF Space secrets at runtime.
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
