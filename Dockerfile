# One image for every part of the app (API, Streamlit UI, MCP server, ingestion).
# docker-compose.yml picks which one to run with `command:`.

# ---- Base: a small official Python image (Debian "slim" = no compilers, docs, etc.) ----
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE: don't write .pyc files into the image.
# PYTHONUNBUFFERED: print logs immediately (otherwise `docker logs` lags behind).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    CHECKPOINT_DB=/app/data/checkpoints.sqlite

WORKDIR /app

# libmagic: file-type detection used by `unstructured` when parsing .docx/.pptx.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies (copied BEFORE the code, so this slow layer is cached) ----
# CPU-only PyTorch for the local embedding fallback: ~200 MB instead of ~2 GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the FlashRank reranker model at build time, so the first query isn't slow.
RUN python -c "import tempfile, os; from flashrank import Ranker; Ranker(cache_dir=os.path.join(tempfile.gettempdir(), 'flashrank'))"

# ---- Application code (changes often, so it comes last) ----
COPY app ./app
COPY ui ./ui
COPY evals ./evals

# ---- Don't run as root: if the app is ever compromised, the attacker isn't root in the container ----
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/processed_data /app/.cache \
    && chown -R appuser:appuser /app /tmp/flashrank
USER appuser

EXPOSE 8000 8501

# Docker restarts/marks the container unhealthy if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# Default: the REST API. Listen on 0.0.0.0 — inside a container, 127.0.0.1 is unreachable from outside.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
