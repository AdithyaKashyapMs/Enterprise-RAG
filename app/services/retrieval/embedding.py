"""
Embedding model selection.

EMBEDDING_PROVIDER=gemini | local | auto (default auto: Gemini if reachable, else local).

The vector dimension is measured from a real embedding instead of being
hard-coded, and Qdrant search verifies it matches the collection — so a silent
switch between providers can never produce garbage or empty results.
"""

import os
import re
import time
from typing import Callable, TypeVar

import logfire

from app.config import settings

BATCH_SIZE = 50
REQUEST_TIMEOUT_S = float(os.getenv("EMBED_TIMEOUT_S", "60"))
T = TypeVar("T")

_active_model = None
_model_type: str | None = None  # "gemini" | "local"
_dim: int | None = None


def _load_gemini():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        # Without a timeout a dropped connection (e.g. laptop sleep) blocks forever on SSL read.
        client_args={"timeout": REQUEST_TIMEOUT_S},
    )
    probe = model.embed_query("probe")
    return model, len(probe)


def _load_local():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
    return model, model.get_sentence_embedding_dimension()


def _init():
    """Initialise the embedding model once per process."""
    global _active_model, _model_type, _dim
    if _active_model is not None:
        return

    provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()

    if provider in ("gemini", "auto"):
        try:
            _active_model, _dim = _load_gemini()
            _model_type = "gemini"
            logfire.info(f"Gemini embeddings ready ({settings.GEMINI_EMBEDDING_MODEL}, {_dim}-dim).")
            return
        except Exception as e:
            if provider == "gemini":
                raise RuntimeError(f"Gemini embeddings unavailable: {e}") from e
            logfire.warning(f"Gemini unavailable ({e}); using local embeddings.")

    _active_model, _dim = _load_local()
    _model_type = "local"
    logfire.info(f"Local embeddings ready ({settings.LOCAL_EMBEDDING_MODEL}, {_dim}-dim).")


def get_embedding_dim() -> int:
    _init()
    return _dim


def get_embedding_model_name() -> str:
    _init()
    return settings.GEMINI_EMBEDDING_MODEL if _model_type == "gemini" else settings.LOCAL_EMBEDDING_MODEL


MAX_ATTEMPTS = int(os.getenv("EMBED_MAX_ATTEMPTS", "6"))
MAX_WAIT_S = 65.0
_RETRY_DELAY_RE = re.compile(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s")


def _with_rate_limit_retry(fn: Callable[[], T], what: str, max_attempts: int = MAX_ATTEMPTS) -> T:
    """
    Retry Gemini rate-limit (429 / RESOURCE_EXHAUSTED) errors and network timeouts.
    Honours the server's retryDelay hint when present (free tier asks for 20-60s),
    otherwise backs off exponentially. Other errors are raised immediately.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            lowered = f"{type(e).__name__} {err}".lower()
            retryable = any(x in lowered for x in ("429", "rate limit", "quota", "resource_exhausted",
                                                   "timeout", "timed out", "connection"))
            if not retryable or attempt == max_attempts:
                raise
            hint = _RETRY_DELAY_RE.search(err)
            wait = min(float(hint.group(1)) + 1 if hint else 2**attempt, MAX_WAIT_S)
            logfire.warning(f"Gemini error on {what} ({type(e).__name__}) — retrying in {wait:.0f}s ({attempt}/{max_attempts}).")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type != "gemini":
        return _active_model.encode(batch, show_progress_bar=False).tolist()
    return _with_rate_limit_retry(lambda: _active_model.embed_documents(batch), "batch")


def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        # Fewer attempts at query time: a user is waiting.
        return _with_rate_limit_retry(lambda: _active_model.embed_query(query), "query", max_attempts=2)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            vectors.extend(_embed_batch(batch))
    return vectors
