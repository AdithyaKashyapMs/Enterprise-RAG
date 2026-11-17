import atexit
from collections import OrderedDict
from threading import Lock

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query, get_embedding_dim


class RetrievalError(RuntimeError):
    """Retrieval could not run (misconfiguration or backend failure) — not 'no results'."""


_client: QdrantClient | None = None
_client_lock = Lock()
_dim_checked = False

_CACHE_MAX = 128
_search_cache: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_search_cache_lock = Lock()


def get_client() -> QdrantClient:
    """Qdrant Cloud/server via QDRANT_CLUSTER_ENDPOINT, or embedded on-disk via QDRANT_PATH."""
    global _client
    with _client_lock:
        if _client is None:
            if settings.QDRANT_PATH:
                _client = QdrantClient(path=settings.QDRANT_PATH)
                # Close explicitly; otherwise its __del__ runs mid-shutdown and prints a traceback.
                atexit.register(_client.close)
            elif settings.QDRANT_URL:
                _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            else:
                raise RetrievalError("Set QDRANT_CLUSTER_ENDPOINT (or QDRANT_PATH for local mode).")
    return _client


def get_collection_dim(client: QdrantClient) -> int:
    return client.get_collection(settings.QDRANT_COLLECTION).config.params.vectors.size


def _ensure_ready(client: QdrantClient, check_dim: bool = True) -> None:
    """Fail loudly if the collection is missing or doesn't match the query embedder."""
    global _dim_checked
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        raise RetrievalError(
            f"Collection '{settings.QDRANT_COLLECTION}' does not exist — run ingestion first."
        )
    if not check_dim or _dim_checked:
        return
    collection_dim, query_dim = get_collection_dim(client), get_embedding_dim()
    if collection_dim != query_dim:
        raise RetrievalError(
            f"Embedding dimension mismatch: collection has {collection_dim}-dim vectors but the "
            f"active embedding model produces {query_dim}-dim. Re-ingest with --wipe or fix "
            f"EMBEDDING_PROVIDER."
        )
    _dim_checked = True


def _source_filter(source: str | None) -> models.Filter | None:
    if not source:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
    )


def _to_doc(point, score: float | None = None) -> dict:
    payload = point.payload or {}
    return {
        "content": payload.get("text", ""),
        "source": payload.get("source", "unknown"),
        "chunk_index": payload.get("chunk_index"),
        "score": float(score if score is not None else getattr(point, "score", 0.0) or 0.0),
    }


def search_enterprise_knowledge(query: str, limit: int = 8, source: str | None = None) -> list[dict]:
    """
    Vector search over the knowledge base, optionally restricted to one source file.
    Returns [{content, source, chunk_index, score}]; raises RetrievalError if search cannot run.
    """
    key = (" ".join(query.casefold().split()), limit, source)
    with _search_cache_lock:
        if key in _search_cache:
            _search_cache.move_to_end(key)
            logfire.info(f"Qdrant cache hit for query: {query}")
            return [dict(item) for item in _search_cache[key]]

    try:
        client = get_client()
        _ensure_ready(client)
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=embed_query(query),
            query_filter=_source_filter(source),
            limit=limit,
            with_payload=True,
        )
    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(f"Qdrant search failed: {e}") from e

    results = [_to_doc(p) for p in response.points]
    with _search_cache_lock:
        _search_cache[key] = [dict(item) for item in results]
        if len(_search_cache) > _CACHE_MAX:
            _search_cache.popitem(last=False)
    return results


def list_sources() -> list[dict]:
    """All indexed source files with their chunk counts: [{source, source_type, chunks}]."""
    try:
        client = get_client()
        _ensure_ready(client, check_dim=False)
        counts: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=["source", "source_type"],
                with_vectors=False,
            )
            for p in points:
                src = (p.payload or {}).get("source", "unknown")
                entry = counts.setdefault(
                    src, {"source": src, "source_type": (p.payload or {}).get("source_type"), "chunks": 0}
                )
                entry["chunks"] += 1
            if offset is None:
                break
    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(f"Listing documents failed: {e}") from e
    return sorted(counts.values(), key=lambda d: d["source"])


def read_chunks(source: str, start_chunk: int = 0, num_chunks: int = 3) -> list[dict]:
    """Contiguous chunks [start_chunk, start_chunk + num_chunks) of one source file, in order."""
    try:
        client = get_client()
        _ensure_ready(client, check_dim=False)
        points, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                    models.FieldCondition(
                        key="chunk_index",
                        range=models.Range(gte=start_chunk, lt=start_chunk + num_chunks),
                    ),
                ]
            ),
            limit=num_chunks,
            with_payload=True,
            with_vectors=False,
        )
    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(f"Reading '{source}' failed: {e}") from e
    return sorted((_to_doc(p, score=0.0) for p in points), key=lambda d: d["chunk_index"] or 0)
