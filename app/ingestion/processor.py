"""
Ingestion pipeline: parse -> chunk -> save JSON snapshot -> embed -> upsert to Qdrant.

Idempotent: each chunk gets a deterministic ID derived from (source file, chunk
index), and a file's old points are deleted before its new ones are written, so
re-running ingestion replaces documents instead of duplicating them.

Usage:
    python -m app.ingestion.processor DATA --wipe
    python -m app.ingestion.processor DATA/true_data true
"""

import json
import os
import sys
from datetime import datetime, timezone

import logfire
from qdrant_client.http import models

from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import (
    embed_texts,
    get_embedding_dim,
    get_embedding_model_name,
)
from app.services.retrieval.qdrant_service import chunk_id, get_client, get_collection_dim

PROCESSED_DATA_DIR = "processed_data"
# Files that failed in this run, so the CLI can report them and exit non-zero.
failed_files: list[tuple[str, str]] = []
already_indexed: set[str] = set()  # filled by --resume


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def extract_text(file_path: str, filename: str) -> str | None:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return parse_pdf(file_path)
    if ext in ("html", "htm"):
        return parse_html(file_path)
    if ext in ("txt", "md"):
        return parse_text(file_path)
    if ext in ("docx", "pptx"):
        from app.ingestion.loaders.office import parse_office  # heavy import, load lazily

        return parse_office(file_path)
    return None


def process_file(file_path: str, filename: str, source_type: str) -> int:
    """Ingest one file. Returns the number of chunks indexed."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            full_text = extract_text(file_path, filename)
            if full_text is None:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return 0
            if not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return 0

            chunks = chunk_text(full_text)
            if not chunks:
                return 0

            local_path = save_processed_locally(
                {"filename": filename, "source_type": source_type, "chunks": chunks},
                source_type,
                filename,
            )
            logfire.info(f"Saved processed data -> {local_path}")

            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                model_name = get_embedding_model_name()
                ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                client = get_client()

                # Remove this file's previous chunks (it may now have fewer).
                client.delete(
                    collection_name=settings.QDRANT_COLLECTION,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source", match=models.MatchValue(value=filename)
                                )
                            ]
                        )
                    ),
                )

                points = [
                    models.PointStruct(
                        id=chunk_id(filename, i),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                            "chunk_index": i,
                            "embedding_model": model_name,
                            # Lets ranking spot new documents (cold start: exploration boost).
                            "ingested_at": ingested_at,
                        },
                    )
                    for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]
                client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
                logfire.info(f"Indexed {len(points)} points from {filename}.")
                return len(points)

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            failed_files.append((filename, str(e)[:200]))
            return 0


def process_directory(dir_path: str, source_type: str) -> int:
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = sorted(
            f
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith(".")
        )
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        total = 0
        for n, f in enumerate(files, 1):
            if f in already_indexed:
                print(f"[{n}/{len(files)}] {f}: already indexed, skipped", flush=True)
                continue
            count = process_file(os.path.join(dir_path, f), f, source_type)
            print(f"[{n}/{len(files)}] {f}: {count} chunks", flush=True)
            total += count
        return total


def ensure_collection(wipe: bool = False) -> None:
    client = get_client()
    name = settings.QDRANT_COLLECTION

    if wipe and client.collection_exists(name):
        client.delete_collection(name)
        logfire.info(f"Collection '{name}' deleted.")

    dim = get_embedding_dim()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        logfire.info(f"Created collection '{name}' ({dim}-dim, cosine).")
    elif get_collection_dim(client) != dim:
        raise SystemExit(
            f"Collection '{name}' has {get_collection_dim(client)}-dim vectors but the active "
            f"embedding model is {dim}-dim. Re-run with --wipe."
        )

    # Indexes for the per-file delete, source-filtered search, and read_document range scans.
    for field, schema in (
        ("source", models.PayloadSchemaType.KEYWORD),
        ("source_type", models.PayloadSchemaType.KEYWORD),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)


def _source_type_for(name: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    lowered = name.lower()
    if "true" in lowered:
        return "true"
    if "noisy" in lowered:
        return "noisy"
    return lowered or "general"


def run_universal_ingestion(base_dir: str, explicit_source_type: str | None = None, wipe: bool = False) -> int:
    """Ingest base_dir. Each sub-folder becomes a source_type; otherwise the folder itself does."""
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        ensure_collection(wipe=wipe)

        subdirs = sorted(
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        )
        if not subdirs:
            source_type = _source_type_for(
                os.path.basename(os.path.normpath(base_dir)), explicit_source_type
            )
            return process_directory(base_dir, source_type)

        return sum(
            process_directory(os.path.join(base_dir, d), _source_type_for(d))
            for d in subdirs
        )


if __name__ == "__main__":
    logfire.configure(
        service_name="enterprise-ingestion-service",
        token=settings.LOGFIRE_TOKEN,
        send_to_logfire="if-token-present",
    )

    wipe_requested = "--wipe" in sys.argv
    if "--resume" in sys.argv and not wipe_requested:
        # Skip files that already have chunks (e.g. after an interrupted run) to save embedding quota.
        from app.services.retrieval.qdrant_service import list_sources

        ensure_collection()
        already_indexed.update(s["source"] for s in list_sources())
        print(f"Resuming: {len(already_indexed)} file(s) already indexed will be skipped.", flush=True)
    args = [a for a in sys.argv[1:] if a not in ("--wipe", "--resume")]
    target_dir = args[0] if args else "DATA"
    explicit_type = args[1] if len(args) > 1 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    total = run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    print(f"Ingestion complete: {total} chunks indexed.")
    if failed_files:
        print(f"{len(failed_files)} file(s) failed — re-run to retry them (ingestion is idempotent):")
        for name, err in failed_files:
            print(f"  - {name}: {err}")
        sys.exit(1)
