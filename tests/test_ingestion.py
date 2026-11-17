from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.processor import _source_type_for, chunk_id


def test_chunker_respects_max_size_without_blank_lines():
    # Loaders often join lines with a single "\n" — the old splitter emitted this as one chunk.
    text = "\n".join(f"Line {i}: kubectl get jobs shows completions and duration." for i in range(400))
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) > 10
    assert max(len(c) for c in chunks) <= 1000


def test_chunker_splits_single_giant_paragraph():
    text = "word " * 5000
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=100)
    assert all(len(c) <= 800 for c in chunks)


def test_chunker_overlaps_neighbours():
    text = " ".join(f"token{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    tail_of_first = chunks[0].split()[-1]
    assert tail_of_first in chunks[1]


def test_chunker_empty():
    assert chunk_text("   \n\n ") == []


def test_chunk_ids_are_deterministic_and_unique():
    assert chunk_id("a.pdf", 0) == chunk_id("a.pdf", 0)
    assert chunk_id("a.pdf", 0) != chunk_id("a.pdf", 1)
    assert chunk_id("a.pdf", 0) != chunk_id("b.pdf", 0)


def test_source_type_mapping():
    assert _source_type_for("true_data") == "true"
    assert _source_type_for("noisy_data") == "noisy"
    assert _source_type_for("whatever", explicit="custom") == "custom"
