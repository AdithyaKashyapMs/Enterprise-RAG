from typing import List

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks no larger than `chunk_size` characters.

    Tries paragraph breaks first, then lines, then sentences, then words — so
    structure is preserved where possible but no chunk can exceed the limit
    (unlike a paragraph-only splitter, which emits whole documents as one chunk
    when the loader doesn't produce blank lines). Overlap keeps facts that
    straddle a boundary retrievable from either side.
    """
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text or not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]
        logfire.info(f"Generated {len(chunks)} chunks")
        return chunks
