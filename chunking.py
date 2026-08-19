"""Chunking module: create overlapping, traceable text chunks from source pages."""

from __future__ import annotations

from document_processing import normalise_whitespace
from models import SourcePage, TextChunk


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text near sentence boundaries while retaining overlap for retrieval context."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than chunk_size.")

    text = normalise_whitespace(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            # Prefer a natural break in the latter half of the candidate span.
            candidates = [text.rfind(separator, start + chunk_size // 2, hard_end) for separator in (". ", "; ", " ")]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def chunk_pages(pages: list[SourcePage], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Give every chunk a stable source/page/index identifier and carry its metadata forward."""
    chunks: list[TextChunk] = []
    for page in pages:
        source_hash = str(page.metadata["source_hash"])[:16]
        page_number = page.metadata["page"]
        for position, text in enumerate(split_text(page.text, chunk_size, chunk_overlap), start=1):
            chunk_id = f"{source_hash}-p{page_number}-c{position:03d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata={**page.metadata, "chunk_id": chunk_id},
                )
            )
    if not chunks:
        raise ValueError("The source documents did not produce any chunks.")
    return chunks
