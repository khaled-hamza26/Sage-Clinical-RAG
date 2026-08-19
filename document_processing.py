"""File-processing module: turn local PDFs and text files into source pages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from models import SourcePage


def supported_source_files(source_directory: Path, allowed_extensions: frozenset[str]) -> list[Path]:
    """Return source files in a stable order and ignore hidden/runtime files."""
    return sorted(
        (
            path
            for path in source_directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ),
        key=lambda path: path.name.lower(),
    )


def process_file(path: Path) -> list[SourcePage]:
    """Extract readable text and source metadata from one supported file."""
    suffix = path.suffix.lower()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    common_metadata = {
        "document": path.stem,
        "source_file": path.name,
        "source_hash": source_hash,
        "section": "Not specified",
    }

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(SourcePage(text=text, metadata={**common_metadata, "page": page_number}))
        if not pages:
            raise ValueError(f"No extractable text was found in {path.name}.")
        return pages

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise ValueError(f"No text was found in {path.name}.")
        return [SourcePage(text=text, metadata={**common_metadata, "page": 1})]

    raise ValueError(f"Unsupported source type: {path.suffix}")


def process_source_directory(source_directory: Path, allowed_extensions: frozenset[str]) -> list[SourcePage]:
    """Process every source file, failing clearly instead of silently skipping one."""
    paths = supported_source_files(source_directory, allowed_extensions)
    if not paths:
        raise ValueError("No source document is available. Upload a PDF, TXT, or Markdown file first.")

    pages: list[SourcePage] = []
    errors: list[str] = []
    for path in paths:
        try:
            pages.extend(process_file(path))
        except Exception as error:
            errors.append(f"{path.name}: {error}")

    if errors:
        raise ValueError("Could not process source document(s): " + "; ".join(errors))
    return pages


def normalise_whitespace(text: str) -> str:
    """Make extraction output predictable without changing the words used as evidence."""
    return re.sub(r"\s+", " ", text).strip()
