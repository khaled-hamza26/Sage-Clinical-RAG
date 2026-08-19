"""Central configuration for the local RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIRECTORY = Path(__file__).resolve().parent
load_dotenv(BASE_DIRECTORY / ".env")


@dataclass(frozen=True)
class Settings:
    base_directory: Path = BASE_DIRECTORY
    source_directory: Path = BASE_DIRECTORY / "data" / "sources"
    vector_directory: Path = BASE_DIRECTORY / "data" / "chroma"
    static_directory: Path = BASE_DIRECTORY / "static"
    collection_name: str = os.getenv("CHROMA_COLLECTION", "sage_clinical_guidelines")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "850"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    max_retrieval_distance: float = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "0.60"))
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "40"))
    allowed_extensions: frozenset[str] = frozenset({".pdf", ".txt", ".md"})
    cors_origins: list[str] = field(
        default_factory=lambda: [
            value.strip()
            for value in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
            if value.strip()
        ]
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        self.source_directory.mkdir(parents=True, exist_ok=True)
        self.vector_directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
