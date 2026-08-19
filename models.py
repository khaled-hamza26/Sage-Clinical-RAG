"""API schemas and small domain objects shared by the RAG modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=30)


class Source(BaseModel):
    title: str
    snippet: str
    meta: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)


class IndexStatus(BaseModel):
    ready: bool
    document_count: int
    chunk_count: int
    generator_configured: bool


class IndexResponse(BaseModel):
    message: str
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class SourcePage:
    """Extracted text for one source page (or one plain-text document)."""

    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float
