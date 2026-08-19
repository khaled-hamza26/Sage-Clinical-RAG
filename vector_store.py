"""Embedding and vector-store module backed by a persistent local Chroma collection."""

from __future__ import annotations

from threading import RLock

import chromadb
from sentence_transformers import SentenceTransformer

from config import Settings
from models import RetrievedChunk, TextChunk


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.vector_directory))
        self._model: SentenceTransformer | None = None
        self._collection = None
        self._lock = RLock()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        with self._lock:
            return self.collection.count()

    def document_count(self) -> int:
        with self._lock:
            if not self.collection.count():
                return 0
            metadata = self.collection.get(include=["metadatas"])["metadatas"]
            return len({item["source_hash"] for item in metadata if item and "source_hash" in item})

    def rebuild(self, chunks: list[TextChunk]) -> None:
        """Replace the generated vector index from the current source chunks."""
        if not chunks:
            raise ValueError("Cannot create a vector store without chunks.")

        with self._lock:
            try:
                self._client.delete_collection(self.settings.collection_name)
            except ValueError:
                pass  # First index build: the collection does not exist yet.

            self._collection = self._client.create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            batch_size = 64
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                embeddings = self.model.encode(
                    [chunk.text for chunk in batch],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self._collection.add(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings.tolist(),
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                )

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        with self._lock:
            count = self.collection.count()
            if not count:
                return []
            embedding = self.model.encode(query, normalize_embeddings=True)
            result = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )

        return [
            RetrievedChunk(
                chunk_id=result["ids"][0][index],
                text=result["documents"][0][index],
                metadata=result["metadatas"][0][index],
                distance=float(result["distances"][0][index]),
            )
            for index in range(len(result["ids"][0]))
        ]
