"""Single entry point for the Sage Clinical RAG web application.

Start the complete application with: ``python app.py``
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import Settings, get_settings
from models import ChatRequest, ChatResponse, IndexResponse, IndexStatus
from rag import IndexNotReadyError, RAGService, RAGServiceUnavailableError

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _safe_filename(filename: str) -> str:
    """Return a harmless filename while keeping its extension."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "uploaded_document.pdf"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        settings.ensure_directories()
        application.state.rag = RAGService(settings)
        logger.info("Sage API is ready. Add a source document from the web UI to begin.")
        yield

    application = FastAPI(
        title="Sage Clinical RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # The UI is served by this same app. This is retained for a separately hosted
    # development UI, and should be narrowed before any public deployment.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def rag_service() -> RAGService:
        return application.state.rag

    @application.get("/api/status", response_model=IndexStatus)
    def get_status() -> IndexStatus:
        return rag_service().status()

    @application.post("/api/documents", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
    async def upload_document(file: UploadFile = File(...)) -> IndexResponse:
        filename = _safe_filename(file.filename or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_extensions:
            allowed = ", ".join(sorted(settings.allowed_extensions))
            raise HTTPException(status_code=415, detail=f"Supported files: {allowed}.")

        contents = await file.read(settings.max_upload_bytes + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(contents) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"The file is larger than {settings.max_upload_mb} MB.",
            )

        destination = settings.source_directory / filename
        previous_contents = destination.read_bytes() if destination.exists() else None
        # Replacing a file with the same name is intentional: the subsequent
        # rebuild guarantees that its old chunks are not left in the index.
        destination.write_bytes(contents)

        try:
            document_count, chunk_count = rag_service().rebuild_index()
        except ValueError as error:
            # Do not discard a previously indexed source if a replacement file is
            # unreadable. The earlier file and its vector index remain usable.
            if previous_contents is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(previous_contents)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:  # pragma: no cover - depends on optional model runtime
            if previous_contents is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(previous_contents)
            logger.exception("Could not index %s", filename)
            raise HTTPException(status_code=500, detail="The document could not be indexed.") from error

        return IndexResponse(
            message=f"Indexed {filename}.",
            document_count=document_count,
            chunk_count=chunk_count,
        )

    @application.post("/api/reindex", response_model=IndexResponse)
    def reindex_documents() -> IndexResponse:
        try:
            document_count, chunk_count = rag_service().rebuild_index()
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:  # pragma: no cover - depends on optional model runtime
            logger.exception("Could not rebuild the vector store")
            raise HTTPException(status_code=500, detail="The knowledge base could not be rebuilt.") from error

        return IndexResponse(
            message="Knowledge base rebuilt.",
            document_count=document_count,
            chunk_count=chunk_count,
        )

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            return rag_service().answer(request.message, request.history)
        except IndexNotReadyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RAGServiceUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:  # pragma: no cover - protects the browser from internal errors
            logger.exception("RAG request failed")
            raise HTTPException(status_code=500, detail="The answer service encountered an unexpected error.") from error

    # This must remain after all /api routes because it is a catch-all mount.
    application.mount("/", StaticFiles(directory=settings.static_directory, html=True), name="static")
    return application


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
