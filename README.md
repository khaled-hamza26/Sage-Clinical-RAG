# Sage Clinical RAG

Sage is a one-command local web application that answers questions only from documents you upload. The browser UI and FastAPI backend are served together on one local address.

## Quick start

1. Install Python 3.11 or newer.
2. In this folder, create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the application packages and configure Groq:

   ```powershell
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

   Edit `.env` and replace `replace_with_your_key` with your `GROQ_API_KEY`.

4. Start the complete program from its one entry point:

   ```powershell
   python app.py
   ```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a PDF/TXT/Markdown source, then ask questions in the UI.

The first document upload downloads and loads the embedding model; this can take longer than later uploads. Uploaded documents are kept in `data/sources/` and the local Chroma vector index in `data/chroma/`, both excluded from version control.

## Project layout

| File | Responsibility |
| --- | --- |
| `document_processing.py` | Extracts text and traceable metadata from PDF, TXT, and Markdown sources. |
| `chunking.py` | Produces overlapping, source/page-aware chunks. |
| `vector_store.py` | Creates BGE embeddings, stores them persistently in Chroma, and retrieves similar chunks. |
| `rag.py` | Enforces retrieval thresholds, calls Groq, validates evidence/citations, and formats browser sources. |
| `app.py` | Single FastAPI entry point, API routes, upload handling, and static UI hosting. |
| `static/index.html` | Browser UI; it uses relative API URLs, so it shares the same port as the backend. |

## Important clinical safeguards

- The model is instructed to answer only from retrieved passages.
- Citations are checked against the metadata actually retrieved.
- If retrieval is weak, the model output is invalid, or evidence is missing, Sage refuses rather than guessing.
- Sage is clinical decision support only. Validate responses against the original sources and current guidance.
