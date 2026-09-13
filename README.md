# 📄 Enterprise Document Assistant

A production-oriented, Retrieval-Augmented Generation (RAG) application for asking
grounded questions over internal PDF, TXT, CSV, and Excel files. Embeddings run
locally with `all-MiniLM-L6-v2`; only retrieved excerpts are sent to Google Gemini.

> **Answers come only from your documents. If the evidence is insufficient, the
> assistant refuses instead of guessing.**

---

## Architecture

```text
 Streamlit UI (app.py)
   │  upload bytes ─ session-scoped knowledge base
   ▼
 ingestion.py ──▶ load_document()  PyPDF / Text / CSV / Excel → Documents
                   chunk_documents() RecursiveCharacterTextSplitter (700/100)
   ▼
 retriever.py ──▶ initialize_vector_store()  persistent ChromaDB (cosine)
                   get_embeddings()           HuggingFace MiniLM (local, CPU)
                   semantic_search()          distance → similarity + threshold
   ▼
 agent.py ─────▶ DocumentAgent
                   ├─ retrieve → assess → refine (≤ 2 rounds) → validate → answer
                   └─ guardrails: citation/quote verification, fallback, injection defense
   ▼
 Gemini (langchain-google-genai, structured output) → cited answer + evidence cards
```

### Retrieval and reasoning flow
1. **Ingest** — Bytes are written to a temporary file (never the request filename),
   parsed, and given safe provenance metadata (source, page/row/sheet, file hash).
   Empty files and whitespace-only files are skipped gracefully.
2. **Chunk** — Text is split into ~700-character chunks with 100-character overlap.
   Excel becomes one document per non-empty row; CSV one document per record.
3. **Embed & store** — Chunks are upserted with deterministic IDs into a persistent
   local ChromaDB collection using cosine space.
4. **Retrieve** — The question is embedded and the top-k chunks are returned.
   Chroma returns a cosine **distance**, converted explicitly to `1 - distance`.
5. **Reason** — Gemini receives the question plus numbered sources and returns a
   structured `GroundedDraft` (sufficiency flag, claims, refined search query).
   If evidence is insufficient, it may request **one** refined retrieval.
6. **Validate** — Every claim must carry a verbatim quote that truly exists in the
   cited source. Invented citations or fabricated quotes trigger the fallback.
   Only then is a cited answer with expandable evidence cards rendered.

### Guardrails against hallucination
- Refusal fallback when no chunk passes the similarity threshold or the model
  reports insufficient evidence.
- Every answer claim is bound to verbatim, programmatically verified quotes.
- Prompt treats the question and documents as untrusted data (prompt-injection
  defense) and states there is no web/shell access.
- Excel row limits and explicit instructions prevent aggregate spreadsheet claims.
- Raw upstream errors are masked; no secrets are echoed to the UI.

---

## Project layout

| File | Purpose |
|---|---|
| `ingestion.py` | Multi-format loaders, limits, provenance metadata, chunking |
| `retriever.py` | MiniLM embeddings, ChromaDB persistence, semantic search |
| `agent.py` | RAG pipeline, bounded agent loop, structured validation, fallback |
| `app.py` | Streamlit sidebar uploads + chat interface with evidence panels |
| `requirements.txt` | Pinned major-version dependencies |
| `.env.example` | Secret template (never commit a real key) |
| `.gitignore` | Python, virtualenv, `.env`, `chroma_db/`, caches |

---

## Local setup

```powershell
# 1. Clone and enter the project
git clone <your-repo-url>
cd capstone_project

# 2. Create and activate a virtual environment (Python 3.11/3.12 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux

# 3. Install dependencies (PyTorch download is large on first install)
python -m pip install -r requirements.txt

# 4. Configure secrets
Copy-Item .env.example .env          # then edit .env
# GOOGLE_API_KEY=your-real-key      (https://aistudio.google.com/app/apikey)
# GEMINI_MODEL=gemini-3.6-flash     (optional; this is the fallback default)

# 5. Run
streamlit run app.py --server.fileWatcherType none
```

### Model configuration
| Setting | Value |
|---|---|
| Default model | `gemini-3.6-flash` |
| Override | set `GEMINI_MODEL` in `.env` or Streamlit secrets |
| Retired (auto-replaced by the default) | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, `gemini-1.5-flash-001`, `gemini-1.5-pro`, `gemini-pro` |

Model resolution is centralized in `agent.resolve_model_name()` and follows
`explicit argument → GEMINI_MODEL env → gemini-3.6-flash`. Blank values and retired
model names are silently swapped for the default, so a stale `.env` entry cannot
produce a `404 NOT_FOUND`. To use a model you have access to, set it explicitly:

```dotenv
GEMINI_MODEL=gemini-3.6-flash
```

Open `http://localhost:8501`, upload documents, click **Index documents**, then ask
questions. The first question downloads the MiniLM model (~90 MB) once; afterwards
it runs offline.

### Deployment (Streamlit Community Cloud / containers)
1. Push the repository (`.env`, `chroma_db/`, and `.venv/` stay ignored).
2. On Streamlit Cloud, add `GOOGLE_API_KEY` (and optionally `GEMINI_MODEL`) under
   **Settings → Secrets**; the app reads secrets first, then `.env`.
3. For containers, install requirements at build time to bake in the embedding
   model, mount a writable volume if you want `chroma_db/` to persist, and expose
   port `8501`: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`.

---

## Limitations

- **Not a source of truth.** It summarizes retrieved excerpts; verify before
  acting on answers.
- **English-centric embeddings.** MiniLM's context window is ~256 word-pieces, so
  long chunks may be partly truncated and non-English content is weaker.
- **No aggregate analytics.** CSV/Excel rows are retrieved individually; total,
  average, and cross-row reasoning over large spreadsheets is unreliable.
- **Scanned PDFs are not OCR'd.** Image-only PDFs yield no text.
- **Retrieval quality depends on chunking and thresholds** (`chunk_size`,
  `chunk_overlap`, `min_similarity`, `k`); tune per corpus.
- **Single-process vector store.** ChromaDB here targets local/single-instance use;
  multi-user scale needs a hosted vector database and authentication.
- **Session-scoped knowledge base.** Each browser session gets its own collection;
  `chroma_db/` persists on disk but is not shared across users.
- **Quota, cost, and privacy.** Retrieved excerpts leave your machine for Gemini;
  do not upload regulated data without an approved Google Cloud/Vertex agreement.
- Verbatim-quote validation reduces fabrication but does not guarantee that a
  correctly quoted sentence truly answers the question.

---

## Security notes

- Never commit `.env`; rotate any key that was exposed.
- Uploaded filenames are sanitized, uploads are size/row/character limited, and
  temporary files are deleted immediately after parsing.
- Upstream exceptions are masked so API details are not rendered.

## License

Released under the MIT License. Add a `LICENSE` file before publishing publicly.
