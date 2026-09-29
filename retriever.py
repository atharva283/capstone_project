'''Local MiniLM embeddings and persistent, session-scoped Chroma retrieval.'''
from __future__ import annotations
import os
# Must be set before chromadb / huggingface are imported: switch off analytics and noisy warnings.
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')  # hides the harmless 'unauthenticated requests' notice
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import logging
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# The "unauthenticated requests to the HF Hub" warning is harmless: the model is public.
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
try:
    from huggingface_hub.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:  # noqa: BLE001 - purely cosmetic
    pass

DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Chroma's default HNSW index (ef_search=100) missed the truly best passage in about 1 of 8 test
# queries: a handful of policy paragraphs hide among thousands of near-identical spreadsheet rows.
# Collections here are small, so a thorough (near-exhaustive) search costs milliseconds; it lowered
# the miss rate on the sample data from 10/80 to 0/128. See DOCUMENTATION.md, challenge 14.
HNSW_SETTINGS = {'space': 'cosine', 'max_neighbors': 64, 'ef_construction': 800, 'ef_search': 2000}


class EmbeddingModelError(RuntimeError):
    '''The local embedding model could not be loaded (usually: no internet on the first run).'''


@dataclass(frozen=True)
class SearchHit:
    document: Document
    similarity: float


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    '''Download once (about 90 MB), then reuse the CPU model; no documents go to Hugging Face.'''
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu', 'trust_remote_code': False},
            encode_kwargs={'normalize_embeddings': True},
        )
    except Exception as exc:
        raise EmbeddingModelError(
            'The local embedding model could not be loaded. The first start downloads it '
            '(about 90 MB), so it needs an internet connection once. Check your connection '
            'or firewall and start the app again.'
        ) from exc


def initialize_vector_store(
    collection_name: str, persist_directory: str | Path = DEFAULT_DB_PATH,
) -> Chroma:
    '''Open or create a persistent cosine-distance collection with a thorough HNSW search.

    Callers MUST supply a trusted collection identifier, not a user-entered name.
    Streamlit generates a fresh random identifier for each browser session.
    '''
    if not re.fullmatch(r'docs_[a-f0-9]{32}', collection_name):
        raise ValueError('Collection name must be docs_ followed by a UUID hex value.')
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(path),
        embedding_function=get_embeddings(),
        collection_configuration={'hnsw': dict(HNSW_SETTINGS)},
    )


def purge_stale_vector_data(persist_directory: str | Path = DEFAULT_DB_PATH) -> None:
    '''Empty chroma_db/ (keeping .gitkeep). Collection ids are random per browser session,
    so data from an earlier run can never be reached again; call this before any store opens.'''
    path = Path(persist_directory)
    if not path.is_dir():
        return
    for item in path.iterdir():
        if item.name == '.gitkeep':
            continue
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except OSError:
            pass  # another running copy of the app may hold the files; leave them alone


def index_chunks(store: Chroma, chunks: list[Document]) -> int:
    '''Upsert deterministic IDs in bounded batches; retries do not duplicate data.

    Returns the number submitted, not the number newly inserted. If a later batch
    fails, earlier batches remain; the caller reports the failure and the user can retry.
    '''
    if not chunks:
        return 0
    ids = [str(chunk.metadata['chunk_id']) for chunk in chunks]
    for start in range(0, len(chunks), 128):
        store.add_documents(documents=chunks[start:start + 128], ids=ids[start:start + 128])
    return len(chunks)


def semantic_search(
    store: Chroma, question: str, k: int = 5, min_similarity: float = 0.30,
) -> list[SearchHit]:
    '''Return relevant chunks using cosine similarity (not a confidence score).'''
    if not question.strip():
        return []
    if not 1 <= k <= 20 or not -1 <= min_similarity <= 1:
        raise ValueError('Require 1 <= k <= 20 and -1 <= min_similarity <= 1.')
    # Chroma returns cosine DISTANCE: smaller is better. Convert explicitly rather
    # than relying on an integration's default relevance-score normalization.
    matches = store.similarity_search_with_score(question, k=k)
    return [
        SearchHit(document=document, similarity=max(-1.0, min(1.0, 1.0 - distance)))
        for document, distance in matches if 1.0 - distance >= min_similarity
    ]
