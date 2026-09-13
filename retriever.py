'''Local MiniLM embeddings and persistent, session-scoped Chroma retrieval.'''
from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

@dataclass(frozen=True)
class SearchHit:
    document: Document
    similarity: float
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    '''Download once, then reuse the CPU model; no documents go to Hugging Face.'''
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu', 'trust_remote_code': False},
        encode_kwargs={'normalize_embeddings': True},
    )


def initialize_vector_store(
    collection_name: str, persist_directory: str | Path = DEFAULT_DB_PATH,
) -> Chroma:
    '''Open or create a persistent cosine-distance collection.

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
        collection_configuration={'hnsw': {'space': 'cosine'}},
    )


def index_chunks(store: Chroma, chunks: list[Document]) -> int:
    '''Upsert deterministic IDs in bounded batches; retries do not duplicate data.

    Returns the number submitted, not the number newly inserted. If a later batch
    fails, earlier batches remain; the UI blocks questions until a retry succeeds
    or the session's collection is cleared.
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
