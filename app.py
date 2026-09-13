'''Streamlit interface for private, session-scoped enterprise document Q&A.'''
from __future__ import annotations
import os
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from agent import AgentError, DocumentAgent, resolve_model_name
from ingestion import IngestionError, chunk_documents, load_document
from retriever import DEFAULT_DB_PATH, index_chunks, initialize_vector_store, semantic_search
load_dotenv()
# Optional hardening: keep uploaded documents and traces out of third-party run logs.
os.environ.setdefault('LANGCHAIN_TRACING_V2', 'false')
os.environ.setdefault('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', '50')
st.set_page_config(page_title='Enterprise Document Assistant', page_icon='📄', layout='wide')
def _secrets() -> tuple[str, str]:
    '''Load configuration strictly from environment variables (.env / system env).'''
    key = str(os.environ.get('GOOGLE_API_KEY', '') or '')
    try:
        key = key or str(st.secrets.get('GOOGLE_API_KEY', ''))
    except Exception:
        pass
    # resolve_model_name guarantees a current, non-retired default (gemini-3.6-flash).
    return key.strip(), resolve_model_name()

def _session() -> None:
    defaults = {
        'messages': [], 'indexed': set(),
        'rag_failed': False, 'total_chunks': 0,
        'collection': f"docs_{uuid.uuid4().hex}",
    }
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)

def _plain_label(text: str) -> str:
    '''Escape HTML then allow only [n] citation markers as sanitized spans.'''
    escaped = re.sub(r'[^0-9A-Za-z]+', '-', text).strip('-')[:60] or 'document'
    return escaped

def _render_sources(answer, label: str) -> None:
    if not answer.sources:
        return
    with st.expander(f'Evidence used for: {_plain_label(label)}'):
        for source in answer.sources:
            meta = source['metadata']
            location = ''
            if 'page' in meta:
                location = f"page {meta['page']}"
            elif 'row' in meta:
                location = f"row {meta['row']}"
            sheet = f" sheet '{meta['sheet']}'" if 'sheet' in meta else ''
            st.markdown(f"**[{source['id']}]** `{_plain_label(str(meta.get('source', 'document')))}`{sheet} {location}")
            for quote in source['quotes']:
                st.caption(f'“{quote}”')
            with st.popover('View excerpt'):
                st.code(source['excerpt'], language=None)

def main() -> None:
    _session()
    st.title('📄 Enterprise Document Assistant')
    st.caption('Grounded question answering over files you ingest. Local embeddings; only retrieved excerpts are sent to Gemini.')
    with st.sidebar:
        st.header('1. Configuration')
        key, resolved_model = _secrets()
        if not key or key.startswith('your_'):
            st.error('Missing `GOOGLE_API_KEY`. Please set it in your `.env` file.')
        else:
            st.success(f'Gemini Connected (`{resolved_model}`)')
        st.divider()
        st.header('2. Upload documents')
        uploads = st.file_uploader(
            'PDF, UTF-8 TXT/CSV, or XLSX (max 20 MiB each)',
            type=['pdf', 'txt', 'csv', 'xlsx'], accept_multiple_files=True,
        )
        index_clicked = st.button('Index documents', type='primary', use_container_width=True)
        clear_clicked = st.button('Clear knowledge base', use_container_width=True)
        st.caption(f'Session scope: {len(st.session_state.indexed)} file(s), {st.session_state.total_chunks} chunk(s).')
    store = initialize_vector_store(st.session_state.collection, DEFAULT_DB_PATH)
    if clear_clicked:
        store.delete_collection()
        st.session_state.indexed, st.session_state.total_chunks = set(), 0
        st.session_state.rag_failed = False
        st.session_state.collection = f'docs_{uuid.uuid4().hex}'
        st.success('Knowledge base cleared for this session.')
        st.rerun()
    if index_clicked:
        if not uploads:
            st.warning('Choose at least one document before indexing.')
        else:
            progress = st.sidebar.progress(0.0, text='Reading documents...')
            all_chunks: list = []
            skipped: list[str] = []
            failures: list[str] = []
            pending: list[tuple[str, str]] = []
            for position, upload in enumerate(uploads, start=1):
                label = Path(upload.name).name
                digest = f"indexed_{upload.size}_{label}"
                progress.progress(position / len(uploads), text=f'Reading {label}...')
                if label in st.session_state.indexed:
                    skipped.append(label)
                    continue
                try:
                    chunks = chunk_documents(load_document(label, upload.getvalue()))
                    if not chunks:
                        skipped.append(f'{label} (empty or no extractable text)')
                    else:
                        all_chunks.extend(chunks)
                        pending.append((digest, label))
                except IngestionError as exc:
                    failures.append(f'{label}: {exc}')
                except Exception:
                    failures.append(f'{label}: unexpected ingestion error')
            progress.empty()
            added = 0
            if all_chunks:
                with st.spinner('Embedding locally and storing in ChromaDB...'):
                    added = index_chunks(store, all_chunks)
                    st.session_state.total_chunks += added
                    for digest, label in pending:
                        st.session_state.indexed.add(label)
                        st.session_state[digest] = True
            st.session_state.rag_failed = False
            if added:
                st.sidebar.success(f'Indexed {added} chunks from {len(pending)} file(s).')
            if skipped:
                st.sidebar.info('Skipped: ' + ', '.join(skipped))
            if failures:
                st.sidebar.error(' | '.join(failures))
            if not all_chunks and not failures:
                st.sidebar.warning('Nothing new was indexed.')
    rag_failed = st.session_state.rag_failed
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message['role'] == 'assistant' and message.get('sources'):
                _render_sources(type('Answer', (), {'sources': message['sources']})(), message.get('question', ''))
    question = st.chat_input('Ask a question about your uploaded documents')
    if question:
        st.session_state.messages.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            key, resolved_model = _secrets()
            ready = key and not key.startswith('your_') and not rag_failed
            hits = semantic_search(store, question) if ready else []
            if not ready:
                response, sources = 'Index at least one document and configure GOOGLE_API_KEY first.', []
            elif not hits:
                response, sources = 'No relevant passages were found in your indexed documents. Re-index a relevant file or rephrase the question.', []
            else:
                try:
                    with st.spinner('Retrieving evidence and reasoning...'):
                        answer = DocumentAgent(key, resolved_model).answer(question, store)
                    response, sources = answer.text, answer.sources
                except (AgentError, ValueError) as exc:
                    response, sources = str(exc), []
                except Exception:
                    response, sources = 'Unexpected error while answering. Please retry.', []
            st.markdown(response)
            _render_sources(type('Answer', (), {'sources': sources})(), question)
        st.session_state.messages.append({'role': 'assistant', 'content': response, 'sources': sources, 'question': question})

if __name__ == '__main__':
    main()
