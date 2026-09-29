'''Setup check: real embeddings + ChromaDB on the sample HR documents. No API key needed.

Run from the project folder:   python tests/verify_setup.py
The first run downloads the small embedding model (about 90 MB), so it needs internet once.
It uses a throw-away temporary database, so chroma_db/ stays empty.
'''
import re
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import FALLBACK_RESPONSE, DocumentAgent
from ingestion import chunk_documents, load_document
from llm import Provider
from retriever import index_chunks, initialize_vector_store, semantic_search

SAMPLES = ROOT / 'sample_documents'


def main() -> None:
    print('1/4  Reading the sample documents...')
    chunks = []
    for name in ('IT_Security_Memo_Sample.txt', 'Employee_Directory.csv'):
        chunks += chunk_documents(load_document(name, (SAMPLES / name).read_bytes()))
    assert len(chunks) > 1000, len(chunks)
    print(f'     OK - {len(chunks)} passages ready')

    print('2/4  Loading the embedding model and storing passages (first run downloads it)...')
    # ignore_cleanup_errors: on Windows ChromaDB may still hold its file for a moment
    with tempfile.TemporaryDirectory(prefix='verify_chroma_', ignore_cleanup_errors=True) as folder:
        store = initialize_vector_store('docs_' + uuid.uuid4().hex, folder)
        for start in range(0, len(chunks), 128):
            index_chunks(store, chunks[start:start + 128])
        print('     OK - stored in a temporary ChromaDB')

        print('3/4  Searching by meaning...')
        # Regression check: 6 policy paragraphs hide among 1000 similar spreadsheet rows. With the
        # default ChromaDB index settings these were sometimes not found at all.
        needles = {'Who should I report suspicious emails to?': 'security@acmecorp-example.com',
                   'Can I use USB storage devices?': 'USB storage'}
        for question, needle in needles.items():
            found = semantic_search(store, question, k=5)
            assert any(needle in hit.document.page_content for hit in found), f'not found: {question}'
        hits = semantic_search(store, 'How often must passwords be changed?', k=3)
        assert hits, 'expected at least one relevant passage'
        assert 'password' in hits[0].document.page_content.lower(), hits[0].document.page_content
        snippet = ' '.join(re.sub(r'^[-=\s]+', '', hits[0].document.page_content).split())[:70]
        print(f'     OK - best match "{snippet}..." (similarity {hits[0].similarity:.2f})')

        print('4/4  Checking the no-evidence guardrail (the LLM must not be called)...')

        def must_not_be_called(provider):
            raise AssertionError('the LLM was called even though nothing relevant exists')

        agent = DocumentAgent([Provider('gemini', 'Gemini', 'unused', 'unused-key-value')],
                              model_factory=must_not_be_called)
        refusal = agent.answer('Who won the 1998 football world cup?', store)
        assert refusal.text == FALLBACK_RESPONSE, refusal.text
        print('     OK - unrelated question refused without calling any LLM')
        store.delete_collection()
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
