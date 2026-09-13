'''Local smoke test: ingestion, chunking, Chroma retrieval, and the no-evidence guardrail.

Run with:  python verify_setup.py  (no API key required)
'''
import uuid
from agent import DocumentAgent
from ingestion import IngestionError, chunk_documents, load_document
from retriever import index_chunks, initialize_vector_store, semantic_search

def main() -> None:
    # 1. Empty and whitespace-only uploads must be skipped, not crash.
    assert load_document('empty.txt', b'') == []
    assert load_document('blank.txt', b'   \n\t') == []
    try:
        load_document('notes.md', b'# unsupported')
        raise AssertionError('unsupported extension should raise IngestionError')
    except IngestionError:
        pass

    # 2. TXT ingestion preserves safe provenance metadata.
    policy = (b'Annual leave is 20 days per year. Managers approve requests. '
              b'Remote work is capped at three days per week.')
    documents = load_document('policy.txt', b'') or load_document('policy.txt', policy)
    assert len(documents) == 1 and documents[0].metadata['source'] == 'policy.txt'
    assert documents[0].metadata['format'] == 'txt'

    # 3. Chunking adds deterministic, content-addressed chunk IDs.
    chunks = chunk_documents(documents)
    assert chunks and all('chunk_id' in chunk.metadata for chunk in chunks)
    assert chunks == chunk_documents(documents)

    # 4. CSV and Excel loaders produce one document per record/row.
    csv_docs = load_document('staff.csv', b'name,team\nAva,Legal\nNoah,Finance\n')
    assert len(csv_docs) == 2, csv_docs
    print('ingestion checks passed:', len(documents), 'text doc,', len(csv_docs), 'csv rows')

    # 5. Real MiniLM embeddings + ChromaDB round trip.
    store = initialize_vector_store('docs_' + uuid.uuid4().hex)
    index_chunks(store, chunks)
    hits = semantic_search(store, 'How many annual leave days?', k=3)
    assert hits, 'expected at least one relevant chunk'
    top = round(hits[0].similarity, 3)

    # 6. Guardrail: an unrelated question finds no evidence and never calls the LLM.
    agent = DocumentAgent.__new__(DocumentAgent)   # bypass API client construction
    agent.generator = None
    refusal = agent.answer('Who won the 1998 football world cup?', store).text
    assert 'could not find enough supporting evidence' in refusal
    store.delete_collection()
    print('retrieval checks passed: top similarity', top, '| refusal guardrail ok')
    print('ALL CHECKS PASSED')

if __name__ == '__main__':
    main()
