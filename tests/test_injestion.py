"""
tmoke test — run this to verify document_processor and vector_store work.
Usage: python test_ingestion.py
"""

import sys
from pipeline.document_processor import DocumentProcessor
from pipeline.vector_store import VectorStore

PDF_PATH = "sample_extract_call.pdf"   # update path if needed

# Step 1: Process PDF
print("=" * 60)
print("STEP 1: Document Processing")
print("=" * 60)

processor = DocumentProcessor(PDF_PATH)
chunks = processor.process()

print(f"Total chunks     : {len(chunks)}")
print(f"Text chunks      : {len(processor.get_text_chunks())}")
print(f"Table chunks     : {len(processor.get_table_chunks())}")

print("\n--- First Table Chunk Preview ---")
tables = processor.get_table_chunks()
if tables:
    print(f"Section : {tables[0].section_heading}")
    print(f"Content :\n{tables[0].content[:500]}")
else:
    print("No tables found!")
    sys.exit(1)

# Step 2: Index into vector store
print("\n" + "=" * 60)
print("STEP 2: Vector Store Indexing")
print("=" * 60)

store = VectorStore()
store.index(chunks)
print(f"Total indexed: {store.count()}")

# Step 3: Test a RAG query
print("\n--- RAG Query Test ---")
results = store.query("temperature limits compliance", n_results=3)
for r in results:
    print(f"\nChunk: {r['chunk_id']} | Section: {r['metadata']['section_heading']}")
    print(f"Content: {r['content'][:200]}")

print("\n✅ Ingestion pipeline working correctly.")
