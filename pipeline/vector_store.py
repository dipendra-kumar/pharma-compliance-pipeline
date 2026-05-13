"""
Vector Store — indexes document chunks into ChromaDB so
Agent 2 can re-validate extracted values against the raw document
using semantic search (RAG).
"""

import chromadb
from chromadb.utils import embedding_functions
from pipeline.document_processor import DocumentChunk


class VectorStore:
    """
    Wraps ChromaDB with sentence-transformer embeddings.
    Provides simple index() and query() methods.
    """

    COLLECTION_NAME = "pharma_compliance_docs"
    EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, good enough for retrieval

    def __init__(self, persist_dir: str = "./output/chroma_db"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBED_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def index(self, chunks: list[DocumentChunk]):
        """
        Adds all document chunks to the vector store.
        Skips chunks already indexed (idempotent).
        """
        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            if not chunk.content.strip():
                continue
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "page_number": chunk.page_number,
                "section_heading": chunk.section_heading,
                "chunk_type": chunk.chunk_type,
            })

        if ids:
            # upsert avoids duplicate errors on re-runs
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            print(f"[VectorStore] Indexed {len(ids)} chunks.")

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Retrieves the most relevant chunks for a given query.

        Returns:
            List of dicts with keys: chunk_id, content, metadata, distance
        """
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self):
        """Clears the collection — useful for re-indexing a new document."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )
        print("[VectorStore] Collection reset.")
