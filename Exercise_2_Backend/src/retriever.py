"""Embedding-based retrieval over the persistent FAISS index."""

from __future__ import annotations

from .embeddings import EmbeddingService
from .preprocessing import preprocess_for_retrieval
from .vector_store import FaissVectorStore


class Retriever:
    def __init__(self, embeddings: EmbeddingService, vector_store: FaissVectorStore) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, object]]:
        cleaned = preprocess_for_retrieval(query)
        if not cleaned:
            return []
        vector = self.embeddings.encode([cleaned])[0]
        return self.vector_store.search(vector, top_k=top_k)

