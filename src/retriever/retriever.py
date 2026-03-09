from typing import List, Optional

from langchain_core.documents import Document

from src.vectorstore.store import VectorStore


class Retriever:
    """Thin retrieval layer over the local VectorStore."""

    def __init__(self, vector_store: VectorStore, default_k: int = 3) -> None:
        if default_k <= 0:
            raise ValueError("default_k must be greater than 0")
        self.vector_store = vector_store
        self.default_k = default_k

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        top_k = k if k is not None else self.default_k
        if top_k <= 0:
            return []
        return self.vector_store.similarity_search(query, k=top_k)
