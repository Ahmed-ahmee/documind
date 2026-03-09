import math
import re
from collections import Counter
from typing import Dict, Iterable, List

from langchain_core.documents import Document


class VectorStore:
    """Simple in-memory vector store for local testing."""

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

    def __init__(self) -> None:
        self._documents: List[Document] = []
        self._vectors: List[Dict[str, float]] = []
        self._norms: List[float] = []

    @classmethod
    def _embed_text(cls, text: str) -> Dict[str, float]:
        tokens = [token.lower() for token in cls._TOKEN_PATTERN.findall(text or "")]
        if not tokens:
            return {}

        counts = Counter(tokens)
        total = float(sum(counts.values()))
        return {term: count / total for term, count in counts.items()}

    @staticmethod
    def _vector_norm(vector: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine_similarity(
        query_vector: Dict[str, float],
        document_vector: Dict[str, float],
        document_norm: float,
    ) -> float:
        if not query_vector or not document_vector or document_norm == 0:
            return 0.0

        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0:
            return 0.0

        dot_product = 0.0
        for term, query_weight in query_vector.items():
            dot_product += query_weight * document_vector.get(term, 0.0)

        return dot_product / (query_norm * document_norm)

    def add_documents(self, documents: Iterable[Document]) -> None:
        for document in documents:
            vector = self._embed_text(document.page_content)
            self._documents.append(document)
            self._vectors.append(vector)
            self._norms.append(self._vector_norm(vector))

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if k <= 0 or not self._documents:
            return []

        query_vector = self._embed_text(query)
        scored = [
            (
                self._cosine_similarity(query_vector, document_vector, document_norm),
                index,
            )
            for index, (document_vector, document_norm) in enumerate(
                zip(self._vectors, self._norms)
            )
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        top_k = min(k, len(self._documents))
        return [self._documents[index] for _, index in scored[:top_k]]

    def get_count(self) -> int:
        return len(self._documents)

