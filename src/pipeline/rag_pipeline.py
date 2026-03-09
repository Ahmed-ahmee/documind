import sys
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.generator.generator import Generator
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader
from src.retriever.retriever import Retriever
from src.vectorstore.store import VectorStore


class RAGPipeline:
    """End-to-end local RAG pipeline used by tests and demos."""

    def __init__(
        self,
        data_path: str = "data/documents/",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ) -> None:
        self.loader = DocumentLoader(data_path)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, default_k=top_k)
        self.generator = Generator()
        self._is_ingested = False

    @property
    def store(self) -> VectorStore:
        """Backward-compatible alias used by older tests/scripts."""
        return self.vector_store

    def ingest(self, force_reindex: bool = False) -> int:
        if self._is_ingested and not force_reindex:
            print("Index already available. Skipping ingestion.")
            return self.vector_store.get_count()

        if force_reindex:
            self.vector_store = VectorStore()
            self.retriever = Retriever(self.vector_store, default_k=self.retriever.default_k)

        print("\nStarting ingestion...")
        documents = self.loader.load()
        chunks = self.chunker.chunk(documents)
        self.vector_store.add_documents(chunks)

        self._is_ingested = True
        count = self.vector_store.get_count()
        print(f"Ingestion complete. Indexed {count} chunks.")
        return count

    def query(self, question: str, k: Optional[int] = None) -> Dict[str, object]:
        if not self._is_ingested and self.vector_store.get_count() == 0:
            self.ingest()

        retrieved_documents: List[Document] = self.retriever.retrieve(question, k=k)
        response = self.generator.generate(question, retrieved_documents)
        return response

    @staticmethod
    def _safe_console(text: str) -> str:
        encoding = sys.stdout.encoding or "utf-8"
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")

    def print_response(self, response: Dict[str, object]) -> None:
        question = str(response.get("question", ""))
        answer = str(response.get("answer", ""))
        sources = response.get("sources", [])

        print("\n" + "=" * 80)
        print(f"Question: {self._safe_console(question)}")
        print("-" * 80)
        print("Answer:")
        print(self._safe_console(answer))
        print("\nSources:")

        if not isinstance(sources, list) or not sources:
            print("  None")
            return

        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                print(f"  {index}. {self._safe_console(str(source))}")
                continue
            source_name = source.get("source", "unknown")
            page = source.get("page", "?")
            chunk_id = source.get("chunk_id", "?")
            print(
                f"  {index}. source={source_name}, "
                f"page={page}, chunk_id={chunk_id}"
            )
