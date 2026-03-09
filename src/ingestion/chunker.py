from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Split loaded documents into overlapping chunks for retrieval."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        print(f"\nChunking {len(documents)} document section(s)...")
        print(
            f"  Settings -> chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        print(f"  -> Produced {len(chunks)} chunks")
        self._print_chunk_stats(chunks)
        return chunks

    def _print_chunk_stats(self, chunks: List[Document]) -> None:
        if not chunks:
            print("  No chunks produced")
            return

        sizes = [len(c.page_content) for c in chunks]
        avg = sum(sizes) / len(sizes)

        print(f"  -> Avg chunk size : {avg:.0f} chars")
        print(f"  -> Min chunk size : {min(sizes)} chars")
        print(f"  -> Max chunk size : {max(sizes)} chars")
        print(f"  -> Total chunks   : {len(chunks)}")
