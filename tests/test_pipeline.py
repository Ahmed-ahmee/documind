import sys
from pathlib import Path

sys.path.append(".")  # makes src/ importable from project root

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker


def _ensure_test_document(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    has_supported_file = any(
        file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt"}
        for file_path in data_dir.iterdir()
    )
    if has_supported_file:
        return

    sample_path = data_dir / "sample.txt"
    sample_path.write_text(
        "DocuMind RAG sample document.\n"
        "This file is auto-created by tests/test_pipeline.py when no input "
        "documents are present.\n",
        encoding="utf-8",
    )
    print(f"Created sample input file: {sample_path}")


def test_ingestion() -> None:
    print("=" * 50)
    print("TEST: Document Ingestion Pipeline")
    print("=" * 50)

    data_dir = Path("data/documents")
    _ensure_test_document(data_dir)

    # Step 1: Load
    loader = DocumentLoader(str(data_dir))
    documents = loader.load()

    print(f"\nLoaded {len(documents)} page(s)")
    print(f"   First 200 chars: {documents[0].page_content[:200]}")
    print(f"   Metadata: {documents[0].metadata}")

    # Step 2: Chunk
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(documents)

    sample_chunk = chunks[min(5, len(chunks) - 1)]
    print("\nSample chunk:")
    print(f"   Content: {sample_chunk.page_content[:200]}")
    print(f"   Metadata: {sample_chunk.metadata}")


if __name__ == "__main__":
    test_ingestion()
