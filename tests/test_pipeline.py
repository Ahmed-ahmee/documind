# tests/test_pipeline.py

import sys
sys.path.append(".")

from src.pipeline.rag_pipeline import RAGPipeline


def _safe_console(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


if __name__ == "__main__":

    pipeline = RAGPipeline()

    # Ingest once — skips automatically if already indexed
    pipeline.ingest()
    
    print("\n--- DEBUG: Raw chunks for optimizer query ---")
    chunks = pipeline.store.similarity_search("Adam optimizer learning rate training", k=4)
    for c in chunks:
        print(f"\nPage {c.metadata['page']} | Chunk {c.metadata['chunk_id']}")
        print(_safe_console(c.page_content[:300]))
        print("---")

    # Ask real questions
    questions = [
        "What is multi-head attention and why is it useful?",
        "What BLEU score did the Transformer achieve on English-German translation?",
        "What optimizer was used during training?",
    ]

    for question in questions:
        response = pipeline.query(question)
        pipeline.print_response(response)
