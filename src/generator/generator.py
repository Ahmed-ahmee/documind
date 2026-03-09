from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    OllamaLLM = None


class RAGGenerator:
    def __init__(self, model: str = "llama3.2"):
        print(f"\nInitializing Generator with model: {model}")
        self.prompt_template = self._build_prompt()
        self.llm = OllamaLLM(model=model) if OllamaLLM is not None else None
        if self.llm is None:
            print("langchain_ollama is not installed; using extractive fallback.")

    def _build_prompt(self) -> PromptTemplate:
        template = """<|system|>
You are a precise document Q&A assistant.
Read the context carefully and answer the question in 2-4 clear sentences.
Use specific facts, numbers, and names from the context.
End your answer with: (Source: Page X)
Never copy text verbatim. Never make up information.
If the answer is not in the context, say: "Not found in document."
<|end|>

<|context|>
{context}
<|end|>

<|question|>
{question}
<|end|>

<|answer|>"""
        return PromptTemplate(input_variables=["context", "question"], template=template)

    def _format_context(self, chunks: List[Document]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            context_parts.append(
                f"[Source {i + 1} | File: {source} | Page: {page}]\n"
                f"{chunk.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _fallback_answer(query: str, chunks: List[Document]) -> str:
        if not chunks:
            return "Not found in document."

        query_terms = {token.lower() for token in query.split()}
        best_chunk = chunks[0]
        best_score = -1

        for chunk in chunks:
            words = {token.lower() for token in chunk.page_content.split()}
            score = len(query_terms.intersection(words))
            if score > best_score:
                best_score = score
                best_chunk = chunk

        page = best_chunk.metadata.get("page", "?")
        preview = best_chunk.page_content.strip().replace("\n", " ")
        preview = preview[:350]
        return f"{preview} (Source: Page {page})"

    def generate(self, query: str, chunks: List[Document]) -> dict:
        context = self._format_context(chunks)
        prompt = self.prompt_template.format(context=context, question=query)

        print(f"\n  [DEBUG] Prompt starts with: {prompt[:80]}")
        print(f"  Sending {len(chunks)} chunks to LLM...")

        if self.llm is None:
            answer = self._fallback_answer(query, chunks)
        else:
            answer = self.llm.invoke(prompt)

        return {
            "question": query,
            "answer": str(answer),
            "sources": [
                {
                    "source": c.metadata.get("source"),
                    "page": c.metadata.get("page"),
                    "chunk_id": c.metadata.get("chunk_id"),
                    "preview": c.page_content[:100],
                }
                for c in chunks
            ],
        }


class Generator(RAGGenerator):
    """Compatibility alias used by pipeline imports."""

