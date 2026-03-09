import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import RAGPipeline


def _safe_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


def _init_state() -> None:
    st.session_state.setdefault("indexed", False)
    st.session_state.setdefault("last_response", None)


st.set_page_config(page_title="DocuMind RAG", layout="wide")
_init_state()

st.title("DocuMind RAG Assistant")
st.caption("Ask questions over files in data/documents")

left_col, right_col = st.columns([2, 1])

with right_col:
    st.subheader("Index")
    data_dir = PROJECT_ROOT / "data" / "documents"
    if data_dir.exists():
        supported_files = [
            p.name for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
        ]
    else:
        supported_files = []

    if supported_files:
        st.write(f"Detected {len(supported_files)} file(s):")
        for name in supported_files:
            st.write(f"- {name}")
    else:
        st.warning("No .pdf or .txt files found in data/documents.")

    if st.button("Build / Refresh Index", use_container_width=True):
        with st.spinner("Indexing documents..."):
            try:
                pipeline = _get_pipeline()
                count = pipeline.ingest(force_reindex=True)
                st.session_state.indexed = True
                st.success(f"Indexed {count} chunks.")
            except Exception as exc:  # noqa: BLE001
                st.session_state.indexed = False
                st.error(f"Indexing failed: {exc}")

    pipeline = _get_pipeline()
    current_count = pipeline.store.get_count()
    st.metric("Chunks in memory", current_count)

with left_col:
    st.subheader("Ask a Question")
    question = st.text_area(
        "Question",
        placeholder="Example: What optimizer was used during training?",
        height=120,
    )

    ask_clicked = st.button("Ask", type="primary")
    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    pipeline = _get_pipeline()
                    response = pipeline.query(question.strip())
                    st.session_state.last_response = response
                    st.session_state.indexed = True
                except Exception as exc:  # noqa: BLE001
                    st.session_state.last_response = None
                    st.error(f"Query failed: {exc}")

    response = st.session_state.last_response
    if response:
        st.markdown("### Answer")
        st.write(_safe_text(str(response.get("answer", ""))))

        sources = response.get("sources", [])
        st.markdown("### Sources")
        if sources:
            for i, source in enumerate(sources, start=1):
                source_name = source.get("source", "unknown")
                page = source.get("page", "?")
                chunk_id = source.get("chunk_id", "?")
                preview = _safe_text(str(source.get("preview", "")))
                with st.expander(f"{i}. {source_name} | page {page} | chunk {chunk_id}"):
                    st.write(preview)
        else:
            st.info("No source chunks returned.")
    else:
        st.info("Build the index and ask a question to see answers here.")

