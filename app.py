"""
Streamlit frontend for the Simple Adaptive RAG chatbot.

Run with:
    streamlit run app.py
"""

from pathlib import Path
import importlib.util
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


if "last_ingested_file" not in st.session_state:
    st.session_state.last_ingested_file = None

if "last_chunk_count" not in st.session_state:
    st.session_state.last_chunk_count = 0

if "question" not in st.session_state:
    st.session_state.question = ""


st.set_page_config(
    page_title="Simple Adaptive RAG",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #eef2f7;
    }
    .main .block-container {
        padding-top: 2.2rem;
        max-width: 1180px;
    }
    .app-hero {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 1.1rem;
        margin-bottom: 1.2rem;
    }
    .app-hero h1 {
        margin-bottom: .35rem;
        font-size: 2.5rem;
        line-height: 1.08;
    }
    .muted {
        color: #6b7280;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .85rem;
        margin: 1.2rem 0 1.7rem;
    }
    .status-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: .9rem 1rem;
        background: #ffffff;
    }
    .status-label {
        color: #6b7280;
        font-size: .8rem;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .status-value {
        color: #111827;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: .2rem;
    }
    .panel {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.15rem;
        background: #ffffff;
        min-height: 310px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        min-height: 2.8rem;
    }
    .answer-box {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        background: #f8fafc;
        line-height: 1.6;
    }
    .hint-box {
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: .85rem 1rem;
        background: #eff6ff;
        color: #1e3a8a;
        margin-top: .75rem;
    }
    .error-help {
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: .9rem 1rem;
        background: #fef2f2;
        color: #991b1b;
    }
    @media (max-width: 800px) {
        .status-grid {
            grid-template-columns: 1fr;
        }
        .app-hero h1 {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_friendly_error(error: Exception) -> None:
    """Show beginner-friendly fixes for common local setup errors."""
    error_message = str(error)

    if "sentence_transformers" in error_message:
        python_executable = sys.executable
        st.markdown(
            """
            <div class="error-help">
            <strong>Missing dependency:</strong> sentence-transformers is not installed in
            the Python environment running this Streamlit app.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Run this command, then restart Streamlit:")
        st.code(f'& "{python_executable}" -m pip install sentence-transformers', language="powershell")
        st.caption(f"Streamlit is currently using: {python_executable}")
        return

    if "Connection refused" in error_message or "Ollama" in error_message:
        st.markdown(
            """
            <div class="error-help">
            <strong>Ollama is not ready:</strong> start Ollama and make sure the llama3
            model is pulled locally.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code("ollama pull llama3", language="powershell")
        return

    st.error(f"Something went wrong: {error_message}")


def package_status(package_name: str) -> str:
    """Return a simple installed/missing label for sidebar diagnostics."""
    if importlib.util.find_spec(package_name) is None:
        return "Missing"

    return "Installed"


def use_example_question(question: str) -> None:
    """Fill the question input from an example button."""
    st.session_state.question = question


with st.sidebar:
    st.header("About")
    st.write(
        "A fully local Adaptive RAG chatbot using LangChain, ChromaDB, "
        "HuggingFace embeddings, Ollama, and LangSmith tracing."
    )
    st.divider()
    st.subheader("Adaptive retrieval")
    st.write("Short questions: k = 2")
    st.write("Medium questions: k = 3")
    st.write("Detailed questions: k = 5")
    st.divider()
    st.subheader("Local stack")
    st.write("Vector DB: ChromaDB")
    st.write("Embeddings: MiniLM-L6-v2")
    st.write("Framework: LangChain")
    st.caption("Default Ollama model: llama3")
    st.divider()
    st.subheader("Environment")
    st.caption(f"Python: {sys.executable}")
    st.caption(f"sentence-transformers: {package_status('sentence_transformers')}")


st.markdown(
    """
    <div class="app-hero">
        <h1>Simple Adaptive RAG Chatbot</h1>
        <p class="muted">
        Upload a PDF, store it locally in ChromaDB, and ask questions using Ollama.
        Retrieval adapts between k=2, k=3, and k=5 based on question complexity.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

document_status = st.session_state.last_ingested_file or "No PDF processed yet"
chunk_status = st.session_state.last_chunk_count or "0"

st.markdown(
    f"""
    <div class="status-grid">
        <div class="status-card">
            <div class="status-label">Document</div>
            <div class="status-value">{document_status}</div>
        </div>
        <div class="status-card">
            <div class="status-label">Stored chunks</div>
            <div class="status-value">{chunk_status}</div>
        </div>
        <div class="status-card">
            <div class="status-label">Inference</div>
            <div class="status-value">Local Ollama llama3</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


upload_column, chat_column = st.columns([1, 2], gap="large")


with upload_column:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("1. Upload PDF")
    st.caption("Your document stays on this machine.")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])

    if uploaded_file is not None:
        pdf_path = DATA_DIR / uploaded_file.name

        if st.button("Process PDF"):
            with st.spinner("Loading, chunking, embedding, and storing PDF..."):
                try:
                    # Import here so dependency errors appear inside the UI instead of
                    # crashing the whole Streamlit app during startup.
                    from ingest import ingest_pdf

                    pdf_path.write_bytes(uploaded_file.getbuffer())
                    chunk_count = ingest_pdf(pdf_path)
                    st.session_state.last_ingested_file = uploaded_file.name
                    st.session_state.last_chunk_count = chunk_count
                    st.success(f"Stored {chunk_count} chunks in ChromaDB.")
                except Exception as error:
                    show_friendly_error(error)
    else:
        st.markdown(
            """
            <div class="hint-box">
            Start by uploading a PDF, then click Process PDF to build the local vector database.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


with chat_column:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("2. Ask a Question")

    example_columns = st.columns(3)
    with example_columns[0]:
        st.button(
            "Summary",
            on_click=use_example_question,
            args=("Summarize this document.",),
        )
    with example_columns[1]:
        st.button(
            "Definition",
            on_click=use_example_question,
            args=("What is scrum?",),
        )
    with example_columns[2]:
        st.button(
            "Steps",
            on_click=use_example_question,
            args=("Explain the key steps in detail.",),
        )

    question = st.text_input(
        "Question",
        placeholder="Example: What is this document about?",
        key="question",
    )

    ask_clicked = st.button("Ask")

    if ask_clicked:
        with st.spinner("Retrieving context and asking Ollama..."):
            try:
                # Import here so the app shell can still load on cloud hosts even
                # when local-only RAG dependencies need setup.
                from rag_pipeline import answer_question

                result = answer_question(question)

                st.markdown("### Answer")
                st.markdown(
                    f"<div class='answer-box'>{result['answer']}</div>",
                    unsafe_allow_html=True,
                )

                st.caption(f"Adaptive retrieval selected k = {result['k']}")

                with st.expander("Retrieved chunks"):
                    retrieved_chunks = result["retrieved_chunks"]

                    if not retrieved_chunks:
                        st.info("No chunks were retrieved.")

                    for index, document in enumerate(retrieved_chunks, start=1):
                        source = document.metadata.get("source", "Unknown source")
                        page = document.metadata.get("page", "Unknown page")
                        st.markdown(f"**Chunk {index}**")
                        st.caption(f"Source: {source} | Page: {page}")
                        st.write(document.page_content)
                        st.divider()

            except Exception as error:
                show_friendly_error(error)

    st.markdown("</div>", unsafe_allow_html=True)
