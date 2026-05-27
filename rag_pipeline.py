"""
Adaptive RAG pipeline for local PDF question answering.

The app uses:
- HuggingFace sentence-transformer embeddings
- ChromaDB persistent vector storage
- Similarity search with dynamic k
- Ollama as the local chat model
- LangSmith tracing through environment variables
"""

from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# Load .env once when this module is imported.
# LangSmith reads LANGCHAIN_* variables from the environment automatically.
load_dotenv()


CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vector_store() -> Chroma:
    """Connect to the persistent ChromaDB vector store."""
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
        collection_name="adaptive_rag_documents",
    )


def get_llm():
    """
    Create the chat model.

    Local mode:
        LLM_PROVIDER=ollama

    Streamlit Cloud mode:
        LLM_PROVIDER=openai
        OPENAI_API_KEY=your_key
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it in Streamlit Cloud secrets "
                "or your local .env file."
            )

        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        return ChatOllama(model=model_name, temperature=0.2)

    raise ValueError("Unsupported LLM_PROVIDER. Use 'ollama' or 'openai'.")


def choose_k(question: str) -> int:
    """
    Beginner-friendly Adaptive RAG logic.

    Instead of always retrieving the same number of chunks, we estimate
    query complexity using word count and a few detail-seeking keywords.
    """
    question_words = question.split()
    word_count = len(question_words)

    detail_keywords = [
        "explain",
        "compare",
        "difference",
        "differences",
        "detailed",
        "steps",
        "summarize",
        "why",
        "how",
    ]

    has_detail_keyword = any(
        keyword in question.lower() for keyword in detail_keywords
    )

    if word_count <= 6 and not has_detail_keyword:
        return 2

    if word_count <= 14:
        return 3

    return 5


def format_documents(documents: List[Document]) -> str:
    """Convert retrieved chunks into a single context string for the LLM."""
    formatted_chunks = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        chunk_text = document.page_content.strip()
        formatted_chunks.append(
            f"Chunk {index}\nSource: {source}\nPage: {page}\nText: {chunk_text}"
        )

    return "\n\n".join(formatted_chunks)


def answer_question(question: str) -> Dict[str, object]:
    """
    Retrieve relevant chunks with adaptive k and ask the selected LLM for an answer.

    Returns a dictionary so Streamlit can display the answer, k value,
    and retrieved chunks separately.
    """
    if not question.strip():
        raise ValueError("Please enter a question.")

    vector_store = get_vector_store()
    k = choose_k(question)

    # Similarity search retrieves the most semantically related chunks.
    retrieved_documents = vector_store.similarity_search(question, k=k)

    if not retrieved_documents:
        return {
            "answer": "I could not find any stored document chunks. Upload and ingest a PDF first.",
            "retrieved_chunks": [],
            "k": k,
        }

    context = format_documents(retrieved_documents)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful local PDF assistant.

Answer the question using only the context below.
If the answer is not in the context, say that the document does not contain enough information.

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "retrieved_chunks": retrieved_documents,
        "k": k,
    }
