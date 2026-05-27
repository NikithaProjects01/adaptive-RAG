"""
PDF ingestion utilities for the Simple Adaptive RAG app.

This file is intentionally small and beginner-friendly:
1. Load PDF pages.
2. Split pages into text chunks.
3. Store those chunks in ChromaDB.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import get_vector_store


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdf(pdf_path: str | Path) -> List[Document]:
    """Load a PDF file and return LangChain Document objects."""
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Split PDF pages into smaller chunks for better semantic search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(documents)


def ingest_pdf(pdf_path: str | Path) -> int:
    """
    Load, chunk, embed, and store a PDF in ChromaDB.

    Returns the number of chunks stored so the UI can show a useful message.
    """
    documents = load_pdf(pdf_path)

    if not documents:
        raise ValueError("No text could be loaded from the PDF.")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("The PDF was loaded, but no text chunks were created.")

    # Add source metadata so retrieved chunks can show where they came from.
    for chunk in chunks:
        chunk.metadata["source"] = str(pdf_path)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)
