# Simple Adaptive RAG Chatbot

This project is a beginner-friendly, fully local Adaptive RAG chatbot. It lets you upload PDF documents, stores their text in a local ChromaDB vector database, and answers questions using a local Ollama model.

## What Is RAG?

RAG means Retrieval-Augmented Generation.

Instead of asking a language model to answer from memory, RAG first retrieves useful text from your documents. The language model then answers using that retrieved context.

Simple workflow:

1. Load a document.
2. Split it into chunks.
3. Convert chunks into embeddings.
4. Store embeddings in a vector database.
5. Retrieve the most similar chunks for a question.
6. Send the chunks and question to the LLM.
7. Display the answer.

## What Is Adaptive RAG?

Naive RAG usually retrieves the same number of chunks for every question.

Adaptive RAG changes retrieval behavior based on the question. This project keeps the logic simple:

| Question type | Retrieved chunks |
| --- | --- |
| Short/simple question | `k = 2` |
| Medium question | `k = 3` |
| Long/detailed question | `k = 5` |

This is still easy to understand, but it is more flexible than always using the same `k` value.

## How This Project Works

When you upload a PDF:

1. `PyPDFLoader` extracts text from the PDF.
2. `RecursiveCharacterTextSplitter` splits the text into chunks.
3. `sentence-transformers/all-MiniLM-L6-v2` creates embeddings.
4. ChromaDB stores the chunks locally in `./chroma_db`.

When you ask a question:

1. The app checks how simple or complex the question is.
2. It chooses `k = 2`, `k = 3`, or `k = 5`.
3. ChromaDB runs similarity search.
4. Retrieved chunks are sent to Ollama.
5. The answer is shown in Streamlit.

## Folder Structure

```text
project_root/
|-- data/
|   `-- sample.pdf
|-- chroma_db/
|-- app.py
|-- rag_pipeline.py
|-- ingest.py
|-- requirements.txt
|-- .env
`-- README.md
```

`chroma_db/` is created automatically when you process your first PDF.

## Requirements

- Python 3.10 or newer
- Ollama installed locally
- The `llama3` Ollama model pulled locally

## Setup Commands

Run these commands from the project folder.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the Ollama model

```bash
ollama pull llama3
```

Make sure Ollama is running. You can test it with:

```bash
ollama run llama3
```

### 4. Add LangSmith settings

Open `.env` and add your LangSmith API key if you want tracing:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Simple-Adaptive-RAG
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

The app also runs without LangSmith if you leave `LANGCHAIN_API_KEY` empty.

### 5. Start Streamlit

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Usage

1. Start the app with `streamlit run app.py`.
2. Upload `data/sample.pdf` or your own PDF.
3. Click **Process PDF**.
4. Ask a question, such as:

```text
What is Adaptive RAG?
```

5. Open **Retrieved chunks** to see what context was sent to the local LLM.

## Notes

- All document storage is local in `./chroma_db`.
- All LLM inference uses local Ollama.
- This project does not use agents, memory, reranking, hybrid search, or advanced RAG techniques.
- The goal is to keep Adaptive RAG simple, readable, and educational.
