<div align="center">

# IntelliDoc - Intelligent Document Q&A Agent

An intelligent, multi-format document question-answering agent that uses semantic search, local embeddings, and state-of-the-art LLMs to ground Q&A in your custom research.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.104+-005571?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.20+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Supported-8E75C2?style=flat-square&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-Supported-f55a42?style=flat-square)](https://groq.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectors-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)

</div>


---

### Project Overview

IntelliDoc is a lightweight, responsive document question-answering application designed to bridge the gap between static documents and actionable insights. Built with a FastAPI backend and a Streamlit frontend, it enables users to build an in-memory knowledge base by uploading local files or automatically ingesting academic literature from ArXiv. 

The application utilizes local SentenceTransformers embeddings (or cloud-based Google Generative AI embeddings) to index document text. When a user asks a question, IntelliDoc performs semantic similarity search over the document chunks, retrieves the most relevant passages, and feeds them directly into an LLM (using Groq or Gemini) to generate accurate, source-attributed answers.

IntelliDoc is specifically tailored for researchers, students, and knowledge workers who need to query complex information locked inside PDFs, Word documents, HTML web pages, and markdown notes. By providing both single-document specificity and cross-document synthesis, it converts hours of manual reading into seconds of semantic exploration.

---

### Why This Project Exists

Information overload is a critical bottleneck in modern research and business workflows. Traditional search engines and standard document readers rely on keyword matching, which fails to capture semantic meaning or synthesize answers across separate files. 

IntelliDoc solves this pain point by providing a complete, self-contained Retrieval-Augmented Generation (RAG) system. Instead of opening multiple browser tabs, reading page by page, and copy-pasting notes, users can directly interrogate their files in natural language, ensuring that the answers generated are grounded explicitly in their documents with zero hallucinations.

---

### Key Features (Implemented Only)

- **Multi-Format Document Ingestion**: Supports `.pdf`, `.docx`, `.html`, `.htm`, `.txt`, and `.md` files out of the box.
- **Intelligent Text Chunking**: Automatically segments document text into 1500-character chunks with a 200-character overlap to preserve semantic context across chunk boundaries.
- **In-Memory Vector Store**: Leverages a fast NumPy-based cosine similarity vector space for search and retrieval with zero infrastructure overhead.
- **Dual LLM Integrations**: Supports fast inference via Groq (`llama3-8b-8192`) and advanced reasoning via Google Gemini (supporting `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite`, `gemini-3-flash-preview`, `gemma-4-31b-it`, and `gemma-4-26b-a4b-it`).
- **Flexible Q&A Modes**: Automatically switches between per-document queries (generating separate answers per file) and combined mode (synthesizing information across all papers) based on query keywords.
- **ArXiv Research Integration**: Search for academic papers by keywords, download their PDFs locally, and ingest their content directly into the vector store.
- **Streamlit Chat Interface**: Provides a user-friendly conversational interface with message history, configuration controls, and file management sidebar.
- **FastAPI REST API**: High-performance backend with custom CORS restrictions, error handling, input validation, and automatic OpenAPI schema documentation.
- **Content-Based Title Extraction**: Automatically parses PDF metadata and content headers to extract the clean, human-readable document titles.

---

### Architecture

IntelliDoc is architected as a decoupled, client-server system consisting of a FastAPI REST backend and a Streamlit web interface. 

#### Request Flow:
1. **Ingestion**: The user uploads files via the Streamlit UI, which forwards them to the FastAPI `/upload` endpoint. `ingest.py` extracts text (utilizing PyMuPDF for text, pdfplumber for tables, python-docx for Word files, and BeautifulSoup4 for HTML) and segments it into overlapping chunks. The chunks are embedded using SentenceTransformers (for Groq) or Google Embeddings (for Gemini) and stored in the NumPy-based `InMemoryVectorStore`.
2. **Q&A**: When the user enters a question, it is sent to `/ask`. The backend embeds the query, queries the vector store via cosine similarity, retrieves the top-K chunks, compiles the grounding prompts, feeds them to the chosen LLM, and returns the response with source documents.

#### ArXiv Flow:
1. The user inputs a query in the ArXiv Search panel. Streamlit calls `/arxiv_search` on the backend.
2. The backend uses the official `arxiv` library to fetch paper metadata and download PDFs.
3. The downloaded PDFs are pushed through the same ingestion pipeline and indexed in the vector store.

```mermaid
graph TD
    UI[Streamlit UI] -->|REST API Requests| API[FastAPI Backend]
    
    subgraph Ingestion Pipeline [/upload]
        API --> Ingest[ingest.py]
        Ingest --> Chunk[chunk_text]
        Chunk --> Embed[Embeddings Generation]
        Embed --> VS[InMemoryVectorStore]
    end
    
    subgraph Query Pipeline [/ask]
        API --> Search[similarity_search]
        Search --> LLM[LLM Inference Groq/Gemini]
        LLM --> Answer[Generate Answer]
    end
    
    subgraph ArXiv Pipeline [/arxiv_search]
        API --> ArXiv[arxiv Library]
        ArXiv --> Down[download_pdf]
        Down --> Ingest
    end
```


---

### Tech Stack

| Technology | Role | Why Chosen |
|---|---|---|
| **FastAPI** | REST API backend | Asynchronous support, high speed, and automatically generates interactive OpenAPI Swagger docs. |
| **Streamlit** | Web frontend | Enables rapid deployment of premium, interactive user interfaces for AI apps using Python. |
| **PyMuPDF (fitz)** | PDF text & image extraction | Exceptionally fast and robust parsing of text, pages, and embedded images in PDF files. |
| **pdfplumber** | PDF table extraction | Provides precise coordinate-based table detection and CSV exporting for PDF tabular data. |
| **python-docx** | DOCX extraction | Clean, standard parser for extracting text content from Microsoft Word document structures. |
| **BeautifulSoup4** | HTML extraction | Offers powerful tree traversal to clean and extract plain text from HTML documents. |
| **SentenceTransformers** | Local embeddings for Groq path | Leverages `all-mpnet-base-v2` locally for high-quality, free, offline semantic embeddings. |
| **Groq SDK** | LLM inference (fast path) | Provides ultra-low latency response generation using Groq's hardware accelerators. |
| **Google Generative AI** | LLM + embeddings (advanced path) | Delivers advanced reasoning capabilities and cloud-scale semantic embedding models. |
| **NumPy** | Vector storage & similarity | Provides high-performance matrix operations for cosine similarity without external database setup. |
| **arxiv** | Research paper client | Integrates directly with the official ArXiv API to search, filter, and fetch academic papers. |
| **python-dotenv** | Environment configuration | Securely loads configuration options and API keys from a local `.env` file. |

---

### Engineering Decisions

1. **In-Memory Vector Store**: We opted for a custom NumPy-based `InMemoryVectorStore` over a dedicated external database (like Pinecone or Chroma). This minimizes setup overhead and system complexity for local prototyping. The trade-off is that the database is lost on server restart, which is acceptable in sandbox environments.
2. **Dual Embedding Isolation**: Groq uses local `SentenceTransformers` embeddings, whereas Gemini uses Google's `text-embedding-004` cloud embeddings. Because the embedding dimensions and vector spaces differ, vectors generated under one model cannot be queried using the other. Users must select their model prior to document ingestion to avoid retrieval mismatches.
3. **Overlapping Text Chunking**: Text is split into 1500-character segments with a 200-character overlap. The overlap acts as a safety buffer, preventing sentences or core concepts from being severed at chunk boundaries, which ensures high semantic continuity.
4. **Intent-Driven Q&A Modes**: Rather than using a complex router, query intent is detected through key phrasing ("all papers", "combined", etc.). This decides whether the engine prompts the LLM to write separate answers per document or perform a unified cross-document synthesis.

---

### AI Components

- **Embedding Models**:
  - `all-mpnet-base-v2` (SentenceTransformers, local) used during Groq execution.
  - `models/text-embedding-004` (Google, API) used during Gemini execution.
- **LLMs for Generation**:
  - `llama3-8b-8192` via Groq.
  - `gemini-2.5-flash` (or customized via `GEMINI_MODEL`) via Google Generative AI.
- **Retrieval**: Custom cosine similarity search executing matrix multiplication on L2-normalized vector stacks in NumPy.
- **Prompt Strategy**: Prompt template injection wrapping extracted contextual chunks alongside the user query to steer generation.

---

### User Flow

1. **Initial Access**: The user opens the web interface in their browser at `http://localhost:8501`.
2. **Document Ingestion**: The user uploads local documents (`.pdf`, `.docx`, `.txt`, `.md`, `.html`) via the sidebar.
3. **Automatic Embedding**: The backend automatically parses, chunks, embeds, and uploads the documents to the vector store.
4. **Literature Search**: Optionally, the user inputs search parameters under the ArXiv sidebar section to list, download, or ingest relevant academic papers.
5. **Chat Interface**: The user types a question in the main chat container.
6. **Retrieval**: The FastAPI backend embeds the query, searches for the top-10 chunks, and compiles the relevant context.
7. **Generation & Grounding**: The selected LLM generates an answer based strictly on the retrieved context.
8. **Attribution**: The answer is formatted and displayed in the chat interface, listing the contributing source documents.

---

### Folder Structure

```
IntelliDoc/
├── backend/
│   ├── main.py          # FastAPI app, endpoints, CORS, startup
│   ├── ingest.py        # Document parsing, chunking, title extraction
│   ├── qa.py            # QAEngine, LLM interfaces, ArXiv integration
│   ├── llm_client.py    # Groq and Gemini client factory functions
│   └── vectorstore.py   # NumPy-based in-memory vector store
├── frontend/
│   └── app.py           # Streamlit UI, chat interface, sidebar controls
├── data/
│   ├── uploads/         # Temporary storage for uploaded files (gitignored)
│   └── assets/          # Extracted PDF images and tables (gitignored)
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

### Installation Guide

#### Prerequisites
- Python 3.8 or higher
- At least one API key: Groq or Google Gemini

#### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd IntelliDoc

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

### Environment Variables

Create a `.env` file in the project root:

```env
# At least one of these is required
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Override the default Gemini model
GEMINI_MODEL=gemini-2.5-flash
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Required for Groq | Grab from [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | Required for Gemini | Grab from [aistudio.google.com](https://aistudio.google.com) |
| `GEMINI_MODEL` | Optional | Override the default model (defaults to `gemini-2.5-flash`) |

*Note: The backend starts successfully with only one of the keys present. If a model is chosen whose key is missing, an informative HTTP 400 error is returned.*

---

### Running The Project

```bash
# Terminal 1 — Start the backend server
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Start the frontend client
cd frontend
streamlit run app.py --server.port 8501
```

Access the application at: `http://localhost:8501`  
Interactive API Swagger docs: `http://localhost:8000/docs`  

⚠️ **Important**: The database is stored in the backend server memory. Do not run the backend with `--reload` in development, and avoid restarting the server, as it wipes all ingested document chunks. If the backend is restarted, you must re-upload/re-ingest your documents.

---

### API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health status check, lists available models |
| `/upload` | POST | Uploads and ingests multiple file uploads |
| `/ask` | POST | Submits a query for retrieval and response generation |
| `/arxiv_search`| POST | Lists, downloads, or ingests ArXiv research literature |

#### Example: Upload a document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"
```

#### Example: Ask a question
```bash
curl -X POST "http://localhost:8000/ask" \
  -F "query=What are the primary metrics?" \
  -F "model=groq" \
  -F "top_k=10"
```

#### Example: Search ArXiv
```bash
curl -X POST "http://localhost:8000/arxiv_search" \
  -F "query=quantum computing cryptography" \
  -F "action=ingest" \
  -F "max_papers=3"
```

---

### Known Limitations

- **Volatile Storage**: Since data is stored in memory, restarting the FastAPI server erases all ingested texts.
- **No Access Controls**: The REST endpoints lack authentication or API keys for user access management.
- **Shared Session Context**: The in-memory vector store is shared across all concurrent user requests.
- **Embedding Vector Mismatches**: Shifting the model between Groq and Gemini after ingestion returns invalid results since the vector models do not share the same dimensions or spatial coordinate definitions.
- **Scalability Ceiling**: The NumPy matrix operations execute on a single CPU thread, which is fine for small collections (<10,000 vectors) but will degrade under enterprise-scale data volumes.

---

### Performance Considerations

- The local SentenceTransformers model (`all-mpnet-base-v2`, ~420MB) loads into memory during the first Groq execution and is cached.
- Cosine similarity search scales linearly with the number of chunks ($O(N)$), ensuring instantaneous retrieval for small to mid-sized databases.
- Groq inference is highly optimized, completing generations in a fraction of the time compared to standard cloud APIs.
- PDF table extraction via `pdfplumber` adds overhead, particularly on scan-heavy documents.

---

### Security Considerations

- All API keys are loaded safely through environment configurations and never exposed in REST responses.
- CORS is configured to block unauthorized requests, restricting origins to `http://localhost:8501` by default.
- No user text is evaluated dynamically; queries are mapped entirely into LLM generation templates.
- Uploaded files are written directly into a restricted workspace directory (`data/uploads/`).

---

### License

MIT License — see LICENSE file for details.