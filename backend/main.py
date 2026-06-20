from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from ingest import extract_documents
from vectorstore import InMemoryVectorStore
from qa import QAEngine, GroqModel, GeminiModel, download_pdf  

import os
app = FastAPI(title="DocuAgent API", version="1.1")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vectorstore = InMemoryVectorStore()

_llm_cache = {}

def get_llm(model_name: str):
    if model_name not in _llm_cache:
        try:
            if model_name == "groq":
                _llm_cache[model_name] = GroqModel()
            elif model_name == "gemini":
                _llm_cache[model_name] = GeminiModel()
        except Exception as e:
            raise ValueError(f"Failed to initialize model '{model_name}': {e}")
    return _llm_cache[model_name]

# Validate that at least one API key is present at startup
if not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("❌ Neither GROQ_API_KEY nor GEMINI_API_KEY is set in environment variables. At least one API key must be configured to start the server.")

default_llm = None
for model in ["groq", "gemini"]:
    try:
        default_llm = get_llm(model)
        break
    except Exception:
        continue

qa_engine = QAEngine(vectorstore=vectorstore, llm=default_llm)

# Startup health message - check environment keys directly to avoid eager instantiation
available = []
if os.getenv("GROQ_API_KEY"):
    available.append("groq")
if os.getenv("GEMINI_API_KEY"):
    available.append("gemini")
print(f"[STARTUP] Available models: {available}")


def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        paths.append(str(file_path))
    return paths

@app.get("/health")
def health():
    available = []
    for m in ["groq", "gemini"]:
        try:
            get_llm(m)
            available.append(m)
        except Exception:
            pass
    return {"status": "ok", "available_models": available}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        paths = save_uploaded_files(files)
        chunks = extract_documents(paths)
        qa_engine.add_documents(chunks)
        return {"message": f"Uploaded {len(files)} files", "chunks_added": len(chunks)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(
    query: str = Form(...),
    model: str = Form("groq"),
    top_k: int = Form(10)
):
    if not query or not query.strip():
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty"})
    try:
        if model not in ["groq", "gemini"]:
            return JSONResponse(status_code=400, content={"error": f"Model must be one of ['groq', 'gemini']"})
        try:
            qa_engine.llm = get_llm(model)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        answer = qa_engine.ask(query, top_k=top_k)
        return {"query": query, "answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/arxiv_search")
async def arxiv_search(
    query: str = Form(...),
    model: str = Form("groq"),
    max_papers: int = Form(3),
    action: str = Form("list"), 
    top_k: int = Form(10)
):
    try:
        if model not in ["groq", "gemini"]:
            return JSONResponse(status_code=400, content={"error": f"Model must be one of ['groq', 'gemini']"})
        try:
            qa_engine.llm = get_llm(model)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        papers = qa_engine.search_and_list_arxiv(query, max_papers=max_papers)

        if action == "list":
            return {"query": query, "papers": papers, "note": "Use action='download' or 'ingest' to proceed."}

        elif action == "download":
            saved_files = []
            for p in papers:
                if p.get("pdf_url"):
                    pdf_path = download_pdf(p["pdf_url"])
                    saved_files.append(pdf_path)
            return {"query": query, "downloaded_files": saved_files}

        elif action == "ingest":
            added_chunks = qa_engine.ingest_papers(papers)
            return {"query": query, "chunks_added": added_chunks, "note": "Papers ingested into QAEngine."}

        else:
            return JSONResponse(status_code=400, content={"error": "Invalid action. Must be one of: list, download, ingest"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})