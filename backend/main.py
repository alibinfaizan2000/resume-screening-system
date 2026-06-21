import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.resume_routes import router as resume_router
from backend.api.matching_routes import router as matching_router
from backend.api.qa_routes import router as qa_router
from backend.core.config import get_settings

settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

app = FastAPI(
    title="Resume Screening & Candidate Matching API",
    description="Production-grade AI-powered resume screening system using RAG, LangGraph, and Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume_router, prefix="/api/v1")
app.include_router(matching_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Resume Screening API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
