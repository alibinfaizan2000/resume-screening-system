# Resume Screening & Candidate Matching System

Production-grade AI-powered resume screening using RAG, LangGraph, Gemini, and Pinecone.

## Architecture

```
PDF Resumes → PyMuPDF → Chunking → Gemini Embeddings → Pinecone
                                                            ↓
Job Description → LangGraph Workflow → Hybrid Search (Dense + BM25)
                                    → Cross-Encoder Reranking
                                    → Gemini 1.5 Pro Generation
                                    → LLM-as-Judge Faithfulness Check
                                    → Cited, Grounded Response
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq (Llama 3.3 70B) — free tier, 14,400 req/day |
| Embeddings | BAAI/bge-base-en-v1.5 — local, sentence-transformers (768-dim, no API limits) |
| Vector DB | Pinecone Serverless |
| Orchestration | LangGraph |
| Framework | LangChain |
| Monitoring | LangSmith |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Sparse Search | BM25 (rank-bm25) |
| Backend | FastAPI |
| Frontend | Streamlit |

## Features

- **Hybrid Search**: Dense vector + BM25 sparse retrieval
- **Cross-Encoder Reranking**: Precision reranking before generation
- **Grounded Generation**: Answers strictly from retrieved chunks
- **LLM-as-Judge**: Second Gemini call scores every answer for faithfulness (0–1) and flags unsupported claims before returning the response
- **Citation Support**: Every answer cites source filename + chunk ID
- **Agentic Workflow**: 8-node LangGraph pipeline
- **Full Observability**: LangSmith tracing per node

## LangGraph Workflow

```
query_understanding → dense_retrieval → sparse_retrieval → merge_results
    → reranking → context_builder → generation → faithfulness_check → END
```

## Quick Start

```bash
git clone https://github.com/yourusername/resume-screening-system.git
cd resume-screening-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# Terminal 1
uvicorn backend.main:app --reload --port 8000

# Terminal 2
cd frontend && streamlit run app.py
```

See `DEPLOYMENT_GUIDE.txt` for full setup and deployment instructions.
See `INTERVIEW_WORKFLOW.txt` for architecture deep-dive and interview prep.
