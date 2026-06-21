from fastapi import APIRouter, HTTPException
from backend.models.schemas import QARequest, QAResponse, SourceChunk, SearchRequest, SearchResponse
from backend.services.retrieval_service import retrieve_and_rerank
from langgraph_workflow.workflow import run_qa_workflow

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    try:
        result = run_qa_workflow(request.question, request.job_description)

        sources = [
            SourceChunk(
                filename=s["filename"],
                chunk_id=s["chunk_id"],
                chunk_text=s["chunk_text"],
                relevance_score=s.get("relevance_score"),
            )
            for s in result.get("sources", [])
        ]

        return QAResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            citations=result.get("citations", []),
            faithfulness_score=result.get("faithfulness_score", 0.0),
            faithfulness_passed=result.get("faithfulness_passed", False),
            unsupported_claims=result.get("unsupported_claims", []),
            workflow_trace={"query_intent": result.get("query_intent", "")},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_resumes(request: SearchRequest):
    try:
        results = retrieve_and_rerank(request.query, top_k_rerank=request.top_k)

        sources = [
            SourceChunk(
                filename=r["filename"],
                chunk_id=r["chunk_id"],
                chunk_text=r["text"],
                relevance_score=r.get("rerank_score", r.get("score")),
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=sources,
            total_results=len(sources),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
