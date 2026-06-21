from fastapi import APIRouter, HTTPException
from datetime import datetime
from backend.services.matching_service import match_candidates
from backend.models.schemas import JobDescriptionRequest, MatchingResponse

router = APIRouter(prefix="/matching", tags=["matching"])


@router.post("/candidates", response_model=MatchingResponse)
async def match_candidates_endpoint(request: JobDescriptionRequest):
    try:
        candidates = match_candidates(request.job_description, top_k=request.top_k)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No candidates found. Please upload resumes first.",
            )
        return MatchingResponse(
            job_description=request.job_description,
            candidates=candidates,
            total_candidates_analyzed=len(candidates),
            generated_at=datetime.utcnow().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
