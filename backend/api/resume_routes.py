from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
from backend.services.ingestion_service import index_resume
from backend.services.pinecone_service import get_index_stats, delete_by_filename
from backend.models.schemas import ResumeUploadResponse, IndexStatusResponse, IndexedResume

router = APIRouter(prefix="/resumes", tags=["resumes"])


@router.post("/upload", response_model=List[ResumeUploadResponse])
async def upload_resumes(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    responses = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            responses.append(
                ResumeUploadResponse(
                    filename=file.filename,
                    status="error",
                    chunks_indexed=0,
                    message="Only PDF files are supported",
                )
            )
            continue

        try:
            content = await file.read()
            chunks_indexed = index_resume(content, file.filename)
            responses.append(
                ResumeUploadResponse(
                    filename=file.filename,
                    status="success",
                    chunks_indexed=chunks_indexed,
                    message=f"Successfully indexed {chunks_indexed} chunks",
                )
            )
        except Exception as e:
            responses.append(
                ResumeUploadResponse(
                    filename=file.filename,
                    status="error",
                    chunks_indexed=0,
                    message=str(e),
                )
            )

    return responses


@router.get("/index-status", response_model=IndexStatusResponse)
async def get_index_status():
    try:
        stats = get_index_stats()
        total_chunks = stats.get("total_vector_count", 0)
        return IndexStatusResponse(
            total_resumes=0,
            indexed_resumes=[],
            total_chunks=total_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename}")
async def delete_resume(filename: str):
    try:
        delete_by_filename(filename)
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
