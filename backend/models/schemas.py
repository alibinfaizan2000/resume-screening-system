from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ResumeUploadResponse(BaseModel):
    filename: str
    status: str
    chunks_indexed: int
    message: str


class JobDescriptionRequest(BaseModel):
    job_description: str = Field(..., min_length=50, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)


class CandidateAnalysis(BaseModel):
    filename: str
    candidate_name: str
    match_percentage: float
    matching_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str
    reasoning: str
    retrieved_chunks: list[dict]


class MatchingResponse(BaseModel):
    job_description: str
    candidates: list[CandidateAnalysis]
    total_candidates_analyzed: int
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class QARequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    job_description: Optional[str] = None


class SourceChunk(BaseModel):
    filename: str
    chunk_id: str
    chunk_text: str
    relevance_score: Optional[float] = None


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    citations: list[str]
    faithfulness_score: float = 0.0
    faithfulness_passed: bool = False
    unsupported_claims: list[str] = []
    workflow_trace: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]
    total_results: int


class IndexedResume(BaseModel):
    filename: str
    total_chunks: int
    indexed_at: str


class IndexStatusResponse(BaseModel):
    total_resumes: int
    indexed_resumes: list[IndexedResume]
    total_chunks: int
