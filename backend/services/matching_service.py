import json
from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.retrieval_service import retrieve_and_rerank
from backend.services.llm_service import get_llm
from backend.models.schemas import CandidateAnalysis
from langsmith import traceable


def group_chunks_by_candidate(chunks: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for chunk in chunks:
        fname = chunk["filename"]
        if fname not in grouped:
            grouped[fname] = []
        grouped[fname].append(chunk)
    return grouped


@traceable(name="analyze_candidate")
def analyze_single_candidate(
    filename: str,
    chunks: list[dict],
    job_description: str,
    llm,
) -> CandidateAnalysis:
    candidate_context = "\n\n".join([c["text"] for c in chunks])

    system_prompt = """You are an expert technical recruiter. Analyze a candidate's resume against a job description.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "candidate_name": "Full Name or 'Unknown'",
  "match_percentage": 0-100,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "recommendation": "Strong Hire | Hire | Maybe | No Hire",
  "reasoning": "2-3 sentence explanation"
}
Base analysis ONLY on the provided resume chunks. Do not fabricate information."""

    user_prompt = f"""Job Description:
{job_description}

Resume Chunks from {filename}:
{candidate_context}

Analyze this candidate strictly based on the provided chunks."""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)

    return CandidateAnalysis(
        filename=filename,
        candidate_name=data.get("candidate_name", "Unknown"),
        match_percentage=float(data.get("match_percentage", 0)),
        matching_skills=data.get("matching_skills", []),
        missing_skills=data.get("missing_skills", []),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        recommendation=data.get("recommendation", "No Hire"),
        reasoning=data.get("reasoning", ""),
        retrieved_chunks=[
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"][:300],
                "score": c.get("rerank_score", c.get("score", 0)),
            }
            for c in chunks
        ],
    )


@traceable(name="match_candidates")
def match_candidates(job_description: str, top_k: int = 5) -> list[CandidateAnalysis]:
    chunks = retrieve_and_rerank(job_description, top_k_rerank=top_k * 3)

    if not chunks:
        return []

    grouped = group_chunks_by_candidate(chunks)
    llm = get_llm()

    analyses = []
    for filename, candidate_chunks in grouped.items():
        try:
            analysis = analyze_single_candidate(filename, candidate_chunks, job_description, llm)
            analyses.append(analysis)
        except Exception as e:
            analyses.append(
                CandidateAnalysis(
                    filename=filename,
                    candidate_name="Parse Error",
                    match_percentage=0,
                    matching_skills=[],
                    missing_skills=[],
                    strengths=[],
                    weaknesses=[],
                    recommendation="Error",
                    reasoning=f"Analysis failed: {str(e)}",
                    retrieved_chunks=[],
                )
            )

    analyses.sort(key=lambda x: x.match_percentage, reverse=True)
    return analyses[:top_k]
