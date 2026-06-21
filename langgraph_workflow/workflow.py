import os
import json
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from backend.services.retrieval_service import (
    dense_retrieval,
    sparse_retrieval,
    rerank_results,
    deduplicate_context,
    hybrid_retrieval,
)
from backend.services.llm_service import get_llm
from backend.core.config import get_settings

settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project


class AgentState(TypedDict):
    query: str
    job_description: Optional[str]
    query_intent: str
    dense_results: list[dict]
    sparse_results: list[dict]
    merged_results: list[dict]
    reranked_results: list[dict]
    context: str
    answer: str
    citations: list[str]
    sources: list[dict]
    citation_verified: bool
    faithfulness_score: float
    unsupported_claims: list[str]
    faithfulness_passed: bool
    error: Optional[str]


def query_understanding_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = f"""Classify this query into one category:
- candidate_ranking: comparing or ranking candidates
- skill_search: finding candidates with specific skills
- candidate_comparison: comparing two specific candidates
- general_qa: general question about resumes

Query: {state['query']}

Respond with only the category name."""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["query_intent"] = response.content.strip()
    return state


def dense_retrieval_node(state: AgentState) -> AgentState:
    enriched_query = state["query"]
    if state.get("job_description"):
        enriched_query = f"{state['query']} {state['job_description'][:200]}"
    state["dense_results"] = dense_retrieval(enriched_query, top_k=settings.top_k_retrieval)
    return state


def sparse_retrieval_node(state: AgentState) -> AgentState:
    corpus = state.get("dense_results", [])
    state["sparse_results"] = sparse_retrieval(state["query"], corpus, top_k=settings.top_k_retrieval)
    return state


def merge_results_node(state: AgentState) -> AgentState:
    seen_ids = set()
    merged = []
    for doc in state["dense_results"] + state["sparse_results"]:
        if doc["chunk_id"] not in seen_ids:
            seen_ids.add(doc["chunk_id"])
            merged.append(doc)
    state["merged_results"] = merged
    return state


def reranking_node(state: AgentState) -> AgentState:
    deduped = deduplicate_context(state["merged_results"])
    state["reranked_results"] = rerank_results(
        state["query"], deduped, top_k=settings.top_k_rerank
    )
    return state


def context_builder_node(state: AgentState) -> AgentState:
    chunks = state["reranked_results"]
    context_parts = []
    sources = []
    citations = []

    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}] File: {chunk['filename']} | Chunk: {chunk['chunk_id']}\n{chunk['text']}"
        )
        sources.append(
            {
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["text"],
                "relevance_score": chunk.get("rerank_score", chunk.get("score")),
            }
        )
        citations.append(f"[{i+1}] {chunk['filename']} — {chunk['chunk_id']}")

    state["context"] = "\n\n---\n\n".join(context_parts)
    state["sources"] = sources
    state["citations"] = citations
    return state


def generation_node(state: AgentState) -> AgentState:
    llm = get_llm()

    system_prompt = """You are an expert AI recruiter assistant. Answer questions STRICTLY based on the provided resume context.

Rules:
1. Only use information from the provided context chunks
2. If information is not in the context, say "No supporting information found in retrieved resumes."
3. Always cite sources using [Source N] notation
4. Be specific, factual, and structured
5. Do not hallucinate or infer beyond what is stated"""

    user_prompt = f"""Query: {state['query']}

Job Description Context: {state.get('job_description', 'Not provided')}

Retrieved Resume Chunks:
{state['context']}

Provide a comprehensive, evidence-based answer with source citations."""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    state["answer"] = response.content
    return state


def faithfulness_check_node(state: AgentState) -> AgentState:
    """LLM-as-judge: verifies every claim in the answer is supported by retrieved context."""
    llm = get_llm()
    answer = state["answer"]
    sources = state["sources"]

    if not sources:
        state["answer"] = "No supporting information found in retrieved resumes."
        state["citation_verified"] = False
        state["faithfulness_score"] = 0.0
        state["unsupported_claims"] = []
        state["faithfulness_passed"] = False
        return state

    context_text = "\n\n".join(
        f"[Source {i+1}] {s['filename']}:\n{s['chunk_text']}"
        for i, s in enumerate(sources)
    )

    judge_prompt = f"""You are a faithfulness judge for a RAG system. Your job is to verify whether the generated answer is fully grounded in the provided context.

CONTEXT (retrieved resume chunks):
{context_text}

GENERATED ANSWER:
{answer}

Instructions:
- Read every claim in the generated answer.
- Check if each claim is explicitly supported by the context above.
- A claim is unsupported if it adds information not present in the context, or contradicts it.
- Do NOT penalize reasonable paraphrasing — only flag actual unsupported facts.

Respond ONLY with a valid JSON object, no markdown:
{{
  "faithfulness_score": <float 0.0 to 1.0>,
  "supported_claims": ["claim1", "claim2"],
  "unsupported_claims": ["claim1", "claim2"],
  "verdict": "PASS" or "FAIL",
  "reasoning": "one sentence explanation"
}}

A score of 1.0 means fully grounded. Below 0.5 is a FAIL."""

    try:
        response = llm.invoke([HumanMessage(content=judge_prompt)])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)

        score = float(result.get("faithfulness_score", 0.0))
        unsupported = result.get("unsupported_claims", [])
        verdict = result.get("verdict", "FAIL")
        passed = verdict == "PASS" and score >= 0.5

        state["faithfulness_score"] = score
        state["unsupported_claims"] = unsupported
        state["faithfulness_passed"] = passed
        state["citation_verified"] = passed

        if not passed:
            disclaimer = (
                f"\n\n⚠️ **Faithfulness Warning** (score: {score:.2f}): "
                f"This answer may contain claims not fully supported by the retrieved resumes. "
                f"Unsupported claims detected: {', '.join(unsupported) if unsupported else 'see reasoning'}."
            )
            state["answer"] = answer + disclaimer

    except (json.JSONDecodeError, KeyError, ValueError):
        state["faithfulness_score"] = 0.0
        state["unsupported_claims"] = []
        state["faithfulness_passed"] = False
        state["citation_verified"] = False

    return state


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("query_understanding", query_understanding_node)
    workflow.add_node("dense_retrieval", dense_retrieval_node)
    workflow.add_node("sparse_retrieval", sparse_retrieval_node)
    workflow.add_node("merge_results", merge_results_node)
    workflow.add_node("reranking", reranking_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("faithfulness_check", faithfulness_check_node)

    workflow.set_entry_point("query_understanding")
    workflow.add_edge("query_understanding", "dense_retrieval")
    workflow.add_edge("dense_retrieval", "sparse_retrieval")
    workflow.add_edge("sparse_retrieval", "merge_results")
    workflow.add_edge("merge_results", "reranking")
    workflow.add_edge("reranking", "context_builder")
    workflow.add_edge("context_builder", "generation")
    workflow.add_edge("generation", "faithfulness_check")
    workflow.add_edge("faithfulness_check", END)

    return workflow.compile()


_compiled_workflow = None


def get_workflow():
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()
    return _compiled_workflow


@traceable(name="resume_qa_workflow")
def run_qa_workflow(question: str, job_description: str = None) -> dict:
    workflow = get_workflow()
    initial_state: AgentState = {
        "query": question,
        "job_description": job_description,
        "query_intent": "",
        "dense_results": [],
        "sparse_results": [],
        "merged_results": [],
        "reranked_results": [],
        "context": "",
        "answer": "",
        "citations": [],
        "sources": [],
        "citation_verified": False,
        "faithfulness_score": 0.0,
        "unsupported_claims": [],
        "faithfulness_passed": False,
        "error": None,
    }
    final_state = workflow.invoke(initial_state)
    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "citations": final_state["citations"],
        "query_intent": final_state["query_intent"],
        "citation_verified": final_state["citation_verified"],
        "faithfulness_score": final_state["faithfulness_score"],
        "unsupported_claims": final_state["unsupported_claims"],
        "faithfulness_passed": final_state["faithfulness_passed"],
    }
