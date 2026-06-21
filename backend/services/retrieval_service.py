from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from backend.services.llm_service import embed_query
from backend.services.pinecone_service import query_vectors, get_index_stats
from backend.core.config import get_settings

settings = get_settings()

_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def dense_retrieval(query: str, top_k: int = None) -> list[dict]:
    k = top_k or settings.top_k_retrieval
    query_vec = embed_query(query)
    matches = query_vectors(query_vec, top_k=k)
    results = []
    for m in matches:
        results.append(
            {
                "chunk_id": m["id"],
                "filename": m["metadata"].get("filename", ""),
                "text": m["metadata"].get("text", ""),
                "score": m["score"],
                "source": "dense",
            }
        )
    return results


def sparse_retrieval(query: str, corpus: list[dict], top_k: int = None) -> list[dict]:
    k = top_k or settings.top_k_retrieval
    if not corpus:
        return []

    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = list(zip(scores, corpus))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored[:k]:
        results.append(
            {
                "chunk_id": doc["chunk_id"],
                "filename": doc["filename"],
                "text": doc["text"],
                "score": float(score),
                "source": "sparse",
            }
        )
    return results


def hybrid_retrieval(query: str, top_k: int = None) -> list[dict]:
    k = top_k or settings.top_k_retrieval

    dense_results = dense_retrieval(query, top_k=k * 2)

    sparse_results = sparse_retrieval(query, dense_results, top_k=k)

    seen_ids = set()
    merged = []
    for doc in dense_results + sparse_results:
        if doc["chunk_id"] not in seen_ids:
            seen_ids.add(doc["chunk_id"])
            merged.append(doc)

    return merged[:k * 2]


def rerank_results(query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
    k = top_k or settings.top_k_rerank
    if not candidates:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [[query, doc["text"]] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(candidates, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:k]


def deduplicate_context(docs: list[dict]) -> list[dict]:
    seen_texts = set()
    deduped = []
    for doc in docs:
        normalized = " ".join(doc["text"].lower().split()[:50])
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            deduped.append(doc)
    return deduped


def retrieve_and_rerank(query: str, top_k_rerank: int = None) -> list[dict]:
    candidates = hybrid_retrieval(query)
    deduped = deduplicate_context(candidates)
    reranked = rerank_results(query, deduped, top_k=top_k_rerank)
    return reranked
