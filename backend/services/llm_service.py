from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from backend.core.config import get_settings

settings = get_settings()

_embedder = None
_llm = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _embedder


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=settings.groq_api_key,
            temperature=0.1,
            max_tokens=8192,
        )
    return _llm


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    embeddings = embedder.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    embedder = get_embedder()
    embedding = embedder.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )[0]
    return embedding.tolist()
