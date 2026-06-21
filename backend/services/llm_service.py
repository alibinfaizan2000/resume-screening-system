import cohere
from langchain_groq import ChatGroq
from backend.core.config import get_settings

settings = get_settings()

_cohere_client = None
_llm = None


def get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
    return _cohere_client


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
    client = get_cohere_client()
    response = client.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return response.embeddings


def embed_query(query: str) -> list[float]:
    client = get_cohere_client()
    response = client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return response.embeddings[0]
