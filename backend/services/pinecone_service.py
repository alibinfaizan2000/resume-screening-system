import os
import time
from pinecone import Pinecone, ServerlessSpec
from backend.core.config import get_settings

settings = get_settings()
_pinecone_client = None
_index = None


def get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def get_pinecone_index():
    global _index
    if _index is not None:
        return _index

    pc = get_pinecone_client()
    index_name = settings.pinecone_index_name

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    _index = pc.Index(index_name)
    return _index


def upsert_vectors(vectors: list[dict]) -> int:
    index = get_pinecone_index()
    batch_size = 100
    upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        upserted += len(batch)
    return upserted


def query_vectors(
    query_vector: list[float],
    top_k: int = 10,
    filter_dict: dict = None,
) -> list[dict]:
    index = get_pinecone_index()
    kwargs = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
    if filter_dict:
        kwargs["filter"] = filter_dict
    result = index.query(**kwargs)
    return result.get("matches", [])


def delete_by_filename(filename: str):
    try:
        index = get_pinecone_index()
        index.delete(filter={"filename": filename})
    except Exception:
        pass


def get_index_stats() -> dict:
    index = get_pinecone_index()
    return index.describe_index_stats()


def list_indexed_resumes() -> list[dict]:
    stats = get_index_stats()
    namespaces = stats.get("namespaces", {})
    return namespaces
