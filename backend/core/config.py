from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    cohere_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "resume-screening"
    pinecone_environment: str = "us-east-1"
    langsmith_api_key: str = ""
    langsmith_project: str = "resume-screening-system"
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    backend_url: str = "http://localhost:8000"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    embedding_dimension: int = 1024

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
