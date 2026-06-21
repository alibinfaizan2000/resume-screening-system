import fitz  # PyMuPDF
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.core.config import get_settings
from backend.services.llm_service import embed_texts
from backend.services.pinecone_service import upsert_vectors, delete_by_filename

settings = get_settings()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len,
)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def chunk_resume(text: str, filename: str) -> list[dict]:
    chunks = splitter.split_text(text)
    result = []
    for i, chunk in enumerate(chunks):
        result.append(
            {
                "chunk_id": f"{filename}__chunk_{i}",
                "text": chunk,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )
    return result


def index_resume(pdf_bytes: bytes, filename: str) -> int:
    delete_by_filename(filename)

    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        raise ValueError(f"No text extracted from {filename}")

    chunks = chunk_resume(text, filename)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append(
            {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "text": chunk["text"][:1000],
                },
            }
        )

    return upsert_vectors(vectors)
