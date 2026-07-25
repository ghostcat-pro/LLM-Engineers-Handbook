import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from qdrant_client.http import exceptions
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, VectorParams
from qdrant_client.models import PointStruct

from llm_engineering.application.networks.embeddings import EmbeddingModelSingleton
from llm_engineering.infrastructure.db.qdrant import connection

SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".txt", ".md"}
LOCAL_SOURCE_COLLECTION = "local_sources"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180


@dataclass(frozen=True)
class LocalSourceDocument:
    path: Path
    source_type: Literal["pdf", "txt", "md"]
    text: str


class LocalSourceChunk(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)
    source_path: str
    source_name: str
    source_type: str
    chunk_index: int
    content: str
    embedding: list[float]

    def to_point(self) -> PointStruct:
        payload = self.model_dump()
        point_id = str(payload.pop("id"))
        vector = payload.pop("embedding")

        return PointStruct(id=point_id, vector=vector, payload=payload)


def extract_source_text(path: Path) -> LocalSourceDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(f"Unsupported source type: {path}")

    if suffix == ".pdf":
        text = _extract_pdf_text(path)
        source_type = "pdf"
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        source_type = "md" if suffix == ".md" else "txt"

    text = normalize_text(text)
    return LocalSourceDocument(path=path, source_type=source_type, text=text)


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n\n".join(pages)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to 0 and smaller than chunk_size.")

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(paragraph, chunk_size=chunk_size, overlap=overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


def load_local_source_documents(source_dir: Path) -> list[LocalSourceDocument]:
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created local sources directory: {source_dir}")
        return []

    documents = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES:
            documents.append(extract_source_text(path))

    return documents


def build_local_source_chunks(
    documents: list[LocalSourceDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[LocalSourceChunk]:
    embedding_model = EmbeddingModelSingleton()
    chunks = []

    for document in documents:
        text_chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        if not text_chunks:
            logger.warning(f"No text chunks extracted from {document.path}.")
            continue

        embeddings = embedding_model(text_chunks, to_list=True)
        for index, (content, embedding) in enumerate(zip(text_chunks, embeddings, strict=True)):
            chunks.append(
                LocalSourceChunk(
                    source_path=str(document.path),
                    source_name=document.path.name,
                    source_type=document.source_type,
                    chunk_index=index,
                    content=content,
                    embedding=embedding,
                )
            )

    return chunks


def reset_local_source_collection() -> None:
    try:
        connection.delete_collection(collection_name=LOCAL_SOURCE_COLLECTION)
    except exceptions.UnexpectedResponse:
        logger.info(f"Collection '{LOCAL_SOURCE_COLLECTION}' did not need deletion.")


def ensure_local_source_collection() -> None:
    embedding_size = EmbeddingModelSingleton().embedding_size
    try:
        connection.get_collection(collection_name=LOCAL_SOURCE_COLLECTION)
    except exceptions.UnexpectedResponse:
        connection.create_collection(
            collection_name=LOCAL_SOURCE_COLLECTION,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )


def upsert_local_source_chunks(chunks: list[LocalSourceChunk]) -> None:
    ensure_local_source_collection()
    if not chunks:
        return

    connection.upsert(collection_name=LOCAL_SOURCE_COLLECTION, points=[chunk.to_point() for chunk in chunks])


def search_local_sources(
    query: str,
    limit: int = 5,
    *,
    source_name: str | None = None,
    exclude_source_name: str | None = None,
) -> list[dict[str, Any]]:
    embedding = EmbeddingModelSingleton()([query], to_list=True)[0]
    query_filter = _build_source_filter(source_name=source_name, exclude_source_name=exclude_source_name)
    try:
        records = connection.search(
            collection_name=LOCAL_SOURCE_COLLECTION,
            query_vector=embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
    except exceptions.UnexpectedResponse:
        return []

    return [{"score": record.score, **(record.payload or {})} for record in records]


def _build_source_filter(source_name: str | None, exclude_source_name: str | None) -> Filter | None:
    must = []
    must_not = []
    if source_name:
        must.append(FieldCondition(key="source_name", match=MatchValue(value=source_name)))
    if exclude_source_name:
        must_not.append(FieldCondition(key="source_name", match=MatchValue(value=exclude_source_name)))

    if not must and not must_not:
        return None

    return Filter(must=must or None, must_not=must_not or None)


def save_source_manifest(source_dir: Path, output_file: Path, documents: list[LocalSourceDocument]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dir": str(source_dir),
        "documents": [
            {
                "path": str(document.path),
                "source_type": document.source_type,
                "characters": len(document.text),
            }
            for document in documents
        ],
    }
    output_file.write_text(json.dumps(manifest, indent=2))
