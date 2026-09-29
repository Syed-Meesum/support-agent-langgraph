"""Vector retrieval over the small policy knowledge base.

Real embeddings, computed locally with Chroma's bundled ONNX model
(all-MiniLM-L6-v2, 384 dimensions). No embedding API key is needed, so the
project keeps running with only ANTHROPIC_API_KEY + DATABASE_URL set, per the
committed `.env.example` - which matters, because the LLM here (Claude) has no
embeddings endpoint of its own.

Storage is a local persistent Chroma collection under backend/.chroma for now.
`search(query, top_k)` is the only thing callers depend on, so the store can be
swapped for Supabase pgvector later. Note for that swap: the declared
`knowledge_chunks.embedding` is `vector(1536)`, an OpenAI-shaped dimension -
moving to pgvector means either widening the schema to vector(384) or changing the
embedder with it.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import chromadb

from app.rag import _chunking
from app.tools.errors import ToolError
from app.tools.models import KnowledgeHit

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

CHROMA_PATH_ENV = "CHROMA_PATH"
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parents[2] / ".chroma"
COLLECTION_NAME = "knowledge_base"

# Cosine similarity floor. Measured with this model: genuinely relevant policy
# chunks score ~0.5-0.85, unrelated text ~0.05-0.15. Below the floor we return
# nothing rather than handing the graph irrelevant context to cite.
MIN_SIMILARITY_ENV = "KB_MIN_SIMILARITY"
DEFAULT_MIN_SIMILARITY = 0.25


def _chroma_path() -> Path:
    return Path(os.environ.get(CHROMA_PATH_ENV, DEFAULT_CHROMA_PATH))


def _min_similarity() -> float:
    raw = os.environ.get(MIN_SIMILARITY_ENV)
    if raw is None:
        return DEFAULT_MIN_SIMILARITY
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_MIN_SIMILARITY


def chunk_id(source: str, position: int, content: str) -> str:
    """The stable id of one chunk. Content-addressed, so an edit retires the old one.

    Public because `app/db/seed.py` writes the same chunks to `knowledge_chunks`
    and the two must agree on what a chunk is.
    """
    digest = hashlib.sha1(f"{source}:{position}:{content}".encode("utf-8")).hexdigest()
    return digest[:16]


def knowledge_chunks() -> list[_chunking.Chunk]:
    """Every chunk the knowledge base currently contains, in file order."""
    chunks: list[_chunking.Chunk] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        chunks.extend(_chunking.split_chunks(path))
    return chunks


def _sync(collection) -> None:
    """Add new chunks and drop stale ones.

    Chunk ids are content hashes, so editing a doc adds a new chunk and retires
    the old one; unchanged docs are never re-embedded.
    """
    desired = {
        chunk_id(chunk.source, chunk.position, chunk.content): chunk
        for chunk in knowledge_chunks()
    }
    existing = set(collection.get(include=[])["ids"])

    missing = [cid for cid in desired if cid not in existing]
    if missing:
        collection.add(
            ids=missing,
            documents=[desired[cid].content for cid in missing],
            metadatas=[
                {"source": desired[cid].source, "position": desired[cid].position}
                for cid in missing
            ],
        )

    stale = existing - set(desired)
    if stale:
        collection.delete(ids=sorted(stale))


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(_chroma_path()))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    _sync(collection)
    return collection


def reload_index() -> None:
    """Drop process-wide caches. Used by tests that re-point CHROMA_PATH."""
    _collection.cache_clear()


def indexed_chunk_count() -> int:
    """How many chunks are in the index."""
    return _collection().count()


def search(query: str, top_k: int = 3) -> list[KnowledgeHit]:
    """Return the best-matching knowledge chunks for a query, best first."""
    if not isinstance(query, str) or not query.strip():
        raise ToolError("invalid_argument", "query must be a non-empty string")
    if top_k < 1:
        raise ToolError("invalid_argument", "top_k must be at least 1", top_k=top_k)

    collection = _collection()
    available = collection.count()
    if available == 0:
        return []

    try:
        result = collection.query(
            query_texts=[query],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # embedding model unavailable, store unwritable, ...
        raise ToolError(
            "knowledge_base_unavailable",
            f"knowledge base query failed: {exc}",
        ) from exc

    threshold = _min_similarity()
    hits: list[KnowledgeHit] = []
    for document, distance, metadata in zip(
        result["documents"][0], result["distances"][0], result["metadatas"][0]
    ):
        score = 1.0 - float(distance)  # cosine space: distance = 1 - similarity
        if score < threshold:
            continue
        hits.append(
            KnowledgeHit(
                content=document,
                source=str((metadata or {}).get("source", "")),
                score=round(score, 6),
            )
        )
    return hits
