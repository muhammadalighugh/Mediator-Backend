"""
vector_store.py
---------------
Chroma wrapper for per-session evidence collections.

Embeddings: sentence-transformers "all-MiniLM-L6-v2" (local, zero API cost).
Each session gets its own Chroma collection persisted at sessions/{id}/chroma.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from models.schemas import EvidenceChunk

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """Lazily initialised — downloads model on first call."""
    logger.info("Loading embedding model %s …", _EMBEDDING_MODEL)
    return SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)


def _chroma_path(session_id: str) -> str:
    import os
    return os.path.join("sessions", session_id, "chroma")


def _get_collection(session_id: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_chroma_path(session_id))
    return client.get_or_create_collection(
        name="evidence",
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(session_id: str, chunks: list[EvidenceChunk]) -> None:
    """Upsert evidence chunks into the session's Chroma collection."""
    if not chunks:
        return
    collection = _get_collection(session_id)
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source_name": c.source_name,
                "chunk_index": c.chunk_index,
                "session_id": c.session_id,
            }
            for c in chunks
        ],
    )
    logger.info(
        "Upserted %d evidence chunks for session %s", len(chunks), session_id
    )


def query(session_id: str, text: str, k: int = 4) -> list[EvidenceChunk]:
    """Retrieve the top-k evidence chunks most similar to `text`."""
    collection = _get_collection(session_id)
    count = collection.count()
    if count == 0:
        return []

    actual_k = min(k, count)
    results = collection.query(query_texts=[text], n_results=actual_k)

    chunks: list[EvidenceChunk] = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for chunk_id, doc, meta in zip(ids, docs, metas):
        chunks.append(
            EvidenceChunk(
                id=chunk_id,
                session_id=meta.get("session_id", session_id),
                source_name=meta.get("source_name", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                text=doc,
            )
        )
    return chunks
