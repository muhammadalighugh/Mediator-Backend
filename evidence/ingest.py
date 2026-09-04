"""
ingest.py
---------
Ingest evidence documents into a session's vector store.

Supported formats: .txt  .md  .csv  .pdf
Chunking: ~500 chars with 50-char overlap (character-based, simple and fast).
"""

from __future__ import annotations

import io
import logging
import re
import uuid
from pathlib import Path

from models.schemas import EvidenceChunk
from evidence.vector_store import add_chunks

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _extract_text(filename: str, content: bytes) -> str:
    """Return plain text from file bytes. Supports txt/md/csv/pdf."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # lazy — only needed for PDFs
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required for PDF ingestion. "
                "Install it with: pip install pypdf"
            ) from exc
        reader = PdfReader(io.BytesIO(content))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)

    # txt / md / csv — decode as UTF-8 with fallback to latin-1
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP  # step back by overlap for next chunk
    return chunks


def ingest_document(
    session_id: str,
    filename: str,
    content: bytes,
) -> list[EvidenceChunk]:
    """Parse, chunk, embed, and store a document. Returns the EvidenceChunks."""
    raw_text = _extract_text(filename, content)
    if not raw_text.strip():
        logger.warning("Document %r produced no text — skipping", filename)
        return []

    text_chunks = _chunk_text(raw_text)

    # Safety guard: ensure chunk text never contains a [source] bracket suffix.
    # Source attribution belongs only in EvidenceChunk.source_name.
    _bracket_suffix = re.compile(r'\s*\[[^\]]+\]\s*$')

    chunks: list[EvidenceChunk] = [
        EvidenceChunk(
            id=str(uuid.uuid4()),
            session_id=session_id,
            source_name=filename,
            chunk_index=i,
            text=_bracket_suffix.sub("", chunk),
        )
        for i, chunk in enumerate(text_chunks)
    ]

    add_chunks(session_id, chunks)
    logger.info(
        "Ingested %r → %d chunks for session %s",
        filename,
        len(chunks),
        session_id,
    )
    return chunks
