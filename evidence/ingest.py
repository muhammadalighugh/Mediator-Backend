"""
ingest.py
---------
Ingest evidence documents into a session's vector store.

Supported formats: .txt  .md  .csv  .pdf

CSV handling:
  - Triggered either by a .csv extension OR by content sniffing: if at least
    3 of the first 5 non-empty lines look like data rows (first cell a
    YYYY-MM-DD date, third cell numeric), the file is routed through the
    verbalization path regardless of extension. This covers users pasting
    CSV rows into a .txt file.
  - Header row is auto-detected and skipped.
  - Each data row is verbalized by the LLM into a natural-language sentence, e.g.
      "2024-03-03,rent,1800,Alex,March rent"
      → "Alex paid the March rent of $1800 on March 3, 2024."
  - If the LLM call fails, the raw row is stored and a WARNING is logged.

Message threads (.txt / .md containing lines like "[Name]: …") pass through
character-chunking unchanged.

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

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CSV detection: by extension OR by content sniffing
# ---------------------------------------------------------------------------

# Heuristic: a line is a CSV data row if it contains at least one comma and
# does NOT look like a message-thread line ("[Speaker]: text").
_MESSAGE_LINE_RE = re.compile(r"^\s*\[.+?\]\s*:")

# A header row likely has no numeric-looking cell and its cells are all words.
_ALL_ALPHA_RE = re.compile(r"^[A-Za-z_ /\-]+$")

# Content-sniffing: first cell looks like an ISO date, third cell is numeric.
_DATE_CELL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Minimum hits (out of the first 5 non-empty lines) to treat a file as CSV-like
# regardless of its extension.
_SNIFF_SAMPLE_SIZE = 5
_SNIFF_MIN_HITS = 3


def _looks_like_header(row: str) -> bool:
    """Return True if every comma-separated cell looks like a column name."""
    cells = [c.strip() for c in row.split(",")]
    if len(cells) < 2:
        return False
    return all(_ALL_ALPHA_RE.match(c) for c in cells if c)


def _is_numeric_cell(cell: str) -> bool:
    """True if a cell looks like a plain number (optionally $-prefixed / with
    thousands separators, e.g. "1800", "$1,800", "120.50")."""
    cell = cell.strip().lstrip("$").replace(",", "")
    if not cell:
        return False
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _looks_like_data_row(line: str) -> bool:
    """True if this line's shape matches an expense-style CSV data row:
    first cell a YYYY-MM-DD date, third cell numeric."""
    cells = [c.strip() for c in line.split(",")]
    if len(cells) < 3:
        return False
    return bool(_DATE_CELL_RE.match(cells[0])) and _is_numeric_cell(cells[2])


def _is_csv_file(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".csv"


def _sniff_csv_like(text: str) -> bool:
    """
    Detect CSV-shaped content regardless of file extension.

    Returns True if at least _SNIFF_MIN_HITS of the first _SNIFF_SAMPLE_SIZE
    non-empty lines look like data rows (see _looks_like_data_row). This
    catches users pasting raw expense rows into a .txt/.md file, where the
    extension-only check would otherwise let raw, unverbalized rows into the
    evidence corpus and break quote validation downstream.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sample = lines[:_SNIFF_SAMPLE_SIZE]
    if not sample:
        return False
    hits = sum(1 for ln in sample if _looks_like_data_row(ln))
    return hits >= _SNIFF_MIN_HITS


def _csv_data_rows(text: str) -> list[str]:
    """
    Return the data rows of a CSV (non-empty, non-header lines).
    The first line is skipped if it looks like a header row.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    start = 1 if _looks_like_header(lines[0]) else 0
    return lines[start:]


_VERBALIZE_SYSTEM = """\
You are a financial-records analyst. Convert each CSV row you receive into a single, \
clear natural-language sentence that preserves every data value exactly — names, dates, \
amounts, and descriptions. Do not add information that is not in the row.

Examples:
  "2024-03-03,rent,1800,Alex,March rent"
  → "Alex paid the March rent of $1800 on March 3, 2024."

  "2024-04-15,groceries,120,Sam,weekly groceries"
  → "Sam paid $120 for weekly groceries on April 15, 2024."

Return only the JSON object — no prose.\
"""

_VERBALIZE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["sentences"],
    "additionalProperties": False,
}


async def _verbalize_csv_rows(rows: list[str]) -> list[str]:
    """
    Call the LLM to convert CSV rows into natural-language sentences.

    Returns a list of sentences parallel to *rows*.
    If the LLM call fails, raises the caught exception so the caller can fall back.
    """
    from reasoning.llm_client import llm_client  # local import avoids circular dep

    # Send all rows in a single call; the model returns one sentence per row.
    user_prompt = "\n".join(f'"{row}"' for row in rows)

    result = await llm_client.complete_json(
        system=_VERBALIZE_SYSTEM,
        user=user_prompt,
        schema=_VERBALIZE_SCHEMA,
        max_tokens=max(800, len(rows) * 80),
    )
    sentences: list[str] = result.get("sentences", [])

    # If the LLM returned fewer sentences than rows, pad with the raw rows.
    if len(sentences) < len(rows):
        logger.warning(
            "Verbalization returned %d sentences for %d rows — padding with raw rows",
            len(sentences),
            len(rows),
        )
        sentences += rows[len(sentences):]

    return sentences[:len(rows)]


# ---------------------------------------------------------------------------
# Character chunking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Safety guard: strip any [source] bracket suffix that crept in
# ---------------------------------------------------------------------------

_BRACKET_SUFFIX = re.compile(r'\s*\[[^\]]+\]\s*$')


def _clean(text: str) -> str:
    return _BRACKET_SUFFIX.sub("", text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest_document(
    session_id: str,
    filename: str,
    content: bytes,
) -> list[EvidenceChunk]:
    """
    Parse, (optionally verbalize), chunk, embed, and store a document.
    Returns the EvidenceChunks created.

    Verbalization triggers on a .csv extension OR on content sniffing (see
    _sniff_csv_like) — so raw expense rows pasted into a .txt/.md file still
    get verbalized instead of landing in the evidence corpus verbatim. If the
    LLM call fails, raw rows are stored and a WARNING is logged. Everything
    else is chunked as plain text.
    """
    raw_text = _extract_text(filename, content)
    if not raw_text.strip():
        logger.warning("Document %r produced no text — skipping", filename)
        return []

    # -----------------------------------------------------------------------
    # CSV (by extension or by content sniffing): verbalize each data row
    # -----------------------------------------------------------------------
    is_csv_ext = _is_csv_file(filename)
    is_sniffed_csv = (not is_csv_ext) and _sniff_csv_like(raw_text)

    if is_csv_ext or is_sniffed_csv:
        if is_sniffed_csv:
            logger.info(
                "Content-sniffed CSV-like data in %r (ext=%s) — routing "
                "through verbalization",
                filename,
                Path(filename).suffix or "<none>",
            )

        rows = _csv_data_rows(raw_text)
        if not rows:
            logger.warning("CSV-like file %r has no data rows — skipping", filename)
            return []

        try:
            sentences = await _verbalize_csv_rows(rows)
            logger.info(
                "Verbalized %d CSV rows from %r for session %s",
                len(sentences),
                filename,
                session_id,
            )
        except Exception as exc:
            logger.warning(
                "CSV verbalization failed for %r (%s) — storing raw rows",
                filename,
                exc,
            )
            sentences = rows  # fall back to raw rows

        # Each verbalized sentence becomes its own chunk (no overlap needed —
        # sentences are already atomic).
        chunks: list[EvidenceChunk] = [
            EvidenceChunk(
                id=str(uuid.uuid4()),
                session_id=session_id,
                source_name=filename,
                chunk_index=i,
                text=_clean(sentence),
            )
            for i, sentence in enumerate(sentences)
            if sentence.strip()
        ]

    # -----------------------------------------------------------------------
    # Non-CSV: chunk as plain text (message threads, markdown, PDFs, etc.)
    # -----------------------------------------------------------------------
    else:
        text_chunks = _chunk_text(raw_text)
        chunks = [
            EvidenceChunk(
                id=str(uuid.uuid4()),
                session_id=session_id,
                source_name=filename,
                chunk_index=i,
                text=_clean(chunk),
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