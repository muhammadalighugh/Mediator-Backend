"""
http_routes.py
--------------
REST endpoints:
  POST /upload-evidence/{session_id}   — multipart file upload
  GET  /report/{session_id}            — MediationReport JSON
  GET  /report/{session_id}/pdf        — MediationReport as a downloadable PDF
  GET  /session/{session_id}           — debug dump of session state

All routes require a valid Bearer JWT (from POST /auth/login or /auth/register).
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
import io

from api.auth_routes import get_current_user
from core.session import session_manager
from evidence.ingest import ingest_document
from models.schemas import MediationReport, User

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_current_user)])

_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}


@router.post(
    "/upload-evidence/{session_id}",
    summary="Upload an evidence document for a session",
    status_code=status.HTTP_201_CREATED,
)
async def upload_evidence(
    session_id: str,
    file: Annotated[UploadFile, File(description="txt / md / csv / pdf")],
) -> dict:
    from pathlib import Path

    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found",
        )

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type {ext!r}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    chunks = await ingest_document(session_id, filename, content)

    # Record in session.evidence for bookkeeping
    for chunk in chunks:
        async with session._lock:
            session.evidence.append(chunk)

    return {
        "session_id": session_id,
        "filename": filename,
        "chunks_ingested": len(chunks),
    }


@router.get(
    "/report/{session_id}",
    summary="Get the final MediationReport for a session",
    response_model=MediationReport,
)
async def get_report(session_id: str) -> MediationReport:
    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found",
        )

    report = getattr(session, "report", None)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not yet generated. Send end_session first.",
        )
    return report


@router.get(
    "/report/{session_id}/pdf",
    summary="Download the MediationReport as a PDF",
    response_class=StreamingResponse,
)
async def get_report_pdf(session_id: str) -> StreamingResponse:
    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found",
        )

    report = getattr(session, "report", None)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not yet generated. Send end_session first.",
        )

    # Build speakers list from the session for speaker-id → name mapping
    speakers = list(session.speakers) if session.speakers else []

    try:
        from reporting.pdf_report import build_pdf
        pdf_bytes = build_pdf(report, speakers)
    except Exception as exc:
        logger.exception("PDF generation failed for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {exc}",
        ) from exc

    safe_id = session_id.replace("/", "_").replace("\\", "_")
    filename = f"mediation-report-{safe_id}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@router.get(
    "/session/{session_id}",
    summary="Debug dump of session state",
)
async def get_session_debug(session_id: str) -> dict:
    session = await session_manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id!r} not found",
        )
    # Include evidence chunk texts so callers can verify verbalization ran.
    evidence_chunks = [
        {
            "id": c.id,
            "source_name": c.source_name,
            "chunk_index": c.chunk_index,
            "text": c.text,
        }
        for c in session.evidence
    ]
    return {
        "id": session.id,
        "created_at": session.created_at.isoformat(),
        "status": session.status.value,
        "speakers": [s.model_dump() for s in session.speakers],
        "utterance_count": len(session.utterances),
        "claim_count": len(session.claims),
        "evidence_chunk_count": len(session.evidence),
        "evidence_chunks": evidence_chunks,
        "has_report": hasattr(session, "report") and session.report is not None,
    }
