"""
session_routes.py
-----------------
Atlas-backed session persistence endpoints.

POST /sessions/persist   — internal helper called after report is built
GET  /sessions/{email}   — list up to 20 past sessions for a user

If MongoDB is unavailable, GET returns an empty list (no 503) so the
/start page degrades silently.  The persist helper also no-ops silently
so a Mongo outage never interrupts the live session flow.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter

from core import database
from models.schemas import MediationReport, SessionSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])

_COLLECTION = "sessions"


# ---------------------------------------------------------------------------
# Internal: called by ws_routes after a report is built
# ---------------------------------------------------------------------------

async def persist_session(
    report: MediationReport,
    speakers: list[str],
    user_email: Optional[str] = None,
) -> None:
    """
    Write a SessionSummary document to Atlas.

    This is fire-and-forget — caller wraps in asyncio.create_task so it never
    blocks the WebSocket response path.  Silently no-ops when Mongo is down.

    speakers: the real display names from session.speakers (e.g. ["Maya", "Daniel"]).
    When names were never bound, the caller passes positional fallbacks like
    "Speaker 1" — those are stored as-is. If the list is empty we store a
    generic "N participants" label instead.
    """
    if database.db is None:
        return

    # Normalise empty / missing names: store a participant count label so the
    # past-sessions card never shows raw "Speaker 1 · Speaker 2".
    if not speakers:
        display_speakers: list[str] = []
    else:
        display_speakers = speakers

    report_kind = getattr(report, "report_kind", "dispute")

    try:
        col = database.db[_COLLECTION]
        doc = {
            "session_id": report.session_id,
            "user_email": user_email,
            "created_at": datetime.now(tz=timezone.utc),
            "speakers": display_speakers,
            "claim_count": len(report.claims),
            "summary": report.summary,
            "report_kind": report_kind,
            "report": report.model_dump(),
        }
        await col.update_one(
            {"session_id": report.session_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info(
            "[DB] persisted session session_id=%s user=%s kind=%s",
            report.session_id, user_email or "guest", report_kind,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DB] session persist failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# GET /sessions/{email}
# ---------------------------------------------------------------------------

@router.get(
    "/{email}",
    response_model=list[SessionSummary],
    summary="List past sessions for a user (up to 20, newest first)",
)
async def list_sessions(email: str) -> list[SessionSummary]:
    if database.db is None:
        return []

    try:
        col = database.db[_COLLECTION]
        cursor = (
            col.find({"user_email": email.strip().lower()})
               .sort("created_at", -1)
               .limit(20)
        )
        docs = await cursor.to_list(length=20)
        return [
            SessionSummary(
                session_id=d["session_id"],
                user_email=d.get("user_email"),
                created_at=d["created_at"],
                speakers=d.get("speakers", []),
                claim_count=d.get("claim_count", 0),
                summary=d.get("summary", ""),
                report_kind=d.get("report_kind", "dispute"),
                report=d.get("report", {}),
            )
            for d in docs
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DB] list_sessions failed (non-fatal): %s", exc)
        return []
