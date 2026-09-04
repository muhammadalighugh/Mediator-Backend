"""
claim_extractor.py
------------------
Extracts atomic, debatable claims from a list of Utterances using the LLM.

Wiring: every 3 new final utterances trigger an async extraction task.
        If a previous extraction task is still in flight for this session,
        the new trigger is skipped (at-most-one-in-flight guard).

Output schema: {"claims": [...]}
Each claim:
  speaker_id, text, statement_type, verbatim_quote, start_ms, confidence
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING

from models.enums import StatementType
from models.schemas import Claim, Utterance
from reasoning.llm_client import llm_client

if TYPE_CHECKING:
    from core.session import Session
    from fastapi import WebSocket

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# JSON schema for the LLM response
# ------------------------------------------------------------------

_CLAIM_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_id": {"type": ["string", "null"]},
        "text": {"type": "string"},
        "statement_type": {
            "type": "string",
            "enum": ["fact", "claim", "assumption", "evidence_ref", "opinion"],
        },
        "verbatim_quote": {"type": "string"},
        "start_ms": {"type": "integer"},
        "confidence": {"type": "number"},
    },
    "required": [
        "speaker_id", "text", "statement_type",
        "verbatim_quote", "start_ms", "confidence",
    ],
    "additionalProperties": False,
}

CLAIMS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": _CLAIM_ITEM_SCHEMA,
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are an expert mediator and debate analyst. Extract atomic, debatable statements \
from the conversation transcript below.

For each statement, classify it as ONE of:
  fact         — directly verifiable against external sources
  claim        — assertion that needs supporting evidence
  assumption   — unstated premise the speaker treats as self-evident
  evidence_ref — a citation of a specific document, message thread, log, or record
  opinion      — a value judgment, preference, or normative position

Rules:
- ONE assertion per claim. SPLIT compound statements:
  "I paid the rent that month and you still owe me half" →
    TWO claims: "Alex paid the rent in March" + "Sam owes Alex half the March rent".
- text field = the claim restated to stand alone, with ALL pronouns and vague references
  resolved using conversation context:
    "I covered that" (Sam, about internet) → "Sam covered the internet bill"
    "I never agreed to that" (Sam) → "Sam never agreed to cover groceries"
  verbatim_quote stays as the exact spoken words.
- Statements that cite a source, document, log, or message thread — e.g. "Check the
  messages from May 18th", "the log shows it", "look at the receipt" — are EVIDENCE_REF,
  not CLAIM or FACT.
- If two utterances assert the same thing in different words, output the claim ONCE.
- Extract only statements that are substantive and potentially contestable.
- SKIP: filler words, pure questions, greetings, agreement rituals ("okay", "sure", "right").
- Each extracted item must include the verbatim quote exactly as spoken.
- speaker_id must match the id field from the provided speakers list (or null if unknown).
- confidence is your certainty that this is a genuinely debatable statement (0.0–1.0).
- Aim for precision over recall: 3–8 high-quality claims is better than 20 weak ones.

Output only the JSON object — no prose.\
"""


def _build_user_prompt(utterances: list[Utterance], speakers_json: str) -> str:
    lines = [f"Speakers: {speakers_json}", "", "Transcript:"]
    for u in utterances:
        spk = u.speaker_id or "unknown"
        lines.append(f"[{spk}] ({u.start_ms}ms) {u.text}")
    return "\n".join(lines)


async def extract_claims(
    utterances: list[Utterance],
    session: "Session",
) -> list[Claim]:
    """Call LLM, parse response, dedupe and store claims on session."""
    if not utterances:
        return []

    speakers_json = json.dumps(
        [{"id": s.id, "display_name": s.display_name} for s in session.speakers]
    )
    user_prompt = _build_user_prompt(utterances, speakers_json)

    result = await llm_client.complete_json(
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        schema=CLAIMS_SCHEMA,
        max_tokens=2000,
    )

    new_claims: list[Claim] = []
    for item in result.get("claims", []):
        try:
            st = StatementType(item["statement_type"])
        except ValueError:
            st = StatementType.CLAIM

        claim = Claim(
            id=str(uuid.uuid4()),
            speaker_id=item.get("speaker_id"),
            text=item["text"],
            statement_type=st,
            verbatim_quote=item["verbatim_quote"],
            start_ms=int(item.get("start_ms", 0)),
            confidence=float(item.get("confidence", 0.7)),
        )
        added = await session.add_claim(claim)
        if added:
            new_claims.append(claim)

    logger.info(
        "Extracted %d new claims for session %s",
        len(new_claims),
        session.id,
    )
    return new_claims


# ------------------------------------------------------------------
# Live extraction coordinator (used from ws_routes)
# ------------------------------------------------------------------

class ClaimExtractionCoordinator:
    """Tracks utterance count and fires extraction tasks with at-most-one-in-flight."""

    BATCH_THRESHOLD = 3  # trigger after every N new final utterances

    def __init__(self, session: "Session") -> None:
        self._session = session
        self._last_processed: int = 0      # index into session.utterances
        self._in_flight: asyncio.Task | None = None
        self._ws_clients: list["WebSocket"] = []
        self._lock = asyncio.Lock()

    def register_ws(self, ws: "WebSocket") -> None:
        self._ws_clients.append(ws)

    def unregister_ws(self, ws: "WebSocket") -> None:
        try:
            self._ws_clients.remove(ws)
        except ValueError:
            pass

    async def on_new_utterance(self) -> None:
        """Called after every new final Utterance is added to the session."""
        async with self._lock:
            total = len(self._session.utterances)
            new_count = total - self._last_processed
            if new_count < self.BATCH_THRESHOLD:
                return
            if self._in_flight and not self._in_flight.done():
                return  # previous run still in flight — skip
            # Snapshot utterances to process
            batch = list(self._session.utterances[self._last_processed:])
            self._last_processed = total
            self._in_flight = asyncio.create_task(
                self._run(batch), name=f"claim_extract_{self._session.id}"
            )

    async def _run(self, batch: list[Utterance]) -> None:
        try:
            new_claims = await extract_claims(batch, self._session)
            if new_claims:
                await self._broadcast_claims()
        except Exception as exc:
            logger.exception("Claim extraction failed: %s", exc)

    async def _broadcast_claims(self) -> None:
        payload = json.dumps({
            "type": "claims_updated",
            "claims": [c.model_dump() for c in self._session.claims],
        })
        dead = []
        for ws in self._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister_ws(ws)
