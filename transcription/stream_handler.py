"""
stream_handler.py
-----------------
Bridges AssemblyAI TurnEvents → Utterance models → WebSocket broadcasts.

The AssemblyAI streaming v3 SDK emits TurnEvent for every "turn" update.
  - end_of_turn == False  → partial (in-progress transcription)
  - end_of_turn == True   → final  (committed, stored as Utterance)

Speaker label handling:
  PATH 1 — turn.speaker_label is a non-empty string:
      resolve it via SpeakerMapper to get speaker_id.
  PATH 2 — turn.speaker_label is None/empty:
      store speaker_id=None; a later async diarization step over the saved
      WAV will backfill attribution.

Garbage filter (final transcripts only — partials are never filtered):
  A final transcript is discarded (not stored, not broadcast) if ANY of:
    - fewer than 3 words
    - purely digits/punctuation once non-alphanumeric characters are stripped
      (e.g. "16124", "...", "12, 34")
    - a single word that matches a known speaker's display name (e.g. "Alex.")
  The single-word-name check is a subset of the <3-word check today, but is
  kept as its own explicit condition so it still holds if the word-count
  threshold ever changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Set

from assemblyai.streaming.v3.models import TurnEvent

from models.schemas import Utterance
from transcription.speaker_mapper import SpeakerMapper

if TYPE_CHECKING:
    from core.session import Session
    from fastapi.websockets import WebSocket

logger = logging.getLogger(__name__)

# Strips anything that isn't a letter/digit/underscore — used both to test
# "purely digits/punctuation" and to normalize a single word before comparing
# it against speaker display names.
_NON_ALNUM_RE = re.compile(r"[^\w]", re.UNICODE)


def _is_purely_digits_or_punct(text: str) -> bool:
    """True for ASR noise like '16124', '...', or '12, 34' — i.e. nothing
    but digits/punctuation once non-alphanumeric characters are stripped."""
    stripped = _NON_ALNUM_RE.sub("", text)
    return stripped == "" or stripped.isdigit()


def _normalize_word(word: str) -> str:
    return _NON_ALNUM_RE.sub("", word).lower()


class StreamHandler:
    """Handles TurnEvents for a single session and broadcasts to WS clients."""

    def __init__(
        self,
        session: "Session",
        speaker_mapper: SpeakerMapper,
    ) -> None:
        self._session = session
        self._mapper = speaker_mapper
        # Registered WebSocket connections for this session
        self._ws_clients: Set["WebSocket"] = set()
        self._ws_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # WebSocket client registry
    # ------------------------------------------------------------------

    async def add_client(self, ws: "WebSocket") -> None:
        async with self._ws_lock:
            self._ws_clients.add(ws)

    async def remove_client(self, ws: "WebSocket") -> None:
        async with self._ws_lock:
            self._ws_clients.discard(ws)

    # ------------------------------------------------------------------
    # AssemblyAI callback (registered via assemblyai_client.on_turn)
    # ------------------------------------------------------------------

    def handle_turn(self, turn: TurnEvent) -> None:
        """Synchronous callback invoked by the SDK; schedules async work."""
        # get_running_loop() is required here — get_event_loop() can return a
        # different or closed loop when called from inside an SDK asyncio task,
        # causing create_task to silently drop the coroutine.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_turn(turn))
        except RuntimeError:
            # No running loop (shouldn't happen in normal operation)
            asyncio.run(self._process_turn(turn))

    # ------------------------------------------------------------------
    # Garbage filter helpers
    # ------------------------------------------------------------------

    def _known_speaker_names(self) -> set[str]:
        """Normalized display names of speakers known to this session."""
        return {
            _normalize_word(name)
            for name in (
                getattr(s, "display_name", None)
                for s in getattr(self._session, "speakers", [])
            )
            if name
        }

    def _is_garbage_final(self, text: str) -> bool:
        text = (text or "").strip()
        words = text.split()

        if len(words) < 3:
            # Covers empty/short fragments AND single-word speaker names
            # (e.g. "Alex.") since a name match is always a single word.
            return True

        if _is_purely_digits_or_punct(text):
            return True

        if len(words) == 1 and _normalize_word(words[0]) in self._known_speaker_names():
            return True

        return False

    # ------------------------------------------------------------------
    # Internal async processing
    # ------------------------------------------------------------------

    async def _process_turn(self, turn: TurnEvent) -> None:
        sid = self._session.id
        raw_label = getattr(turn, "speaker_label", None)
        is_final = turn.end_of_turn

        # ---- Speaker resolution ----------------------------------------
        # PATH 1: SDK provided a speaker label → resolve via mapper
        # PATH 2: no label → None; post-session WAV diarization will backfill
        speaker_id: str | None = self._mapper.resolve(raw_label)

        # start_ms / end_ms: derive from Word timestamps when available
        words = turn.words or []
        start_ms = words[0].start if words else 0
        end_ms = words[-1].end if words else 0

        text_preview = (turn.transcript or "")[:40]

        if is_final:
            label_repr = repr(raw_label) if raw_label is not None else "MISSING"
            logger.info(
                "[RT] [%s] FINAL received: speaker=%s text=%r",
                sid, label_repr, text_preview,
            )

            # ---- Garbage filter ----------------------------------------
            if self._is_garbage_final(turn.transcript):
                words_count = len((turn.transcript or "").split())
                reason = (
                    f"len={words_count}<3" if words_count < 3
                    else "digits/punct" if _is_purely_digits_or_punct(turn.transcript)
                    else "bare-name"
                )
                logger.info(
                    "[FILTER] [%s] garbage filter: dropped %r (reason: %s)",
                    sid, text_preview, reason,
                )
                return

            logger.info("[FILTER] [%s] garbage filter: kept %r", sid, text_preview)

            # ---- Persist as Utterance ----------------------------------
            utterance = Utterance(
                id=str(uuid.uuid4()),
                session_id=sid,
                speaker_id=speaker_id,
                text=turn.transcript,
                is_final=True,
                start_ms=start_ms,
                end_ms=end_ms,
                turn_order=turn.turn_order,
            )
            await self._session.add_utterance(utterance)
            logger.info(
                "[UTT] [%s] stored utterance: speaker=%s start=%d",
                sid, speaker_id, start_ms,
            )

            await self._broadcast(
                {
                    "type": "transcript_final",
                    "speaker_id": speaker_id,
                    "text": turn.transcript,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "utterance_id": utterance.id,
                }
            )
        else:
            logger.info("[RT] [%s] partial received: %r", sid, text_preview)
            # ---- Partial transcript: broadcast only, do NOT store --------
            await self._broadcast({
                "type": "transcript_partial",
                "speaker_id": speaker_id,
                "text": turn.transcript,
                "start_ms": start_ms,
                "end_ms": end_ms,
            })

    async def _broadcast(self, message: dict) -> None:
        """Send a JSON frame to all connected WebSocket clients."""
        text = json.dumps(message)
        async with self._ws_lock:
            clients = list(self._ws_clients)

        logger.info(
            "[BCAST] [%s] broadcast to %d client(s): %s",
            self._session.id, len(clients), message.get("type"),
        )

        dead: list = []
        for ws in clients:
            try:
                await ws.send_text(text)
            except Exception:
                # Client disconnected mid-broadcast — remove on next cleanup
                dead.append(ws)

        if dead:
            async with self._ws_lock:
                for ws in dead:
                    self._ws_clients.discard(ws)