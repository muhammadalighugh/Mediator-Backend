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
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Callable, Set

from assemblyai.streaming.v3.models import TurnEvent

from models.schemas import Utterance
from transcription.speaker_mapper import SpeakerMapper

if TYPE_CHECKING:
    from core.session import Session
    from fastapi.websockets import WebSocket


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
    # Internal async processing
    # ------------------------------------------------------------------

    async def _process_turn(self, turn: TurnEvent) -> None:
        # ---- Speaker resolution ----------------------------------------
        # PATH 1: SDK provided a speaker label
        # PATH 2: no label → defer to post-session diarization
        speaker_id: str | None = self._mapper.resolve(
            getattr(turn, "speaker_label", None)
        )

        # start_ms / end_ms: derive from Word timestamps when available
        words = turn.words or []
        start_ms = words[0].start if words else 0
        end_ms = words[-1].end if words else 0

        is_final = turn.end_of_turn

        payload: dict = {
            "speaker_id": speaker_id,
            "text": turn.transcript,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

        if is_final:
            # ---- Garbage filter: discard final transcripts that are
            #      under 3 words, purely numeric, or a bare name/punctuation.
            text = (turn.transcript or "").strip()
            text_words = text.split()
            if (
                len(text_words) < 3
                or text.replace(".", "").replace(",", "").isdigit()
            ):
                # Silently drop — do not store or broadcast
                return

            # ---- Final transcript: persist as Utterance -----------------
            utterance = Utterance(
                id=str(uuid.uuid4()),
                session_id=self._session.id,
                speaker_id=speaker_id,
                text=turn.transcript,
                is_final=True,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            await self._session.add_utterance(utterance)

            await self._broadcast(
                {
                    "type": "transcript_final",
                    **payload,
                    "utterance_id": utterance.id,
                }
            )
        else:
            # ---- Partial transcript: broadcast only, do NOT store --------
            await self._broadcast({"type": "transcript_partial", **payload})

    async def _broadcast(self, message: dict) -> None:
        """Send a JSON frame to all connected WebSocket clients."""
        text = json.dumps(message)
        async with self._ws_lock:
            clients = list(self._ws_clients)

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
