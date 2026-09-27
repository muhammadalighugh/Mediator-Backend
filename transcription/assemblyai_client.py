"""
assemblyai_client.py
--------------------
Thin factory around AssemblyAI's AsyncStreamingClient (SDK v1.x, streaming.v3).

Each session gets its OWN AsyncStreamingClient instance (single-use by SDK design).
Callers:
  1. Create an instance: client = AssemblyAIClient(api_key)
  2. Register callbacks BEFORE connecting:
       client.on_turn(handler)   # called with (TurnEvent)
       client.on_error(handler)  # called with (RealTimeError)
  3. await client.connect(sample_rate, session_id=..., num_speakers=...)
  4. await client.stream(pcm_bytes)  — repeatedly, from audio chunks
  5. await client.disconnect()       — graceful teardown

Diarization notes (see AssemblyAI streaming diarization docs):
  - Real-time speaker_labels are only supported on specific streaming
    models (u3-rt-pro, universal-streaming-english,
    universal-streaming-multilingual). We pin SPEECH_MODEL explicitly
    rather than relying on the SDK default, which is not guaranteed to be
    one of these and can silently degrade diarization quality.
  - max_speakers is a real accuracy lever ("Setting this accurately can
    improve assignment accuracy when you know the speaker count in
    advance") — we now pass the declared participant count through.
  - Labels (A, B, …) are assigned by the model's own voice clustering, NOT
    by declared participant order — callers must not assume label "A" ==
    "the first declared speaker". See speaker_mapper.bind_labels() for the
    enrollment-based binding that resolves this correctly.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from assemblyai.streaming.v3.async_client import AsyncStreamingClient
from assemblyai.streaming.v3.models import (
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    SpeakerRevisionEvent,
    TurnEvent,
)

logger = logging.getLogger(__name__)

# Diarization is only supported on these streaming models as of the current
# AssemblyAI docs. Pinning explicitly avoids silently landing on a model
# that ignores speaker_labels.
SPEECH_MODEL = "u3-rt-pro"


class AssemblyAIClient:
    """Per-session wrapper around AsyncStreamingClient."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        # A new AsyncStreamingClient instance per session (SDK is single-use)
        self._client: AsyncStreamingClient = AsyncStreamingClient(api_key=api_key)

        self._turn_handler: Optional[Callable[[TurnEvent], None]] = None
        self._error_handler: Optional[Callable[[RealTimeError], None]] = None
        self._revision_handler: Optional[Callable[[SpeakerRevisionEvent], None]] = None

    # ------------------------------------------------------------------
    # Callback registration (must be called before connect)
    #
    # The SDK calls every registered handler as handler(client, payload).
    # We wrap single-argument user callbacks so callers don't need to know
    # about the internal client reference.
    # ------------------------------------------------------------------

    def on_turn(self, handler: Callable[[TurnEvent], None]) -> None:
        """Register a callback for Turn events (both partial and final)."""
        self._turn_handler = handler
        self._client.on(RealTimeEvents.Turn, lambda _client, turn: handler(turn))

    def on_error(self, handler: Callable[[RealTimeError], None]) -> None:
        """Register a callback for error events."""
        self._error_handler = handler
        self._client.on(RealTimeEvents.Error, lambda _client, err: handler(err))

    def on_speaker_revision(self, handler: Callable[[SpeakerRevisionEvent], None]) -> None:
        """Register a callback for SpeakerRevision events.

        Fired by the SDK after end-of-session offline reclustering — carries
        revised speaker labels for previously-emitted turns, keyed by turn_order.
        """
        self._revision_handler = handler
        self._client.on(
            RealTimeEvents.SpeakerRevision,
            lambda _client, rev: handler(rev),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        sample_rate: int = 16000,
        session_id: str = "",
        num_speakers: int | None = None,
    ) -> None:
        """
        Open the WebSocket connection and start read/write loops.

        num_speakers, when provided, is passed as max_speakers to improve
        diarization assignment accuracy — pass the count of DECLARED
        participants for this session (e.g. len(display_names)).
        """
        params_kwargs: dict = dict(
            sample_rate=sample_rate,
            speech_model=SPEECH_MODEL,  # required for real-time speaker_labels
            speaker_labels=True,        # request per-speaker labels when available
        )
        if num_speakers is not None:
            params_kwargs["max_speakers"] = num_speakers

        params = RealTimeParameters(**params_kwargs)
        await self._client.connect(params)
        logger.info(
            "[RT] [%s] transcriber connected (sample_rate=%d, speech_model=%s, "
            "max_speakers=%s)",
            session_id, sample_rate, SPEECH_MODEL, num_speakers,
        )

    async def stream(self, pcm_bytes: bytes) -> None:
        """Send a chunk of raw 16-bit PCM to AssemblyAI."""
        await self._client.stream(pcm_bytes)

    async def disconnect(self) -> None:
        """Gracefully terminate the session (sends TerminateSession frame)."""
        await self._client.disconnect(terminate=True)