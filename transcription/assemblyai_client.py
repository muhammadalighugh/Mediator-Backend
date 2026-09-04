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
  3. await client.connect(sample_rate)
  4. await client.stream(pcm_bytes)  — repeatedly, from audio chunks
  5. await client.disconnect()       — graceful teardown
"""

from __future__ import annotations

from typing import Callable, Optional

from assemblyai.streaming.v3.async_client import AsyncStreamingClient
from assemblyai.streaming.v3.models import (
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    TurnEvent,
)


class AssemblyAIClient:
    """Per-session wrapper around AsyncStreamingClient."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        # A new AsyncStreamingClient instance per session (SDK is single-use)
        self._client: AsyncStreamingClient = AsyncStreamingClient(api_key=api_key)

        self._turn_handler: Optional[Callable[[TurnEvent], None]] = None
        self._error_handler: Optional[Callable[[RealTimeError], None]] = None

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, sample_rate: int = 16000) -> None:
        """Open the WebSocket connection and start read/write loops."""
        params = RealTimeParameters(
            sample_rate=sample_rate,
            speaker_labels=True,    # request per-speaker labels when available
        )
        await self._client.connect(params)

    async def stream(self, pcm_bytes: bytes) -> None:
        """Send a chunk of raw 16-bit PCM to AssemblyAI."""
        await self._client.stream(pcm_bytes)

    async def disconnect(self) -> None:
        """Gracefully terminate the session (sends TerminateSession frame)."""
        await self._client.disconnect(terminate=True)
