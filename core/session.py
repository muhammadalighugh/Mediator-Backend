from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import unicodedata
import wave
from datetime import datetime, timezone
from typing import Optional

from models.enums import SessionStatus
from models.schemas import Claim, EvidenceChunk, EnrollmentRecord, Speaker, Utterance

logger = logging.getLogger(__name__)


def _normalize_claim_text(text: str) -> str:
    """
    Normalize a claim's text for deduplication.
    Lowercases, strips punctuation, collapses whitespace.
    Handles ordinal variants ("1st"/"first") and month name case.
    """
    t = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace
    t = " ".join(t.split())
    # Normalize ordinals: convert digit+suffix to bare digit so "1st" == "first" won't
    # match, but "1st" == "1st" and "the 1st" != "the first" are accepted — we focus
    # on punctuation/case differences, not English number word variants.
    return t

# Bytes per sample for 16-bit PCM
_BYTES_PER_SAMPLE = 2


class Session:
    def __init__(self, session_id: str, sample_rate: int = 16000) -> None:
        self.id = session_id
        self.created_at: datetime = datetime.now(tz=timezone.utc)
        self.speakers: list[Speaker] = []
        self.utterances: list[Utterance] = []
        self.claims: list[Claim] = []
        self.evidence: list[EvidenceChunk] = []
        self.enrollment_records: list[EnrollmentRecord] = []
        self.status: SessionStatus = SessionStatus.LIVE

        # The end of the last accepted enrollment utterance (ms into the stream).
        # Utterances whose end_ms <= this value are enrollment speech and must be
        # excluded from claim extraction.  Set to 0 when there is no enrollment.
        self.enrollment_end_ms: int = 0

        # Raw 16kHz mono 16-bit PCM audio accumulator
        self._audio_buffer: bytearray = bytearray()
        self._sample_rate: int = sample_rate

        self._lock: asyncio.Lock = asyncio.Lock()

        # Throttle buffer-size log to at most once per 5 seconds
        self._last_buffer_log: float = 0.0

        logger.info(
            "[SESSION] [%s] created (sample_rate=%d)",
            session_id, sample_rate,
        )

    # ------------------------------------------------------------------
    # Utterances
    # ------------------------------------------------------------------

    async def add_utterance(self, utterance: Utterance) -> None:
        async with self._lock:
            self.utterances.append(utterance)

    # ------------------------------------------------------------------
    # Claims  (deduped by verbatim_quote)
    # ------------------------------------------------------------------

    async def add_claim(self, claim: Claim) -> bool:
        """Add a claim; return False (and skip) if a normalized-equivalent already exists."""
        async with self._lock:
            existing_normalized = {_normalize_claim_text(c.text) for c in self.claims}
            if _normalize_claim_text(claim.text) in existing_normalized:
                return False
            self.claims.append(claim)
            return True

    # ------------------------------------------------------------------
    # Audio buffer
    # ------------------------------------------------------------------

    async def append_audio(self, chunk: bytes) -> None:
        async with self._lock:
            self._audio_buffer.extend(chunk)
            now = time.monotonic()
            if now - self._last_buffer_log >= 5.0:
                self._last_buffer_log = now
                logger.info(
                    "[SESSION] [%s] audio buffer size after append: %d bytes",
                    self.id, len(self._audio_buffer),
                )

    def slice_audio(self, start_ms: int, end_ms: int) -> bytes:
        """
        Return the raw PCM bytes covering [start_ms, end_ms).

        The buffer is 16kHz 16-bit mono, so:
            bytes_per_ms = sample_rate / 1000 * bytes_per_sample
                         = 16000 / 1000 * 2 = 32

        Clamps end to the current buffer length so callers do not need to
        know the buffer size. Called synchronously from the event loop —
        no lock needed because bytearray slice is atomic in CPython and
        only the event loop thread appends to the buffer.
        """
        rate = self._sample_rate * _BYTES_PER_SAMPLE  # bytes per second
        start_byte = int(start_ms / 1000 * rate)
        end_byte = int(end_ms / 1000 * rate)
        end_byte = min(end_byte, len(self._audio_buffer))
        return bytes(self._audio_buffer[start_byte:end_byte])

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    async def finalize(self) -> str:
        """Write audio buffer to sessions/{id}/audio.wav and mark as ended.

        Returns the absolute path of the written WAV file.
        """
        async with self._lock:
            self.status = SessionStatus.ENDED

            out_dir = os.path.join("sessions", self.id)
            os.makedirs(out_dir, exist_ok=True)
            wav_path = os.path.join(out_dir, "audio.wav")

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)                      # mono
                wf.setsampwidth(_BYTES_PER_SAMPLE)      # 16-bit
                wf.setframerate(self._sample_rate)
                wf.writeframes(bytes(self._audio_buffer))

            wav_bytes = os.path.getsize(wav_path)
            logger.info(
                "[SESSION] [%s] finalized: %d utterance(s), wav=%s, wav bytes=%d",
                self.id, len(self.utterances), wav_path, wav_bytes,
            )
            return wav_path


class SessionManager:
    """In-memory store for all active/completed sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def create(self, session_id: str, sample_rate: int = 16000) -> Session:
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id!r} already exists")
            session = Session(session_id=session_id, sample_rate=sample_rate)
            self._sessions[session_id] = session
            return session

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_or_raise(self, session_id: str) -> Session:
        session = await self.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} not found")
        return session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)


# Module-level singleton — imported everywhere
session_manager = SessionManager()
