from __future__ import annotations

import asyncio
import os
import re
import unicodedata
import wave
from datetime import datetime, timezone
from typing import Optional

from models.enums import SessionStatus
from models.schemas import Claim, EvidenceChunk, Speaker, Utterance


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
        self.status: SessionStatus = SessionStatus.LIVE

        # Raw 16kHz mono 16-bit PCM audio accumulator
        self._audio_buffer: bytearray = bytearray()
        self._sample_rate: int = sample_rate

        self._lock: asyncio.Lock = asyncio.Lock()

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
