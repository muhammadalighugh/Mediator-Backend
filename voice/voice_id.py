"""
voice/voice_id.py
-----------------
Voice identification against enrolled speaker voiceprints using resemblyzer.

Pipeline per utterance:
    raw 16kHz 16-bit mono PCM bytes
    → float32 in [-1, 1]
    → resemblyzer VoiceEncoder.embed_utterance()
    → cosine similarity against all enrolled embeddings
    → return (best_name, similarity) if above thresholds, else (None, 0.0)

Enrollment:
    VoiceID.enroll(name, pcm_bytes) stores one embedding per name.
    Re-enrolling the same name overwrites the prior embedding.

Identification rules:
    1. Clip shorter than MIN_DURATION_S seconds → (None, 0.0) — embeddings
       from very short audio are unreliable.
    2. best cosine similarity < SIM_THRESHOLD → (None, 0.0)
    3. best − second_best < MARGIN_THRESHOLD → (None, 0.0) — too ambiguous.
    4. Otherwise return (best_name, similarity).

All thresholds are module-level constants — tune without changing call sites.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Minimum cosine similarity to accept an identification.
SIM_THRESHOLD: float = 0.65

# The winner must beat the runner-up by at least this margin.
MARGIN_THRESHOLD: float = 0.08

# Clips shorter than this (in seconds) are skipped — embeddings are unreliable.
MIN_DURATION_S: float = 1.0

# Sample rate this module expects (must match Session._sample_rate).
EXPECTED_SAMPLE_RATE: int = 16_000

# ---------------------------------------------------------------------------
# Lazy encoder singleton — loaded once on first use to avoid import-time cost
# and to fail gracefully if resemblyzer is not installed.
# ---------------------------------------------------------------------------

_encoder = None  # type: ignore[assignment]


def _get_encoder():
    """Return the shared VoiceEncoder, loading it on first call."""
    global _encoder
    if _encoder is None:
        try:
            from resemblyzer import VoiceEncoder  # type: ignore[import]
            _encoder = VoiceEncoder(device="cpu")
            logger.info("VoiceEncoder loaded (CPU)")
        except ImportError:
            logger.warning(
                "resemblyzer is not installed — voice identification disabled. "
                "Add 'resemblyzer' to requirements.txt and reinstall."
            )
            _encoder = None
    return _encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit signed PCM bytes to a float32 array in [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float arrays."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# VoiceID
# ---------------------------------------------------------------------------

class VoiceID:
    """
    Enroll known speakers and identify utterances against their voiceprints.

    Thread safety: all mutations are protected by no async primitives because
    the encoder and dict accesses happen synchronously inside the event loop.
    This is safe as long as callers await their own coordination (the WS route
    and report pipeline both run on the same asyncio event loop thread).
    """

    def __init__(self) -> None:
        # name → d-vector embedding (numpy array of shape [256])
        self._embeddings: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(self, name: str, pcm_bytes: bytes) -> bool:
        """
        Enroll a speaker from raw 16kHz 16-bit mono PCM audio.

        Returns True on success, False if the clip is too short or the encoder
        is unavailable (log messages explain why in both cases).
        """
        encoder = _get_encoder()
        if encoder is None:
            return False

        samples = _pcm_to_float(pcm_bytes)
        duration_s = len(samples) / EXPECTED_SAMPLE_RATE

        if duration_s < MIN_DURATION_S:
            logger.warning(
                "VoiceID.enroll: clip for %r is %.2fs — shorter than minimum %.1fs; skipping",
                name, duration_s, MIN_DURATION_S,
            )
            return False

        try:
            embedding = encoder.embed_utterance(samples)
            self._embeddings[name] = embedding
            logger.info(
                "VoiceID: enrolled %r (%.2fs, embedding shape %s)",
                name, duration_s, embedding.shape,
            )
            return True
        except Exception as exc:
            logger.warning("VoiceID.enroll failed for %r: %s", name, exc)
            return False

    def identify(self, pcm_bytes: bytes) -> tuple[str | None, float]:
        """
        Identify the speaker from raw 16kHz 16-bit mono PCM audio.

        Returns (best_name, similarity) if confident, (None, 0.0) otherwise.
        """
        encoder = _get_encoder()
        if encoder is None or not self._embeddings:
            return None, 0.0

        samples = _pcm_to_float(pcm_bytes)
        duration_s = len(samples) / EXPECTED_SAMPLE_RATE

        if duration_s < MIN_DURATION_S:
            logger.debug(
                "VoiceID.identify: clip too short (%.2fs < %.1fs) — skipping",
                duration_s, MIN_DURATION_S,
            )
            return None, 0.0

        try:
            query = encoder.embed_utterance(samples)
        except Exception as exc:
            logger.warning("VoiceID.identify: embed failed: %s", exc)
            return None, 0.0

        # Rank all enrolled speakers by cosine similarity
        scores: list[tuple[float, str]] = []
        for enrolled_name, emb in self._embeddings.items():
            sim = _cosine_sim(query, emb)
            scores.append((sim, enrolled_name))

        scores.sort(reverse=True)
        best_sim, best_name = scores[0]

        if best_sim < SIM_THRESHOLD:
            logger.debug(
                "VoiceID.identify: best sim %.3f for %r below threshold %.2f — no match",
                best_sim, best_name, SIM_THRESHOLD,
            )
            return None, 0.0

        if len(scores) >= 2:
            second_sim = scores[1][0]
            margin = best_sim - second_sim
            if margin < MARGIN_THRESHOLD:
                logger.debug(
                    "VoiceID.identify: margin %.3f between %r and %r below %.2f — ambiguous",
                    margin, best_name, scores[1][1], MARGIN_THRESHOLD,
                )
                return None, 0.0

        logger.debug(
            "VoiceID.identify: identified %r sim=%.3f", best_name, best_sim
        )
        return best_name, best_sim

    @property
    def enrolled_names(self) -> list[str]:
        return list(self._embeddings.keys())
