from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from models.schemas import Speaker, Utterance

if TYPE_CHECKING:
    from models.schemas import EnrollmentRecord

logger = logging.getLogger(__name__)

# Default palette — cycles when more than len(COLORS) speakers join
_COLORS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
]

# Synthetic speaker used when an utterance never got a diarization label at
# all (speaker_id is None). This is distinct from "an extra diarized label
# we dropped" — we have zero signal here, so we label it rather than guess
# or discard content outright.
UNKNOWN_SPEAKER_ID = "speaker_unknown"
UNKNOWN_DISPLAY_NAME = "Unknown"


class SpeakerMapper:
    """Maps AssemblyAI speaker labels (e.g. 'A', 'B') to Speaker objects.

    Speaker objects are created up-front from display names sent by the client.
    The realtime API may emit a `speaker` field on FinalTranscript events.
    When it does, we map that label to the pre-created Speaker.
    When it does not (field is None/absent), speaker_id stays None and will be
    resolved later via async diarization of the saved WAV.
    """

    def __init__(self, display_names: list[str]) -> None:
        # Ordered list of Speaker objects, one per participant
        self.speakers: list[Speaker] = [
            Speaker(
                id=f"speaker_{i}",
                display_name=name,
                color=_COLORS[i % len(_COLORS)],
            )
            for i, name in enumerate(display_names)
        ]

        # AssemblyAI realtime labels are single uppercase letters: A, B, C …
        # Build the mapping eagerly. Labels beyond the declared roster are
        # NOT added on the fly — unknown labels return UNKNOWN_SPEAKER_ID.
        self._label_to_speaker: dict[str, Speaker] = {
            chr(ord("A") + i): spk for i, spk in enumerate(self.speakers)
        }
        self._log_label_map("__init__")

    def _log_label_map(self, context: str) -> None:
        """Log the current label→speaker mapping for diagnostics."""
        mapping = {
            lbl: f"{spk.id} ({spk.display_name!r})"
            for lbl, spk in self._label_to_speaker.items()
        }
        logger.info("SpeakerMapper [%s] label map: %s", context, mapping)

    def resolve(self, label: str | None) -> str | None:
        """Return the internal speaker_id for an AssemblyAI label, or None.

        PATH 1 — label is present and known: return speaker_id.
        PATH 2 — label is present but unknown (unexpected extra label):
            Log a WARNING with the actual label value so the mismatch is
            immediately visible. Return the UNKNOWN_SPEAKER_ID rather than
            creating a synthetic "Speaker N" name — that name would propagate
            into the claim board and report with no real identity behind it.
        PATH 3 — label is None/empty: return None (deferred attribution).
        """
        if not label:
            return None

        normalised = label.strip().upper()
        if normalised not in self._label_to_speaker:
            logger.warning(
                "SpeakerMapper.resolve: unknown label %r (normalised %r) — "
                "not in label map %s; assigning Unknown. "
                "If this label appears frequently, check enrollment binding.",
                label, normalised, list(self._label_to_speaker.keys()),
            )
            # Return UNKNOWN rather than fabricating a name — callers that
            # display speaker_id will fall back to "Unknown" via get_speaker().
            return UNKNOWN_SPEAKER_ID

        return self._label_to_speaker[normalised].id

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        return next((s for s in self.speakers if s.id == speaker_id), None)

    def bind_labels(self, records: list["EnrollmentRecord"]) -> None:
        """
        Rebuild the realtime-label → Speaker mapping from enrollment records.

        Each EnrollmentRecord carries the AssemblyAI realtime label that was
        active when that speaker said their name. This is the ground-truth
        binding: label "A" spoke during slot-1 enrollment → that label IS
        speaker_0 (whose display_name is the enrolled name).

        After this call:
          - self.speakers is rebuilt in slot order (record[0] → speaker_0, etc.)
          - self._label_to_speaker maps each label to the correct Speaker
          - Any label NOT covered by enrollment falls back to positional mapping

        No-op if records is empty — caller keeps the original positional map.
        """
        if not records:
            return

        # Rebuild speakers from enrollment order, preserving colors
        new_speakers: list[Speaker] = []
        new_label_map: dict[str, Speaker] = {}

        for i, rec in enumerate(records):
            spk = Speaker(
                id=f"speaker_{i}",
                display_name=rec.speaker_name,
                color=_COLORS[i % len(_COLORS)],
            )
            new_speakers.append(spk)
            label = rec.realtime_label.strip().upper()
            new_label_map[label] = spk
            logger.info(
                "Enrollment binding: realtime label %r → speaker_%d (%r)",
                label, i, rec.speaker_name,
            )

        # Fill in any positional slots not covered by enrollment records
        # (defensive — normally len(records) == len(declared speakers))
        for i in range(len(records), len(self.speakers)):
            label = chr(ord("A") + i)
            if label not in new_label_map:
                # Reuse the existing positionally-created speaker for this slot.
                # Do NOT synthesise a "Speaker N" name — if we don't know who
                # this is, keep the declared name from the original SpeakerMapper
                # constructor (which came from start_session).
                existing = self.speakers[i] if i < len(self.speakers) else Speaker(
                    id=f"speaker_{i}",
                    display_name=UNKNOWN_DISPLAY_NAME,
                    color=_COLORS[i % len(_COLORS)],
                )
                new_speakers.append(existing)
                new_label_map[label] = existing
                logger.warning(
                    "bind_labels: slot %d (label %r) not covered by enrollment records — "
                    "keeping declared name %r",
                    i, label, existing.display_name,
                )

        self.speakers = new_speakers
        self._label_to_speaker = new_label_map
        self._log_label_map("after bind_labels")


# ---------------------------------------------------------------------------
# Speaker consolidation
# ---------------------------------------------------------------------------
#
# Diarization (live label assignment via SpeakerMapper.resolve, or the async
# WAV diarization step) can surface MORE distinct speakers than were declared
# for the session — e.g. a TTS/system voice bleeding into the mic, or a brief
# cross-talk artifact getting its own label. consolidate_speakers() is the
# single place that reconciles that back down to the declared roster.
#
# Call sites (both use this same function — it is intentionally standalone
# rather than a SpeakerMapper method, so it can run over a full canonical
# transcript that wasn't necessarily produced by a single SpeakerMapper
# instance):
#   1. mediation_report.py — on the canonical diarized transcript, before
#      claim re-extraction.
#   2. Wherever live utterances are read for the live Claim Board, if more
#      distinct speaker_id values are present than declared_speakers.


def consolidate_speakers(
    utterances: list[Utterance],
    declared_speakers: list[Speaker],
) -> list[Utterance]:
    """
    Collapse diarization output down to at most len(declared_speakers) real
    speakers, and guarantee every surviving utterance has a non-blank
    speaker_id.

    Word count (not utterance count) decides which labels are "real":
    diarization sometimes fragments one person into many short utterances,
    so a per-utterance count would unfairly penalize a real speaker who
    happened to speak in short bursts. Total words spoken is more robust.

    Behavior:
      - distinct labels <= len(declared_speakers): no-op (aside from the
        None-speaker guarantee below).
      - distinct labels > len(declared_speakers): keep only the top-N labels
        by total word count (N = len(declared_speakers)). Utterances
        belonging to a dropped label are REMOVED ENTIRELY — never reassigned
        to a kept speaker. Reassigning would silently merge two genuinely
        different speakers if diarization ever legitimately splits one real
        speaker into two labels; dropping the rare label is the safer
        failure mode for a system whose whole job is reporting who said
        what.
      - Utterances with speaker_id=None (never resolved by diarization at
        all) are not counted as an extra "label" for the threshold check
        above — we have no signal to say they're noise the way an excess
        diarized label is — but they are still guaranteed not to survive
        with a blank speaker: they're relabeled to a synthetic "Unknown"
        speaker rather than dropped.

    Logs at INFO which labels were kept vs. dropped, with word counts,
    whenever any label is actually dropped.

    Does not mutate the input list; returns a new list.
    """
    declared_ids = {s.id for s in declared_speakers}
    n_declared = len(declared_speakers)

    # ---- 1. total words spoken per speaker_id (None handled separately) ---
    word_counts: dict[str, int] = defaultdict(int)
    for u in utterances:
        if u.speaker_id is None:
            continue
        word_counts[u.speaker_id] += len((u.text or "").split())

    distinct_labels = set(word_counts.keys())

    # ---- 2. nothing to consolidate -----------------------------------------
    if len(distinct_labels) <= n_declared:
        keep_ids = distinct_labels
    else:
        # ---- 3. keep top-N labels by word count ----------------------------
        ranked = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
        keep_ids = {label for label, _ in ranked[:n_declared]}
        dropped = [(label, count) for label, count in ranked[n_declared:]]

        # ---- 4. log what happened ------------------------------------------
        kept_summary = ", ".join(
            f"{label}({count}w)" for label, count in ranked[:n_declared]
        )
        dropped_summary = ", ".join(f"{label}({count}w)" for label, count in dropped)
        logger.info(
            "Speaker consolidation: %d distinct speaker labels found, "
            "%d declared. Kept: %s. Dropped: %s.",
            len(distinct_labels),
            n_declared,
            kept_summary,
            dropped_summary,
        )

        if not (keep_ids <= declared_ids):
            # Defensive: the labels surviving word-count ranking should
            # already be the declared roster in the normal case. If they
            # aren't, something upstream isn't resolving labels the way
            # this function assumes — surface it loudly rather than silently
            # mislabeling the report.
            logger.warning(
                "Speaker consolidation kept %d label(s) not in the declared "
                "roster (%s) — check upstream label resolution.",
                len(keep_ids - declared_ids),
                sorted(keep_ids - declared_ids),
            )

    # ---- 5. build the consolidated list ------------------------------------
    consolidated: list[Utterance] = []
    for u in utterances:
        if u.speaker_id is not None and u.speaker_id not in keep_ids:
            continue  # dropped excess label — removed entirely, not reassigned

        if u.speaker_id is None:
            # Never attributed by diarization — guarantee non-blank speaker
            # without guessing who it was.
            u = u.model_copy(update={"speaker_id": UNKNOWN_SPEAKER_ID})

        consolidated.append(u)

    return consolidated