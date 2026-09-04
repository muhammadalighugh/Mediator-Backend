from __future__ import annotations

from models.schemas import Speaker

# Default palette — cycles when more than len(COLORS) speakers join
_COLORS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
]


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
        # Build the mapping eagerly; extend on demand if more labels appear.
        self._label_to_speaker: dict[str, Speaker] = {
            chr(ord("A") + i): spk for i, spk in enumerate(self.speakers)
        }

    def resolve(self, label: str | None) -> str | None:
        """Return the internal speaker_id for an AssemblyAI label, or None.

        PATH 1 — label is present:  look up and return speaker_id.
        PATH 2 — label is None/empty: return None (deferred attribution).
        """
        if not label:
            return None

        label = label.strip().upper()
        if label not in self._label_to_speaker:
            # Unexpected extra label — create a new Speaker on the fly
            idx = len(self.speakers)
            new_speaker = Speaker(
                id=f"speaker_{idx}",
                display_name=f"Speaker {idx + 1}",
                color=_COLORS[idx % len(_COLORS)],
            )
            self.speakers.append(new_speaker)
            self._label_to_speaker[label] = new_speaker

        return self._label_to_speaker[label].id

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        return next((s for s in self.speakers if s.id == speaker_id), None)
