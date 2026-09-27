from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
from .enums import StatementType, VerdictType


# ---------------------------------------------------------------------------
# Auth / user
# ---------------------------------------------------------------------------

class User(BaseModel):
    id: str                    # UUID, stable per email
    name: str
    email: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Persisted session summary (written to Atlas after report is built)
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    session_id: str
    user_email: Optional[str]  # None for guest / unauthenticated sessions
    created_at: datetime
    speakers: list[str]
    claim_count: int
    summary: str               # the one-line dispute summary from MediationReport
    report_kind: Literal["dispute", "conversation"] = "dispute"
    report: dict               # full MediationReport as a plain dict


class Speaker(BaseModel):
    id: str
    display_name: str
    color: str = "#6366f1"  # default indigo — caller may override


class Utterance(BaseModel):
    id: str
    session_id: str
    speaker_id: Optional[str]  # None until diarization resolves attribution
    text: str
    is_final: bool
    start_ms: int
    end_ms: int


class Claim(BaseModel):
    id: str
    speaker_id: Optional[str]
    text: str
    statement_type: StatementType
    verbatim_quote: str
    start_ms: int
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceChunk(BaseModel):
    id: str
    session_id: str
    source_name: str
    chunk_index: int
    text: str


class EvidenceLink(BaseModel):
    claim_id: str
    evidence_chunk_id: str
    quote: str
    verdict: VerdictType
    reasoning: str


class ContradictionFlag(BaseModel):
    claim_id_a: str
    claim_id_b: str
    description: str
    resolved: bool = False


class EnrollmentRecord(BaseModel):
    """Binding between a spoken name and the realtime speaker label that said it."""
    speaker_name: str          # extracted name, e.g. "Sam"
    realtime_label: str        # AssemblyAI label, e.g. "A" or "B"
    start_ms: int              # start of the enrollment utterance
    end_ms: int                # end of the enrollment utterance


class SpeakerAssessment(BaseModel):
    """Deterministic per-speaker verdict tally computed from EvidenceLink list."""
    speaker_id: str
    supported: int
    contradicted: int
    uncertain: int             # UNCERTAIN + INSUFFICIENT_EVIDENCE combined
    checkable_total: int       # supported + contradicted (excludes uncertain)


class MediationReport(BaseModel):
    session_id: str
    claims: list[Claim]
    verdicts: list[EvidenceLink]
    contradictions: list[ContradictionFlag]
    agreements: list[str]           # free-text agreement summaries
    dispute_type: str
    summary: str
    assessments: list[SpeakerAssessment] = Field(default_factory=list)
    report_kind: Literal["dispute", "conversation"] = "dispute"
