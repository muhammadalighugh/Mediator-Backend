from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from .enums import StatementType, VerdictType


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


class MediationReport(BaseModel):
    session_id: str
    claims: list[Claim]
    verdicts: list[EvidenceLink]
    contradictions: list[ContradictionFlag]
    agreements: list[str]           # free-text agreement summaries
    dispute_type: str
    summary: str
