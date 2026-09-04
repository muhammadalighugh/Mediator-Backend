from enum import Enum


class StatementType(str, Enum):
    FACT = "fact"
    CLAIM = "claim"
    ASSUMPTION = "assumption"
    EVIDENCE_REF = "evidence_ref"
    OPINION = "opinion"


class VerdictType(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SessionStatus(str, Enum):
    LIVE = "live"
    ENDED = "ended"
