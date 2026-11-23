from dataclasses import dataclass

@dataclass(frozen=True)
class EvidenceToActionContract:
    evidence_ids: frozenset
    target: str
    scope: str
    expected_benefit: str
    operational_risk: float
    authorization_level: str
    expiry_seconds: int
    rollback_rule: str
    independence_ratio: float
