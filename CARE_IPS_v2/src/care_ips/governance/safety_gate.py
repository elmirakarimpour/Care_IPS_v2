from care_ips.governance.eac import EvidenceToActionContract
from care_ips.config import SafetyGateConfig

def evidence_sufficiency(contract):
    return bool(contract.evidence_ids) and bool(contract.expected_benefit.strip())

def policy_compliant(contract):
    return (0.0 <= contract.operational_risk <= 1.0 and
            contract.expiry_seconds > 0 and
            contract.authorization_level in {"approved", "human_approved"})

def least_privilege(contract):
    return bool(contract.target.strip()) and bool(contract.scope.strip()) and len(contract.scope) <= 500

def reversible(contract):
    return bool(contract.rollback_rule and contract.rollback_rule.strip())

def authorize(contract: EvidenceToActionContract, cfg: SafetyGateConfig) -> str:
    checks = [
        evidence_sufficiency(contract),
        contract.independence_ratio >= cfg.ir_threshold,
        policy_compliant(contract),
        least_privilege(contract),
        reversible(contract),
    ]
    return "AUTHORIZE" if all(checks) else "ESCALATE"
