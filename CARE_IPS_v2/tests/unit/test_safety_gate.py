from care_ips.config import SafetyGateConfig
from care_ips.governance.eac import EvidenceToActionContract
from care_ips.governance.safety_gate import authorize
def test_incomplete_contract_escalates():
    c = EvidenceToActionContract(frozenset(), "", "", "", 1.2, "unknown", 0, "", 0.1)
    assert authorize(c, SafetyGateConfig()) == "ESCALATE"
