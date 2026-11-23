import numpy as np
from care_ips.config import CACMConfig

def compute_error_correlation(agent_i_errors, agent_j_errors) -> float:
    a, b = np.asarray(agent_i_errors, float), np.asarray(agent_j_errors, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Error histories must be 1-D arrays of equal length")
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(max(0.0, np.corrcoef(a, b)[0, 1]))

def compute_evidence_overlap(evidence_ids_i: frozenset, evidence_ids_j: frozenset) -> float:
    union = evidence_ids_i | evidence_ids_j
    return len(evidence_ids_i & evidence_ids_j) / len(union) if union else 0.0

def compute_dependency_matrix(agents, cacm_config: CACMConfig):
    # agents: sequence of dicts with keys errors, evidence_ids, model_signature, provenance
    n = len(agents)
    D = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rho = compute_error_correlation(agents[i]["errors"], agents[j]["errors"])
            overlap = compute_evidence_overlap(agents[i]["evidence_ids"], agents[j]["evidence_ids"])
            model_sim = float(agents[i]["model_signature"] == agents[j]["model_signature"])
            prov_sim = float(agents[i]["provenance"] == agents[j]["provenance"])
            d = (cacm_config.w1*rho + cacm_config.w2*overlap +
                 cacm_config.w3*model_sim + cacm_config.w4*prov_sim)
            D[i, j] = D[j, i] = float(np.clip(d, 0, 1))
    return D
