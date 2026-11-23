import numpy as np
from care_ips.consensus.psd_projection import project_to_psd

def build_dpp_kernel(D_psd: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    c = np.asarray(confidences, dtype=float)
    if np.any(c < 0) or np.any(c > 1):
        raise ValueError("Confidences must be in [0, 1]")
    L = np.outer(c, c) * np.asarray(D_psd, dtype=float)
    np.fill_diagonal(L, c ** 2)
    return L

def independence_ratio(D: np.ndarray, confidences: np.ndarray) -> float:
    c = np.asarray(confidences, dtype=float)
    if D.shape != (len(c), len(c)):
        raise ValueError("D and confidence dimensions do not match")
    denom = float(np.prod(c ** 2))
    if denom <= 0:
        return 0.0
    L = build_dpp_kernel(project_to_psd(D), c)
    sign, logdet = np.linalg.slogdet(L)
    if sign <= 0:
        return 0.0
    value = float(np.exp(logdet - np.log(denom)))
    return float(np.clip(value, 0.0, 1.0))
