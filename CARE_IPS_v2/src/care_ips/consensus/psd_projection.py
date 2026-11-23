import numpy as np

def project_to_psd(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Dependency matrix must be square")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("Dependency matrix must be symmetric")
    eigvals, eigvecs = np.linalg.eigh(D)
    clipped = np.clip(eigvals, 0, None)
    out = eigvecs @ np.diag(clipped) @ eigvecs.T
    diag = np.diag(out)
    scale = 1.0 / np.sqrt(np.where(diag > 1e-12, diag, 1.0))
    out = out * np.outer(scale, scale)
    np.fill_diagonal(out, 1.0)
    return out
