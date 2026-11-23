from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

def classification_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def expected_calibration_error(y_true, probabilities, classes, n_bins=15):
    import numpy as np
    y_true = np.asarray(y_true)
    probs = np.asarray(probabilities)
    confidence = probs.max(axis=1)
    predicted = np.asarray(classes)[probs.argmax(axis=1)]
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)
