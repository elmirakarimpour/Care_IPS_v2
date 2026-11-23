from dataclasses import dataclass
import shap

@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    feature_name: str
    shap_value: float
    agent_id: str

def extract_top_evidence(agent, event_row, event_id: str, predicted_class_index: int, top_k: int = 5):
    explainer = shap.TreeExplainer(agent.model)
    explanation = explainer(event_row[agent.feature_columns])
    values = explanation.values
    if values.ndim == 3:
        values = values[0, :, predicted_class_index]
    elif values.ndim == 2:
        values = values[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    ranked = sorted(zip(agent.feature_columns, values), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return [EvidenceItem(f"{event_id}:{name}", name, float(val), agent.agent_id) for name, val in ranked]
