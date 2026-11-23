from care_ips.evidence.attribution import EvidenceItem

def check_faithfulness(agent, event_row, evidence: EvidenceItem, baseline_value: float,
                       confidence_drop_threshold: float = 0.05) -> bool:
    original = agent.predict(event_row)
    perturbed = event_row.copy()
    perturbed.loc[:, evidence.feature_name] = baseline_value
    # Compare probability for the ORIGINAL predicted class, not max probability
    # after perturbation (which may belong to a different class).
    class_order = list(agent.model.classes_)
    class_idx = class_order.index(original.verdict)
    p_before = float(agent.model.predict_proba(event_row[agent.feature_columns])[0, class_idx])
    p_after = float(agent.model.predict_proba(perturbed[agent.feature_columns])[0, class_idx])
    return (p_before - p_after) >= confidence_drop_threshold

def filter_faithful_evidence(agent, event_row, evidence_items, feature_baselines,
                             confidence_drop_threshold: float = 0.05):
    return [e for e in evidence_items if check_faithfulness(
        agent, event_row, e, feature_baselines[e.feature_name], confidence_drop_threshold)]
