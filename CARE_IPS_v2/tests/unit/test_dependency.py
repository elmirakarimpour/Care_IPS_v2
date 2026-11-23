from care_ips.consensus.dependency import compute_evidence_overlap, compute_error_correlation
def test_jaccard():
    assert compute_evidence_overlap(frozenset({"a","b"}), frozenset({"b","c"})) == 1/3
def test_constant_error_history_has_zero_correlation():
    assert compute_error_correlation([0,0,0], [1,1,1]) == 0
