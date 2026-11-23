import numpy as np
from care_ips.consensus.cacm import independence_ratio
def test_independent_agents_have_ir_one():
    assert abs(independence_ratio(np.eye(3), np.array([.9,.8,.95])) - 1.0) < 1e-8
def test_ir_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.uniform(0,1,(4,4))
        D = (A + A.T)/2
        np.fill_diagonal(D,1)
        ir = independence_ratio(D, rng.uniform(.1,1,4))
        assert -1e-9 <= ir <= 1 + 1e-9
