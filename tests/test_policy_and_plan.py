from pathlib import Path

from nbs_runtime.contract import fingerprint_holds, weight_holds
from nbs_runtime.plan import plan

CFG = Path(__file__).resolve().parents[1] / "config" / "runs" / "agroforestry_planted_first_pass.toml"


def test_policy_is_internally_consistent(pkg):
    policy = pkg.policy("planted_terrain_soil_v1")
    assert [c.weight for c in policy.criteria] == [0.5, 0.25, 0.25]
    assert weight_holds(policy) == []
    assert fingerprint_holds(pkg, policy) == []
    assert len(policy.gate_rule_ids) == 2


def test_template_run_is_on_hold(pkg):
    p = plan(CFG, pkg)
    assert not p.ready
    text = " ".join(p.holds)
    for needle in ("AOI", "applicability", "soil", "legal"):
        assert needle in text
