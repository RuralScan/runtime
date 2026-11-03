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


def test_plan_warns_about_a_placeholder_box_and_a_large_grid(pkg):
    from nbs_runtime.runconfig import RunConfig

    cfg = RunConfig.load(CFG, {"aoi": {"source": "GADM 4.1"}, "grid": {"resolution_m": 30}})
    p = plan(cfg, pkg)
    assert not any("AOI boundary" in h for h in p.holds)      # source recorded: that hold is gone
    text = " ".join(p.warnings)
    assert "bounding box" in text and "cells" in text
    assert p.config_fingerprint and p.aoi_fingerprint


def test_a_malformed_config_is_a_hold(pkg):
    from nbs_runtime.runconfig import RunConfig

    p = plan(RunConfig.load(CFG, {"simulation": {"scale_mode": "mixed"}}), pkg)
    assert any(h.startswith("config:") for h in p.holds)


def test_exploratory_mode_is_announced_in_the_plan(pkg):
    from nbs_runtime.runconfig import RunConfig

    assert any("EXPLORATORY" in w for w in plan(RunConfig.load(CFG), pkg).warnings)
    pol = plan(RunConfig.load(CFG, {"run": {"mode": "policy"}}), pkg)
    assert not any("EXPLORATORY" in w for w in pol.warnings)
