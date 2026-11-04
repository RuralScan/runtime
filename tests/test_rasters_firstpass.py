import random

import pytest

np = pytest.importorskip("numpy")

from nbs_runtime.aggregate import combine
from nbs_runtime.data.synthetic import synthetic_layers
from nbs_runtime.firstpass import STATUS, Layers, run_first_pass
from nbs_runtime.grid import Grid
from nbs_runtime.rasters import (
    block_mean,
    gate_class_groups,
    host_fractions_from_classes,
    level_index,
    score_array,
    soil_value_warnings,
    thickness_weighted,
)
from nbs_runtime.responses import score_value
from nbs_runtime.simulate import build_level_space

POLICY = "planted_terrain_soil_v1"


@pytest.fixture(scope="module")
def space(pkg):
    return build_level_space(pkg, pkg.policy(POLICY))


def test_score_array_matches_the_scalar_rule_including_endpoints_and_unknowns(space):
    rng = np.random.default_rng(0)
    for opt, hi in zip(space.options, (50.0, 60.0, 11.0), strict=True):
        edges = [v for iv in opt["intervals"] for v in (iv.get("lower"), iv.get("upper")) if v is not None]
        values = np.concatenate([rng.uniform(-1, hi, 400), np.array(edges), [np.nan]])
        got = score_array(opt["intervals"], values)
        want = [score_value(opt["intervals"], None if np.isnan(v) else float(v)) for v in values]
        for g, w in zip(got, want, strict=True):
            assert (np.isnan(g) and w is None) or g == pytest.approx(w)


def test_level_index_maps_scores_and_marks_unknown():
    idx = level_index((0.25, 0.5, 0.75, 0.9), np.array([0.25, 0.9, np.nan, 0.6]))
    assert list(idx) == [0, 3, -1, -1]


def test_peter_worked_example_for_depths():
    # DATASET_INPUTS.md: SOC 12, 9, 6 g/kg -> 8 g/kg; pH 5, 6, 7 -> 6.333333
    soc = thickness_weighted([np.array([12.0]), np.array([9.0]), np.array([6.0])])
    ph = thickness_weighted([np.array([5.0]), np.array([6.0]), np.array([7.0])])
    assert soc[0] == pytest.approx(8.0)
    assert ph[0] == pytest.approx(6.333333, abs=1e-6)


def test_a_missing_depth_makes_the_cell_unknown_without_renormalising():
    out = thickness_weighted([np.array([12.0]), np.array([np.nan]), np.array([6.0])])
    assert np.isnan(out[0])


def test_soil_conversion_guards():
    assert soil_value_warnings(np.array([8.0]), np.array([6.5])) == []
    assert soil_value_warnings(np.array([0.3]), np.array([6.5]))      # scaled twice
    assert soil_value_warnings(np.array([8.0]), np.array([65.0]))     # not divided by 10


def test_gate_class_groups_agree_with_the_contract_gate(pkg):
    from nbs_runtime.gates import evaluate_gate

    params = pkg.gate_params(pkg.policy(POLICY))
    groups = gate_class_groups(params)
    for code in groups["eligible"]:
        assert evaluate_gate(params, [code])[0] == [1.0]
    for code in groups["ineligible"]:
        assert evaluate_gate(params, [code])[0] == [0.0]
    for code in groups["unknown"]:
        assert evaluate_gate(params, [code])[0] == [None]
    assert groups == {"eligible": [40], "ineligible": [50, 70, 80], "unknown": [30]}


def test_host_fractions_from_a_fine_class_map(pkg):
    groups = gate_class_groups(pkg.gate_params(pkg.policy(POLICY)))
    fine = np.array([[40, 40, 50, 50], [40, 10, 80, 30], [40, 40, 40, 40], [10, 10, 40, 40]])
    fe, fu = host_fractions_from_classes(groups, fine, 2)
    assert fe.shape == (2, 2)
    assert fe[0, 0] == pytest.approx(0.75)          # 3 of 4 cells are cropland
    assert fu[0, 0] == pytest.approx(0.25)          # class 10 is not in the gate: unknown
    assert fe[1, 0] == pytest.approx(0.5)
    assert block_mean(np.ones((4, 4)), 2).shape == (2, 2)


def test_haiti_grid_matches_the_prototype_shape():
    g = Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000)
    assert g.shape == (47, 65)
    assert g.crs_transform[0] == pytest.approx(5000 / 111_320)
    assert g.extent[3] == 20.1


def _pixel_layers(slope, soc, ph, fe, fu):
    one = lambda v: np.array([[v]], dtype=float)
    return Layers(one(slope), one(soc), one(ph), one(fe), one(fu))


@pytest.mark.parametrize("mode", ["exploratory", "policy"])
def test_first_pass_agrees_with_the_pixel_rule_in_aggregate_combine(pkg, space, mode):
    policy = pkg.policy(POLICY)
    rng = random.Random(4)
    for _ in range(300):
        slope = rng.choice([1.0, 3.0, 7.0, 12.0, 17.0, 25.0, 45.0])
        soc = rng.choice([3.0, 8.0, 15.0, 25.0, 40.0])
        ph = rng.choice([4.5, 5.5, 6.5, 7.5, 8.5])
        fe = rng.choice([0.0, 0.3, 1.0])
        fu = 0.0 if fe >= 1.0 else rng.choice([0.0, 0.2])
        fp = run_first_pass(space, _pixel_layers(slope, soc, ph, fe, fu), mode)
        scores = {c.rule_id: score_value(space.options[i]["intervals"], v)
                  for i, (c, v) in enumerate(zip(policy.criteria, (slope, soc, ph), strict=True))}
        host = True if fe > 0 else (False if fu == 0 else None)
        excluded = None if mode == "policy" else False
        want = combine(policy, scores, host, excluded, fe)
        got = {0: "scored", 1: "excluded", 2: "unknown", 3: "unknown", 4: "unknown", 5: "unknown"}[int(fp.status[0, 0])]
        assert got == want["status"]
        if want["status"] == "scored":
            assert fp.pete[0, 0] == pytest.approx(want["score"])


def test_policy_mode_scores_nothing_until_the_legal_mask_is_assessed(space):
    lay = synthetic_layers(Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000), seed=3)
    pol = run_first_pass(space, lay, "policy")
    exp = run_first_pass(space, lay, "exploratory")
    assert pol.status_counts()[STATUS[0]] == 0
    assert exp.status_counts()[STATUS[0]] > 0
    assert set(np.unique(pol.status)) <= {1, 2, 4}


def test_scored_pixels_have_a_combo_and_everything_else_does_not(space):
    lay = synthetic_layers(Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000), seed=3)
    fp = run_first_pass(space, lay, "exploratory")
    assert (fp.combo_id[fp.scored] >= 0).all() and (fp.combo_id[~fp.scored] == -1).all()
    assert np.isfinite(fp.pete[fp.scored]).all() and np.isnan(fp.pete[~fp.scored]).all()
    assert sum(fp.combo_counts(space.n_combos).values()) == int(fp.scored.sum())


def test_depth_sensitivity_keeps_the_three_depths(space):
    lay = synthetic_layers(Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000), seed=3)
    fp = run_first_pass(space, lay, "exploratory")
    assert set(fp.depth_index) == {"0-5", "5-15", "15-30"}
    assert fp.depth_range is not None and np.nanmax(fp.depth_range) > 0


def test_layers_on_different_grids_are_refused(space):
    bad = Layers(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        run_first_pass(space, bad)


def test_synthetic_layers_are_deterministic_and_plausible():
    g = Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000)
    a, b = synthetic_layers(g, seed=5), synthetic_layers(g, seed=5)
    assert np.array_equal(a.slope_deg, b.slope_deg) and a.synthetic
    assert 0 <= np.nanmin(a.slope_deg) and np.nanmax(a.slope_deg) <= 40
    assert 4 < np.nanmedian(a.soc_g_kg) < 40 and 4 < np.nanmedian(a.ph) < 9


# ---- why a cell is not scored ----

def test_reason_map_separates_no_data_from_gaps_in_the_functions(space):
    from nbs_runtime.firstpass import REASONS, explain_unscored, reason_map

    def pixel(slope, soc, ph):
        return Layers(*(np.array([[v]], dtype=float) for v in (slope, soc, ph, 1.0, 0.0)))

    cases = {
        (8.0, 25.0, 6.8): 0,            # all known, inside the functions
        (np.nan, 25.0, 6.8): 1,         # no data
        (25.0, 25.0, 6.8): 2,           # slope 20-44 deg: gap
        (8.0, 25.0, 6.0): 3,            # pH 5.55-6.45: gap (two sources disagree)
        (25.0, 25.0, 6.0): 5,           # two gaps
    }
    for (slope, soc, ph), want in cases.items():
        assert int(reason_map(space, pixel(slope, soc, ph))[0, 0]) == want, (slope, soc, ph)
    lay = synthetic_layers(Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000), seed=3)
    out = explain_unscored(space, lay)
    assert set(out) == set(REASONS.values())
    assert sum(n for n, _ in out.values()) == lay.slope_deg.size
    assert sum(share for _, share in out.values()) == pytest.approx(1.0)


def test_a_cell_is_scored_exactly_when_the_reason_map_says_so(space):
    from nbs_runtime.firstpass import reason_map

    lay = synthetic_layers(Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000), seed=3)
    lay.eligible_fraction = np.ones_like(lay.slope_deg)
    lay.unknown_fraction = np.zeros_like(lay.slope_deg)
    fp = run_first_pass(space, lay, "exploratory")
    assert np.array_equal(reason_map(space, lay) == 0, fp.scored)


def test_no_data_is_kept_apart_from_a_value_in_a_function_gap(space):
    def pixel(slope, soc, ph):
        return Layers(*(np.array([[v]], dtype=float) for v in (slope, soc, ph, 1.0, 0.0)))

    assert int(run_first_pass(space, pixel(np.nan, 25.0, 6.8), "exploratory").status[0, 0]) == 5   # no data
    assert int(run_first_pass(space, pixel(8.0, 25.0, 6.0), "exploratory").status[0, 0]) == 3      # pH gap
    assert int(run_first_pass(space, pixel(8.0, 25.0, 6.8), "exploratory").status[0, 0]) == 0      # scored
