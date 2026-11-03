import random

import pytest

from nbs_runtime.simulate import (
    build_level_space,
    draw_scales,
    pixels_to_counts,
    quantile,
    simulate,
)

POLICY = "planted_terrain_soil_v1"


@pytest.fixture(scope="module")
def space(pkg):
    return build_level_space(pkg, pkg.policy(POLICY))


def test_level_space_matches_ciat_policy(space):
    assert space.names == ("slope", "soil_organic_carbon", "soil_ph")
    assert space.levels == ((0.25, 0.5, 0.75, 0.9), (0.25, 0.5, 0.75, 0.9), (0.5, 0.75, 0.9))
    assert space.weights == (0.5, 0.25, 0.25)
    assert space.n_combos == 48
    assert len(space.combos()) == 48


def test_baseline_reproduces_ciat_index(space):
    # slope 12 deg -> 0.5, SOC 15 g/kg -> 0.75, pH 6.5 -> 0.9  => 0.6625 (checked by hand)
    combo = space.combo_of([12, 15, 6.5])
    assert combo == (1, 2, 2)
    assert space.baseline(combo) == pytest.approx(0.6625)


def test_unknown_value_has_no_combo(space):
    assert space.combo_of([25, 15, 6.5]) is None      # slope 25 deg is unknown in the pool
    assert space.combo_of([None, 15, 6.5]) is None


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_every_scenario_keeps_the_class_order_and_range(space, mode):
    rng = random.Random(1)
    for _ in range(500):
        scales = draw_scales(space, rng, mode)
        for lv, s in zip(space.levels, scales, strict=True):
            assert len(s) == len(lv)
            assert s == sorted(s)
            assert all(0.0 <= v <= 1.0 for v in s)


def test_shared_mode_uses_one_scale_per_class_label(space):
    scales = draw_scales(space, random.Random(2), "shared")
    slope, soc, ph = scales
    assert slope == soc                      # same four labels, same four numbers
    assert ph == slope[1:]                   # pH classes are CIAT's 0.5, 0.75, 0.9 labels


def test_same_seed_same_result_and_other_seed_differs(space):
    a = simulate(space, n_draws=400, seed=7)
    b = simulate(space, n_draws=400, seed=7)
    c = simulate(space, n_draws=400, seed=8)
    assert a.values == b.values
    assert a.values != c.values


def test_summary_is_ordered_and_bounded(space):
    res = simulate(space, n_draws=1000, seed=3)
    for row in res.table():
        assert 0.0 <= row["p05"] <= row["median"] <= row["p95"] <= 1.0


def test_dominated_pixel_never_beats_the_dominating_one(space):
    res = simulate(space, n_draws=1000, seed=4)
    a = space.combo_of([3, 15, 6.5])     # better slope, same soil
    b = space.combo_of([12, 15, 6.5])
    assert res.prob_greater(a, b) == 1.0
    assert res.prob_greater(b, a) == 0.0


def test_non_dominated_pair_is_uncertain(space):
    res = simulate(space, n_draws=3000, seed=5)
    b = space.combo_of([12, 15, 6.5])    # better soil
    c = space.combo_of([3, 3, 5.5])      # better slope
    assert 0.05 < res.prob_greater(b, c) < 0.95


def test_shared_scale_gives_a_wider_range_than_independent(space):
    ind = simulate(space, n_draws=4000, seed=6, mode="independent")
    sha = simulate(space, n_draws=4000, seed=6, mode="shared")
    combo = space.combo_of([12, 15, 6.5])
    wi = ind.summary(combo)["p95"] - ind.summary(combo)["p05"]
    ws = sha.summary(combo)["p95"] - sha.summary(combo)["p05"]
    assert ws > wi


def test_varying_weights_still_gives_valid_indices(space):
    res = simulate(space, n_draws=500, seed=9, weight_concentration=20)
    for row in res.table():
        assert 0.0 <= row["p05"] <= row["p95"] <= 1.0


def test_prob_top_third_extremes(space):
    res = simulate(space, n_draws=500, seed=10)
    counts = {c: 1 for c in space.combos()}               # one pixel in every combination
    top = res.prob_top_third(counts)
    best, worst = (3, 3, 2), (0, 0, 0)
    assert top[best] == 1.0
    assert top[worst] == 0.0


def test_pixels_to_counts_skips_unknown(space):
    counts = pixels_to_counts(space, [[3, 15, 6.5], [3, 15, 6.5], [25, 15, 6.5], [12, 15, 6.5]])
    assert sum(counts.values()) == 3
    assert counts[(3, 2, 2)] == 2


def test_quantile_interpolates():
    assert quantile([0.0, 1.0], 0.5) == 0.5
    assert quantile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert quantile([1.0, 2.0, 3.0], 1.0) == 3.0


def test_invalid_options_are_rejected(space):
    with pytest.raises(ValueError):
        draw_scales(space, random.Random(0), "other")
    with pytest.raises(ValueError):
        draw_scales(space, random.Random(0), "independent", prior="centered")


def test_metadata_records_what_a_manifest_needs(space):
    meta = simulate(space, n_draws=50, seed=11).metadata()
    for key in ("n_draws", "seed", "scale_mode", "prior", "weight_concentration", "levels"):
        assert key in meta
