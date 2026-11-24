import numpy as np
import pytest

from nbs_runtime.recipes import load_family
from nbs_runtime.suitability import (
    NO_DATA,
    NOT_ELIGIBLE,
    OUTSIDE_EVIDENCE,
    SCORED,
    aggregate_families,
    classify,
    compute,
)
from recipe_helpers import toy_family


@pytest.fixture
def fam(tmp_path):
    root, roles = toy_family(tmp_path)
    return load_family("toy", "fam", root, roles)


def layers(rain, temp, slope, ndvi):
    return {k: np.array(v, dtype=float) for k, v in
            {"rain": rain, "temp": temp, "slope": slope, "ndvi": ndvi}.items()}


def test_good_terrain_does_not_compensate_bad_climate(fam):
    # pixel 0: poor rain (0.2), perfect slope and ndvi; pixel 1: good rain (1.0), same terrain
    r = compute(fam, layers([500, 2000], [20, 20], [0, 0], [1, 1]))
    assert r.M == pytest.approx([1.0, 1.0])
    assert r.S == pytest.approx([0.2, 1.0])          # a weighted sum would have given ~0.68 to pixel 0
    assert list(r.status) == [SCORED, SCORED]


def test_formula_by_hand(fam):
    # rain 1000 -> 0.8, temp 10 -> 0.5 (L = 0.5, set by temp); slope 10 -> 0.5, ndvi 0.5 -> 0.5
    r = compute(fam, layers([1000], [10], [10], [0.5]))
    assert r.L[0] == pytest.approx(0.5)
    assert r.M[0] == pytest.approx(0.6 * 0.5 + 0.4 * 0.5)
    assert r.S[0] == pytest.approx(0.25)
    assert r.limiting[r.driver[0]] == "temp"


def test_weights_are_renormalised_among_modulating(fam):
    r = compute(fam, layers([2000], [20], [0], [0]))   # slope 1.0, ndvi 0.0
    assert r.weights == pytest.approx({"slope": 0.6, "ndvi": 0.4})
    assert r.M[0] == pytest.approx(0.6)
    eq = compute(fam, layers([2000], [20], [0], [0]), weights="equal")
    assert eq.M[0] == pytest.approx(0.5)


def test_pixel_states(fam):
    r = compute(
        fam,
        layers([1000, 1000, np.nan, 3000], [20, 20, 20, 20], [5, 5, 5, 5], [0.5, 0.5, 0.5, 0.5]),
        eligible=np.array([True, False, True, True]),
    )
    assert list(r.status) == [SCORED, NOT_ELIGIBLE, NO_DATA, OUTSIDE_EVIDENCE]
    assert np.isnan(r.S[1:]).all()


def test_clamp_scores_values_beyond_the_evidence(fam):
    r = compute(fam, layers([3000], [20], [5], [0.5]), outside="clamp")
    assert r.status[0] == SCORED and r.L[0] == pytest.approx(1.0)


def test_drop_a_variable(fam):
    base = compute(fam, layers([1000], [10], [10], [0.5]))
    no_temp = compute(fam, layers([1000], [10], [10], [0.5]), drop=("temp",))
    assert no_temp.L[0] == pytest.approx(0.8) and no_temp.S[0] > base.S[0]
    no_ndvi = compute(fam, layers([1000], [10], [10], [0.5]), drop=("ndvi",))
    assert no_ndvi.weights == {"slope": 1.0}


def test_outside_share(fam):
    lay = layers([1000, 3000, 400, 1500], [20] * 4, [5] * 4, [0.5] * 4)
    r = compute(fam, lay)
    assert r.outside_share(lay)["rain"] == pytest.approx(0.5)


def test_aggregate_families():
    from types import SimpleNamespace
    a = SimpleNamespace(S=np.array([0.0, 0.6, np.nan]))
    b = SimpleNamespace(S=np.array([0.4, 0.8, np.nan]))
    agg = aggregate_families({"a": a, "b": b})
    assert agg["mean"][:2] == pytest.approx([0.2, 0.7]) and np.isnan(agg["mean"][2])
    assert agg["breadth"] == pytest.approx([0.5, 1.0, 0.0])
    assert agg["quality"][:2] == pytest.approx([0.4, 0.7])


def test_classes_zero_is_its_own_class():
    S = np.array([np.nan, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    c = classify(S)
    assert c.classes[0] == -1 and c.classes[1] == 0
    assert c.classes[2] == 1 and c.classes[-1] == 5
    assert c.summary["max"] == pytest.approx(1.0)


def test_classes_fall_back_with_ties():
    c = classify(np.array([0.5] * 10 + [0.9] * 10))
    assert c.k < 5 and c.note
