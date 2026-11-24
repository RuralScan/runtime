import numpy as np
import pytest

from nbs_runtime.recipes import load_family
from nbs_runtime.sensitivity import dirichlet, equal_weights, leave_one_out, rankdata, spearman
from nbs_runtime.suitability import compute
from recipe_helpers import toy_family


@pytest.fixture
def setup(tmp_path):
    root, roles = toy_family(tmp_path)
    fam = load_family("toy", "fam", root, roles)
    rng = np.random.default_rng(1)
    n = (30, 40)
    lay = {
        "rain": rng.uniform(500, 2000, n),
        "temp": rng.uniform(10, 30, n),
        "slope": rng.uniform(0, 30, n),
        "ndvi": rng.uniform(0, 1, n),
    }
    return fam, lay


def test_rank_helpers():
    assert list(rankdata(np.array([3, 1, 1, 2]))) == [4, 1.5, 1.5, 3]
    assert spearman(np.arange(10), np.arange(10) ** 2) == pytest.approx(1.0)


def test_equal_weights(setup):
    fam, lay = setup
    out = equal_weights(fam, lay)
    assert out["weights"] == pytest.approx({"slope": 0.5, "ndvi": 0.5})
    assert -1 <= out["spearman"] <= 1


def test_dirichlet_fixed_weights_reproduce_the_base(setup):
    fam, lay = setup
    base = compute(fam, lay)
    d = dirichlet(fam, lay, n_draws=20, concentration=float("inf"), base=base)
    sc = base.scored
    assert np.allclose(d.median[sc], base.S[sc])
    assert np.allclose(d.p05[sc], d.p95[sc])
    assert np.nanmedian(d.spearman) == pytest.approx(1.0)


def test_dirichlet_spreads_with_low_concentration(setup):
    fam, lay = setup
    tight = dirichlet(fam, lay, n_draws=200, concentration=500, seed=3)
    loose = dirichlet(fam, lay, n_draws=200, concentration=2, seed=3)
    assert loose.summary()["mean_range_5_95"] > tight.summary()["mean_range_5_95"]
    assert 0 <= loose.summary()["base_top_that_is_robust"] <= 1
    assert loose.weights.sum(axis=1) == pytest.approx(np.ones(200))


def test_leave_one_out(setup):
    fam, lay = setup
    out = leave_one_out(fam, lay)
    assert set(out) == {"rain", "temp", "slope", "ndvi"}
    assert out["slope"]["role"] == "modulating"
    assert out["slope"]["mean_abs_change"] > 0
