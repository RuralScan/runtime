"""plan(), eligibility, the offline end-to-end run, saving, and the figures."""
from pathlib import Path

import numpy as np
import pytest

from nbs_runtime import eligibility as E
from nbs_runtime.data.variables_loader import LoadContext
from nbs_runtime.grid import Grid
from nbs_runtime.overview import block_mean, block_share, make_overview
from nbs_runtime.plan import plan
from nbs_runtime.recipes import load_family
from nbs_runtime.runconfig import RunConfig
from nbs_runtime.sensitivity import dirichlet, equal_weights, leave_one_out
from nbs_runtime.suitability import NO_DATA, NOT_ELIGIBLE, SCORED, classify, compute
from nbs_runtime.workflow import explain_pixel, load_inputs, save_run
from recipe_helpers import write_recipe, write_roles

CFG = Path(__file__).resolve().parents[1] / "config" / "runs" / "agroforestry_cross_family.toml"
GRID = Grid.from_bbox((-72.6, 18.3, -72.0, 18.8), 2000)


# ------------------------------------------------------------------ plan

def test_the_repo_run_is_ready():
    p = plan(RunConfig.load(CFG))
    assert p.ready, p.holds
    assert p.family is not None and {r.variable for r in p.family.limiting} == {"annual_precipitation", "elevation"}
    assert any("ignored for limiting" in w for w in p.warnings)


def test_unit_mismatch_and_missing_catalog_entry_are_holds(tmp_path):
    root = tmp_path / "recipes"
    write_recipe(root, "agro", "fam", "slope", [0, 10], [1, 0], weight=1, unit="percent")
    write_recipe(root, "agro", "fam", "rootstock", [0, 1], [0, 1], weight=1)
    roles = write_roles(tmp_path / "roles.toml", {"slope": "modulating", "rootstock": "modulating"})
    cfg = RunConfig.from_dict({"run": {"nbs": "agro", "family": "fam"}, "aoi": {"bbox": [-72, 18, -71, 19]}})
    holds = " ".join(plan(cfg, root, roles).holds)
    assert "slope recipe is in 'percent'" in holds and "no catalog entry" in holds


def test_a_broken_recipe_is_a_hold_not_a_crash(tmp_path):
    root = tmp_path / "recipes"
    write_recipe(root, "agro", "fam", "slope", [3, 1], [1, 0], weight=1)
    roles = write_roles(tmp_path / "roles.toml", {"slope": "modulating"})
    cfg = RunConfig.from_dict({"run": {"nbs": "agro", "family": "fam"}, "aoi": {"bbox": [-72, 18, -71, 19]}})
    p = plan(cfg, root, roles)
    assert not p.ready and "strictly increasing" in p.holds[0]


# ------------------------------------------------------------------ eligibility

def test_eligibility_from_shares():
    shares = {30: np.array([0.3, 0.0, np.nan]), 40: np.array([0.3, 0.2, np.nan]), 10: np.array([0.4, 0.8, np.nan])}
    e = E.from_shares(shares, (30, 40), 0.5)
    assert list(e.G[:2]) == [1.0, 0.0] and np.isnan(e.G[2])
    assert e.share[0] == pytest.approx(0.6)


def test_eligibility_from_dominant_class_and_unknown_land_cover():
    e = E.from_classes(np.array([40.0, 10.0, np.nan]), (30, 40))
    assert list(e.G[:2]) == [1.0, 0.0] and np.isnan(e.G[2])


def test_unknown_land_cover_is_no_data_not_ineligible(tmp_path):
    fam = load_family("agroforestry", "cross_family")
    lay = {"annual_precipitation": np.full(3, 900.0), "elevation": np.full(3, 800.0),
           "slope": np.full(3, 5.0), "ndvi": np.full(3, 0.4)}
    r = compute(fam, lay, eligible=np.array([1.0, 0.0, np.nan]))
    assert list(r.status) == [SCORED, NOT_ELIGIBLE, NO_DATA]


# ------------------------------------------------------------------ end to end, offline

@pytest.fixture(scope="module")
def run():
    fam = load_family("agroforestry", "cross_family")
    inputs = load_inputs(fam, LoadContext(GRID, synthetic=True, verbose=False))
    base = compute(fam, inputs.layers, inputs.eligibility.G)
    return fam, inputs, base


def test_synthetic_inputs_are_complete_and_labelled(run):
    fam, inputs, _ = run
    assert set(inputs.layers) == set(fam.variables) and not inputs.errors
    assert inputs.synthetic and "land_cover" in inputs.data
    assert inputs.eligibility.G.shape == GRID.shape


def test_end_to_end_gives_states_classes_and_sensitivity(run):
    fam, inputs, base = run
    assert base.scored.any()
    cls = classify(base.S)
    assert cls.classes.shape == GRID.shape and (cls.classes[base.scored] >= 0).all()
    eq = equal_weights(fam, inputs.layers, inputs.eligibility.G, base=base)
    assert -1 <= eq["spearman"] <= 1
    d = dirichlet(fam, inputs.layers, inputs.eligibility.G, n_draws=50, base=base)
    assert np.isnan(d.median[~base.scored]).all()
    loo = leave_one_out(fam, inputs.layers, inputs.eligibility.G, base=base)
    assert set(loo) == set(fam.variables)


def test_explain_pixel_spells_out_the_formula(run):
    fam, inputs, base = run
    r, c = np.argwhere(base.scored)[0]
    text = explain_pixel(base, fam, inputs.layers, int(r), int(c))
    assert "L = min(" in text and "M = " in text and "S = L x G x M" in text
    off = np.argwhere(~base.scored)
    if off.size:
        assert "S =" not in explain_pixel(base, fam, inputs.layers, *map(int, off[0]))


def test_save_run_writes_arrays_and_manifest(run, tmp_path):
    import json

    fam, _, base = run
    files = save_run(tmp_path, GRID, {"family": fam.manifest()}, base, classify(base.S))
    names = {f.name for f in files}
    assert {"arrays.npz", "manifest.json"} <= names
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["family"]["recipes"]["slope"]["sha256"]
    with np.load(tmp_path / "arrays.npz") as z:
        assert z["S"].shape == GRID.shape and "z_slope" in z.files


def test_overview_blocks(run):
    _, _, base = run
    ov = make_overview(base.S, GRID, 4000)
    assert ov.S.shape == (-(-GRID.height // 2), -(-GRID.width // 2))
    with pytest.raises(ValueError):
        make_overview(base.S, GRID, 3000)
    a = np.array([[1.0, 3.0], [np.nan, 5.0]])
    assert block_mean(np.nan_to_num(a), ~np.isnan(a), 2)[0, 0] == pytest.approx(3.0)
    assert block_share(np.ones((3, 3), bool), 2)[1, 1] == pytest.approx(1.0)


def test_figures_are_written(run, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from nbs_runtime import viz

    fam, inputs, base = run
    cls = classify(base.S)
    d = dirichlet(fam, inputs.layers, inputs.eligibility.G, n_draws=20, base=base)
    loo = leave_one_out(fam, inputs.layers, inputs.eligibility.G, base=base)
    for name, fig in {
        "curves": lambda p: viz.plot_curves(fam, inputs.layers, p),
        "scores": lambda p: viz.plot_scores(base, GRID, True, p),
        "index": lambda p: viz.plot_index(base, GRID, True, p),
        "classes": lambda p: viz.plot_classes(cls, GRID, True, p),
        "sensitivity": lambda p: viz.plot_sensitivity(base, d, loo, GRID, True, p),
    }.items():
        out = tmp_path / f"{name}.png"
        fig(out)
        assert out.stat().st_size > 5_000, name
