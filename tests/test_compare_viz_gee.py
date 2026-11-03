import pytest

np = pytest.importorskip("numpy")

from nbs_runtime.compare import compare, rankdata, spearman
from nbs_runtime.data import gee
from nbs_runtime.data.synthetic import synthetic_layers
from nbs_runtime.firstpass import run_first_pass
from nbs_runtime.grid import Grid
from nbs_runtime.simulate import build_level_space, simulate

GRID = Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000)


@pytest.fixture(scope="module")
def run(pkg):
    space = build_level_space(pkg, pkg.policy("planted_terrain_soil_v1"))
    fp = run_first_pass(space, synthetic_layers(GRID, seed=3), "exploratory")
    res = simulate(space, n_draws=600, seed=1)
    return fp, res, compare(fp, res)


def test_rankdata_and_spearman():
    assert list(rankdata(np.array([10.0, 20.0, 20.0, 30.0]))) == [1.0, 2.5, 2.5, 4.0]
    assert spearman(np.arange(10.0), np.arange(10.0) * 2) == pytest.approx(1.0)
    assert spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.ones(5), np.arange(5.0)))


def test_comparison_only_fills_scored_pixels(run):
    fp, _, comp = run
    for arr in (comp.median, comp.p05, comp.p95, comp.p_top):
        assert np.isfinite(arr[fp.scored]).all() and np.isnan(arr[~fp.scored]).all()
    assert (comp.p05[fp.scored] <= comp.median[fp.scored]).all()
    assert (comp.median[fp.scored] <= comp.p95[fp.scored]).all()
    assert ((comp.p_top[fp.scored] >= 0) & (comp.p_top[fp.scored] <= 1)).all()


def test_the_simulation_tracks_ciat_ranks_closely_but_not_exactly(run):
    stats = run[2].stats
    assert 0.8 < stats["spearman_pete_vs_median"] <= 1.0
    assert 0.5 < stats["top_third_agreement"] <= 1.0
    assert stats["n_scored"] == int(run[0].scored.sum())


def test_a_pixel_better_in_every_class_has_a_higher_chance_of_the_top_third(run, pkg):
    fp, res, _ = run
    combos = res.space.combos()
    counts = fp.combo_counts(res.space.n_combos)
    present = [i for i, n in counts.items() if n > 0]
    best, worst = max(present, key=lambda i: res.space.baseline(combos[i])), min(present, key=lambda i: res.space.baseline(combos[i]))
    p = res.prob_top_third({combos[i]: counts[i] for i in counts})
    assert p[combos[best]] >= p[combos[worst]]


def test_plot_runs_and_writes_a_file(run, tmp_path):
    pytest.importorskip("matplotlib")
    from nbs_runtime.viz import plot_comparison

    fp, _, comp = run
    out = tmp_path / "maps.png"
    plot_comparison(fp, comp, GRID, "SYNTHETIC test", "pipeline test only", out)
    assert out.exists() and out.stat().st_size > 10_000


def test_aggregation_stages_respect_earth_engines_limit():
    target = 5000 / 111_320
    stages = gee.aggregation_stages(gee.SRTM_NATIVE_DEG, target)
    assert len(stages) == 2 and stages[-1] == target
    prev = gee.SRTM_NATIVE_DEG
    for s in stages:
        assert 1 < s / prev <= 32.0 + 1e-9
        prev = s
    assert gee.aggregation_stages(gee.SRTM_NATIVE_DEG, gee.SRTM_NATIVE_DEG) == []
    wc = gee.aggregation_stages(gee.WORLDCOVER_NATIVE_DEG, target)                  # 10 m to 5 km
    assert len(wc) == 2 and wc[-1] == target
    assert wc[0] / gee.WORLDCOVER_NATIVE_DEG <= 32.0 and wc[1] / wc[0] <= 32.0


def test_soilgrids_band_names():
    assert gee.soilgrids_band("soc", "0-5") == "soc_0-5cm_mean"
    assert gee.soilgrids_band("phh2o", "15-30") == "phh2o_15-30cm_mean"


def test_layer_maps_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from nbs_runtime.viz import plot_layers

    out = tmp_path / "layers.png"
    plot_layers(synthetic_layers(GRID, seed=3), GRID, "SYNTHETIC test", out)
    assert out.exists() and out.stat().st_size > 10_000
