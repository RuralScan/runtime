import itertools

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
    stats = run[2].statistics
    assert 0.8 < stats["spearman_baseline_vs_median"] <= 1.0
    assert 0.5 < stats["top_third_agreement"] <= 1.0
    assert stats["n_scored"] == int(run[0].scored.sum())


def test_comparison_exports_original_and_neutral_measures(run):
    comp = run[2]
    original = [
        "spearman_pete_vs_median", "pete_top_third_threshold", "share_pete_top_third",
        "share_of_pete_top_third_that_is_fragile", "share_pete_above_p95",
    ]
    assert [key for key in comp.stats if "pete" in key] == original
    statistics = comp.statistics
    assert list(statistics) == [key.replace("pete", "baseline") for key in comp.stats]
    for key in original:
        assert statistics[key.replace("pete", "baseline")] == comp.stats[key]


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


# ---- request tiling: the fix for "User memory limit exceeded" ----

def test_grid_windows_stay_aligned_with_the_global_grid():
    w = GRID.window(10, 20, 5, 30)
    assert w.shape == (10, 25) and w.resolution_deg == GRID.resolution_deg
    assert w.crs_transform[2] == pytest.approx(GRID.crs_transform[2] + 5 * GRID.resolution_deg)
    assert w.crs_transform[5] == pytest.approx(GRID.crs_transform[5] - 10 * GRID.resolution_deg)
    assert GRID.window(0, GRID.height, 0, GRID.width).crs_transform == pytest.approx(GRID.crs_transform)


@pytest.mark.parametrize("native", [gee.SRTM_NATIVE_DEG, gee.SOILGRIDS_NATIVE_DEG, gee.WORLDCOVER_NATIVE_DEG])
def test_tiles_cover_the_grid_exactly_once_and_respect_the_pixel_budget(native):
    tiles = gee.tile_plan(GRID, native)
    cover = np.zeros(GRID.shape, dtype=int)
    for r0, r1, c0, c1 in tiles:
        cover[r0:r1, c0:c1] += 1
        side = max(r1 - r0, c1 - c0)
        px_per_cell = max(1.0, (GRID.resolution_deg / native) ** 2)
        assert side <= gee.MAX_TILE_CELLS
        assert side == 1 or side * side * px_per_cell <= gee.MAX_TILE_INPUT_PIXELS * 1.0001
    assert (cover == 1).all()


def test_heavier_inputs_get_smaller_tiles():
    slope = gee.tile_plan(GRID, gee.SRTM_NATIVE_DEG)
    host = gee.tile_plan(GRID, gee.WORLDCOVER_NATIVE_DEG)
    soil = gee.tile_plan(GRID, gee.SOILGRIDS_NATIVE_DEG)
    assert len(host) > len(slope) > len(soil)


@pytest.mark.parametrize("ratio", [33, 161.7, 500, 5000, 100_000])
def test_aggregation_stages_nest_in_whole_steps(ratio):
    native = 1.0
    stages = gee.aggregation_stages(native, native * ratio)
    assert stages[-1] == pytest.approx(ratio)
    assert 1.0 <= stages[0] / native <= 32.0 + 1e-9
    for prev, cur in itertools.pairwise(stages):
        step = cur / prev
        assert step <= 32.0 + 1e-9 and step == pytest.approx(round(step))


def test_stitching_tiles_gives_the_same_array_as_one_request(monkeypatch):
    def field(grid):
        lon, lat = grid.cell_centres()
        return np.add.outer(lat * 3.0, lon * 7.0)

    monkeypatch.setattr(gee, "_open", lambda image, sub: {"v": field(sub)})
    whole = field(GRID)
    for budget in (10_000_000, 400_000, 1):                 # one tile, several tiles, single cells
        got = gee._fetch(None, GRID, gee.SRTM_NATIVE_DEG, "test", verbose=False, max_input_pixels=budget)["v"]
        assert got.shape == GRID.shape
        assert np.allclose(got, whole)


# ---- parallel requests, progress and the request plan ----

def test_request_plan_for_haiti_at_5_km():
    plan = gee.request_plan(GRID, host="apply")
    assert plan["slope"] == 48 and plan["soil (6 bands)"] == 6 and plan["land cover (11 classes)"] == 352
    assert plan["total"] == 48 + 6 + 352
    skip = gee.request_plan(GRID, host="skip")
    assert "land cover (11 classes)" not in skip and skip["total"] == 54


def _field(grid):
    lon, lat = grid.cell_centres()
    return np.add.outer(lat * 3.0, lon * 7.0)


def test_parallel_fetch_gives_the_same_array_as_sequential(monkeypatch):
    monkeypatch.setattr(gee, "_open", lambda image, sub: {"v": _field(sub)})
    seq = gee._fetch(None, GRID, gee.SRTM_NATIVE_DEG, "t", verbose=False, workers=1)["v"]
    par = gee._fetch(None, GRID, gee.SRTM_NATIVE_DEG, "t", verbose=False, workers=4)["v"]
    assert np.array_equal(seq, par) and np.allclose(par, _field(GRID))


def test_a_failing_tile_is_raised_in_parallel_mode_too(monkeypatch):
    calls = {"n": 0}

    def flaky(image, sub):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("User memory limit exceeded")
        return {"v": _field(sub)}

    monkeypatch.setattr(gee, "_open", flaky)
    with pytest.raises(RuntimeError, match="memory"):
        gee._fetch(None, GRID, gee.SRTM_NATIVE_DEG, "t", verbose=False, workers=4)


def test_progress_reports_time_left(monkeypatch, capsys):
    monkeypatch.setattr(gee, "_open", lambda image, sub: {"v": _field(sub)})
    gee._fetch(None, GRID, gee.SRTM_NATIVE_DEG, "slope", verbose=True, workers=2)
    out = capsys.readouterr().out
    assert "tiles" in out and "left" in out and "done in" in out


# ---- Earth Engine start-up: project id from .env, authenticate once ----

def test_env_file_is_read_without_extra_dependencies(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comment\n\nEE_PROJECT = "my-project"\nHF_TOKEN=abc\nnot a pair\n', encoding="utf-8")
    assert gee.read_env_file(f) == {"EE_PROJECT": "my-project", "HF_TOKEN": "abc"}
    assert gee.read_env_file(tmp_path / "missing.env") == {}


def test_project_comes_from_the_argument_then_the_environment_then_the_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("EE_PROJECT=from-file\n", encoding="utf-8")
    assert gee.resolve_project("from-arg", {"EE_PROJECT": "from-env"}, f) == "from-arg"
    assert gee.resolve_project(None, {"EE_PROJECT": "from-env"}, f) == "from-env"
    assert gee.resolve_project(None, {}, f) == "from-file"
    assert gee.resolve_project("  ", {}, tmp_path / "none.env") is None


def test_init_ee_authenticates_once_when_the_machine_is_not_signed_in(monkeypatch, tmp_path):
    import sys
    import types

    class EEException(Exception):
        pass

    calls = []
    fake = types.ModuleType("ee")
    fake.EEException = EEException

    def initialize(project=None):
        calls.append(("init", project))
        if calls.count(("init", project)) == 1:
            raise EEException("Please authorize access to your Earth Engine account")

    fake.Initialize = initialize
    fake.Authenticate = lambda: calls.append(("auth", None))
    monkeypatch.setitem(sys.modules, "ee", fake)
    monkeypatch.setenv("EE_PROJECT", "p1")
    assert gee.init_ee() == "p1"
    assert calls == [("init", "p1"), ("auth", None), ("init", "p1")]
    with pytest.raises(EEException):
        calls.clear()
        gee.init_ee("p2", authenticate_if_needed=False)


def test_init_ee_explains_a_missing_project(monkeypatch):
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.setattr(gee, "REPO_ROOT", __import__("pathlib").Path("/nonexistent"))
    with pytest.raises(RuntimeError, match="EE_PROJECT"):
        gee.init_ee()


# ---- night lights: a year the asset does not have, and Earth Engine's own exceptions ----

def _fake_ee(monkeypatch, years, bands=("average",)):
    import sys
    import types
    from datetime import UTC, datetime

    class EEException(Exception):
        pass

    seen = {}

    class Image:
        def __init__(self, ok=True):
            self.ok = ok

        def bandNames(self):
            def info():
                if not self.ok:
                    raise EEException("Image.bandNames: Parameter 'image' is required and may not be null.")
                return list(bands)

            return types.SimpleNamespace(getInfo=info)

        def select(self, band):
            return self

    class Collection:
        def __init__(self, asset):
            self.window = None

        def aggregate_array(self, prop):
            ms = [datetime(y, 1, 1, tzinfo=UTC).timestamp() * 1000 for y in years]
            return types.SimpleNamespace(getInfo=lambda: ms)

        def filterDate(self, a, b):
            seen["window"] = (a, b)
            self.window = (a, b)
            return self

        def first(self):
            return Image(ok=bool(years))

    fake = types.ModuleType("ee")
    fake.EEException = EEException
    fake.ImageCollection = Collection
    fake.Image = lambda asset: Image()
    monkeypatch.setitem(sys.modules, "ee", fake)
    monkeypatch.setattr(gee, "load_image_band", lambda grid, img, native, label, **k: np.full(grid.shape, 0.5))
    return seen, EEException


def test_night_lights_use_the_latest_year_the_asset_has_by_default(monkeypatch):
    seen, _ = _fake_ee(monkeypatch, range(2012, 2022))
    assert gee.collection_years("x") == list(range(2012, 2022))
    out = gee.load_viirs(GRID, "x", "average")
    assert out.shape == GRID.shape and seen["window"] == ("2021-01-01", "2022-01-01")


def test_a_year_the_asset_does_not_have_is_explained(monkeypatch):
    _fake_ee(monkeypatch, range(2012, 2022))
    with pytest.raises(ValueError) as e:
        gee.load_viirs(GRID, "NOAA/VIIRS/DNB/ANNUAL_V21", "average", 2023)
    msg = str(e.value)
    assert "no composite for 2023" in msg and "2012 to 2021" in msg and "VIIRS_YEAR" in msg


def test_an_empty_collection_and_an_unreadable_image_say_what_happened(monkeypatch):
    _fake_ee(monkeypatch, [])
    with pytest.raises(ValueError, match="has no images"):
        gee.load_viirs(GRID, "x", "average")

    import sys

    _fake_ee(monkeypatch, [2020])
    ee = sys.modules["ee"]
    image = ee.ImageCollection("x").first()                         # an image with the band "average"
    gee._require_band(image, "average", "x")
    with pytest.raises(ValueError, match="band 'avg_rad' is not in"):
        gee._require_band(image, "avg_rad", "x")
    _fake_ee(monkeypatch, [])
    empty = sys.modules["ee"].ImageCollection("x").first()          # what an empty filter returns: a null image
    with pytest.raises(ValueError, match="could not read"):
        gee._require_band(empty, "average", "x")
