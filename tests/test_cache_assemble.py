import pytest

np = pytest.importorskip("numpy")

from nbs_runtime.data import gee
from nbs_runtime.data.assemble import assemble_layers
from nbs_runtime.data.cache import LayerCache
from nbs_runtime.grid import Grid
from nbs_runtime.rasters import (
    TOPSOIL_LAYERS,
    WORLDCOVER_CODES,
    block_mean,
    gate_class_groups,
    host_fractions_from_classes,
    host_fractions_from_shares,
    share_sum_warnings,
)
from nbs_runtime.runconfig import RunConfig

GRID = Grid.from_bbox((-72.45, 18.45, -72.25, 18.65), 250)
GROUPS = {"eligible": [40], "ineligible": [50, 70, 80], "unknown": [30]}


def _shares(shape, rng):
    raw = rng.random((len(WORLDCOVER_CODES), *shape))
    raw /= raw.sum(axis=0)
    return {c: raw[i] for i, c in enumerate(WORLDCOVER_CODES)}


# ---- class shares -> host fractions ----

def test_default_resolution_is_250_m():
    assert RunConfig.from_dict({}).grid.resolution_m == 250
    assert GRID.shape == (90, 90)                 # a 0.2 degree box at 250 m (rounded up)


def test_shares_give_the_same_fractions_as_counting_fine_cells(pkg):
    groups = gate_class_groups(pkg.gate_params(pkg.policy("planted_terrain_soil_v1")))
    rng = np.random.default_rng(1)
    fine = rng.choice(list(WORLDCOVER_CODES), size=(40, 40))
    shares = {c: block_mean((fine == c).astype(float), 4) for c in WORLDCOVER_CODES}
    fe, fu = host_fractions_from_shares(groups, shares)
    fe2, fu2 = host_fractions_from_classes(groups, fine, 4)
    assert np.allclose(fe, fe2) and np.allclose(fu, fu2)
    assert share_sum_warnings(shares) == []


def test_a_cell_without_valid_pixels_is_unknown_host():
    shares = {c: np.full((2, 2), np.nan) for c in WORLDCOVER_CODES}
    fe, fu = host_fractions_from_shares(GROUPS, shares)
    assert np.isnan(fe).all() and np.isnan(fu).all()


def test_wrong_class_list_is_flagged():
    shares = {c: np.full((2, 2), 0.05) for c in WORLDCOVER_CODES}          # adds up to 0.55
    assert share_sum_warnings(shares)


# ---- assembling layers: units converted once ----

def test_assemble_converts_units_once_and_weights_depths():
    shape = (3, 3)
    soc_raw = {d: np.full(shape, 75.0) for d in TOPSOIL_LAYERS}          # dg/kg -> 7.5 g/kg
    ph_raw = {d: np.full(shape, 65.0) for d in TOPSOIL_LAYERS}           # pH x10 -> 6.5
    shares = _shares(shape, np.random.default_rng(2))
    lay = assemble_layers(np.full(shape, 9.0), soc_raw, ph_raw, shares, GROUPS)
    assert np.allclose(lay.soc_g_kg, 7.5) and np.allclose(lay.ph, 6.5)
    assert lay.notes == [] and not lay.synthetic
    assert set(lay.soc_by_depth) == set(TOPSOIL_LAYERS)


def test_assemble_leaves_a_cell_unknown_when_a_depth_is_missing():
    shape = (2, 2)
    soc_raw = {d: np.full(shape, 75.0) for d in TOPSOIL_LAYERS}
    soc_raw["5-15"][0, 0] = np.nan
    ph_raw = {d: np.full(shape, 65.0) for d in TOPSOIL_LAYERS}
    lay = assemble_layers(np.ones(shape), soc_raw, ph_raw, _shares(shape, np.random.default_rng(3)), GROUPS)
    assert np.isnan(lay.soc_g_kg[0, 0]) and not np.isnan(lay.soc_g_kg[1, 1])


def test_assemble_warns_about_double_scaling():
    shape = (2, 2)
    soc_raw = {d: np.full(shape, 3.0) for d in TOPSOIL_LAYERS}           # 0.3 g/kg after conversion
    ph_raw = {d: np.full(shape, 65.0) for d in TOPSOIL_LAYERS}
    lay = assemble_layers(np.ones(shape), soc_raw, ph_raw, _shares(shape, np.random.default_rng(4)), GROUPS)
    assert any("scaled twice" in n for n in lay.notes)


# ---- the layer cache ----

def test_cache_round_trip_and_summary(tmp_path):
    cache = LayerCache(tmp_path, GRID, gee.ASSET_IDS)
    a = np.arange(12.0).reshape(3, 4)
    a[0, 0] = np.nan
    cache.put("slope", {"slope": a})
    got = cache.get("slope")
    assert np.allclose(got["slope"], a, equal_nan=True)
    assert cache.has("slope") and not cache.has("other")
    assert cache.summary()["layers"]["slope"]["unknown_share"] == pytest.approx(1 / 12, abs=1e-3)
    assert not list(cache.dir.glob("*.tmp.npz"))                          # no half-written files


def test_cache_key_follows_the_grid_and_the_assets(tmp_path):
    base = LayerCache(tmp_path, GRID, gee.ASSET_IDS).key
    assert base == LayerCache(tmp_path, GRID, gee.ASSET_IDS).key
    assert base != LayerCache(tmp_path, Grid.from_bbox(GRID.bbox, 500), gee.ASSET_IDS).key
    assert base != LayerCache(tmp_path, Grid.from_bbox((-72.5, 18.45, -72.25, 18.65), 250), gee.ASSET_IDS).key
    assert base != LayerCache(tmp_path, GRID, {**gee.ASSET_IDS, "host": "another asset"}).key


def test_refresh_ignores_what_is_stored(tmp_path):
    LayerCache(tmp_path, GRID, gee.ASSET_IDS).put("slope", {"slope": np.ones((2, 2))})
    assert LayerCache(tmp_path, GRID, gee.ASSET_IDS, refresh=True).get("slope") is None


# ---- load_layers with fake Earth Engine loaders: cache hit, resume, refresh ----

@pytest.fixture
def fake_ee(monkeypatch):
    calls = {"slope": 0, "soil": 0, "host": 0}
    rng = np.random.default_rng(5)

    def slope(grid, **kw):
        calls["slope"] += 1
        return np.full(grid.shape, 8.0)

    def soil(grid, variable, depth, **kw):
        calls["soil"] += 1
        return np.full(grid.shape, 75.0 if variable == "soc" else 65.0)

    def host(grid, **kw):
        calls["host"] += 1
        return _shares(grid.shape, rng)

    monkeypatch.setattr(gee, "load_slope", slope)
    monkeypatch.setattr(gee, "load_soilgrids_band", soil)
    monkeypatch.setattr(gee, "load_worldcover_shares", host)
    return calls


def test_second_run_reads_the_cache_and_never_calls_earth_engine(tmp_path, fake_ee):
    first = gee.load_layers(GRID, GROUPS, LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    assert fake_ee == {"slope": 1, "soil": 6, "host": 1}
    second = gee.load_layers(GRID, GROUPS, LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    assert fake_ee == {"slope": 1, "soil": 6, "host": 1}                  # no new calls
    assert np.allclose(first.slope_deg, second.slope_deg)
    assert np.allclose(first.eligible_fraction, second.eligible_fraction, atol=1e-6)
    assert np.allclose(second.soc_g_kg, 7.5) and np.allclose(second.ph, 6.5)


def test_an_interrupted_run_resumes_from_the_layers_already_saved(tmp_path, fake_ee, monkeypatch):
    def failing_host(grid, **kw):
        raise RuntimeError("User memory limit exceeded")

    ok_host = gee.load_worldcover_shares
    monkeypatch.setattr(gee, "load_worldcover_shares", failing_host)
    with pytest.raises(RuntimeError):
        gee.load_layers(GRID, GROUPS, LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    assert fake_ee["slope"] == 1 and fake_ee["soil"] == 6                  # these were saved
    monkeypatch.setattr(gee, "load_worldcover_shares", ok_host)
    gee.load_layers(GRID, GROUPS, LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    assert fake_ee["slope"] == 1 and fake_ee["soil"] == 6 and fake_ee["host"] == 1


def test_refresh_fetches_everything_again(tmp_path, fake_ee):
    gee.load_layers(GRID, GROUPS, LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    gee.load_layers(GRID, GROUPS, LayerCache(tmp_path, GRID, gee.ASSET_IDS, refresh=True), verbose=False)
    assert fake_ee == {"slope": 2, "soil": 12, "host": 2}
