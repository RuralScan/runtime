import pytest

np = pytest.importorskip("numpy")

from nbs_runtime.compare import compare
from nbs_runtime.data import gee
from nbs_runtime.data.assemble import assemble_layers
from nbs_runtime.data.cache import LayerCache
from nbs_runtime.data.synthetic import synthetic_layers
from nbs_runtime.firstpass import run_first_pass
from nbs_runtime.grid import Grid
from nbs_runtime.overview import block_mean, block_share, make_overview
from nbs_runtime.rasters import TOPSOIL_LAYERS, WORLDCOVER_CODES
from nbs_runtime.simulate import build_level_space, simulate

GRID = Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000)
GROUPS = {"eligible": [40], "ineligible": [50, 70, 80], "unknown": [30]}


# ---- block aggregation, checked by hand ----

def test_block_mean_uses_only_valid_cells():
    a = np.array([[1.0, 2.0, 9.0, 9.0],
                  [3.0, 4.0, 9.0, 9.0]])
    valid = np.array([[True, True, False, False],
                      [True, False, False, False]])
    out = block_mean(a, valid, 2)
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx((1 + 2 + 3) / 3)        # 4.0 is not valid
    assert np.isnan(out[0, 1])                                  # nothing valid in that block


def test_block_share_ignores_padding_cells():
    mask = np.ones((3, 3), dtype=bool)                          # 3 x 3 padded to 4 x 4
    assert np.allclose(block_share(mask, 2), 1.0)               # padding must not lower the share
    mask[0, 0] = False
    assert block_share(mask, 2)[0, 0] == pytest.approx(0.75)


@pytest.fixture(scope="module")
def result(pkg):
    space = build_level_space(pkg, pkg.policy("planted_terrain_soil_v1"))
    grid = Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 250)
    fp = run_first_pass(space, synthetic_layers(grid, seed=3), "exploratory")
    res = simulate(space, n_draws=400, seed=1)
    return grid, fp, res, compare(fp, res)


def test_overview_covers_the_grid_and_hides_thin_blocks(result):
    grid, fp, _, comp = result
    ov = make_overview(fp, comp, grid, 5000, min_scored_share=0.5)
    assert ov.factor == 20 and ov.shape == (47, 65)
    assert ov.extent[0] == grid.bbox[0] and ov.extent[3] == grid.bbox[3]
    assert np.isnan(ov.baseline_index[ov.coverage < 0.5]).all()
    full = ov.coverage >= 0.5
    assert np.isfinite(ov.baseline_index[full]).all()
    assert ((ov.p_top[full] >= 0) & (ov.p_top[full] <= 1)).all()
    assert (ov.p05[full] <= ov.median[full] + 1e-9).all()
    strict = make_overview(fp, comp, grid, 5000, min_scored_share=0.95)
    assert np.isfinite(strict.baseline_index).sum() <= np.isfinite(ov.baseline_index).sum()


def test_overview_must_be_a_whole_multiple_of_the_resolution(result):
    grid, fp, _, comp = result
    with pytest.raises(ValueError):
        make_overview(fp, comp, grid, 1100)


def test_a_block_mean_is_the_mean_of_its_scored_cells(result):
    grid, fp, _, comp = result
    ov = make_overview(fp, comp, grid, 5000, min_scored_share=0.0)
    r = c = None
    for i in range(ov.shape[0]):
        for j in range(ov.shape[1]):
            if ov.coverage[i, j] > 0.5:
                r, c = i, j
                break
        if r is not None:
            break
    blk = fp.baseline_index[r * 20:(r + 1) * 20, c * 20:(c + 1) * 20]
    assert ov.baseline_index[r, c] == pytest.approx(np.nanmean(blk))


# ---- host gate skipped, and cache-only ----

def test_assemble_without_shares_treats_every_cell_as_host():
    shape = (3, 3)
    soc = {d: np.full(shape, 75.0) for d in TOPSOIL_LAYERS}
    ph = {d: np.full(shape, 65.0) for d in TOPSOIL_LAYERS}
    lay = assemble_layers(np.full(shape, 8.0), soc, ph, None, GROUPS)
    assert not lay.host_gate_applied
    assert (lay.eligible_fraction == 1).all() and (lay.unknown_fraction == 0).all()
    assert any("NOT applied" in n for n in lay.notes)


def test_first_pass_without_the_gate_scores_wherever_the_three_values_are_known(pkg):
    space = build_level_space(pkg, pkg.policy("planted_terrain_soil_v1"))
    lay = synthetic_layers(GRID, seed=3)
    lay.eligible_fraction = np.ones_like(lay.slope_deg)
    lay.unknown_fraction = np.zeros_like(lay.slope_deg)
    fp = run_first_pass(space, lay, "exploratory")
    assert (fp.status != 1).all() and (fp.status != 2).all()
    known = ~np.isnan(lay.slope_deg) & ~np.isnan(lay.soc_g_kg) & ~np.isnan(lay.ph)
    assert (fp.scored <= known).all()


@pytest.fixture
def fake_ee(monkeypatch):
    calls = {"slope": 0, "soil": 0, "host": 0}

    def slope(grid, **kw):
        calls["slope"] += 1
        return np.full(grid.shape, 8.0)

    def soil(grid, variable, depth, **kw):
        calls["soil"] += 1
        return np.full(grid.shape, 75.0 if variable == "soc" else 65.0)

    def host(grid, **kw):
        calls["host"] += 1
        return {c: np.full(grid.shape, 1 / len(WORLDCOVER_CODES)) for c in WORLDCOVER_CODES}

    monkeypatch.setattr(gee, "load_slope", slope)
    monkeypatch.setattr(gee, "load_soilgrids_band", soil)
    monkeypatch.setattr(gee, "load_worldcover_shares", host)
    return calls


SMALL = Grid.from_bbox((-72.45, 18.45, -72.25, 18.65), 250)


def test_host_skip_never_reads_worldcover(tmp_path, fake_ee):
    lay = gee.load_layers(SMALL, GROUPS, LayerCache(tmp_path, SMALL, gee.ASSET_IDS), verbose=False, host="skip")
    assert fake_ee["host"] == 0 and fake_ee["slope"] == 1 and fake_ee["soil"] == 6
    assert not lay.host_gate_applied


def test_cache_only_uses_what_is_stored_and_never_calls_earth_engine(tmp_path, fake_ee):
    cache = LayerCache(tmp_path, SMALL, gee.ASSET_IDS)
    gee.load_layers(SMALL, GROUPS, cache, verbose=False, host="skip")            # fills slope and soil
    before = dict(fake_ee)
    lay = gee.load_layers(SMALL, GROUPS, LayerCache(tmp_path, SMALL, gee.ASSET_IDS), verbose=False,
                          host="skip", cache_only=True)
    assert fake_ee == before and not lay.host_gate_applied
    with pytest.raises(gee.MissingLayersError, match="worldcover_shares"):
        gee.load_layers(SMALL, GROUPS, LayerCache(tmp_path, SMALL, gee.ASSET_IDS), verbose=False,
                        host="apply", cache_only=True)
    assert fake_ee == before                                                     # still no calls


def test_a_partial_cache_is_enough_for_an_overview_run(tmp_path, fake_ee, monkeypatch):
    """Slope and soil finished, WorldCover did not: the overview must still work without it."""
    cache = LayerCache(tmp_path, SMALL, gee.ASSET_IDS)
    monkeypatch.setattr(gee, "load_worldcover_shares", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("slow")))
    with pytest.raises(RuntimeError):
        gee.load_layers(SMALL, GROUPS, cache, verbose=False, host="apply")
    lay = gee.load_layers(SMALL, GROUPS, LayerCache(tmp_path, SMALL, gee.ASSET_IDS), verbose=False,
                          host="skip", cache_only=True)
    assert np.allclose(lay.soc_g_kg, 7.5) and not lay.host_gate_applied


# ---- figures ----

def test_overview_and_ladder_figures_are_written(result, tmp_path):
    pytest.importorskip("matplotlib")
    from nbs_runtime.viz import plot_combo_ladder, plot_overview

    grid, fp, res, comp = result
    ov = make_overview(fp, comp, grid, 5000)
    out1, out2 = tmp_path / "overview.png", tmp_path / "ladder.png"
    plot_overview(ov, "SYNTHETIC overview", "test", out1)
    plot_combo_ladder(res, fp.combo_counts(res.space.n_combos), "SYNTHETIC ladder", "test", out2)
    assert out1.stat().st_size > 10_000 and out2.stat().st_size > 10_000


def test_reason_map_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from nbs_runtime.contract import load_package
    from nbs_runtime.viz import plot_reasons

    pkg = load_package()
    space = build_level_space(pkg, pkg.policy("planted_terrain_soil_v1"))
    out = tmp_path / "reasons.png"
    plot_reasons(synthetic_layers(GRID, seed=3), space, GRID, "SYNTHETIC test", "", out)
    assert out.stat().st_size > 10_000
