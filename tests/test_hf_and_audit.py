import sqlite3

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("geopandas")

import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import LineString

from nbs_runtime import variables as V
from nbs_runtime.audit import audit_data, describe, summary_rows
from nbs_runtime.data import gee, hf
from nbs_runtime.data.cache import LayerCache
from nbs_runtime.data.synthetic import synthetic_variable
from nbs_runtime.data.variables_loader import LoadContext, load_variable
from nbs_runtime.grid import Grid

BBOX = (-73.0, 18.0, -72.0, 19.0)
GRID = Grid.from_bbox(BBOX, 5000)


def write_tif(path, array, bbox=BBOX, nodata=-9999.0, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape
    transform = from_origin(bbox[0], bbox[3], (bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype=dtype, crs="EPSG:4326",
                       transform=transform, nodata=nodata) as dst:
        dst.write(array.astype(dtype), 1)
    return path


# ---------------------------------------------------------------- file finding
def test_local_copy_is_found_with_or_without_the_raw_folder(tmp_path):
    f = tmp_path / "raw" / "hwsd_v2" / "HWSD2.hdr"
    f.parent.mkdir(parents=True)
    f.write_text("x")
    assert hf.find_local("raw/hwsd_v2/HWSD2.hdr", [tmp_path]) == f
    g = tmp_path / "data" / "aridity" / "a.tif"
    g.parent.mkdir(parents=True)
    g.write_text("x")
    assert hf.find_local("raw/aridity/a.tif", [tmp_path / "data"]) == g
    assert hf.find_local("raw/none.tif", [tmp_path]) is None


def test_raster_source_prefers_local_then_streams_from_the_hub(tmp_path):
    f = tmp_path / "raw" / "x" / "a.tif"
    f.parent.mkdir(parents=True)
    f.write_text("x")
    assert hf.raster_source("raw/x/a.tif", ("local", "hf"), [tmp_path]) == (str(f), "local")
    src, origin = hf.raster_source("raw/y/b.tif", ("local", "hf"), [tmp_path], mode="stream")
    assert origin == "hf-stream" and src == "/vsicurl/https://huggingface.co/datasets/dagudelo30/nbs-ruralscan-data/resolve/main/raw/y/b.tif"
    with pytest.raises(ValueError):
        hf.raster_source("raw/y/b.tif", mode="copy")
    with pytest.raises(hf.FileNotAvailable, match="not found locally"):
        hf.raster_source("raw/y/b.tif", ("local",), [tmp_path])


def test_download_path_is_used_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(hf, "download", lambda rel, repo=hf.DEFAULT_REPO: tmp_path / "cached.tif")
    assert hf.raster_source("raw/y/b.tif", ("hf",), mode="download") == (str(tmp_path / "cached.tif"), "hf-download")
    path, origin = hf.resolve_file("raw/y/b.sqlite", ("hf",))
    assert origin == "hf-download" and path == tmp_path / "cached.tif"


def test_a_missing_file_explains_where_it_looked(tmp_path, monkeypatch):
    def boom(rel, repo=hf.DEFAULT_REPO):
        raise OSError("404")

    monkeypatch.setattr(hf, "download", boom)
    with pytest.raises(hf.FileNotAvailable) as e:
        hf.resolve_file("raw/z/q.shp", ("local", "hf"), [tmp_path], siblings=["raw/z/q.shx"])
    msg = str(e.value)
    assert "not found" in msg and "could not download raw/z/q.shp" in msg


def test_check_access_reports_unreachable_networks(monkeypatch):
    import urllib.error
    import urllib.request

    def refuse(*a, **k):
        raise urllib.error.URLError("proxy blocked")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert "not reachable" in hf.check_access()


# ---------------------------------------------------------------- raster to grid
def test_a_raster_window_is_averaged_onto_the_grid(tmp_path):
    h = w = 100                                           # 0.01 degree cells over a 1 x 1 degree box
    arr = np.tile(np.arange(w, dtype="float32"), (h, 1))   # value = column index
    path = write_tif(tmp_path / "a.tif", arr)
    out = hf.read_raster_to_grid(str(path), GRID)
    assert out.shape == GRID.shape
    ok = np.isfinite(out)
    assert ok.sum() > 0
    # the first grid column spans about 0.045 degrees = the first ~4.5 source columns: mean near 2
    assert 1.0 < out[3, 0] < 3.5
    assert out[3, 0] < out[3, 5] < out[3, 10]             # increases eastwards, as the source does


def test_nodata_becomes_unknown_and_scale_is_applied_once(tmp_path):
    arr = np.full((100, 100), 8000.0, dtype="float32")
    arr[:, :50] = -9999.0                                  # the west half has no data
    path = write_tif(tmp_path / "ai.tif", arr)
    out = hf.read_raster_to_grid(str(path), GRID, scale=0.0001)
    assert np.isnan(out[:, 0]).all() and np.isfinite(out[:, -2]).all()
    assert np.nanmax(out) == pytest.approx(0.8, abs=1e-4)


def test_class_codes_are_never_averaged(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.choice([10.0, 40.0, 80.0], size=(100, 100)).astype("float32")
    path = write_tif(tmp_path / "c.tif", arr)
    out = hf.read_raster_to_grid(str(path), GRID, "mode")
    assert set(np.unique(out[np.isfinite(out)])) <= {10.0, 40.0, 80.0}
    with pytest.raises(ValueError):
        hf.read_raster_to_grid(str(path), GRID, "cubic")


# ---------------------------------------------------------------- HWSD v2
def make_hwsd_db(path, quirk=True):
    con = sqlite3.connect(path)
    cur = con.cursor()
    if quirk:    # as in the official file: the column called CODE holds the text, SYMBOL holds the number
        cur.execute("CREATE TABLE D_DRAINAGE (CODE TEXT, SYMBOL INTEGER, VALUE TEXT)")
        cur.executemany("INSERT INTO D_DRAINAGE VALUES (?,?,?)", [("E", 1, "x"), ("W", 3, "x"), ("P", 6, "x")])
    else:
        cur.execute("CREATE TABLE D_DRAINAGE (SYMBOL TEXT, CODE INTEGER, VALUE TEXT)")
        cur.executemany("INSERT INTO D_DRAINAGE VALUES (?,?,?)", [("E", 1, "x"), ("W", 3, "x"), ("P", 6, "x")])
    cur.execute("CREATE TABLE HWSD2_SMU (HWSD2_SMU_ID INTEGER, DRAINAGE TEXT, SHARE REAL)")
    cur.executemany("INSERT INTO HWSD2_SMU VALUES (?,?,?)", [
        (101, "E", 30.0), (101, "P", 70.0),                 # unit 101: the dominant soil (70%) is poorly drained
        (102, "W", 100.0), (103, None, 100.0)])
    con.commit()
    con.close()
    return path


@pytest.mark.parametrize("quirk", [True, False])
def test_hwsd_lookup_takes_the_dominant_soil_whatever_the_column_names(tmp_path, quirk):
    db = make_hwsd_db(tmp_path / "h.sqlite", quirk)
    assert hf.hwsd_drainage_lookup(db) == {101: 6.0, 102: 3.0}


def test_soil_unit_ids_become_drainage_classes_and_unknown_stays_unknown():
    smu = np.array([[101.0, 102.0], [np.nan, 999.0]])
    out = hf.drainage_from_smu(smu, {101: 6.0, 102: 3.0})
    assert out[0, 0] == 6.0 and out[0, 1] == 3.0 and np.isnan(out[1, 0]) and np.isnan(out[1, 1])
    assert np.isnan(hf.drainage_from_smu(smu, {})).all()


# ---------------------------------------------------------------- roads
def test_utm_zone_of_haiti():
    assert hf.utm_epsg(-72.5, 19.0) == 32618 and hf.utm_epsg(30.0, -10.0) == 32736


def make_roads(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"fclass": ["primary", "track"]},
                           geometry=[LineString([(-73.2, 18.5), (-71.8, 18.5)]), LineString([(-73.2, 18.9), (-71.8, 18.9)])],
                           crs="EPSG:4326")
    gdf.to_file(path)
    return path


def test_distance_to_the_nearest_road_in_kilometres(tmp_path):
    shp = make_roads(tmp_path / "roads.shp")
    grid = Grid.from_bbox((-72.8, 18.0, -72.2, 19.2), 5000)
    d = hf.road_distance_km(shp, grid)
    assert d.shape == grid.shape and np.isfinite(d).all()
    _, lat = grid.cell_centres()
    row_far = int(np.argmin(np.abs(lat - 19.2)))              # about 0.3 degrees north of the 18.9 line
    assert d[row_far, 3] == pytest.approx((lat[row_far] - 18.9) * 110.7, rel=0.02)
    row_on = int(np.argmin(np.abs(lat - 18.5)))
    assert d[row_on, 3] < 3.0                                  # within half a cell of the 18.5 line
    only_primary = hf.road_distance_km(shp, grid, fclasses=["primary"])
    assert only_primary[row_far, 3] > d[row_far, 3]            # the nearer 'track' road is left out


# ---------------------------------------------------------------- end to end from a local copy of the dataset
@pytest.fixture
def local_dataset(tmp_path):
    root = tmp_path / "data"
    h = w = 100
    ramp = np.tile(np.arange(w, dtype="float32"), (h, 1))            # a gentle west-to-east gradient
    write_tif(root / "raw/worldclim_v2_bioclim_precip/wc2.1_30s_bio_12.tif", 1400.0 + ramp)
    write_tif(root / "raw/worldclim_v2_bioclim_temp/wc2.1_30s_bio_1.tif", 24.0 + ramp / 100.0)
    write_tif(root / "raw/global_aridity_index_v3/ai_v31_yr.tif", 8200.0 + ramp)
    smu = np.full((h, w), 101.0)
    smu[:, 60:] = 102.0
    write_tif(root / "raw/hwsd_v2/HWSD2.bil", smu)                 # a stand-in raster in the same place
    make_hwsd_db(root / "raw/hwsd_v2/HWSD2.sqlite")
    make_roads(root / "raw/osm_road_network/gis_osm_roads_free_1.shp")
    return root


def test_the_five_hugging_face_variables_load_from_a_local_copy_and_look_right(local_dataset):
    ctx = LoadContext(GRID, order=("local",), local_roots=[local_dataset], verbose=False)
    audits = {}
    for vid in ("annual_precipitation", "mean_annual_temperature", "aridity_index", "soil_drainage", "distance_to_road"):
        data = load_variable(V.BY_ID[vid], ctx)
        assert data.ok, data.error
        assert data.origin == "local"
        audits[vid] = audit_data(data)
        assert audits[vid].flags == [] or vid == "distance_to_road", audits[vid].flags
    assert audits["aridity_index"].vmean == pytest.approx(0.825, abs=0.005)      # about 8250 x 0.0001, scaled once
    assert audits["annual_precipitation"].vmean == pytest.approx(1450.0, abs=15.0)
    assert set(audits["soil_drainage"].class_shares) == {3, 6}


def test_a_missing_file_is_reported_not_replaced_by_made_up_data(tmp_path):
    ctx = LoadContext(GRID, order=("local",), local_roots=[tmp_path], verbose=False)
    data = load_variable(V.BY_ID["aridity_index"], ctx)
    assert not data.ok and "not found" in data.error and not data.synthetic
    a = audit_data(data)
    assert a.values is None and a.flags and a.flags[0].startswith("not loaded")
    assert "NOT LOADED" in describe(a)


# ---------------------------------------------------------------- Earth Engine variables (fake loaders)
@pytest.fixture
def fake_ee(monkeypatch):
    calls = []
    shape_of = lambda g: g.shape
    monkeypatch.setattr(gee, "load_slope", lambda g, **k: calls.append("slope") or np.full(shape_of(g), 12.0))
    monkeypatch.setattr(gee, "load_soilgrids_band",
                        lambda g, var, d, **k: calls.append(f"{var}_{d}") or np.full(shape_of(g), 65.0 if var == "phh2o" else 75.0))
    from nbs_runtime.rasters import WORLDCOVER_CODES

    def shares(g, **k):
        calls.append("shares")
        out = {c: np.zeros(shape_of(g)) for c in WORLDCOVER_CODES}
        out[40] = np.full(shape_of(g), 0.7)
        out[10] = np.full(shape_of(g), 0.3)
        return out

    monkeypatch.setattr(gee, "load_worldcover_shares", shares)
    monkeypatch.setattr(gee, "load_tree_canopy", lambda g, a, b, **k: calls.append("canopy") or np.full(shape_of(g), 35.0))
    monkeypatch.setattr(gee, "load_accessibility", lambda g, a, b, **k: calls.append("access") or np.full(shape_of(g), 90.0))
    monkeypatch.setattr(gee, "load_viirs", lambda g, a, b, y, **k: calls.append(f"viirs{y}") or np.full(shape_of(g), 0.4))
    monkeypatch.setattr(gee, "load_wdpa_share", lambda g, a, **k: calls.append("wdpa") or np.full(shape_of(g), 0.1))
    return calls


def test_earth_engine_variables_are_converted_once_and_combined(fake_ee, tmp_path):
    ctx = LoadContext(GRID, cache=LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    ph = load_variable(V.BY_ID["soil_ph"], ctx)
    soc = load_variable(V.BY_ID["soil_organic_carbon"], ctx)
    slope = load_variable(V.BY_ID["slope"], ctx)
    lc = load_variable(V.BY_ID["land_cover"], ctx)
    assert np.allclose(ph.values, 6.5) and np.allclose(soc.values, 7.5) and np.allclose(slope.values, 12.0)
    assert np.all(lc.values == 40.0) and lc.shares[40][0, 0] == pytest.approx(0.7)
    for vid in ("tree_canopy_cover", "accessibility_travel_time", "electrification_index", "protected_area_status"):
        assert load_variable(V.BY_ID[vid], ctx).ok
    assert "viirsNone" in fake_ee                              # no year given: the latest the asset has


def test_a_second_run_reads_the_cache_and_does_not_call_earth_engine_again(fake_ee, tmp_path):
    cache_dir = tmp_path
    for _ in range(2):
        ctx = LoadContext(GRID, cache=LayerCache(cache_dir, GRID, gee.ASSET_IDS), verbose=False)
        for vid in ("slope", "soil_ph", "tree_canopy_cover", "land_cover"):
            assert load_variable(V.BY_ID[vid], ctx).ok
        if _ == 0:
            first_calls = len(fake_ee)
    assert len(fake_ee) == first_calls


# ---------------------------------------------------------------- audit
def test_audit_catches_a_missing_scale_factor():
    spec = V.BY_ID["aridity_index"]
    raw = np.full(GRID.shape, 8200.0)                       # forgot x 0.0001
    from nbs_runtime.data.variables_loader import VariableData

    a = audit_data(VariableData(spec, raw, "local"))
    assert any("outside" in f for f in a.flags)


def test_audit_reports_no_data_constant_values_and_odd_classes():
    from nbs_runtime.data.variables_loader import VariableData

    spec = V.BY_ID["slope"]
    v = np.full(GRID.shape, np.nan)
    assert any("no data anywhere" in f for f in audit_data(VariableData(spec, v, "gee")).flags)
    v[:5, :5] = 3.0
    flags = " ".join(audit_data(VariableData(spec, v, "gee")).flags)
    assert "every value is the same" in flags and "no data (sea" in flags
    odd = np.full(GRID.shape, 40.0)
    odd[0, 0] = 77.0
    assert any("not in the legend" in f for f in audit_data(VariableData(V.BY_ID["land_cover"], odd, "gee")).flags)


def test_summary_keeps_groups_together_and_shows_the_role_from_the_package(pkg):
    ctx = LoadContext(GRID, synthetic=True)
    audits = {v.id: audit_data(load_variable(v, ctx)) for v in V.CATALOG}
    rows = summary_rows(audits, V.contract_roles(pkg), V.policy_variables(pkg, "planted_terrain_soil_v1"))
    assert [r["group"] for r in rows] == [v.group for v in V.CATALOG]
    by_var = {r["variable"]: r for r in rows}
    assert by_var["slope"]["role_in_ciat_package"] == "scores the index (first-pass policy)"
    assert by_var["annual_precipitation"]["role_in_ciat_package"] == "a score in the package, not in the first-pass policy"
    assert by_var["aridity_index"]["role_in_ciat_package"] == "not in the package"
    assert all(r["origin"] == "synthetic" for r in rows)


def test_synthetic_variables_are_reproducible_plausible_and_never_unlabelled():
    for v in V.CATALOG:
        a = synthetic_variable(v, GRID, seed=3)
        b = synthetic_variable(v, GRID, seed=3)
        assert np.array_equal(a, b, equal_nan=True) and a.shape == GRID.shape
        if v.plausible:
            known = a[np.isfinite(a)]
            assert known.min() >= v.plausible[0] - 1e-9 and known.max() <= v.plausible[1] + 1e-9
    data = load_variable(V.BY_ID["slope"], LoadContext(GRID, synthetic=True))
    assert data.synthetic and data.origin == "synthetic"


# ---------------------------------------------------------------- figures
def test_variable_and_group_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from nbs_runtime.viz import plot_group_overview, plot_variable_audit

    ctx = LoadContext(GRID, synthetic=True)
    audits = [audit_data(load_variable(V.BY_ID[i], ctx)) for i in ("soil_ph", "soil_drainage", "land_cover")]
    for k, a in enumerate(audits):
        out = tmp_path / f"v{k}.png"
        plot_variable_audit(a, GRID, out)
        assert out.stat().st_size > 8_000
    out = tmp_path / "group.png"
    plot_group_overview(audits, GRID, "Soil", out)
    assert out.stat().st_size > 8_000
    bad = audit_data(load_variable(V.BY_ID["aridity_index"], LoadContext(GRID, order=("local",), local_roots=[tmp_path], verbose=False)))
    plot_variable_audit(bad, GRID)                                  # a variable that did not load still draws


def test_a_token_is_read_from_the_environment_or_the_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(gee, "REPO_ROOT", tmp_path)
    assert hf.hf_token() is None
    (tmp_path / ".env").write_text("HF_TOKEN=from-file\n", encoding="utf-8")
    assert hf.hf_token() == "from-file"
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf.hf_token() == "from-env"


def test_an_earth_engine_error_is_reported_for_that_variable_and_does_not_stop_the_rest(fake_ee, monkeypatch, tmp_path):
    class EEException(Exception):             # Earth Engine's own type: not an OSError, not a ValueError
        pass

    def boom(*a, **k):
        raise EEException("Image.bandNames: Parameter 'image' is required and may not be null.")

    monkeypatch.setattr(gee, "load_viirs", boom)
    ctx = LoadContext(GRID, cache=LayerCache(tmp_path, GRID, gee.ASSET_IDS), verbose=False)
    bad = load_variable(V.BY_ID["electrification_index"], ctx)
    assert not bad.ok and bad.error.startswith("EEException: ") and "may not be null" in bad.error
    assert load_variable(V.BY_ID["tree_canopy_cover"], ctx).ok                 # the next variable still loads
    assert audit_data(bad).flags[0].startswith("not loaded")
