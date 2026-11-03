import json
from pathlib import Path

import pytest

from nbs_runtime.aoi import AOI
from nbs_runtime.runconfig import RunConfig

CFG = Path(__file__).resolve().parents[1] / "config" / "runs" / "agroforestry_planted_first_pass.toml"
HAITI = (-74.5, 18.0, -71.6, 20.1)


def test_haiti_box_area_and_cells():
    aoi = AOI(bbox=HAITI)
    assert aoi.validate() == []
    assert 65_000 < aoi.area_km2() < 78_000
    assert 2_500 < aoi.estimated_cells(5000) < 3_300
    assert aoi.estimated_cells(30) > 50_000_000


@pytest.mark.parametrize(
    "bbox",
    [(-74.5, 20.1, -71.6, 18.0), (-200.0, 18.0, -71.6, 20.1), (-74.5, 18.0, -71.6, 95.0)],
)
def test_bad_bboxes_are_reported(bbox):
    assert AOI(bbox=bbox).validate()


def test_kinds_need_their_fields():
    assert AOI(kind="geojson").validate()
    assert AOI(kind="admin").validate()
    assert AOI(kind="other").validate()


def test_geojson_bounds(tmp_path):
    poly = {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[-74.0, 18.5], [-72.0, 18.5], [-72.0, 19.9], [-74.0, 18.5]]]}}
    f = tmp_path / "aoi.geojson"
    f.write_text(json.dumps(poly), encoding="utf-8")
    aoi = AOI(kind="geojson", geojson_path=str(f))
    assert aoi.resolved_bbox() == (-74.0, 18.5, -72.0, 19.9)


def test_fingerprint_changes_with_the_area_and_its_source():
    a = AOI(bbox=HAITI)
    assert a.fingerprint() == AOI(bbox=HAITI).fingerprint()
    assert a.fingerprint() != AOI(bbox=(-74.4, 18.0, -71.6, 20.1)).fingerprint()
    assert a.fingerprint() != AOI(bbox=HAITI, source="GADM 4.1").fingerprint()


def test_config_loads_and_validates():
    cfg = RunConfig.load(CFG)
    assert cfg.validate() == []
    assert cfg.aoi.bbox == HAITI
    assert cfg.simulation.n_draws == 5000 and cfg.simulation.seed == 20261103
    assert cfg.data.order == ("local", "hf", "gee")


def test_overrides_are_merged_by_section():
    cfg = RunConfig.load(CFG, {"grid": {"resolution_m": 1000}, "simulation": {"seed": 1}})
    assert cfg.grid.resolution_m == 1000 and cfg.simulation.seed == 1
    assert cfg.simulation.n_draws == 5000          # untouched keys survive


def test_a_typo_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"simulation": {"n_draw": 10}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"simulaton": {}})


def test_invalid_values_are_reported():
    bad = RunConfig.from_dict({"aoi": {"bbox": list(HAITI)},
                               "simulation": {"scale_mode": "mixed", "n_draws": 0},
                               "grid": {"resolution_m": 0}, "data": {"order": ["local", "disk"]}})
    assert len(bad.validate()) == 4


def test_an_empty_aoi_is_reported():
    assert RunConfig.from_dict({}).validate() == ["aoi.bbox must be [minx, miny, maxx, maxy]"]


def test_fingerprint_follows_the_config():
    a = RunConfig.load(CFG)
    assert a.fingerprint() == RunConfig.load(CFG).fingerprint()
    assert a.fingerprint() != RunConfig.load(CFG, {"simulation": {"seed": 2}}).fingerprint()
    assert json.loads(a.to_json())["aoi"]["bbox"] == list(HAITI)


def test_run_mode_is_validated_and_defaults_to_exploratory():
    assert RunConfig.load(CFG).run.mode == "exploratory"
    assert RunConfig.load(CFG, {"run": {"mode": "policy"}}).validate() == []
    assert any("run.mode" in m for m in RunConfig.load(CFG, {"run": {"mode": "quick"}}).validate())
