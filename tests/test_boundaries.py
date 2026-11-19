import pytest

gpd = pytest.importorskip("geopandas")
from shapely import box

from nbs_runtime.aoi import AOI
from nbs_runtime.data import boundaries


@pytest.fixture
def cache(tmp_path, monkeypatch):
    gpd.GeoDataFrame({"ISO_A3": ["HTI", "ESP", "ESP"]}, crs=4326,
                     geometry=[box(-74.5, 18.0, -71.6, 20.1), box(-9.3, 36.0, 3.3, 43.8), box(-5.4, 35.8, -5.2, 35.9)],
                     ).to_parquet(tmp_path / "wb_admin0.parquet")
    gpd.GeoDataFrame({"ISO_A3": ["KEN", "KEN", "KEN", "UGA"],
                      "ADM1CD_c": ["KEN001", "KEN001", "KEN002", "UGA001"],
                      "ADM2CD_c": ["KEN001001", "KEN001002", "KEN002001", "UGA001001"]}, crs=4326,
                     geometry=[box(36, -1, 37, 0), box(37, -1, 38, 0), box(36, 0, 37, 1), box(32, 0, 33, 1)],
                     ).to_parquet(tmp_path / "wb_admin2.parquet")
    monkeypatch.setattr(boundaries, "CACHE_DIR", tmp_path)


def test_code_selects_every_unit_inside_it(cache):
    def codes(code):
        return list(boundaries.admin(code, 2)["ADM2CD_c"])

    assert codes("ken") == ["KEN001001", "KEN001002", "KEN002001"]
    assert codes("KEN001") == ["KEN001001", "KEN001002"]
    assert codes("KEN002001") == ["KEN002001"]
    with pytest.raises(ValueError, match="no admin 2 units"):
        boundaries.admin("XXX", 2)


def test_rows_sharing_an_iso3_are_all_returned(cache):
    assert len(boundaries.admin("ESP")) == 2                 # mainland plus Ceuta


def test_admin_aoi(cache):
    assert AOI(kind="admin", admin_code="HTI", admin_level=3).validate() == ["aoi.admin_level must be 0, 1 or 2"]
    aoi = AOI(kind="admin", admin_code="HTI")
    assert aoi.validate() == []
    assert aoi.resolved_bbox() == (-74.5, 18.0, -71.6, 20.1)
    assert 65_000 < aoi.area_km2() < 78_000
    assert AOI(kind="admin", admin_code="KEN", admin_level=2).resolved_bbox() == (36, -1, 38, 1)
