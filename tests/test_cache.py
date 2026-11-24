import numpy as np
import pytest

from nbs_runtime.data import gee
from nbs_runtime.data.cache import LayerCache
from nbs_runtime.grid import Grid
from nbs_runtime.rasters import WORLDCOVER_CODES, share_sum_warnings, thickness_weighted

GRID = Grid.from_bbox((-72.45, 18.45, -72.25, 18.65), 250)


def test_grid_shape():
    assert GRID.shape == (90, 90)                 # a 0.2 degree box at 250 m (rounded up)


def test_thickness_weighting_and_a_missing_depth():
    a, b, c = np.array([10.0, 10.0]), np.array([20.0, np.nan]), np.array([30.0, 30.0])
    out = thickness_weighted([a, b, c])
    assert out[0] == pytest.approx((5 * 10 + 10 * 20 + 15 * 30) / 30)
    assert np.isnan(out[1])                         # no renormalisation onto the other depths


def test_class_shares_should_add_up_to_one():
    rng = np.random.default_rng(0)
    raw = rng.random((len(WORLDCOVER_CODES), 4, 4))
    raw /= raw.sum(axis=0)
    shares = {c: raw[i] for i, c in enumerate(WORLDCOVER_CODES)}
    assert share_sum_warnings(shares) == []
    del shares[40]
    assert share_sum_warnings(shares)


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
