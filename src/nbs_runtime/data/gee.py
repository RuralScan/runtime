"""Earth Engine adapter for CIAT's first pass. Thin on purpose: no analytical rule lives here.

It follows DATASET_INPUTS.md (developer-v0.1.5):
  slope  USGS/SRTMGL1_003, `elevation` band. Slope in degrees is derived at the native ~30 m scale and
         only then aggregated (mean) to the output grid. A coarser grid must not redefine the
         derivative scale, so the elevation is never resampled before the slope is taken.
  soil   projects/soilgrids-isric/soc_mean and phh2o_mean, depths 0-5, 5-15 and 15-30 cm. Values are
         returned as stored; conversion (dg/kg x 0.1, pH x10 x 0.1) happens once, in rasters.py.
  host   ESA/WorldCover/v200 `Map` band. Class codes are never interpolated: the share of fine cells in
         each group (eligible / unknown) is averaged to the output grid instead.

Masked or missing values come back as NaN (unknown). Nothing is filled with a mean or a mode.

NOT TESTED OFFLINE: this module needs earthengine-api, xee and an authenticated session. The notebook's
quick check (section 3a) pulls a small box and maps every layer, so read it before a full run.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..firstpass import Layers
from ..grid import METRES_PER_DEGREE, Grid
from ..rasters import (
    PH_X10_TO_PH,
    SOC_DG_PER_KG_TO_G_PER_KG,
    TOPSOIL_LAYERS,
    soil_value_warnings,
    thickness_weighted,
)

SRTM = "USGS/SRTMGL1_003"
SOILGRIDS_PREFIX = "projects/soilgrids-isric"
WORLDCOVER = "ESA/WorldCover/v200"
SRTM_NATIVE_DEG = 1.0 / 3600.0                      # 1 arc-second
SOILGRIDS_NATIVE_DEG = 250.0 / METRES_PER_DEGREE
WORLDCOVER_NATIVE_DEG = 10.0 / METRES_PER_DEGREE
MAX_PIXELS_PER_OUTPUT = 1024                         # Earth Engine's limit for reduceResolution


def soilgrids_band(variable: str, depth: str) -> str:
    """Band name for a SoilGrids mean prediction, for example soc_0-5cm_mean."""
    return f"{variable}_{depth}cm_mean"


def aggregation_stages(native_deg: float, target_deg: float, max_ratio: float = 32.0) -> list[float]:
    """Cell sizes of the successive mean-aggregations from the native grid up to the target.

    Earth Engine's reduceResolution accepts at most 1024 input pixels per output pixel (32 x 32), so
    a large jump (30 m to 5 km) is done in several stages. Returns [] when no aggregation is needed.
    """
    if target_deg <= native_deg * 1.0001:
        return []
    ratio = target_deg / native_deg
    n = max(1, math.ceil(math.log(ratio) / math.log(max_ratio)))
    stages = [native_deg * ratio ** (k / n) for k in range(1, n + 1)]
    stages[-1] = target_deg
    return stages


def init_ee(project: str | None = None) -> None:
    import ee

    ee.Initialize(project=project or None)


def _aggregate(image, grid: Grid, native_deg: float):
    import ee

    minx, _, _, maxy = grid.bbox
    out = image
    for res in aggregation_stages(native_deg, grid.resolution_deg):
        out = out.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=MAX_PIXELS_PER_OUTPUT).reproject(
            crs="EPSG:4326", crsTransform=[res, 0, minx, 0, -res, maxy]
        )
    return out


def _to_array(image, grid: Grid) -> np.ndarray:
    """Open one single-band image on the output grid. xee wants shape_2d as (width, height)."""
    import ee
    import xarray as xr
    import xee  # noqa: F401  registers the "ee" backend

    ds = xr.open_dataset(
        ee.ImageCollection([image]),
        engine="ee",
        crs="EPSG:4326",
        crs_transform=tuple(grid.crs_transform),
        shape_2d=(grid.width, grid.height),
    )
    da = ds[next(iter(ds.data_vars))]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    da = da.rename({k: v for k, v in (("lon", "x"), ("lat", "y")) if k in da.dims})
    da = da.transpose("y", "x").sortby("y", ascending=False)          # north up
    arr = np.asarray(da.values, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"Earth Engine returned {arr.shape}, expected {grid.shape} (rows, columns)")
    return arr


def load_slope(grid: Grid) -> np.ndarray:
    import ee

    elevation = ee.Image(SRTM).select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope")                 # degrees, native scale
    return _to_array(_aggregate(slope, grid, SRTM_NATIVE_DEG), grid)


def load_soilgrids(grid: Grid, variable: str, depths: Sequence[str] = TOPSOIL_LAYERS) -> dict[str, np.ndarray]:
    """Raw SoilGrids mean predictions by depth, in the stored units (not yet converted)."""
    import ee

    image = ee.Image(f"{SOILGRIDS_PREFIX}/{variable}_mean")
    available = image.bandNames().getInfo()
    out = {}
    for d in depths:
        band = soilgrids_band(variable, d)
        if band not in available:
            raise ValueError(f"band {band!r} not in {SOILGRIDS_PREFIX}/{variable}_mean; bands: {available}")
        out[d] = _to_array(_aggregate(image.select(band), grid, SOILGRIDS_NATIVE_DEG), grid)
    return out


def load_host_fractions(grid: Grid, groups: Mapping[str, Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Share of fine cells that are host-eligible and that are unknown, per output cell."""
    import ee

    wc = ee.ImageCollection(WORLDCOVER).first().select("Map")

    def indicator(codes: Sequence[int]):
        codes = list(codes)
        return wc.remap(codes, [1] * len(codes), 0) if codes else ee.Image(0).updateMask(wc.mask())

    eligible = indicator(groups["eligible"]).rename("eligible")
    known = indicator(list(groups["eligible"]) + list(groups["ineligible"]))
    unknown = ee.Image(1).subtract(known).updateMask(wc.mask()).rename("unknown")
    fe = _to_array(_aggregate(eligible, grid, WORLDCOVER_NATIVE_DEG), grid)
    fu = _to_array(_aggregate(unknown, grid, WORLDCOVER_NATIVE_DEG), grid)
    return fe, fu


def load_layers(grid: Grid, groups: Mapping[str, Sequence[int]]) -> Layers:
    slope = load_slope(grid)
    soc_raw = load_soilgrids(grid, "soc")
    ph_raw = load_soilgrids(grid, "phh2o")
    soc_by = {d: a * SOC_DG_PER_KG_TO_G_PER_KG for d, a in soc_raw.items()}     # converted once
    ph_by = {d: a * PH_X10_TO_PH for d, a in ph_raw.items()}
    soc = thickness_weighted([soc_by[d] for d in TOPSOIL_LAYERS])
    ph = thickness_weighted([ph_by[d] for d in TOPSOIL_LAYERS])
    fe, fu = load_host_fractions(grid, groups)
    return Layers(
        slope_deg=slope, soc_g_kg=soc, ph=ph, eligible_fraction=fe, unknown_fraction=fu,
        soc_by_depth=soc_by, ph_by_depth=ph_by, synthetic=False,
        notes=[
            f"slope: {SRTM}, ee.Terrain.slope at native scale, then staged mean to {grid.resolution_m} m",
            ("soil: SoilGrids mean predictions 0-5, 5-15, 15-30 cm, thickness-weighted "
             "(scoping proposal planted_topsoil_0_30_v1, not yet selected by CIAT)"),
            f"host: {WORLDCOVER} Map band, fractions at 10 m averaged to the grid",
            *soil_value_warnings(soc, ph),
        ],
    )
