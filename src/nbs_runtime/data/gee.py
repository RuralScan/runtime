"""Earth Engine adapter for CIAT's first pass. Thin on purpose: no analytical rule lives here.

It follows DATASET_INPUTS.md (developer-v0.1.5):
  slope  USGS/SRTMGL1_003, `elevation` band. Slope in degrees is derived at the native ~30 m scale and
         only then aggregated (mean) to the output grid. A coarser grid must not redefine the
         derivative scale, so the elevation is never resampled before the slope is taken.
  soil   projects/soilgrids-isric/soc_mean and phh2o_mean, depths 0-5, 5-15 and 15-30 cm. Values are
         returned as stored; conversion (dg/kg x 0.1, pH x10 x 0.1) happens once, in rasters.py.
  host   ESA/WorldCover/v200 `Map` band. Class codes are never interpolated: the share of fine cells in
         EACH class is averaged to the output grid. Which classes count as host is decided later, from
         CIAT's gate, so a change in the gate never needs a new extraction.

Masked or missing values come back as NaN (unknown). Nothing is filled with a mean or a mode.

Requests are split into small tiles of the output grid. Taking the slope at 30 m means Earth Engine reads
every fine pixel under each output cell, so one request for a whole country exceeds the memory Earth Engine
gives a user ("User memory limit exceeded"). Each tile is sized so that it reads at most
MAX_TILE_INPUT_PIXELS fine pixels; lower it if a tile still fails.

Each layer is extracted once and saved in a LayerCache (see cache.py); a second run, or a new CIAT
package version, reads the cache and does not call Earth Engine.

NOT TESTED OFFLINE: this module needs earthengine-api, xee and an authenticated session. The notebook's
quick check (section 3a) pulls a small box and maps every layer, so read it before a full run.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..firstpass import Layers
from ..grid import METRES_PER_DEGREE, Grid
from ..rasters import TOPSOIL_LAYERS, WORLDCOVER_CODES
from .assemble import assemble_layers
from .cache import LayerCache

SRTM = "USGS/SRTMGL1_003"
SOILGRIDS_PREFIX = "projects/soilgrids-isric"
WORLDCOVER = "ESA/WorldCover/v200"
SRTM_NATIVE_DEG = 1.0 / 3600.0                      # 1 arc-second
SOILGRIDS_NATIVE_DEG = 250.0 / METRES_PER_DEGREE
WORLDCOVER_NATIVE_DEG = 10.0 / METRES_PER_DEGREE
# what is recorded in the cache key: change any of these and a new cache folder is used
ASSET_IDS = {
    "slope": f"{SRTM} elevation, ee.Terrain.slope at native scale, mean to grid",
    "soc": f"{SOILGRIDS_PREFIX}/soc_mean depths 0-5, 5-15, 15-30 cm, as stored",
    "phh2o": f"{SOILGRIDS_PREFIX}/phh2o_mean depths 0-5, 5-15, 15-30 cm, as stored",
    "host": f"{WORLDCOVER} Map, share of each class, mean to grid",
}
MAX_PIXELS_PER_OUTPUT = 1024                         # Earth Engine's limit for reduceResolution
MAX_TILE_INPUT_PIXELS = 2_500_000                    # fine pixels one tile may read
MAX_TILE_CELLS = 256                                 # at most 256 x 256 output cells per tile


def soilgrids_band(variable: str, depth: str) -> str:
    """Band name for a SoilGrids mean prediction, for example soc_0-5cm_mean."""
    return f"{variable}_{depth}cm_mean"


def aggregation_stages(native_deg: float, target_deg: float, max_ratio: float = 32.0) -> list[float]:
    """Cell sizes of the successive mean-aggregations from the native grid up to the target.

    Earth Engine's reduceResolution accepts at most 1024 input pixels per output pixel (32 x 32), so
    a large jump (30 m to 5 km) is done in several stages. Each stage after the first is an exact integer
    multiple of the previous one, so the fine cells nest inside the output cells. Returns [] when no
    aggregation is needed.
    """
    if target_deg <= native_deg * 1.0001:
        return []
    ratio = target_deg / native_deg
    if ratio <= max_ratio:
        return [target_deg]
    n = math.ceil(math.log(ratio) / math.log(max_ratio))
    factor = math.ceil(ratio ** (1.0 / n))
    return [target_deg / factor**k for k in range(n - 1, -1, -1)]


def tile_plan(grid: Grid, native_deg: float, max_input_pixels: int = MAX_TILE_INPUT_PIXELS,
              max_cells: int = MAX_TILE_CELLS) -> list[tuple[int, int, int, int]]:
    """Split the output grid into square blocks (row0, row1, col0, col1) small enough for one request."""
    px_per_cell = max(1.0, (grid.resolution_deg / native_deg) ** 2)
    side = int(max(1, min(max_cells, math.floor(math.sqrt(max_input_pixels / px_per_cell)))))
    return [
        (r0, min(r0 + side, grid.height), c0, min(c0 + side, grid.width))
        for r0 in range(0, grid.height, side)
        for c0 in range(0, grid.width, side)
    ]


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


def _open(image, grid: Grid) -> dict[str, np.ndarray]:
    """Open an image on a (sub-)grid with xee: one array per band, north up. shape_2d is (width, height)."""
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
    out = {}
    for name in ds.data_vars:
        da = ds[name]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        da = da.rename({k: v for k, v in (("lon", "x"), ("lat", "y")) if k in da.dims})
        da = da.transpose("y", "x").sortby("y", ascending=False)          # north up
        arr = np.asarray(da.values, dtype=float)
        if arr.shape != grid.shape:
            raise ValueError(f"Earth Engine returned {arr.shape}, expected {grid.shape} (rows, columns)")
        out[str(name)] = arr
    return out


def _fetch(image, grid: Grid, native_deg: float, label: str, verbose: bool = True,
           max_input_pixels: int = MAX_TILE_INPUT_PIXELS) -> dict[str, np.ndarray]:
    """Fetch an image on the whole grid, one small tile at a time, and stitch the tiles together."""
    tiles = tile_plan(grid, native_deg, max_input_pixels)
    out: dict[str, np.ndarray] = {}
    for k, (r0, r1, c0, c1) in enumerate(tiles, 1):
        if verbose:
            print(f"  {label}: tile {k}/{len(tiles)}", end="\r")
        for name, arr in _open(image, grid.window(r0, r1, c0, c1)).items():
            out.setdefault(name, np.full(grid.shape, np.nan))[r0:r1, c0:c1] = arr
    if verbose:
        print(f"  {label}: {len(tiles)} tile(s) done      ")
    return out


def load_slope(grid: Grid, verbose: bool = True, max_input_pixels: int = MAX_TILE_INPUT_PIXELS) -> np.ndarray:
    import ee

    elevation = ee.Image(SRTM).select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope")                 # degrees, native scale
    img = _aggregate(slope, grid, SRTM_NATIVE_DEG)
    return _fetch(img, grid, SRTM_NATIVE_DEG, "slope", verbose, max_input_pixels)["slope"]


def load_soilgrids_band(grid: Grid, variable: str, depth: str, verbose: bool = True,
                        max_input_pixels: int = MAX_TILE_INPUT_PIXELS) -> np.ndarray:
    """One SoilGrids mean prediction at one depth, in the stored units (not yet converted)."""
    import ee

    image = ee.Image(f"{SOILGRIDS_PREFIX}/{variable}_mean")
    available = image.bandNames().getInfo()
    band = soilgrids_band(variable, depth)
    if band not in available:
        raise ValueError(f"band {band!r} not in {SOILGRIDS_PREFIX}/{variable}_mean; bands: {available}")
    img = _aggregate(image.select(band), grid, SOILGRIDS_NATIVE_DEG)
    return next(iter(_fetch(img, grid, SOILGRIDS_NATIVE_DEG, band, verbose, max_input_pixels).values()))


def load_worldcover_shares(grid: Grid, verbose: bool = True,
                           max_input_pixels: int = MAX_TILE_INPUT_PIXELS) -> dict[int, np.ndarray]:
    """Share of fine cells in each WorldCover class, per output cell (one request per tile)."""
    import ee

    wc = ee.ImageCollection(WORLDCOVER).first().select("Map")
    bands = [wc.eq(code).rename(f"c{code}") for code in WORLDCOVER_CODES]     # masked pixels stay masked
    img = _aggregate(ee.Image.cat(bands), grid, WORLDCOVER_NATIVE_DEG)
    out = _fetch(img, grid, WORLDCOVER_NATIVE_DEG, "WorldCover classes", verbose, max_input_pixels)
    return {code: out[f"c{code}"] for code in WORLDCOVER_CODES}


def load_layers(grid: Grid, groups: Mapping[str, Sequence[int]], cache: LayerCache | None = None,
                verbose: bool = True, max_input_pixels: int = MAX_TILE_INPUT_PIXELS) -> Layers:
    """All first-pass layers on the grid. With a cache, each layer is read from it or fetched and saved."""

    def cached(name: str, fetch):
        if cache is not None:
            got = cache.get(name)
            if got is not None:
                if verbose:
                    print(f"  {name}: from cache")
                return got
        arrays = fetch()
        if cache is not None:
            cache.put(name, arrays)
        return arrays

    kw = {"verbose": verbose, "max_input_pixels": max_input_pixels}
    slope = cached("slope", lambda: {"slope": load_slope(grid, **kw)})["slope"]
    soc_raw = {d: cached(f"soc_{d}", lambda d=d: {"v": load_soilgrids_band(grid, "soc", d, **kw)})["v"]
               for d in TOPSOIL_LAYERS}
    ph_raw = {d: cached(f"phh2o_{d}", lambda d=d: {"v": load_soilgrids_band(grid, "phh2o", d, **kw)})["v"]
              for d in TOPSOIL_LAYERS}
    got = cached("worldcover_shares",
                 lambda: {str(c): a for c, a in load_worldcover_shares(grid, **kw).items()})
    shares = {int(k): v for k, v in got.items()}
    notes = [
        f"slope: {SRTM}, ee.Terrain.slope at native scale, then mean to {grid.resolution_m} m",
        ("soil: SoilGrids mean predictions 0-5, 5-15, 15-30 cm, thickness-weighted "
         "(scoping proposal planted_topsoil_0_30_v1, not yet selected by CIAT)"),
        f"host: {WORLDCOVER} Map band, share of each class averaged to the grid",
    ]
    if cache is not None:
        notes.append(f"layer cache: {cache.dir}")
    return assemble_layers(slope, soc_raw, ph_raw, shares, groups, notes)
