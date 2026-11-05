"""Load one catalog variable on the grid, from wherever it lives, and say where it came from.

There is no silent fallback to made-up data: if a variable cannot be loaded, the reason is returned so it can be
read. Made-up arrays exist only for offline tests and are labelled as synthetic everywhere.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..grid import Grid
from ..rasters import (
    PH_X10_TO_PH,
    SOC_DG_PER_KG_TO_G_PER_KG,
    TOPSOIL_LAYERS,
    WORLDCOVER_CODES,
    thickness_weighted,
)
from ..variables import GEE_CACHED, GEE_IMAGE, HF_RASTER, HWSD, OSM_ROADS, VarSpec
from . import gee, hf
from .cache import LayerCache
from .synthetic import synthetic_variable


@dataclass
class LoadContext:
    grid: Grid
    order: Sequence[str] = ("local", "hf")
    local_roots: Sequence[Path | str] = ()
    hf_repo: str = hf.DEFAULT_REPO
    hf_mode: str = "stream"
    cache: LayerCache | None = None
    workers: int = 1
    max_input_pixels: int = gee.MAX_TILE_INPUT_PIXELS
    verbose: bool = True
    synthetic: bool = False
    seed: int = 1


@dataclass
class VariableData:
    spec: VarSpec
    values: np.ndarray | None
    origin: str = ""
    notes: list[str] = field(default_factory=list)
    error: str = ""
    shares: dict[int, np.ndarray] | None = None
    synthetic: bool = False

    @property
    def ok(self) -> bool:
        return self.values is not None


def _kw(ctx: LoadContext) -> dict:
    return {"verbose": ctx.verbose, "max_input_pixels": ctx.max_input_pixels, "workers": ctx.workers}


def _hf_raster(spec: VarSpec, ctx: LoadContext) -> VariableData:
    src, origin = hf.raster_source(spec.detail["rel"], ctx.order, ctx.local_roots, ctx.hf_repo, ctx.hf_mode)
    values = hf.read_raster_to_grid(src, ctx.grid, "average", scale=float(spec.detail.get("scale", 1.0)))
    return VariableData(spec, values, origin)


def _hwsd(spec: VarSpec, ctx: LoadContext) -> VariableData:
    src, origin = hf.raster_source(spec.detail["rel"], ctx.order, ctx.local_roots, ctx.hf_repo, ctx.hf_mode)
    db, _ = hf.resolve_file(spec.detail["sqlite"], ctx.order, ctx.local_roots, ctx.hf_repo)
    smu = hf.read_raster_to_grid(src, ctx.grid, "mode")                      # whole soil-unit ids, never averaged
    lookup = hf.hwsd_drainage_lookup(db)
    values = hf.drainage_from_smu(smu, lookup)
    note = f"{len(lookup):,} soil units with a drainage class in the attribute database"
    return VariableData(spec, values, origin, [note])


def _osm(spec: VarSpec, ctx: LoadContext) -> VariableData:
    rel = spec.detail["rel"]
    stem = rel.rsplit(".", 1)[0]
    shp, origin = hf.resolve_file(rel, ctx.order, ctx.local_roots, ctx.hf_repo,
                                  siblings=[f"{stem}.{e}" for e in ("shx", "dbf", "prj", "cpg")])
    return VariableData(spec, hf.road_distance_km(shp, ctx.grid), origin)


def _gee_cached(spec: VarSpec, ctx: LoadContext) -> VariableData:
    grid, cache, kw = ctx.grid, ctx.cache, _kw(ctx)
    origin = "gee (cache)" if cache is not None else "gee"
    if spec.id == "slope":
        got = gee.cached_arrays(cache, "slope", lambda: {"slope": gee.load_slope(grid, **kw)}, ctx.verbose)
        return VariableData(spec, got["slope"], origin)
    if spec.id in ("soil_ph", "soil_organic_carbon"):
        var, factor = ("phh2o", PH_X10_TO_PH) if spec.id == "soil_ph" else ("soc", SOC_DG_PER_KG_TO_G_PER_KG)
        layers = []
        for d in TOPSOIL_LAYERS:
            got = gee.cached_arrays(cache, f"{var}_{d}", lambda d=d: {"v": gee.load_soilgrids_band(grid, var, d, **kw)},
                                    ctx.verbose)
            layers.append(got["v"] * factor)                                 # converted once
        return VariableData(spec, thickness_weighted(layers), origin)
    if spec.id == "land_cover":
        got = gee.cached_arrays(cache, "worldcover_shares",
                                lambda: {str(c): a for c, a in gee.load_worldcover_shares(grid, **kw).items()}, ctx.verbose)
        shares = {int(k): v for k, v in got.items()}
        stack = np.stack([shares[c] for c in WORLDCOVER_CODES])
        dominant = np.asarray(WORLDCOVER_CODES, dtype=float)[np.argmax(np.nan_to_num(stack, nan=-1.0), axis=0)]
        dominant[np.isnan(stack).all(axis=0)] = np.nan
        return VariableData(spec, dominant, origin, shares=shares)
    raise KeyError(f"no loader for {spec.id}")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[-40:]


def _gee_image(spec: VarSpec, ctx: LoadContext) -> VariableData:
    grid, cache, kw, d = ctx.grid, ctx.cache, _kw(ctx), spec.detail
    origin = "gee (cache)" if cache is not None else "gee"
    if spec.id == "tree_canopy_cover":
        fetch = lambda: {"v": gee.load_tree_canopy(grid, d["asset"], d["band"], **kw)}
    elif spec.id == "accessibility_travel_time":
        fetch = lambda: {"v": gee.load_accessibility(grid, d["asset"], d["band"], **kw)}
    elif spec.id == "electrification_index":
        fetch = lambda: {"v": gee.load_viirs(grid, d["asset"], d["band"], int(d["year"]), **kw)}
    elif spec.id == "protected_area_status":
        fetch = lambda: {"v": gee.load_wdpa_share(grid, d["asset"], **kw)}
    else:
        raise KeyError(f"no loader for {spec.id}")
    name = f"{spec.id}_{_slug(d['asset'])}" + (f"_{d['year']}" if "year" in d else "")
    return VariableData(spec, gee.cached_arrays(cache, name, fetch, ctx.verbose)["v"], origin)


_LOADERS = {HF_RASTER: _hf_raster, HWSD: _hwsd, OSM_ROADS: _osm, GEE_CACHED: _gee_cached, GEE_IMAGE: _gee_image}


def load_variable(spec: VarSpec, ctx: LoadContext) -> VariableData:
    """Load one variable. Returns the values and where they came from, or the reason it could not be loaded."""
    if ctx.synthetic:
        values = synthetic_variable(spec, ctx.grid, ctx.seed)
        return VariableData(spec, values, "synthetic", ["SYNTHETIC: a pipeline test, not real data"], synthetic=True)
    try:
        data = _LOADERS[spec.source](spec, ctx)
    except (OSError, ImportError, ValueError, KeyError, RuntimeError) as exc:
        return VariableData(spec, None, error=f"{type(exc).__name__}: {exc}")
    shape = data.values.shape if data.values is not None else None
    if shape is not None and shape != ctx.grid.shape:
        return VariableData(spec, None, error=f"loaded {shape}, expected {ctx.grid.shape}")
    return data
