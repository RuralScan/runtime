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
from ..rasters import TOPSOIL_LAYERS, WORLDCOVER_CODES, thickness_weighted
from ..variables import FILE, GEE, VarSpec
from . import gee, hf
from .cache import LayerCache
from .synthetic import synthetic_variable


@dataclass
class LoadContext:
    grid: Grid
    order: Sequence[str] = ("local", "hf")
    local_roots: Sequence[Path | str] = ()
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


def _file(spec: VarSpec, ctx: LoadContext) -> VariableData:
    """A raster file, read onto the grid and converted to its unit."""
    src, origin = hf.raster_source(spec.path, ctx.order, ctx.local_roots, ctx.hf_mode)
    return VariableData(spec, hf.read_raster_to_grid(src, ctx.grid, spec.resample, scale=spec.scale), origin)


def _gee(spec: VarSpec, ctx: LoadContext) -> VariableData:
    """A single Earth Engine image, cached under its id, asset and parameters."""
    name = "_".join([spec.id, _slug(spec.path), *(str(v or "latest") for v in spec.params.values())])
    fetch = lambda: {"v": gee.load_gee_image(spec, ctx.grid, **_kw(ctx))}
    return VariableData(spec, gee.cached_arrays(ctx.cache, name, fetch, ctx.verbose)["v"], _gee_origin(ctx))


# ---- Named derivation steps (VarSpec.derive) that need more than one image or file ----
def _gee_origin(ctx: LoadContext) -> str:
    return "gee (cache)" if ctx.cache is not None else "gee"


def _slope(spec: VarSpec, ctx: LoadContext) -> VariableData:
    # cached as "slope" so the index calculation and the audit share one extraction
    got = gee.cached_arrays(ctx.cache, "slope", lambda: {"slope": gee.load_slope(ctx.grid, **_kw(ctx))}, ctx.verbose)
    return VariableData(spec, got["slope"], _gee_origin(ctx))


def _soilgrids_0_30(spec: VarSpec, ctx: LoadContext) -> VariableData:
    var = spec.params["variable"]
    layers = []
    for d in TOPSOIL_LAYERS:
        got = gee.cached_arrays(ctx.cache, f"{var}_{d}",
                                lambda d=d: {"v": gee.load_soilgrids_band(ctx.grid, var, d, **_kw(ctx))}, ctx.verbose)
        layers.append(got["v"] * spec.scale)                                 # converted once
    return VariableData(spec, thickness_weighted(layers), _gee_origin(ctx))


def _class_shares(spec: VarSpec, ctx: LoadContext) -> VariableData:
    got = gee.cached_arrays(ctx.cache, "worldcover_shares",
                            lambda: {str(c): a for c, a in gee.load_worldcover_shares(ctx.grid, **_kw(ctx)).items()},
                            ctx.verbose)
    shares = {int(k): v for k, v in got.items()}
    stack = np.stack([shares[c] for c in WORLDCOVER_CODES])
    dominant = np.asarray(WORLDCOVER_CODES, dtype=float)[np.argmax(np.nan_to_num(stack, nan=-1.0), axis=0)]
    dominant[np.isnan(stack).all(axis=0)] = np.nan
    return VariableData(spec, dominant, _gee_origin(ctx), shares=shares)


def _hwsd_drainage(spec: VarSpec, ctx: LoadContext) -> VariableData:
    src, origin = hf.raster_source(spec.path, ctx.order, ctx.local_roots, ctx.hf_mode)
    db, _ = hf.resolve_file(spec.params["sqlite"], ctx.order, ctx.local_roots)
    smu = hf.read_raster_to_grid(src, ctx.grid, spec.resample)     # "mode": whole soil-unit ids, never averaged
    lookup = hf.hwsd_drainage_lookup(db)
    values = hf.drainage_from_smu(smu, lookup)
    note = f"{len(lookup):,} soil units with a drainage class in the attribute database"
    return VariableData(spec, values, origin, [note])


def _distance_to_lines(spec: VarSpec, ctx: LoadContext) -> VariableData:
    stem = spec.path.rsplit(".", 1)[0]
    shp, origin = hf.resolve_file(spec.path, ctx.order, ctx.local_roots,
                                  siblings=[f"{stem}.{e}" for e in ("shx", "dbf", "prj", "cpg")])
    return VariableData(spec, hf.road_distance_km(shp, ctx.grid), origin)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[-40:]


# A step named here replaces the plain read; any other derive value is an image step in gee.IMAGE_STEPS.
_STEPS = {
    "slope": _slope, "soilgrids_0_30": _soilgrids_0_30, "class_shares": _class_shares,
    "hwsd_drainage": _hwsd_drainage, "distance_to_lines": _distance_to_lines,
}
_SOURCES = {FILE: _file, GEE: _gee}


def load_variable(spec: VarSpec, ctx: LoadContext) -> VariableData:
    """Load one variable. Returns the values and where they came from, or the reason it could not be loaded."""
    if ctx.synthetic:
        values = synthetic_variable(spec, ctx.grid, ctx.seed)
        return VariableData(spec, values, "synthetic", ["SYNTHETIC: a pipeline test, not real data"], synthetic=True)
    try:
        data = _STEPS.get(spec.derive, _SOURCES[spec.source])(spec, ctx)
    except Exception as exc:        # noqa: BLE001
        # Earth Engine raises its own exception types (not OSError or ValueError), and a failure of one variable
        # must be reported with its reason, not stop the notebook. Ctrl-C is not an Exception and still stops it.
        return VariableData(spec, None, error=f"{type(exc).__name__}: {exc}")
    shape = data.values.shape if data.values is not None else None
    if shape is not None and shape != ctx.grid.shape:
        return VariableData(spec, None, error=f"loaded {shape}, expected {ctx.grid.shape}")
    return data
