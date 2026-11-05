"""Synthetic layers for testing the pipeline offline. NOT real data: never report these results."""
from __future__ import annotations

import zlib

import numpy as np

from ..firstpass import Layers
from ..grid import Grid
from ..rasters import TOPSOIL_LAYERS, thickness_weighted


def smooth_field(rng: np.random.Generator, shape: tuple[int, int], blobs: int = 14) -> np.ndarray:
    """A smooth random field scaled to [0, 1] (sum of Gaussian blobs)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy / max(h - 1, 1)
    xx = xx / max(w - 1, 1)
    f = np.zeros(shape)
    for _ in range(blobs):
        cx, cy = rng.random(2)
        s = rng.uniform(0.08, 0.3)
        f += rng.uniform(0.4, 1.0) * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
    f -= f.min()
    return f / f.max() if f.max() > 0 else f


def synthetic_layers(grid: Grid, seed: int = 1, nodata_share: float = 0.02) -> Layers:
    rng = np.random.default_rng(seed)
    shape = grid.shape
    f1, f2, f3, f4, f5 = (smooth_field(rng, shape) for _ in range(5))

    slope = 40.0 * f1**2                                   # degrees, many gentle cells
    soc_base = 4.0 + 36.0 * f2                             # g/kg
    ph_base = 4.6 + 3.8 * f3
    soc_by = dict(zip(TOPSOIL_LAYERS, (soc_base * 1.35, soc_base, soc_base * 0.7), strict=True))
    ph_by = dict(zip(TOPSOIL_LAYERS, (ph_base - 0.15, ph_base, ph_base + 0.2), strict=True))

    gaps = rng.random(shape) < nodata_share                # soil nodata, as SoilGrids has over water
    for d in TOPSOIL_LAYERS:
        soc_by[d] = np.where(gaps, np.nan, soc_by[d])
        ph_by[d] = np.where(gaps, np.nan, ph_by[d])

    eligible = np.clip((f4 - 0.2) * 1.5, 0.0, 1.0)
    unknown = np.clip(0.3 * (1.0 - eligible) * f5, 0.0, 1.0 - eligible)
    return Layers(
        slope_deg=slope,
        soc_g_kg=thickness_weighted([soc_by[d] for d in TOPSOIL_LAYERS]),
        ph=thickness_weighted([ph_by[d] for d in TOPSOIL_LAYERS]),
        eligible_fraction=eligible,
        unknown_fraction=unknown,
        soc_by_depth=soc_by,
        ph_by_depth=ph_by,
        synthetic=True,
        notes=["SYNTHETIC layers: pipeline test only, not Haiti"],
    )


# ---------------------------------------------------------------------------
# One synthetic array per catalog variable, for testing the variable audit offline
# ---------------------------------------------------------------------------
# typical ranges for Haiti-like conditions; the values are made up and are never reported as real
_SYNTH_RANGE = {
    "annual_precipitation": (500.0, 2800.0), "mean_annual_temperature": (14.0, 27.0), "aridity_index": (0.3, 1.4),
    "soil_ph": (4.8, 8.2), "soil_organic_carbon": (8.0, 70.0), "slope": (0.0, 30.0),
    "tree_canopy_cover": (0.0, 95.0), "accessibility_travel_time": (15.0, 900.0),
    "distance_to_road": (0.0, 25.0), "electrification_index": (0.0, 6.0), "protected_area_status": (0.0, 1.0),
}


def synthetic_variable(spec, grid: Grid, seed: int = 1, nodata_share: float = 0.03) -> np.ndarray:
    """A made-up array with a plausible range for one catalog variable (a pipeline test, never real data)."""
    rng = np.random.default_rng(zlib.crc32(spec.id.encode()) + seed)       # stable across runs, unlike hash()
    f = smooth_field(rng, grid.shape)
    if spec.id == "land_cover":
        codes = np.array(sorted(spec.class_labels))
        out = codes[np.minimum((f * len(codes)).astype(int), len(codes) - 1)].astype(float)
    elif spec.id == "soil_drainage":
        out = np.minimum(1 + (f * 7).astype(int), 7).astype(float)
    else:
        lo, hi = _SYNTH_RANGE[spec.id]
        shaped = f**2 if spec.id in ("slope", "electrification_index", "protected_area_status") else f
        out = lo + (hi - lo) * shaped
    gaps = rng.random(grid.shape) < nodata_share
    return np.where(gaps, np.nan, out)
