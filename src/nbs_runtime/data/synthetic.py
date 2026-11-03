"""Synthetic layers for testing the pipeline offline. NOT real data: never report these results."""
from __future__ import annotations

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
