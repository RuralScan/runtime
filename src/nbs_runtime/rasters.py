"""Small raster helpers shared by the data loaders: soil depth weighting and land-cover class shares.

Everything here is plain numpy and offline. Unknown is NaN and is never converted to zero or filled
with a mean.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)     # ESA WorldCover v200 classes
TOPSOIL_LAYERS = ("0-5", "5-15", "15-30")                            # SoilGrids depths, cm
TOPSOIL_THICKNESS_CM = (5.0, 10.0, 15.0)


def thickness_weighted(layers: Sequence[np.ndarray],
                       thickness: Sequence[float] = TOPSOIL_THICKNESS_CM) -> np.ndarray:
    """0-30 cm thickness-weighted value: (5*x0_5 + 10*x5_15 + 15*x15_30) / 30, from converted layers.

    A cell missing any layer is unknown: no renormalisation onto the remaining depths.
    """
    stack = np.stack([np.asarray(a, dtype=float) for a in layers])
    w = np.asarray(thickness, dtype=float).reshape(-1, *([1] * (stack.ndim - 1)))
    out = (stack * w).sum(axis=0) / w.sum()
    out[np.isnan(stack).any(axis=0)] = np.nan
    return out


def share_sum_warnings(shares: Mapping[int, np.ndarray]) -> list[str]:
    """The class shares of a cell should add up to 1 wherever the cell has data."""
    total = sum(shares.values())
    ok = total[~np.isnan(total)]
    if ok.size and not (0.99 <= float(np.median(ok)) <= 1.01):
        return [f"land-cover shares add up to {float(np.median(ok)):.3f} (median), not 1: check the class list"]
    return []
