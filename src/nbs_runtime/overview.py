"""A coarse overview of a fine-grid result: compute at 250 m, then summarise blocks of cells.

The method is applied where the data supports it (every fine cell is scored on its own) and only the
RESULT is summarised, so a block is never scored from an averaged input. A block's value is the mean over
its scored cells; a block with fewer than `min_scored_share` scored cells is left unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._result_fields import accept_baseline_keyword
from .compare import Comparison
from .firstpass import FirstPass
from .grid import Grid


def _blocks(a: np.ndarray, factor: int, fill: float) -> np.ndarray:
    h, w = a.shape
    ph, pw = (-h) % factor, (-w) % factor
    a = np.pad(a, ((0, ph), (0, pw)), constant_values=fill)
    return a.reshape(a.shape[0] // factor, factor, a.shape[1] // factor, factor)


def block_mean(values: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    """Mean of `values` over the cells of each block that are `valid`; NaN where none is."""
    v = _blocks(np.where(valid, values, 0.0), factor, 0.0)
    n = _blocks(valid.astype(float), factor, 0.0)
    count = n.sum(axis=(1, 3))
    total = v.sum(axis=(1, 3))
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def block_share(mask: np.ndarray, factor: int) -> np.ndarray:
    """Share of the REAL cells of each block for which `mask` is true (padding is not counted)."""
    real = _blocks(np.ones(mask.shape), factor, 0.0).sum(axis=(1, 3))
    return _blocks(mask.astype(float), factor, 0.0).sum(axis=(1, 3)) / real


@accept_baseline_keyword("ciat")
@dataclass
class Overview:
    factor: int
    resolution_m: int
    extent: tuple[float, float, float, float]
    min_scored_share: float
    ciat: np.ndarray                     # Archived field backing baseline_index
    median: np.ndarray
    p05: np.ndarray
    p95: np.ndarray
    p_top: np.ndarray
    coverage: np.ndarray             # share of the block's cells that were scored

    @property
    def baseline_index(self) -> np.ndarray:
        """Mean baseline score of the scored cells in each displayed block."""
        return self.ciat

    @baseline_index.setter
    def baseline_index(self, value: np.ndarray) -> None:
        self.ciat = value

    @property
    def width(self) -> np.ndarray:
        return self.p95 - self.p05

    @property
    def shape(self) -> tuple[int, int]:
        return self.baseline_index.shape


def make_overview(fp: FirstPass, comp: Comparison, grid: Grid, overview_m: int,
                  min_scored_share: float = 0.5) -> Overview:
    factor = round(overview_m / grid.resolution_m)
    if factor < 1 or abs(factor * grid.resolution_m - overview_m) > 1e-6:
        raise ValueError(f"overview_m ({overview_m}) must be a whole multiple of the grid resolution "
                         f"({grid.resolution_m} m)")
    scored = fp.scored
    coverage = block_share(scored, factor)
    keep = coverage >= min_scored_share

    def mean_of(a: np.ndarray) -> np.ndarray:
        out = block_mean(a, scored, factor)
        return np.where(keep, out, np.nan)

    nh, nw = coverage.shape
    minx, _, _, maxy = grid.bbox
    r = grid.resolution_deg * factor
    return Overview(
        factor=factor, resolution_m=overview_m,
        extent=(minx, minx + nw * r, maxy - nh * r, maxy),
        min_scored_share=min_scored_share,
        baseline_index=mean_of(fp.baseline_index), median=mean_of(comp.median), p05=mean_of(comp.p05),
        p95=mean_of(comp.p95), p_top=mean_of(comp.p_top),
        coverage=coverage,
    )
