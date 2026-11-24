"""Summarise a fine-grid result in coarse blocks, for a quick look at a large area.

The calculation stays on the fine grid; only the RESULT is summarised. A block's value is the mean over
its scored cells, and a block with too few scored cells stays unknown. A block is never scored from an
averaged input: averaging the input of a curve hides what is inside the block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid


def _blocks(a: np.ndarray, factor: int, fill: float) -> np.ndarray:
    h, w = a.shape
    ph, pw = (-h) % factor, (-w) % factor
    a = np.pad(a, ((0, ph), (0, pw)), constant_values=fill)
    return a.reshape(a.shape[0] // factor, factor, a.shape[1] // factor, factor)


def block_mean(values: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    """Mean of the valid cells of each block; NaN where a block has none."""
    v = _blocks(np.where(valid, values, 0.0), factor, 0.0).sum(axis=(1, 3))
    n = _blocks(valid.astype(float), factor, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, v / n, np.nan)


def block_share(mask: np.ndarray, factor: int) -> np.ndarray:
    """Share of the real (not padding) cells of each block where mask is True."""
    real = _blocks(np.ones(mask.shape), factor, 0.0).sum(axis=(1, 3))
    hit = _blocks(mask.astype(float), factor, 0.0).sum(axis=(1, 3))
    return hit / real


@dataclass
class Overview:
    S: np.ndarray                 # mean S of the scored cells, NaN for thin blocks
    scored_share: np.ndarray      # share of each block that is scored
    grid: Grid
    factor: int
    min_scored_share: float


def make_overview(S: np.ndarray, grid: Grid, overview_m: int, min_scored_share: float = 0.25) -> Overview:
    if overview_m % grid.resolution_m:
        raise ValueError(f"overview_m ({overview_m}) must be a whole multiple of the grid ({grid.resolution_m} m)")
    factor = overview_m // grid.resolution_m
    scored = ~np.isnan(S)
    share = block_share(scored, factor)
    mean = np.where(share >= min_scored_share, block_mean(np.nan_to_num(S), scored, factor), np.nan)
    h, w = mean.shape
    minx, _, _, maxy = grid.bbox
    r = grid.resolution_deg * factor
    coarse = Grid((minx, maxy - h * r, minx + w * r, maxy), overview_m, r, w, h)
    return Overview(mean, share, coarse, factor, min_scored_share)
