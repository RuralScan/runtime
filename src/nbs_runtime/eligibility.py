"""G, the eligibility of each cell: is the land cover one where the NbS can be implemented?

Land cover is read as the share of each ESA WorldCover class in every cell (data/gee.py), so the eligible
classes are a run parameter and changing them never needs a new extraction. A cell is eligible when the
eligible classes cover at least ``min_share`` of it. Without class shares (synthetic data) the dominant
class is used. A cell with no land-cover data is unknown (NaN), never "not eligible".
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Eligibility:
    G: np.ndarray                  # 1 eligible, 0 not eligible, NaN unknown (no land-cover data)
    share: np.ndarray              # share of each cell in the eligible classes (NaN where unknown)
    classes: tuple[int, ...]
    min_share: float
    notes: list[str] = field(default_factory=list)


def from_shares(shares: Mapping[int, np.ndarray], classes: Sequence[int], min_share: float = 0.5) -> Eligibility:
    stack = np.stack([np.asarray(a, dtype=float) for a in shares.values()])
    no_data = np.isnan(stack).all(axis=0)
    zero = np.zeros(stack.shape[1:])
    share = sum((np.nan_to_num(np.asarray(shares[c], dtype=float)) for c in classes if c in shares), zero)
    share = np.where(no_data, np.nan, share)
    return Eligibility(_gate(share, min_share), share, tuple(classes), min_share,
                       [f"eligible when classes {list(classes)} cover at least {min_share:.0%} of the cell"])


def from_classes(dominant: np.ndarray, classes: Sequence[int], min_share: float = 0.5) -> Eligibility:
    d = np.asarray(dominant, dtype=float)
    share = np.where(np.isnan(d), np.nan, np.isin(d, list(classes)).astype(float))
    return Eligibility(_gate(share, min_share), share, tuple(classes), min_share,
                       [f"eligible when the dominant class is one of {list(classes)} (no class shares)"])


def everywhere(shape: tuple[int, ...]) -> Eligibility:
    return Eligibility(np.ones(shape), np.ones(shape), (), 0.0,
                       ["eligibility SKIPPED: every cell treated as eligible land cover"])


def _gate(share: np.ndarray, min_share: float) -> np.ndarray:
    return np.where(np.isnan(share), np.nan, (share >= min_share - 1e-9).astype(float))
