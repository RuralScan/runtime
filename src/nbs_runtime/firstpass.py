"""CIAT's partial terrain-and-soil index, computed over rasters.

Same rules as aggregate.combine, on arrays: host gate and legal mask first, then the weighted sum of
the three scores, with unknown propagated and no renormalisation. Four pixel states are kept apart.

Two modes:
  "policy"       the legal-exclusion mask must be assessed. It is not yet, so every pixel that is not
                 excluded by the host gate comes out unknown, exactly as CIAT's policy cases say.
  "exploratory"  the legal mask is NOT applied (treated as no exclusion) so the maps can be
                 compared. Results are labelled exploratory and are not policy-conformant.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .rasters import level_index, score_array
from .simulate import LevelSpace

STATUS = {
    0: "scored",
    1: "excluded: no eligible host",
    2: "unknown: host",
    3: "unknown: a score",
    4: "unknown: legal mask not assessed",
}
MODES = ("policy", "exploratory")


@dataclass
class Layers:
    slope_deg: np.ndarray
    soc_g_kg: np.ndarray
    ph: np.ndarray
    eligible_fraction: np.ndarray
    unknown_fraction: np.ndarray
    soc_by_depth: dict[str, np.ndarray] | None = None
    ph_by_depth: dict[str, np.ndarray] | None = None
    synthetic: bool = False
    host_gate_applied: bool = True          # False: every cell is treated as host (exploratory overview)
    notes: list[str] = field(default_factory=list)

    def check_shapes(self) -> None:
        shapes = {a.shape for a in (self.slope_deg, self.soc_g_kg, self.ph,
                                    self.eligible_fraction, self.unknown_fraction)}
        if len(shapes) != 1:
            raise ValueError(f"layers are not on one grid: shapes {sorted(shapes)}")


@dataclass
class FirstPass:
    status: np.ndarray                      # uint8, codes in STATUS
    pete: np.ndarray                        # CIAT index, NaN unless scored
    combo_id: np.ndarray                    # class-combination id, -1 unless scored
    eligible_fraction: np.ndarray
    mode: str
    depth_index: dict[str, np.ndarray] | None = None
    depth_range: np.ndarray | None = None   # max - min of the index across the three depths

    @property
    def scored(self) -> np.ndarray:
        return self.status == 0

    def status_counts(self) -> dict[str, int]:
        return {STATUS[k]: int((self.status == k).sum()) for k in STATUS}

    def combo_counts(self, n_combos: int) -> dict[int, int]:
        ids = self.combo_id[self.scored]
        c = np.bincount(ids, minlength=n_combos)
        return {i: int(n) for i, n in enumerate(c)}


def _scores(space: LevelSpace, slope, soc, ph) -> list[np.ndarray]:
    return [score_array(opt["intervals"], arr) for opt, arr in zip(space.options, (slope, soc, ph), strict=True)]


def _index(space: LevelSpace, scores: list[np.ndarray]) -> np.ndarray:
    total = sum(w * s for w, s in zip(space.weights, scores, strict=True))
    return np.where(np.any([np.isnan(s) for s in scores], axis=0), np.nan, total)


def run_first_pass(space: LevelSpace, layers: Layers, mode: str = "exploratory") -> FirstPass:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    layers.check_shapes()
    fe = np.asarray(layers.eligible_fraction, dtype=float)
    fu = np.asarray(layers.unknown_fraction, dtype=float)

    host_nan = np.isnan(fe) | np.isnan(fu)
    host_true = ~host_nan & (fe > 0)
    host_false = ~host_nan & (fe <= 0) & (fu <= 0)
    host_unknown = ~host_true & ~host_false

    scores = _scores(space, layers.slope_deg, layers.soc_g_kg, layers.ph)
    scores_known = ~np.any([np.isnan(s) for s in scores], axis=0)
    index = _index(space, scores)

    status = np.zeros(fe.shape, dtype=np.uint8)
    status[host_unknown] = 2
    if mode == "policy":
        status[host_true] = 4                       # legal mask not assessed -> unknown
    else:
        status[host_true & ~scores_known] = 3
        status[host_true & scores_known] = 0
    status[host_false] = 1                          # no host: excluded whatever the rest is

    scored = status == 0
    idx = [level_index(lv, s) for lv, s in zip(space.levels, scores, strict=True)]
    combo = np.full(fe.shape, -1, dtype=np.int32)
    if scored.any():
        flat = np.ravel_multi_index(tuple(i[scored] for i in idx), tuple(len(lv) for lv in space.levels))
        combo[scored] = flat
    pete = np.where(scored, index, np.nan)

    depth_index = depth_range = None
    if layers.soc_by_depth and layers.ph_by_depth:
        depth_index = {}
        for d in layers.soc_by_depth:
            sc = _scores(space, layers.slope_deg, layers.soc_by_depth[d], layers.ph_by_depth[d])
            depth_index[d] = np.where(scored, _index(space, sc), np.nan)
        stack = np.stack(list(depth_index.values()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            depth_range = np.where(np.isnan(stack).any(axis=0), np.nan,
                                   np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0))
    return FirstPass(status, pete, combo, fe, mode, depth_index, depth_range)
