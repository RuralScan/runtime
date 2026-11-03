"""Put the simulation next to CIAT's index, pixel by pixel.

The simulation is computed once per class combination; here each scored pixel looks up its own
combination. Nothing in this module changes CIAT's index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .firstpass import FirstPass
from .simulate import SimResult


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    x = np.asarray(x, dtype=float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = np.cumsum(obs)
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return (0.5 * (count[dense] + count[dense - 1] + 1))[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class Comparison:
    median: np.ndarray
    p05: np.ndarray
    p95: np.ndarray
    p_top: np.ndarray            # chance of being in the top third of scored pixels
    stats: dict[str, Any]

    @property
    def width(self) -> np.ndarray:
        return self.p95 - self.p05


def _lookup(res: SimResult, key: str) -> np.ndarray:
    return np.array([row[key] for row in res.table()], dtype=float)


def compare(fp: FirstPass, res: SimResult) -> Comparison:
    n = res.space.n_combos
    scored = fp.scored
    ids = fp.combo_id
    shape = fp.pete.shape

    def spread(table: np.ndarray) -> np.ndarray:
        out = np.full(shape, np.nan)
        out[scored] = table[ids[scored]]
        return out

    median, p05, p95 = (spread(_lookup(res, k)) for k in ("median", "p05", "p95"))

    counts = fp.combo_counts(n)
    combos = res.space.combos()
    p_by_combo = res.prob_top_third({combos[i]: c for i, c in counts.items()})
    p_table = np.array([p_by_combo[combos[i]] for i in range(n)])
    p_top = spread(p_table)

    stats: dict[str, Any] = {"n_scored": int(scored.sum()), "n_cells": int(scored.size)}
    if scored.any():
        pete = fp.pete[scored]
        thr = float(np.quantile(pete, 2 / 3))
        pete_top = pete >= thr
        likely_top = p_top[scored] > 0.5
        uncertain = (p_top[scored] > 0.2) & (p_top[scored] < 0.8)
        width = (p95 - p05)[scored]
        stats.update(
            spearman_pete_vs_median=spearman(pete, median[scored]),
            pete_top_third_threshold=thr,
            share_pete_top_third=float(pete_top.mean()),
            top_third_agreement=float((pete_top == likely_top).mean()),
            share_of_pete_top_third_that_is_fragile=float((~likely_top[pete_top]).mean()) if pete_top.any() else float("nan"),
            share_uncertain_top_third=float(uncertain.mean()),
            width_median=float(np.median(width)),
            width_p90=float(np.quantile(width, 0.9)),
            share_pete_above_p95=float((pete > p95[scored]).mean()),
        )
    return Comparison(median, p05, p95, p_top, stats)
