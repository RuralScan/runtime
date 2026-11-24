"""Sensitivity of the suitability map to the weights and to each variable.

Three tests, none of which needs anything beyond the recipes (no low/high bands):

1. equal_weights   the same map with equal modulating weights (the methodology's benchmark).
2. dirichlet       many draws of the modulating weights around the recipe weights. Per pixel:
                   median, 5-95% range and the probability of being in the top quintile.
3. leave_one_out   the map recomputed without each variable in turn (weights renormalised).

Only the modulating weights move in tests 1 and 2, so L and G stay fixed and the work is a matrix
product. Draw-level statistics (top-quintile threshold, rank agreement) use a random sample of the
scored pixels, as the methodology does for class limits.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .recipes import Family
from .suitability import Result, compute

TOP_Q = 0.8      # top quintile


def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (pure numpy)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_mask(S: np.ndarray, q: float = TOP_Q) -> np.ndarray:
    """Pixels in the top quintile of the positive scores."""
    pos = S[(S > 0) & ~np.isnan(S)]
    if not pos.size:
        return np.zeros(S.shape, dtype=bool)
    return (S >= np.quantile(pos, q)) & (S > 0)


def retained(base_top: np.ndarray, other_top: np.ndarray) -> float:
    """Share of the base top-quintile pixels still in the top quintile."""
    n = int(base_top.sum())
    return float((base_top & other_top).sum() / n) if n else float("nan")


# --------------------------------------------------------------------------- 1. equal weights

def equal_weights(family: Family, layers: Mapping[str, Any], eligible: Any = None,
                  base: Result | None = None, outside: str = "unknown") -> dict[str, Any]:
    base = base or compute(family, layers, eligible, outside=outside)
    eq = compute(family, layers, eligible, weights="equal", outside=outside)
    return {
        "result": eq,
        "weights": eq.weights,
        "spearman": spearman(base.S, eq.S),
        "top_retained": retained(top_mask(base.S), top_mask(eq.S)),
        "mean_abs_change": float(np.nanmean(np.abs(eq.S - base.S))),
    }


# --------------------------------------------------------------------------- 2. Dirichlet

@dataclass
class DirichletResult:
    median: np.ndarray
    p05: np.ndarray
    p95: np.ndarray
    p_top: np.ndarray            # probability of being in the top quintile, per pixel
    base_top: np.ndarray         # top quintile of the base map
    spearman: np.ndarray         # rank agreement of each draw with the base map
    weights: np.ndarray          # drawn weights, shape (n_draws, n_modulating)
    modulating: tuple[str, ...]
    concentration: float

    def summary(self, robust: float = 0.8) -> dict[str, float]:
        bt = self.base_top
        n = max(int(bt.sum()), 1)
        return {
            "spearman_median": float(np.nanmedian(self.spearman)),
            "spearman_p05": float(np.nanpercentile(self.spearman, 5)),
            "base_top_that_is_robust": float((self.p_top[bt] >= robust).sum() / n),
            "mean_range_5_95": float(np.nanmean(self.p95 - self.p05)),
        }


def dirichlet(family: Family, layers: Mapping[str, Any], eligible: Any = None,
              n_draws: int = 200, concentration: float = 50.0, seed: int = 0,
              outside: str = "unknown", sample_size: int = 50_000, chunk: int = 200_000,
              base: Result | None = None) -> DirichletResult:
    """Draw the modulating weights from Dirichlet(concentration * recipe weights).

    A larger concentration keeps the draws closer to the recipe weights; float('inf') fixes them.
    Zero weights get a small floor so the distribution is defined.
    """
    base = base or compute(family, layers, eligible, outside=outside)
    mods = base.modulating
    shape = base.S.shape
    sc = base.scored.ravel()
    idx_scored = np.flatnonzero(sc)
    Z = np.stack([base.z[v].ravel()[idx_scored] for v in mods]) if mods else np.ones((1, idx_scored.size))
    Lv = base.L.ravel()[idx_scored]
    w0 = np.array([base.weights[v] for v in mods]) if mods else np.array([1.0])

    rng = np.random.default_rng(seed)
    if len(w0) <= 1 or not np.isfinite(concentration):
        W = np.tile(w0, (n_draws, 1))
    else:
        W = rng.dirichlet(np.maximum(concentration * w0, 1e-2), size=n_draws)

    # draw-level statistics on a sample of the scored pixels
    samp = rng.choice(idx_scored.size, size=min(sample_size, idx_scored.size), replace=False)
    S_samp = (W @ Z[:, samp]) * Lv[samp]
    thr = np.array([np.quantile(r[r > 0], TOP_Q) if (r > 0).any() else np.inf for r in S_samp])
    base_samp = base.S.ravel()[idx_scored][samp]
    rho = np.array([spearman(base_samp, r) for r in S_samp])

    med, p05, p95, ptop = (np.full(idx_scored.size, np.nan) for _ in range(4))
    for a in range(0, idx_scored.size, chunk):
        b = min(a + chunk, idx_scored.size)
        Sc = (W @ Z[:, a:b]) * Lv[a:b]
        med[a:b] = np.median(Sc, axis=0)
        p05[a:b], p95[a:b] = np.percentile(Sc, [5, 95], axis=0)
        ptop[a:b] = ((Sc >= thr[:, None]) & (Sc > 0)).mean(axis=0)

    def full(v):
        out = np.full(sc.size, np.nan)
        out[idx_scored] = v
        return out.reshape(shape)

    return DirichletResult(full(med), full(p05), full(p95), full(ptop), top_mask(base.S),
                           rho, W, mods, concentration)


# --------------------------------------------------------------------------- 3. leave one out

def leave_one_out(family: Family, layers: Mapping[str, Any], eligible: Any = None,
                  base: Result | None = None, outside: str = "unknown") -> dict[str, dict[str, Any]]:
    """Recompute without each variable. Compared on the pixels scored in both maps."""
    base = base or compute(family, layers, eligible, outside=outside)
    btop = top_mask(base.S)
    out = {}
    for v in family.variables:
        if len(family.variables) == 1:
            break
        r = compute(family, layers, eligible, outside=outside, drop=(v,))
        both = base.scored & r.scored
        out[v] = {
            "role": family.roles[v],
            "spearman": spearman(np.where(both, base.S, np.nan), np.where(both, r.S, np.nan)),
            "top_retained": retained(btop, top_mask(r.S)),
            "mean_abs_change": float(np.nanmean(np.abs(np.where(both, r.S - base.S, np.nan))))
                               if both.any() else float("nan"),
            "newly_scored": int((r.scored & ~base.scored).sum()),
            "result": r,
        }
    return out
