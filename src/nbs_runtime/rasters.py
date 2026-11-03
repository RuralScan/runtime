"""Array versions of CIAT's rules, plus the input conversions Pete's brief spells out.

Everything here is plain numpy and offline. Unknown is NaN (or -1 for class indices) and is
never converted to zero or filled with a mean.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# ---- SoilGrids conversions: "convert each native value once" (DATASET_INPUTS.md) ----
SOC_DG_PER_KG_TO_G_PER_KG = 0.1      # soc_mean is stored in dg/kg
PH_X10_TO_PH = 0.1                   # phh2o_mean is stored as pH x 10
TOPSOIL_LAYERS = ("0-5", "5-15", "15-30")
TOPSOIL_THICKNESS_CM = (5.0, 10.0, 15.0)


def score_array(intervals: Sequence[Mapping[str, Any]], values: Any) -> np.ndarray:
    """Vectorised twin of responses.score_value: stepwise, endpoint-aware, unknown stays NaN.

    A value in no interval is NaN. Two intervals that both contain a value but disagree on the
    score, or a matching interval with no score, also give NaN.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    seen = np.zeros(x.shape, dtype=bool)
    conflict = np.zeros(x.shape, dtype=bool)
    finite = ~np.isnan(x)
    for iv in intervals:
        lo, hi = iv.get("lower"), iv.get("upper")
        m = finite.copy()
        if lo is not None:
            m &= (x > lo) | ((x == lo) & bool(iv.get("lower_inclusive")))
        if hi is not None:
            m &= (x < hi) | ((x == hi) & bool(iv.get("upper_inclusive")))
        score = iv.get("score")
        if score is None:
            conflict |= m
            continue
        fresh = m & ~seen
        clash = m & seen & ~np.isclose(out, score, equal_nan=False)
        out[fresh] = score
        seen |= fresh
        conflict |= clash
    out[conflict] = np.nan
    return out


def level_index(levels: Sequence[float], scores: np.ndarray) -> np.ndarray:
    """Position of each score among a criterion's ascending levels; -1 where unknown."""
    lv = np.asarray(levels, dtype=float)
    s = np.asarray(scores, dtype=float)
    idx = np.searchsorted(lv, np.nan_to_num(s, nan=-1.0))
    idx = np.clip(idx, 0, len(lv) - 1)
    ok = ~np.isnan(s) & np.isclose(lv[idx], s)
    return np.where(ok, idx, -1).astype(np.int16)


def thickness_weighted(layers: Sequence[np.ndarray],
                       thickness: Sequence[float] = TOPSOIL_THICKNESS_CM) -> np.ndarray:
    """Pete's 0-30 cm proposal: (5*x0_5 + 10*x5_15 + 15*x15_30) / 30, already converted layers.

    A cell missing any layer is unknown: no renormalisation onto the remaining depths.
    """
    stack = np.stack([np.asarray(a, dtype=float) for a in layers])
    w = np.asarray(thickness, dtype=float).reshape(-1, *([1] * (stack.ndim - 1)))
    out = (stack * w).sum(axis=0) / w.sum()
    out[np.isnan(stack).any(axis=0)] = np.nan
    return out


def soil_value_warnings(soc_g_kg: np.ndarray, ph: np.ndarray) -> list[str]:
    """Cheap guards against converting twice, or not at all (physical plausibility only)."""
    warnings = []
    soc = soc_g_kg[~np.isnan(soc_g_kg)]
    p = ph[~np.isnan(ph)]
    if soc.size:
        if np.median(soc) < 1.0:
            warnings.append("SOC median below 1 g/kg: values may have been scaled twice")
        if np.median(soc) > 600.0:
            warnings.append("SOC median above 600 g/kg: values may not have been converted")
    if p.size:
        if np.median(p) > 11.0:
            warnings.append("pH median above 11: values may not have been divided by 10")
        if np.median(p) < 2.0:
            warnings.append("pH median below 2: values may have been scaled twice")
    return warnings


def gate_class_groups(params: Mapping[str, Any],
                      provided_masks: Sequence[str] = ()) -> dict[str, list[int]]:
    """Split a host gate's class codes into eligible, ineligible and unknown, from the contract.

    A class that needs an extra host input (for example managed grassland without grazing) is
    unknown until that input is provided. Codes outside the gate's class map are unknown too.
    """
    class_map = {int(k): v for k, v in params["class_map"].items()}
    needs = {int(k): list(v) for k, v in params.get("class_requirements", {}).items()}
    eligible, ineligible, unknown = [], [], []
    for code, v in sorted(class_map.items()):
        if v == 0:
            ineligible.append(code)
        elif needs.get(code) and not all(n in provided_masks for n in needs[code]):
            unknown.append(code)
        else:
            eligible.append(code)
    return {"eligible": eligible, "ineligible": ineligible, "unknown": unknown}


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Mean over factor x factor blocks (offline stand-in for the aggregation done in Earth Engine)."""
    h, w = a.shape
    h2, w2 = h - h % factor, w - w % factor
    return a[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def host_fractions_from_classes(
    groups: Mapping[str, Sequence[int]], classes: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Eligible and unknown fractions per output cell from a fine class map (offline stand-in)."""
    c = np.asarray(classes)
    eligible = np.isin(c, list(groups["eligible"])).astype(float)
    known = np.isin(c, list(groups["eligible"]) + list(groups["ineligible"]))
    unknown = (~known).astype(float)
    return block_mean(eligible, factor), block_mean(unknown, factor)
