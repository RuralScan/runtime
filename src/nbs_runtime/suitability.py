"""Suitability index, approach 3 of the RuralScan MCDA methodology.

    S = L * G * M

    L  minimum of the scores of the limiting variables (no weights). The most limiting factor rules,
       so good terrain can never compensate for a climate where the practice does not work.
    G  eligibility (land cover allowed for the NbS): 1 or 0, NaN where land cover is unknown.
       No other hard limits.
    M  weighted sum of the scores of the modulating variables, with the recipe weights renormalised
       among them so that they sum to 1. Weights are never renormalised per pixel.

A family with no limiting variable has L = 1; one with no modulating variable has M = 1.

Pixel states (precedence from top to bottom):
    NOT_ELIGIBLE      the land cover is not eligible for this NbS
    NO_DATA           a required input (or the land cover) has no data
    OUTSIDE_EVIDENCE  a value lies outside the nodes of its recipe: there is no evidence for it
    SCORED            everything known; S can still be 0 if a curve reaches 0
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .curves import evaluate
from .recipes import Family

SCORED, NOT_ELIGIBLE, NO_DATA, OUTSIDE_EVIDENCE = 0, 1, 2, 3
STATUS = {
    SCORED: "scored",
    NOT_ELIGIBLE: "not eligible (land cover)",
    NO_DATA: "not evaluated: an input has no data",
    OUTSIDE_EVIDENCE: "not evaluated: a value is outside the evidence of its recipe",
}


@dataclass
class Result:
    S: np.ndarray                         # suitability, NaN unless scored
    L: np.ndarray                         # limiting score (NaN where unknown)
    M: np.ndarray                         # modulating score (NaN where unknown)
    G: np.ndarray                         # eligibility, 1 or 0
    z: dict[str, np.ndarray]              # 0-1 score of each variable
    status: np.ndarray                    # uint8, codes in STATUS
    driver: np.ndarray                    # index into `limiting` of the variable that set L, -1 if none
    limiting: tuple[str, ...]
    modulating: tuple[str, ...]
    weights: dict[str, float]             # modulating weights actually used (sum to 1)
    weight_mode: str
    outside: str
    dropped: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def scored(self) -> np.ndarray:
        return self.status == SCORED

    def status_counts(self) -> dict[str, int]:
        return {STATUS[k]: int((self.status == k).sum()) for k in STATUS}

    def outside_share(self, inputs: Mapping[str, Any]) -> dict[str, float]:
        """Share of eligible pixels with data whose value is outside each recipe's nodes."""
        elig = self.G > 0          # NaN compares False: unknown land cover is left out
        out = {}
        for v, zv in self.z.items():
            x = np.asarray(inputs[v], dtype=float)
            has = elig & ~np.isnan(x)
            out[v] = float((np.isnan(zv) & has).sum() / max(int(has.sum()), 1))
        return out

    def driver_counts(self) -> dict[str, int]:
        """How many scored pixels each limiting variable rules."""
        sc = self.scored
        return {name: int(((self.driver == i) & sc).sum()) for i, name in enumerate(self.limiting)}


def _stack(arrays: list[np.ndarray], shape) -> np.ndarray:
    return np.stack(arrays) if arrays else np.empty((0, *shape))


def compute(
    family: Family,
    layers: Mapping[str, Any],
    eligible: Any = None,
    weights: str = "recipe",
    custom_weights: Mapping[str, float] | None = None,
    outside: str = "unknown",
    drop: tuple[str, ...] = (),
) -> Result:
    """Suitability of one family on a grid.

    layers          variable name -> array of raw values (all on the same grid)
    eligible        G: boolean array, or 1/0/NaN (NaN = land cover unknown); None = everything eligible
    weights         "recipe" (renormalised recipe weights) or "equal"
    custom_weights  explicit modulating weights (normalised here); overrides `weights`
    drop            variables left out (used by the leave-one-out sensitivity test)
    """
    used = [v for v in family.variables if v not in drop]
    if not used:
        raise ValueError("every variable was dropped")
    missing = [v for v in used if v not in layers]
    if missing:
        raise KeyError(f"no layer for: {', '.join(missing)}")
    arrays = {v: np.asarray(layers[v], dtype=float) for v in used}
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"layers are not on one grid: {sorted(shapes)}")
    shape = shapes.pop()

    # eligible: booleans, or 1/0 with NaN where the land cover is unknown
    G = np.ones(shape) if eligible is None else np.asarray(eligible, dtype=float)
    if G.shape != shape:
        raise ValueError("eligible mask is not on the same grid as the layers")

    z = {v: evaluate(family.recipe(v), arrays[v], outside) for v in used}
    lim = [r.variable for r in family.by_role("limiting", drop)]
    mod = [r.variable for r in family.by_role("modulating", drop)]

    if custom_weights is not None:
        tot = sum(custom_weights[v] for v in mod)
        w = {v: custom_weights[v] / tot for v in mod}
        mode = "custom"
    else:
        w = family.modulating_weights(weights, drop)
        mode = weights

    # L: minimum of the limiting scores; unknown if any is unknown
    Z_lim = _stack([z[v] for v in lim], shape)
    if lim:
        lim_known = ~np.isnan(Z_lim).any(axis=0)
        L = np.where(lim_known, np.nanmin(np.where(np.isnan(Z_lim), np.inf, Z_lim), axis=0), np.nan)
        driver = np.where(lim_known, np.argmin(np.where(np.isnan(Z_lim), np.inf, Z_lim), axis=0), -1)
    else:
        L = np.ones(shape)
        driver = np.full(shape, -1)

    # M: weighted sum of the modulating scores; unknown if any is unknown (no renormalisation)
    M = sum(w[v] * z[v] for v in mod) if mod else np.ones(shape)
    M = np.asarray(M, dtype=float)

    S = L * M * G
    nodata = np.any([np.isnan(arrays[v]) for v in used], axis=0) | np.isnan(G)
    unknown = np.isnan(S) | np.isnan(L) | np.isnan(M)

    status = np.full(shape, SCORED, dtype=np.uint8)
    status[unknown] = OUTSIDE_EVIDENCE
    status[nodata] = NO_DATA
    status[G == 0] = NOT_ELIGIBLE
    S = np.where(status == SCORED, S, np.nan)

    notes = []
    if outside == "clamp":
        notes.append("outside='clamp': values beyond the evidence take the end-node score")
    if mode != "recipe":
        notes.append(f"weights: {mode}")
    if drop:
        notes.append(f"dropped: {', '.join(drop)}")
    return Result(S, L, M, G, z, status, driver.astype(np.int16), tuple(lim), tuple(mod), w, mode,
                  outside, tuple(drop), notes)


# --------------------------------------------------------------------------- across families

def aggregate_families(results: Mapping[str, Result]) -> dict[str, np.ndarray]:
    """Per-pixel summary over families, as the worked example does over species.

    mean     mean S over the families evaluated in the pixel (a family scoring 0 counts as 0)
    breadth  share of all families with S > 0
    quality  mean S of the families with S > 0 (0 if none)
    n_evaluated  number of families evaluated in the pixel; mean is NaN where it is 0

    Provisional: the rule for going from families to one layer per practice is still to be agreed.
    """
    stack = np.stack([r.S for r in results.values()])
    n = stack.shape[0]
    evaluated = ~np.isnan(stack)
    n_eval = evaluated.sum(axis=0)
    filled = np.where(evaluated, stack, 0.0)
    positive = filled > 0
    n_pos = positive.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_eval > 0, filled.sum(axis=0) / n_eval, np.nan)
        quality = np.where(n_pos > 0, np.where(positive, filled, 0).sum(axis=0) / n_pos, 0.0)
    return {"mean": mean, "breadth": n_pos / n, "quality": quality, "n_evaluated": n_eval}


# --------------------------------------------------------------------------- classes

@dataclass
class Classes:
    classes: np.ndarray        # -1 not scored, 0 score 0, 1..k from low to high
    edges: np.ndarray          # class limits over the positive scores
    k: int                     # number of classes above 0 actually used
    summary: dict[str, float]  # absolute min, median and max of the positive scores
    note: str = ""


def classify(S: np.ndarray, k: int = 5, edges: np.ndarray | None = None) -> Classes:
    """Score 0 is a class of its own; positive scores are split into k quantile classes.

    If the quantiles cannot separate k classes (ties), 3 are tried, then 1, and the note says so.
    Pass `edges` to use fixed class limits instead (for example ones stored with the recipe).
    """
    S = np.asarray(S, dtype=float)
    pos = S[(S > 0) & ~np.isnan(S)]
    summary = ({"min": float(pos.min()), "median": float(np.median(pos)), "max": float(pos.max())}
               if pos.size else {"min": np.nan, "median": np.nan, "max": np.nan})
    note = ""
    if edges is None:
        for kk in (k, 3, 1):
            e = np.quantile(pos, np.linspace(0, 1, kk + 1)) if pos.size else np.array([0.0, 0.0])
            if np.unique(e).size == kk + 1 or kk == 1:
                break
        if kk != k:
            note = f"{k} quantile classes were not separable (ties); {kk} used"
        edges, k = e, kk
    else:
        edges = np.asarray(edges, dtype=float)
        k = edges.size - 1
    cls = np.full(S.shape, -1, dtype=np.int8)
    cls[S == 0] = 0
    m = (S > 0) & ~np.isnan(S)
    cls[m] = np.clip(np.digitize(S[m], edges[1:-1], right=True) + 1, 1, k)
    return Classes(cls, edges, k, summary, note)
