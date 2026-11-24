"""Turn raw values into 0-1 scores with a recipe's linear function.

Between the first and the last node the score is interpolated linearly. Outside the nodes there is
no evidence: by default the score is unknown (NaN), and the pixel is reported as "not evaluated".
``outside="clamp"`` repeats the score of the nearest end node instead; it exists for comparison and
must be labelled when used. A missing input (NaN) is always unknown, never zero.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .recipes import Recipe

OUTSIDE = ("unknown", "clamp")


def evaluate_nodes(x: Sequence[float], score: Sequence[float], values, outside: str = "unknown") -> np.ndarray:
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}")
    xs = np.asarray(x, dtype=float)
    ss = np.asarray(score, dtype=float)
    v = np.asarray(values, dtype=float)
    out = np.interp(v, xs, ss)
    out = np.where(np.isnan(v), np.nan, out)
    if outside == "unknown":
        out = np.where((v < xs[0]) | (v > xs[-1]), np.nan, out)
    return out


def evaluate(recipe: Recipe, values, outside: str = "unknown") -> np.ndarray:
    return evaluate_nodes(recipe.x, recipe.score, values, outside)
