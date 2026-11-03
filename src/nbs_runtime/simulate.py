"""Ordinal simulation: how much does the index depend on the numbers given to the classes?

CIAT's functions give each class a score by convention (0.25, 0.5, 0.75, 0.9). Those numbers are
a convention, not a measurement. Here the class ORDER stays fixed, and the numbers are redrawn many
times in increasing order. The index of a pixel then becomes a distribution (median, 5-95% range)
instead of one number. CIAT's own index is kept next to it as the baseline; this does not replace it.

What stays fixed: class limits, the class a pixel falls in, and (by default) the weights.
What is drawn: the score of each class, per variable ("independent") or one common scale for the
same class label across variables ("shared").

A pixel's index depends only on the class it falls in for each variable, so everything is computed
once per class combination (4 x 4 x 3 = 48 here) and a map is a lookup. Pure Python on purpose.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from .contract import Package, Policy
from .responses import score_value

Combo = tuple[int, ...]


@dataclass(eq=False)
class LevelSpace:
    """The ordered classes of each criterion, as CIAT's policy defines them."""

    criteria: tuple[str, ...]
    names: tuple[str, ...]
    levels: tuple[tuple[float, ...], ...]     # CIAT scores per criterion, ascending
    weights: tuple[float, ...]
    options: tuple[dict[str, Any], ...] = field(default=(), repr=False)

    @property
    def n_combos(self) -> int:
        n = 1
        for lv in self.levels:
            n *= len(lv)
        return n

    def combos(self) -> list[Combo]:
        return list(product(*[range(len(lv)) for lv in self.levels]))

    def baseline(self, combo: Combo) -> float:
        """CIAT's index for a class combination (their scores, their weights)."""
        return sum(w * lv[i] for w, lv, i in zip(self.weights, self.levels, combo, strict=True))

    def combo_of(self, values: Sequence[Any]) -> Combo | None:
        """Class combination of one pixel's raw values; None if any value is unknown."""
        idx = []
        for j, v in enumerate(values):
            s = score_value(self.options[j]["intervals"], v)
            if s is None:
                return None
            idx.append(self.levels[j].index(s))
        return tuple(idx)


def build_level_space(pkg: Package, policy: Policy) -> LevelSpace:
    options, levels = [], []
    for c in policy.criteria:
        opt = pkg.option(c.rule_id, c.function_id)
        scores = sorted({iv["score"] for iv in opt["intervals"] if iv.get("score") is not None})
        options.append(opt)
        levels.append(tuple(float(s) for s in scores))
    return LevelSpace(
        criteria=tuple(c.rule_id for c in policy.criteria),
        names=tuple(c.rule_id.split("__")[-1] for c in policy.criteria),
        levels=tuple(levels),
        weights=tuple(c.weight for c in policy.criteria),
        options=tuple(options),
    )


def draw_scales(
    space: LevelSpace, rng: random.Random, mode: str = "independent", prior: str = "uniform"
) -> list[list[float]]:
    """One scenario: ascending scores for every class of every criterion, all in [0, 1]."""
    if prior != "uniform":
        raise ValueError("only the 'uniform' prior is implemented")
    if mode == "independent":
        return [sorted(rng.random() for _ in lv) for lv in space.levels]
    if mode == "shared":
        vocab = sorted({s for lv in space.levels for s in lv})
        common = sorted(rng.random() for _ in vocab)
        return [[common[vocab.index(s)] for s in lv] for lv in space.levels]
    raise ValueError(f"mode must be 'independent' or 'shared', got {mode!r}")


def _draw_weights(base: Sequence[float], concentration: float, rng: random.Random) -> list[float]:
    if concentration <= 0:
        return list(base)
    g = [rng.gammavariate(b * concentration, 1.0) for b in base]
    total = sum(g)
    return [v / total for v in g]


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""
    if not sorted_values:
        raise ValueError("empty sample")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


@dataclass
class SimResult:
    space: LevelSpace
    n_draws: int
    seed: int
    mode: str
    prior: str
    weight_concentration: float
    values: dict[Combo, list[float]]

    def summary(self, combo: Combo) -> dict[str, float]:
        v = sorted(self.values[combo])
        return {
            "baseline": self.space.baseline(combo),
            "p05": quantile(v, 0.05),
            "median": quantile(v, 0.50),
            "p95": quantile(v, 0.95),
        }

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for combo in self.space.combos():
            row: dict[str, Any] = {"combo": combo}
            row.update(self.summary(combo))
            rows.append(row)
        return rows

    def prob_greater(self, a: Combo, b: Combo) -> float:
        """Share of scenarios in which combination a has a higher index than b."""
        pairs = zip(self.values[a], self.values[b], strict=True)
        return sum(1 for x, y in pairs if x > y) / self.n_draws

    def prob_top_third(self, counts: dict[Combo, int]) -> dict[Combo, float]:
        """Probability that a class combination falls in the top third of the pixel population.

        `counts` is how many pixels fall in each combination (from a raster, later). In each
        scenario the combinations are ranked by index and the top third of PIXELS is marked.
        """
        total = sum(counts.values())
        target = total / 3.0
        hits = dict.fromkeys(counts, 0)
        keys = [c for c, n in counts.items() if n > 0]
        for d in range(self.n_draws):
            ranked = sorted(keys, key=lambda c: self.values[c][d], reverse=True)
            before = 0
            for c in ranked:
                if before >= target:
                    break
                hits[c] += 1
                before += counts[c]
        return {c: hits[c] / self.n_draws for c in counts}

    def metadata(self) -> dict[str, Any]:
        """What a run manifest should record about this simulation."""
        return {
            "n_draws": self.n_draws,
            "seed": self.seed,
            "scale_mode": self.mode,
            "prior": self.prior,
            "weight_concentration": self.weight_concentration,
            "criteria": list(self.space.criteria),
            "levels": [list(lv) for lv in self.space.levels],
            "weights": list(self.space.weights),
        }


def simulate(
    space: LevelSpace,
    n_draws: int = 5000,
    seed: int = 20261103,
    mode: str = "independent",
    prior: str = "uniform",
    weight_concentration: float = 0.0,
) -> SimResult:
    """Run the simulation for every class combination. Same seed, same result."""
    rng = random.Random(seed)
    combos = space.combos()
    values: dict[Combo, list[float]] = {c: [] for c in combos}
    for _ in range(n_draws):
        scales = draw_scales(space, rng, mode, prior)
        w = _draw_weights(space.weights, weight_concentration, rng)
        for c in combos:
            values[c].append(sum(w[j] * scales[j][c[j]] for j in range(len(c))))
    return SimResult(space, n_draws, seed, mode, prior, weight_concentration, values)


def pixels_to_counts(space: LevelSpace, pixels: Iterable[Sequence[Any]]) -> dict[Combo, int]:
    """Count pixels per class combination; pixels with any unknown value are left out."""
    counts: dict[Combo, int] = dict.fromkeys(space.combos(), 0)
    for values in pixels:
        combo = space.combo_of(values)
        if combo is not None:
            counts[combo] += 1
    return counts
