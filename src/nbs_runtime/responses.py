"""Numeric response evaluation from a package function option's intervals.

Rules taken from CIAT's contract:
- intervals are stepwise: never interpolated, never extrapolated;
- endpoint inclusivity is respected;
- a value in no interval is unknown (None);
- two intervals that both contain a value but disagree on the score are unknown;
- None / NaN input is unknown, never zero.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

Unknown = None


def is_unknown(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _contains(iv: dict[str, Any], x: float) -> bool:
    lo, hi = iv.get("lower"), iv.get("upper")
    ok_lo = lo is None or x > lo or (x == lo and bool(iv.get("lower_inclusive")))
    ok_hi = hi is None or x < hi or (x == hi and bool(iv.get("upper_inclusive")))
    return ok_lo and ok_hi


def score_value(intervals: Sequence[dict[str, Any]], x: Any) -> float | None:
    if is_unknown(x):
        return Unknown
    hits = [iv.get("score") for iv in intervals if _contains(iv, float(x))]
    if not hits:
        return Unknown
    if any(h is None for h in hits) or len(set(hits)) > 1:
        return Unknown
    return float(hits[0])


def evaluate_option(option: dict[str, Any], values: Iterable[Any]) -> list[float | None]:
    intervals = option["intervals"]
    return [score_value(intervals, v) for v in values]
