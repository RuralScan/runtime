"""Combine criterion scores under an approved policy.

Behaviour inferred from CIAT's six policy_conformance cases (confirm any other reading
with the method lead before relying on it):
- host False, or legal exclusion True            -> status "excluded"
- otherwise a host or legal value that is unknown -> "unknown"
- otherwise any required criterion score unknown  -> "unknown" (no renormalisation)
- otherwise                                       -> "scored": weighted sum plus the minimum
Masks have no weight. eligible_fraction is passed through, never folded into the score.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contract import Policy
from .responses import is_unknown


def combine(
    policy: Policy,
    scores: Mapping[str, float | None],
    host: bool | None,
    excluded: bool | None,
    eligible_fraction: float | None = None,
) -> dict[str, Any]:
    if host is False or excluded is True:
        return _out("excluded")
    if host is None or excluded is None:
        return _out("unknown")
    vals = [scores.get(c.rule_id) for c in policy.criteria]
    if any(is_unknown(v) for v in vals):
        return _out("unknown")
    weighted = sum(c.weight * float(scores[c.rule_id]) for c in policy.criteria)
    return {
        "status": "scored",
        "score": round(weighted, 10),
        "minimum": min(float(v) for v in vals),
        "eligible_fraction": eligible_fraction,
    }


def _out(status: str) -> dict[str, Any]:
    return {"status": status, "score": None, "minimum": None, "eligible_fraction": None}
