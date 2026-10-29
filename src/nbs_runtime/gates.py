"""Categorical host gates (eligibility). They carry no weight in the MCDA.

Each gate maps exact class codes (for example ESA WorldCover) to 1 (host allowed) or 0
(host excluded). Some classes need an extra host input (a supplied mask or a crop fraction).
Unmapped classes and missing inputs stay unknown; unknown is never converted to zero.

Statuses: mapped, excluded, host_absent, missing_host_context, unmapped_class.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .responses import is_unknown


def evaluate_gate(
    params: Mapping[str, Any],
    classes: Sequence[Any],
    host_masks: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[list[float | None], list[str]]:
    class_map = {str(k): v for k, v in params["class_map"].items()}
    requirements = {str(k): list(v) for k, v in params.get("class_requirements", {}).items()}
    host_masks = host_masks or {}
    values: list[float | None] = []
    statuses: list[str] = []
    for i, code in enumerate(classes):
        if is_unknown(code) or str(int(code)) not in class_map:
            values.append(None)
            statuses.append("unmapped_class")
            continue
        key = str(int(code))
        base = class_map[key]
        if base == 0:
            values.append(0.0)
            statuses.append("excluded")
            continue
        needed = requirements.get(key, [])
        if not needed:
            values.append(float(base))
            statuses.append("mapped")
            continue
        parts = [host_masks.get(name, [None] * len(classes))[i] for name in needed]
        if any(is_unknown(p) for p in parts):
            values.append(None)
            statuses.append("missing_host_context")
            continue
        v = min(float(p) for p in parts)  # AND across required inputs
        values.append(v)
        statuses.append("mapped" if v > 0 else "host_absent")
    return values, statuses
