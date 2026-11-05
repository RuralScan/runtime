"""Audit one variable: numbers that show whether it loaded sensibly, and flags for what looks wrong.

The point is to look at a variable with your own eyes before it enters anything else: its range, how much of the
area has no data, and whether the values are in a plausible range for their unit (which catches a missing or a
doubled scale factor).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .data.variables_loader import VariableData
from .variables import CATALOG, VarSpec, role_text


@dataclass
class Audit:
    spec: VarSpec
    origin: str
    synthetic: bool
    values: np.ndarray | None
    n_cells: int = 0
    n_unknown: int = 0
    vmin: float = float("nan")
    vmax: float = float("nan")
    vmean: float = float("nan")
    vmedian: float = float("nan")
    share_outside: float = 0.0
    class_shares: dict[int, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def unknown_share(self) -> float:
        return self.n_unknown / self.n_cells if self.n_cells else float("nan")

    @property
    def ok(self) -> bool:
        return self.values is not None


def audit_data(data: VariableData, unknown_flag_share: float = 0.5, outside_flag_share: float = 0.005) -> Audit:
    spec = data.spec
    if data.values is None:
        return Audit(spec, "", data.synthetic, None, error=data.error, flags=[f"not loaded: {data.error}"])
    v = np.asarray(data.values, dtype=float)
    known = np.isfinite(v)
    a = Audit(spec, data.origin, data.synthetic, v, n_cells=int(v.size), n_unknown=int((~known).sum()),
              notes=list(data.notes))
    if not known.any():
        a.flags.append("no data anywhere in this area")
        return a
    x = v[known]
    a.vmin, a.vmax, a.vmean, a.vmedian = float(x.min()), float(x.max()), float(x.mean()), float(np.median(x))
    if spec.kind == "categorical":
        codes, counts = np.unique(x.astype(int), return_counts=True)
        a.class_shares = {int(c): float(n) / x.size for c, n in zip(codes, counts, strict=True)}
        if spec.class_labels:
            odd = sorted(set(a.class_shares) - set(spec.class_labels))
            if odd:
                a.flags.append(f"class codes not in the legend: {odd}")
    if spec.plausible is not None:
        lo, hi = spec.plausible
        a.share_outside = float(((x < lo) | (x > hi)).mean())
        if a.share_outside > outside_flag_share:
            a.flags.append(f"{100 * a.share_outside:.1f}% of values are outside {lo:g} to {hi:g} {spec.unit}: "
                           "check the units and the scale factor")
    if a.vmax == a.vmin:
        a.flags.append("every value is the same")
    if a.unknown_share > unknown_flag_share:
        a.flags.append(f"{100 * a.unknown_share:.0f}% of cells have no data (sea, water, or no coverage)")
    return a


def summary_rows(audits: Mapping[str, Audit], roles: Mapping[str, list[str]] | None = None,
                 policy_vars: set[str] | None = None) -> list[dict]:
    """One row per variable, in catalog order (so the groups and subgroups stay together)."""
    rows = []
    for spec in CATALOG:
        a = audits.get(spec.id)
        if a is None:
            continue
        rows.append({
            "group": spec.group, "subgroup": spec.subgroup, "variable": spec.id, "unit": spec.unit,
            "dataset": spec.dataset, "origin": a.origin or "not loaded",
            "native_m": spec.native_m, "min": a.vmin, "max": a.vmax, "mean": a.vmean,
            "unknown_pct": 100 * a.unknown_share if a.n_cells else float("nan"),
            "role_in_ciat_package": role_text(spec.id, roles, policy_vars) if roles is not None else "",
            "flags": "; ".join(a.flags),
        })
    return rows


def describe(a: Audit) -> str:
    """The lines printed for one variable (the same facts the earlier audit notebook showed)."""
    s = a.spec
    lines = [f"--- {s.id}: {s.label} ---",
             f"  group:             {s.group} > {s.subgroup}",
             f"  dataset:           {s.dataset}",
             f"  origin:            {a.origin or 'not loaded'}" + ("  (SYNTHETIC)" if a.synthetic else ""),
             f"  native resolution: {s.native_m} m"]
    if a.values is None:
        lines.append(f"  NOT LOADED: {a.error}")
        return "\n".join(lines)
    lines += [f"  shape:             {a.values.shape}",
              f"  min / max / mean:  {a.vmin:.4g} / {a.vmax:.4g} / {a.vmean:.4g}   ({s.unit})",
              f"  cells with no data: {100 * a.unknown_share:.1f}%"]
    if a.class_shares:
        top = sorted(a.class_shares.items(), key=lambda kv: -kv[1])[:5]
        labels = s.class_labels or {}
        lines.append("  main classes:      " + ", ".join(f"{labels.get(c, c)} {100 * f:.0f}%" for c, f in top))
    for n in a.notes:
        lines.append(f"  note:              {n}")
    for f in a.flags:
        lines.append(f"  CHECK:             {f}")
    return "\n".join(lines)
