"""A plain-language summary of a run, ready to paste into a slide, and the gaps in CIAT's functions.

A *gap* is a range of a variable where CIAT's function gives no score: the sources disagree and no median is
defined, a source marks the range unsuitable while another gives a score, or no source covers it. Cells whose
value falls in a gap stay unknown. The gaps are read from the package itself, so they are exact and reproducible.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .compare import Comparison
from .contract import Package
from .firstpass import REASONS, FirstPass, Layers, explain_unscored
from .grid import Grid
from .plan import Plan
from .responses import _contains
from .runconfig import RunConfig
from .simulate import LevelSpace, SimResult

UNITS = {"slope": "degrees", "soil_organic_carbon": "g/kg", "soil_ph": "pH"}
LABELS = {"slope": "Slope", "soil_organic_carbon": "Soil organic carbon", "soil_ph": "Soil pH"}
INDEX_LABELS = {"slope": "slope", "soil_organic_carbon": "soil organic carbon", "soil_ph": "soil pH"}


def _reason(hits: Sequence[Mapping[str, Any]]) -> str:
    if not hits:
        return "no source covers this range"
    flags = {f for iv in hits for f in iv.get("flags", [])}
    if "median_tie" in flags:
        return "the sources disagree and no median is defined (median tie)"
    if "exclusion_conflict" in flags:
        return "one source marks it unsuitable while another gives it a score"
    if "no_evidence_at_value" in flags:
        return "no source has evidence at these values"
    return "no score is defined here" + (f" (flags: {', '.join(sorted(flags))})" if flags else "")


def function_gaps(intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ranges (lo, hi) with no score, each with its reason. lo or hi is None when the range is unbounded.

    Isolated exact values that carry a score inside a gap are not listed separately.
    """
    pts = sorted({float(v) for iv in intervals for v in (iv.get("lower"), iv.get("upper")) if v is not None})
    if not pts:
        return []
    edges: list[float | None] = [None, *pts, None]
    gaps: list[dict[str, Any]] = []
    for a, b in itertools.pairwise(edges):
        x = pts[0] - 1.0 if a is None else pts[-1] + 1.0 if b is None else (a + b) / 2.0
        hits = [iv for iv in intervals if _contains(iv, x)]
        scores = {iv.get("score") for iv in hits}
        if hits and None not in scores and len(scores) == 1:
            continue
        reason = _reason(hits)
        if gaps and gaps[-1]["hi"] == a and gaps[-1]["reason"] == reason:
            gaps[-1]["hi"] = b
        else:
            gaps.append({"lo": a, "hi": b, "reason": reason})
    return gaps


def describe_gap(gap: Mapping[str, Any], unit: str) -> str:
    lo, hi = gap["lo"], gap["hi"]
    if lo is None:
        span = f"below {hi:g}"
    elif hi is None:
        span = f"above {lo:g}"
    else:
        span = f"{lo:g} to {hi:g}"
    return f"{span} {unit}: {gap['reason']}"


def gaps_by_variable(space: LevelSpace) -> dict[str, list[dict[str, Any]]]:
    return {name: function_gaps(opt["intervals"]) for name, opt in zip(space.names, space.options, strict=True)}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_summary(cfg: RunConfig, pkg: Package, plan: Plan, layers: Layers, fp: FirstPass, comp: Comparison,
                  res: SimResult, space: LevelSpace, grid: Grid, data_source: str = "gee") -> str:
    flags = []
    if cfg.run.mode == "exploratory":
        flags.append("**exploratory**: the legal-exclusion mask is not applied and decisions are open")
    if not layers.host_gate_applied:
        flags.append("**host gate not applied**: every cell counts as host")
    if layers.synthetic:
        flags.append("**SYNTHETIC data**: a pipeline test, not real results")
    if not cfg.aoi.source:
        flags.append("the AOI is a placeholder box, not the official boundary")
    lines = [f"# Run summary: {cfg.aoi.name}", ""]
    if flags:
        lines += ["**Read this first.** " + "; ".join(flags) + ".", ""]

    cells = fp.status.size
    lines += [
        "## What was run", "",
        "| | |", "|---|---|",
        f"| CIAT package | {pkg.version} @ {pkg.pin['ref'][:8]}, policy `{plan.policy_id}` |",
        "| Index | partial terrain-and-soil index: "
        + ", ".join(f"{INDEX_LABELS.get(n, n)} {int(100 * w)}%" for n, w in zip(space.names, space.weights, strict=True)) + " |",
        f"| Grid | {grid.shape[0]} x {grid.shape[1]} cells at {cfg.grid.resolution_m} m ({cells:,} cells) |",
        f"| Data | {data_source}; slope derived at 30 m and averaged, soil 0-30 cm thickness-weighted |",
        f"| Simulation | {res.n_draws:,} draws, {res.mode} scales, seed {res.seed} |",
        f"| Run fingerprint | `{cfg.fingerprint()[:10]}` |", "",
    ]

    reasons = explain_unscored(space, layers)
    no_data = reasons[REASONS[1]][0]
    with_data = cells - no_data
    scored = int(fp.scored.sum())
    lines += ["## Where the index can be computed", "", "| Cells | Count | Share |", "|---|---:|---:|"]
    for reason, (n, share) in reasons.items():
        lines.append(f"| {reason} | {n:,} | {_pct(share)} |")
    in_functions = reasons[REASONS[0]][0]
    if with_data:
        sentence = (f"Of the {with_data:,} cells that have data, **{in_functions:,} ({_pct(in_functions / with_data)}) "
                    "have all three values inside CIAT's functions**.")
        if layers.host_gate_applied:
            sentence += f" After the host gate, {scored:,} cells are scored."
        lines += ["", sentence]
    lines.append("")

    gaps = gaps_by_variable(space)
    lines += ["## Gaps in CIAT's functions", "",
              "A value that falls in a gap gets no score, and its cell stays unknown. Read from the package:", ""]
    any_gap = False
    for name in space.names:
        for gap in gaps[name]:
            any_gap = True
            lines.append(f"- **{LABELS.get(name, name)}**, {describe_gap(gap, UNITS.get(name, ''))}")
    if not any_gap:
        lines.append("- none")
    lines.append("")

    if scored:
        s = comp.stats
        pete = fp.pete[fp.scored]
        lines += [
            "## CIAT's index and how much the class numbers matter", "",
            (f"- CIAT's index over the scored cells: from {float(pete.min()):.2f} to {float(pete.max()):.2f} "
            f"(median {float(np.median(pete)):.2f})."),
            (f"- Rank agreement between CIAT's index and the simulation median (Spearman): "
            f"{s['spearman_pete_vs_median']:.2f}."),
            (f"- Width of the simulated 5-95% range of one cell: median {s['width_median']:.2f}, 90th percentile "
            f"{s['width_p90']:.2f} (on a 0-1 scale)."),
            (f"- Membership of the top third: the two methods agree for {_pct(s['top_third_agreement'])} of the scored cells; "
            f"{_pct(s['share_uncertain_top_third'])} have an uncertain top third (chance between 20% and 80%)."),
            (f"- {_pct(s['share_of_pete_top_third_that_is_fragile'])} of the cells in CIAT's top third have less than a "
            "50% chance of being there once the class numbers vary."), "",
        ]

    lines += ["## Points to raise with CIAT", ""]
    if any_gap:
        lines.append("- The gaps above leave part of the area unscored. Are they intended, or can the sources' "
                     "disagreement be carried as a range instead of leaving the cell unknown?")
    lines += [
        ("- Aggregation: at coarse resolution the inputs are averaged per cell before the functions apply. Is the mean "
        "slope the intended statistic, or should each fine cell be scored and the scores averaged?"),
        (f"- Resolution: SoilGrids is native 250 m and soil is half of the index weight. Is {cfg.grid.resolution_m} m acceptable, "
        "or should the first pass run at 250 m?"),
        "- Open decisions that keep this run exploratory: " + (", ".join(
            h.split(":", 1)[-1].strip() for h in plan.holds if not h.startswith("upstream")) or "none") + ".", "",
        "## Limits of this view", "",
        ("- The simulation draws class scores with no prior information, so its median sits below CIAT's convention "
        "(0.25, 0.5, 0.75, 0.9); compare ranks and ranges, not levels."),
        "- It measures the effect of the numbers given to the classes. It does not validate the index.",
    ]
    return "\n".join(lines) + "\n"
