"""Figures for the notebooks: recipes, inputs, the index (L, G, M, S), classes and sensitivity.

Needs matplotlib (optional dependency). Every function returns the figure and saves it when out_path
is given. Maps are drawn north-up on the grid's extent; grey means unknown.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .grid import Grid
from .suitability import STATUS

GREY = "#d9d9d9"
STATUS_COLORS = {0: "#1F8A70", 1: "#C3CCD2", 2: "#8C8C8C", 3: "#E6A23C"}


def _save(fig, out_path):
    if out_path is not None:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig


def _map(ax, a, grid: Grid, title: str, cmap="YlGn", vmin=None, vmax=None, label="", boundary=None):
    import matplotlib.pyplot as plt

    cm = plt.get_cmap(cmap).with_extremes(bad=GREY)
    im = ax.imshow(np.ma.masked_invalid(a), cmap=cm, vmin=vmin, vmax=vmax, extent=grid.extent,
                   origin="upper", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046).set_label(label)
    if boundary is not None:
        draw_boundary(ax, boundary)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    return im


def _prefix(synthetic: bool) -> str:
    return "SYNTHETIC TEST: " if synthetic else ""


# --------------------------------------------------------------------------- recipes

def plot_curves(family, layers: Mapping[str, np.ndarray] | None = None, out_path: Path | None = None):
    """Each recipe's function, with its role and weight. With layers, the distribution of the AOI's values
    is drawn under the curve, so the share outside the evidence (the nodes) can be seen."""
    import matplotlib.pyplot as plt

    recs = family.recipes
    w = family.modulating_weights()
    n = len(recs)
    cols = min(n, 4)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, r in zip(axes.ravel(), recs, strict=False):
        role = family.roles[r.variable]
        color = "#A8402F" if role == "limiting" else "#156052"
        x = np.asarray(r.x)
        lo, hi = x[0], x[-1]
        if layers is not None and r.variable in layers:
            v = np.asarray(layers[r.variable], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = min(lo, np.percentile(v, 0.5)), max(hi, np.percentile(v, 99.5))
                ax2 = ax.twinx()
                ax2.hist(v, bins=40, range=(lo, hi), color="#9FB1BD", alpha=0.45)
                ax2.set_yticks([])
                out = np.mean((v < x[0]) | (v > x[-1]))
                ax.text(0.02, 0.04, f"{out:.0%} of cells outside the evidence", transform=ax.transAxes,
                        fontsize=8, color="#555555")
        if r.low and r.high:
            ax.fill_between(x, r.low, r.high, color=color, alpha=0.15, lw=0)
        ax.plot(x, r.score, "-o", color=color, lw=2, ms=4, zorder=3)
        ax.axvspan(lo, x[0], color=GREY, alpha=0.5, lw=0)
        ax.axvspan(x[-1], hi, color=GREY, alpha=0.5, lw=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.02)
        ax.set_zorder(ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        tag = "limiting" if role == "limiting" else f"modulating, w = {w[r.variable]:.2f}"
        ax.set_title(f"{r.variable}\n({tag})", fontsize=10, color=color)
        ax.set_xlabel(r.unit, fontsize=8)
        ax.set_ylabel("score z", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Recipes: {family.nbs}/{family.family}  (grey = outside the evidence)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- scores and index

def plot_scores(result, grid: Grid, synthetic: bool = False, out_path: Path | None = None, boundary=None):
    """The 0-1 score z of each variable, one map each."""
    import matplotlib.pyplot as plt

    names = list(result.z)
    cols = min(len(names), 4)
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax, v in zip(axes.ravel(), names, strict=False):
        role = "limiting" if v in result.limiting else "modulating"
        _map(ax, result.z[v], grid, f"z: {v} ({role})", vmin=0, vmax=1, boundary=boundary)
    fig.suptitle(_prefix(synthetic) + "Score of each variable (grey: no data or outside the evidence)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_index(result, grid: Grid, synthetic: bool = False, out_path: Path | None = None, boundary=None):
    """L, G, M, S, the pixel states and which limiting variable sets L."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    _map(axes[0, 0], result.L, grid, "L: minimum of the limiting scores", vmin=0, vmax=1, boundary=boundary)
    _map(axes[0, 1], result.G, grid, "G: eligible land cover (1) or not (0)", cmap="Greys", vmin=0, vmax=1,
         boundary=boundary)
    _map(axes[0, 2], result.M, grid, "M: weighted sum of the modulating scores", vmin=0, vmax=1,
         boundary=boundary)
    _map(axes[1, 0], result.S, grid, "S = L x G x M", vmin=0, vmax=1, label="suitability", boundary=boundary)

    codes = sorted(STATUS)
    cm = mcolors.ListedColormap([STATUS_COLORS[c] for c in codes])
    ax = axes[1, 1]
    ax.imshow(result.status, cmap=cm, vmin=-0.5, vmax=len(codes) - 0.5, extent=grid.extent, origin="upper",
              interpolation="nearest")
    ax.legend(handles=[mpatches.Patch(color=STATUS_COLORS[c], label=STATUS[c]) for c in codes],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7, frameon=False)
    ax.set_title("Pixel state", fontsize=10)
    ax.tick_params(labelsize=7)

    ax = axes[1, 2]
    if result.limiting:
        d = np.where(result.scored, result.driver, np.nan).astype(float)
        colors = _category_colors(len(result.limiting), "tab10")
        ax.imshow(np.ma.masked_invalid(d), cmap=mcolors.ListedColormap(colors), vmin=-0.5,
                  vmax=len(result.limiting) - 0.5, extent=grid.extent, origin="upper", interpolation="nearest")
        ax.legend(handles=[mpatches.Patch(color=colors[i], label=v) for i, v in enumerate(result.limiting)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no limiting variable", ha="center", transform=ax.transAxes)
    ax.set_title("Which limiting variable sets L", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.suptitle(_prefix(synthetic) + "Suitability index, approach 3", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_classes(classes, grid: Grid, synthetic: bool = False, out_path: Path | None = None, boundary=None):
    """Score 0 as its own class, then the quantile classes, with the absolute range of the scores."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    k = classes.k
    names = ["score 0"] + ([f"class {i}" for i in range(1, k + 1)] if k > 1 else ["score > 0"])
    if k == 5:
        names = ["score 0", "very low", "low", "medium", "high", "very high"]
    colors = ["#C3CCD2"] + [plt.get_cmap("YlGn")(0.25 + 0.75 * i / max(k - 1, 1)) for i in range(k)]
    c = np.where(classes.classes < 0, np.nan, classes.classes).astype(float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.imshow(np.ma.masked_invalid(c), cmap=mcolors.ListedColormap(colors).with_extremes(bad=GREY),
               vmin=-0.5, vmax=k + 0.5, extent=grid.extent, origin="upper", interpolation="nearest")
    if boundary is not None:
        draw_boundary(ax1, boundary)
    ax1.legend(handles=[mpatches.Patch(color=col, label=n) for col, n in zip(colors, names, strict=True)],
               loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax1.set_title("Priority classes (grey: not scored)", fontsize=10)
    counts = [int((classes.classes == i).sum()) for i in range(k + 1)]
    ax2.bar(names, counts, color=colors)
    ax2.set_ylabel("cells")
    ax2.tick_params(axis="x", labelsize=8, rotation=30)
    s = classes.summary
    ax2.set_title(f"absolute S over positive scores: min {s['min']:.2f}, median {s['median']:.2f}, "
                  f"max {s['max']:.2f}", fontsize=9)
    fig.suptitle(_prefix(synthetic) + "Classes" + (f" ({classes.note})" if classes.note else ""),
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- sensitivity

def plot_sensitivity(base, dres, loo: Mapping[str, Mapping], grid: Grid, synthetic: bool = False,
                     out_path: Path | None = None, boundary=None):
    """Dirichlet: median, 5-95% range and probability of the top quintile; leave one out: rank agreement."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    _map(axes[0], dres.median, grid, "Median S over the weight draws", vmin=0, vmax=1, boundary=boundary)
    _map(axes[1], dres.p95 - dres.p05, grid, "Width of the 5-95% range", cmap="magma_r", vmin=0,
         boundary=boundary)
    _map(axes[2], dres.p_top, grid, "Probability of the top quintile", cmap="RdYlGn", vmin=0, vmax=1,
         boundary=boundary)
    ax = axes[3]
    names = list(loo)
    rho = [loo[v]["spearman"] for v in names]
    kept = [loo[v]["top_retained"] for v in names]
    y = np.arange(len(names))
    ax.barh(y - 0.2, rho, height=0.4, color="#37388C", label="rank agreement (Spearman)")
    ax.barh(y + 0.2, kept, height=0.4, color="#1F8A70", label="top quintile retained")
    ax.set_yticks(y, [f"without {v}" for v in names], fontsize=8)
    ax.set_xlim(min(0.0, np.nanmin(rho + kept)), 1.0)
    ax.axvline(1.0, color="#999999", lw=0.8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.set_title("Leave one variable out", fontsize=10)
    fig.suptitle(_prefix(synthetic) + f"Sensitivity ({len(dres.spearman)} weight draws, concentration "
                 f"{dres.concentration:g})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------------- variable audit

def _category_colors(n: int, cmap: str):
    import matplotlib.pyplot as plt

    base = plt.get_cmap(cmap)
    if base.N <= 20:                                   # a qualitative map: one colour per class
        return [base(i % base.N) for i in range(n)]
    return [base(i / max(n - 1, 1)) for i in range(n)]  # a continuous map: spread the classes along it


def draw_boundary(ax, units) -> None:
    """Outline every admin unit of a GeoDataFrame on a map axis, keeping the axis limits."""
    ax.autoscale(False)
    units.boundary.plot(ax=ax, color="#222222", lw=0.8, add_labels=False)


def plot_variable_audit(audit, grid: Grid, out_path: Path | None = None, boundary=None):
    """A map of one variable and its distribution (a histogram, or the share of each class)."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    s = audit.spec
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    prefix = "SYNTHETIC TEST: " if audit.synthetic else ""
    fig.suptitle(f"{prefix}{s.label} ({s.unit})", fontsize=13, fontweight="bold", x=0.02, ha="left")
    sub = f"{s.group} > {s.subgroup}  ·  {s.dataset}  ·  {audit.origin or 'not loaded'}"
    if audit.values is None:
        ax1.axis("off")
        ax2.axis("off")
        ax1.text(0.0, 0.6, "Not loaded.\n\n" + audit.error, fontsize=11, wrap=True, va="center")
        fig.text(0.02, 0.905, sub, fontsize=9, color="#555555")
        return fig
    v = audit.values
    extent = grid.extent
    if s.kind == "categorical" and s.class_labels:
        codes = sorted(s.class_labels)
        colors = _category_colors(len(codes), s.cmap)
        idx = np.full(v.shape, np.nan)
        for i, c in enumerate(codes):
            idx[v == c] = i
        ax1.imshow(np.ma.masked_invalid(idx), cmap=mcolors.ListedColormap(colors), vmin=-0.5, vmax=len(codes) - 0.5,
                   extent=extent, origin="upper", interpolation="nearest")
        present = [c for c in codes if c in audit.class_shares]
        ax1.legend(handles=[mpatches.Patch(color=colors[codes.index(c)], label=s.class_labels[c]) for c in present],
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
        shares = sorted(audit.class_shares.items(), key=lambda kv: kv[1])
        ax2.barh([s.class_labels.get(c, str(c)) for c, _ in shares], [100 * f for _, f in shares],
                 color=[colors[codes.index(c)] if c in codes else "#999999" for c, _ in shares])
        ax2.set_xlabel("share of cells with data (%)")
    else:
        cm = plt.get_cmap(s.cmap).with_extremes(bad="#d9d9d9")
        im = ax1.imshow(np.ma.masked_invalid(v), cmap=cm, extent=extent, origin="upper")
        plt.colorbar(im, ax=ax1, fraction=0.046).set_label(s.unit)
        known = v[np.isfinite(v)]
        ax2.hist(known, bins=30, color="#0F6E56", edgecolor="white")
        ax2.set_xlabel(s.unit)
        ax2.set_ylabel("cells")
        if s.plausible is not None:
            for edge in s.plausible:
                if known.min() <= edge <= known.max():
                    ax2.axvline(edge, color="#D9822B", ls="--", lw=1)
    if boundary is not None:
        draw_boundary(ax1, boundary)
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    fig.text(0.02, 0.905, sub, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
    return fig


def plot_group_overview(audits, grid: Grid, title: str, out_path: Path | None = None, boundary=None):
    """Small maps of every variable of a group side by side."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    audits = list(audits)
    n = len(audits)
    cols = min(3, max(n, 1))
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, a in zip(axes.ravel(), audits, strict=False):
        ax.axis("on")
        s = a.spec
        ax.set_title(f"{s.label}\n({s.unit})", fontsize=10)
        if a.values is None:
            ax.text(0.5, 0.5, "not loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        if s.kind == "categorical" and s.class_labels:
            codes = sorted(s.class_labels)
            idx = np.full(a.values.shape, np.nan)
            for i, c in enumerate(codes):
                idx[a.values == c] = i
            cm = mcolors.ListedColormap(_category_colors(len(codes), s.cmap))
            ax.imshow(np.ma.masked_invalid(idx), cmap=cm, vmin=-0.5, vmax=len(codes) - 0.5, extent=grid.extent,
                      origin="upper", interpolation="nearest")
        else:
            cm = plt.get_cmap(s.cmap).with_extremes(bad="#d9d9d9")
            im = ax.imshow(np.ma.masked_invalid(a.values), cmap=cm, extent=grid.extent, origin="upper")
            plt.colorbar(im, ax=ax, fraction=0.046)
        if boundary is not None:
            draw_boundary(ax, boundary)
        ax.tick_params(labelsize=8)
    prefix = "SYNTHETIC TEST: " if any(a.synthetic for a in audits) else ""
    fig.suptitle(prefix + title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
    return fig
