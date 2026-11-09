"""Maps that put CIAT's index and the ordinal simulation side by side. Needs matplotlib (optional)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .compare import Comparison
from .firstpass import STATUS, FirstPass
from .grid import Grid


def plot_comparison(fp: FirstPass, comp: Comparison, grid: Grid, title: str,
                    footnote: str = "", out_path: Path | None = None):
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    extent = grid.extent
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    cmap_idx = plt.get_cmap("YlGn").with_extremes(bad="#d9d9d9")

    def draw(ax, data, name, cmap, vmin, vmax, label):
        cmap = plt.get_cmap(cmap).with_extremes(bad="#d9d9d9") if isinstance(cmap, str) else cmap
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, origin="upper")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("longitude")
        cb = plt.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label(label)

    draw(axes[0, 0], fp.baseline_index, "CIAT: partial terrain-and-soil index", cmap_idx, 0, 1, "index (0-1)")
    draw(axes[0, 1], comp.median, "Simulation: median", cmap_idx, 0, 1, "index (0-1)")
    draw(axes[0, 2], comp.width, "Simulation: 5-95% width", "magma", 0, max(0.5, np.nanmax(comp.width) if np.isfinite(comp.width).any() else 0.5), "range of the index")
    diff = comp.median - fp.baseline_index
    lim = max(0.1, float(np.nanmax(np.abs(diff)))) if np.isfinite(diff).any() else 0.1
    draw(axes[1, 0], diff, "Median minus CIAT", "RdBu", -lim, lim, "difference")
    draw(axes[1, 1], comp.p_top, "Chance of being in the top third", "viridis", 0, 1, "probability")

    ax = axes[1, 2]
    colors = ["#2e7d32", "#9e9e9e", "#fb8c00", "#ef5350", "#8e24aa", "#bdbdbd"]
    cmap_s = mcolors.ListedColormap(colors)
    ax.imshow(fp.status, cmap=cmap_s, vmin=-0.5, vmax=5.5, extent=extent, origin="upper",
              interpolation="nearest")
    ax.set_title("Pixel state", fontsize=11)
    ax.set_xlabel("longitude")
    ax.legend(handles=[mpatches.Patch(color=colors[k], label=STATUS[k]) for k in STATUS],
              loc="upper left", bbox_to_anchor=(0.0, -0.12), fontsize=8, frameon=False, ncol=1)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_layers(layers, grid: Grid, title: str, out_path: Path | None = None):
    """One map per input layer, so each variable can be checked by eye before it enters the index."""
    import matplotlib.pyplot as plt

    if layers.synthetic:
        title = "SYNTHETIC TEST: " + title
    extent = grid.extent
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0))
    unknown_any = np.isnan(layers.slope_deg) | np.isnan(layers.soc_g_kg) | np.isnan(layers.ph)
    panels = [
        (layers.slope_deg, "Slope (degrees)", "terrain", None, None),
        (layers.soc_g_kg, "Soil organic carbon, 0-30 cm (g/kg)", "copper", None, None),
        (layers.ph, "Soil pH, 0-30 cm", "RdYlGn", 4.0, 9.0),
        (layers.eligible_fraction, "Host-eligible share of each cell" + ("" if layers.host_gate_applied else "\n(gate NOT applied: all 1)"), "Greens", 0.0, 1.0),
        (layers.unknown_fraction, "Unknown-host share of each cell" + ("" if layers.host_gate_applied else "\n(gate NOT applied: all 0)"), "Greys", 0.0, 1.0),
        (unknown_any.astype(float), "Cells with an unknown slope, SOC or pH", "Reds", 0.0, 1.0),
    ]
    for ax, (data, name, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        cm = plt.get_cmap(cmap).with_extremes(bad="#d9d9d9")
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cm, vmin=vmin, vmax=vmax, extent=extent, origin="upper")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("longitude")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if layers.synthetic:
        fig.text(0.5, 0.005, "SYNTHETIC DATA: pipeline test, not real", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_overview(ov, title: str, footnote: str = "", out_path: Path | None = None):
    """Coarse overview: where the index is high, how much it depends on the class numbers, and coverage."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    diff = ov.median - ov.baseline_index
    lim = max(0.1, float(np.nanmax(np.abs(diff)))) if np.isfinite(diff).any() else 0.1
    wmax = max(0.5, float(np.nanmax(ov.width))) if np.isfinite(ov.width).any() else 0.5
    panels = [
        (ov.baseline_index, "CIAT: partial terrain-and-soil index", "YlGn", 0, 1, "mean index of scored cells"),
        (ov.median, "Simulation: median", "YlGn", 0, 1, "mean of the medians"),
        (ov.width, "How much the class numbers matter\n(5-95% width)", "magma", 0, wmax, "range of the index"),
        (diff, "Median minus CIAT", "RdBu", -lim, lim, "difference"),
        (ov.p_top, "Chance of being in the top third", "viridis", 0, 1, "mean probability"),
        (ov.coverage, "Share of the block that was scored", "Blues", 0, 1, "coverage"),
    ]
    for ax, (data, name, cmap, vmin, vmax, label) in zip(axes.ravel(), panels, strict=True):
        cm = plt.get_cmap(cmap).with_extremes(bad="#d9d9d9")
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cm, vmin=vmin, vmax=vmax, extent=ov.extent,
                       origin="upper", interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("longitude")
        plt.colorbar(im, ax=ax, fraction=0.046).set_label(label)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_combo_ladder(res, counts: dict[int, int], title: str, footnote: str = "",
                      out_path: Path | None = None):
    """One row per class combination present in the data: CIAT's index, the simulated median and range.

    Rows are ordered by CIAT's index. Where the bars of two rows overlap, the order of those classes
    depends on the numbers given to the classes.
    """
    import matplotlib.pyplot as plt

    combos = res.space.combos()
    rows = sorted(((combos[i], n) for i, n in counts.items() if n > 0),
                  key=lambda t: res.space.baseline(t[0]), reverse=True)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.3 * len(rows) + 1.8)))
    for y, (combo, n) in enumerate(rows):
        s = res.summary(combo)
        ax.hlines(y, s["p05"], s["p95"], color="#4C3FA8", lw=4, alpha=0.45)
        ax.plot(s["median"], y, "o", color="#4C3FA8", ms=6)
        ax.plot(s["baseline"], y, "D", color="#0F6E56", ms=6)
    labels = [" | ".join(f"{res.space.levels[j][i]:.2f}" for j, i in enumerate(c)) + f"   n={n:,}"
              for c, n in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("index (0-1)")
    ax.set_ylabel(" | ".join(res.space.names) + "  (CIAT class scores)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        plt.Line2D([], [], marker="D", color="#0F6E56", ls="", label="CIAT index"),
        plt.Line2D([], [], marker="o", color="#4C3FA8", ls="", label="simulation median"),
        plt.Line2D([], [], color="#4C3FA8", lw=4, alpha=0.45, label="simulation 5-95% range"),
    ], loc="lower right", fontsize=9, frameon=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_reasons(layers, space, grid: Grid, title: str, footnote: str = "", out_path: Path | None = None):
    """A map of why each cell is or is not scored, with the share of cells for each reason."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from .firstpass import REASONS, reason_map

    codes = reason_map(space, layers)
    if layers.synthetic:
        title = "SYNTHETIC TEST: " + title
    colors = ["#2e7d32", "#bdbdbd", "#fb8c00", "#8e24aa", "#6d4c41", "#c62828"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(codes, cmap=mcolors.ListedColormap(colors), vmin=-0.5, vmax=5.5, extent=grid.extent,
              origin="upper", interpolation="nearest")
    ax.set_xlabel("longitude")
    ax.set_title(title, fontsize=13, fontweight="bold")
    handles = [mpatches.Patch(color=colors[k], label=f"{REASONS[k]}  ({100 * (codes == k).mean():.1f}%)")
               for k in REASONS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)
    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Variable audit figures
# ---------------------------------------------------------------------------
def _category_colors(n: int, cmap: str):
    import matplotlib.pyplot as plt

    base = plt.get_cmap(cmap)
    if base.N <= 20:                                   # a qualitative map: one colour per class
        return [base(i % base.N) for i in range(n)]
    return [base(i / max(n - 1, 1)) for i in range(n)]  # a continuous map: spread the classes along it


def plot_variable_audit(audit, grid: Grid, out_path: Path | None = None):
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
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    fig.text(0.02, 0.905, sub, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
    return fig


def plot_group_overview(audits, grid: Grid, title: str, out_path: Path | None = None):
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
        ax.tick_params(labelsize=8)
    prefix = "SYNTHETIC TEST: " if any(a.synthetic for a in audits) else ""
    fig.suptitle(prefix + title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
    return fig
