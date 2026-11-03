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

    draw(axes[0, 0], fp.pete, "CIAT: partial terrain-and-soil index", cmap_idx, 0, 1, "index (0-1)")
    draw(axes[0, 1], comp.median, "Simulation: median", cmap_idx, 0, 1, "index (0-1)")
    draw(axes[0, 2], comp.width, "Simulation: 5-95% width", "magma", 0, max(0.5, np.nanmax(comp.width) if np.isfinite(comp.width).any() else 0.5), "range of the index")
    diff = comp.median - fp.pete
    lim = max(0.1, float(np.nanmax(np.abs(diff)))) if np.isfinite(diff).any() else 0.1
    draw(axes[1, 0], diff, "Median minus CIAT", "RdBu", -lim, lim, "difference")
    draw(axes[1, 1], comp.p_top, "Chance of being in the top third", "viridis", 0, 1, "probability")

    ax = axes[1, 2]
    colors = ["#2e7d32", "#9e9e9e", "#fb8c00", "#ef5350", "#8e24aa"]
    cmap_s = mcolors.ListedColormap(colors)
    ax.imshow(fp.status, cmap=cmap_s, vmin=-0.5, vmax=4.5, extent=extent, origin="upper",
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
        (layers.eligible_fraction, "Host-eligible share of each cell", "Greens", 0.0, 1.0),
        (layers.unknown_fraction, "Unknown-host share of each cell", "Greys", 0.0, 1.0),
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
    diff = ov.median - ov.ciat
    lim = max(0.1, float(np.nanmax(np.abs(diff)))) if np.isfinite(diff).any() else 0.1
    wmax = max(0.5, float(np.nanmax(ov.width))) if np.isfinite(ov.width).any() else 0.5
    panels = [
        (ov.ciat, "CIAT: partial terrain-and-soil index", "YlGn", 0, 1, "mean index of scored cells"),
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
