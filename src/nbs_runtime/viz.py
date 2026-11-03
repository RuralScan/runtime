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
