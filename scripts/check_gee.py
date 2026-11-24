"""Check that every variable of a family loads from its source on a tiny area, before a full run.

  uv run python scripts/check_gee.py --nbs agroforestry --family cross_family [--project YOUR_EE_PROJECT]

Prints the shape, range and share of unknown cells of each variable and of the land cover. Read the
numbers against the units: slope in degrees (0 to a few tens), elevation in metres, NDVI between -1 and 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbs_runtime.data import gee
from nbs_runtime.data.variables_loader import LoadContext, load_variable
from nbs_runtime.grid import Grid
from nbs_runtime.recipes import load_family
from nbs_runtime.variables import BY_ID, GEE


def describe(name: str, a: np.ndarray | None, error: str = "") -> None:
    if a is None:
        print(f"  {name:<24} NOT LOADED: {error}")
        return
    ok = a[~np.isnan(a)]
    rng = f"{ok.min():.4g} to {ok.max():.4g} (median {np.median(ok):.4g})" if ok.size else "no data"
    print(f"  {name:<24} shape {a.shape} | {rng} | unknown {100 * np.isnan(a).mean():.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbs", default="agroforestry")
    ap.add_argument("--family", default="cross_family")
    ap.add_argument("--project", default=None, help="Earth Engine project id (default: EE_PROJECT in .env)")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-72.45, 18.45, -72.25, 18.65],
                    metavar=("MINX", "MINY", "MAXX", "MAXY"), help="a small box near Port-au-Prince")
    ap.add_argument("--resolution", type=int, default=1000)
    a = ap.parse_args()

    fam = load_family(a.nbs, a.family)
    specs = [BY_ID[v] for v in fam.variables] + [BY_ID["land_cover"]]
    grid = Grid.from_bbox(tuple(a.bbox), a.resolution)
    print("grid:", grid.shape, "rows x columns at", a.resolution, "m")
    if any(s.source == GEE for s in specs):
        print("Earth Engine project:", gee.init_ee(a.project))
        print("requests:", gee.request_plan(specs, grid))
    ctx = LoadContext(grid, verbose=False)
    for s in specs:
        d = load_variable(s, ctx)
        describe(f"{s.id} ({s.unit})", d.values, d.error)


if __name__ == "__main__":
    main()
