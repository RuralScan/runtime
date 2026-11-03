"""Check the Earth Engine adapter on a tiny area before a full run.

  python scripts/check_gee.py --project YOUR_EE_PROJECT

Prints the grid, the bands SoilGrids really has, and the shape, range and share of unknown cells of each
layer. Read the numbers: slope should be degrees (0 to a few tens), SOC a few to tens of g/kg after
conversion, pH 4 to 9, host fractions between 0 and 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbs_runtime.contract import load_package
from nbs_runtime.data import gee
from nbs_runtime.grid import Grid
from nbs_runtime.rasters import gate_class_groups


def describe(name: str, a: np.ndarray) -> None:
    ok = a[~np.isnan(a)]
    rng = f"{ok.min():.3g} to {ok.max():.3g} (median {np.median(ok):.3g})" if ok.size else "no data"
    print(f"  {name:<22} shape {a.shape} | {rng} | unknown {100 * np.isnan(a).mean():.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="Earth Engine project id")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-72.45, 18.45, -72.25, 18.65],
                    metavar=("MINX", "MINY", "MAXX", "MAXY"), help="a small box near Port-au-Prince")
    ap.add_argument("--resolution", type=int, default=1000)
    a = ap.parse_args()

    gee.init_ee(a.project)
    grid = Grid.from_bbox(tuple(a.bbox), a.resolution)
    print("grid:", grid.shape, "rows x columns at", a.resolution, "m")
    print("aggregation stages for slope (deg):", [round(s, 5) for s in gee.aggregation_stages(gee.SRTM_NATIVE_DEG, grid.resolution_deg)])

    pkg = load_package()
    gate = pkg.gate_params(pkg.policy("planted_terrain_soil_v1"))
    groups = gate_class_groups(gate)
    print("host class groups:", groups)

    layers = gee.load_layers(grid, groups)
    for n in layers.notes:
        print("note:", n)
    describe("slope (deg)", layers.slope_deg)
    describe("SOC 0-30 cm (g/kg)", layers.soc_g_kg)
    describe("pH 0-30 cm", layers.ph)
    describe("eligible fraction", layers.eligible_fraction)
    describe("unknown fraction", layers.unknown_fraction)


if __name__ == "__main__":
    main()
