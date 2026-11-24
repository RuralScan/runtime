"""The steps of a run that sit between the pieces: load the inputs a family needs, explain one pixel,
write the outputs. The notebook calls these so each of its cells stays short.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from . import eligibility as elig
from .data.variables_loader import LoadContext, VariableData, load_variable
from .grid import Grid
from .recipes import Family
from .suitability import STATUS, Classes, Result
from .variables import BY_ID

# --------------------------------------------------------------------------- inputs


@dataclass
class Inputs:
    layers: dict[str, np.ndarray]                  # variable -> raw values on the grid
    data: dict[str, VariableData]                  # variable -> what was loaded and from where
    eligibility: elig.Eligibility
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        return any(d.synthetic for d in self.data.values())


def load_inputs(family: Family, ctx: LoadContext, eligibility_mode: str = "apply",
                classes: tuple[int, ...] = (30, 40), min_share: float = 0.5) -> Inputs:
    """Load every variable of the family, and the land cover for G. A variable that fails is reported."""
    layers, data, errors = {}, {}, {}
    for v in family.variables:
        spec = BY_ID.get(v)
        if spec is None:
            errors[v] = "no catalog entry in variables.py"
            continue
        d = load_variable(spec, ctx)
        data[v] = d
        if d.ok:
            layers[v] = d.values
        else:
            errors[v] = d.error

    if eligibility_mode == "skip":
        e = elig.everywhere(ctx.grid.shape)
    else:
        lc = load_variable(BY_ID["land_cover"], ctx)
        data["land_cover"] = lc
        if not lc.ok:
            errors["land_cover"] = lc.error
            e = elig.everywhere(ctx.grid.shape)
            e.notes.append(f"land cover could not be loaded ({lc.error}): eligibility NOT applied")
        elif lc.shares is not None:
            e = elig.from_shares(lc.shares, classes, min_share)
        else:
            e = elig.from_classes(lc.values, classes, min_share)
    return Inputs(layers, data, e, errors)


# --------------------------------------------------------------------------- one pixel


def explain_pixel(result: Result, family: Family, layers: Mapping[str, Any], row: int, col: int) -> str:
    """The whole calculation for one pixel, step by step (as in the 12-pixel worked example)."""
    st = int(result.status[row, col])
    lines = [f"Pixel row {row}, column {col}: {STATUS[st]}"]
    g = result.G[row, col]
    lines.append(f"  G (eligible land cover) = {'unknown' if np.isnan(g) else int(g)}")
    for v in family.variables:
        if v in result.dropped:
            continue
        r = family.recipe(v)
        x = float(np.asarray(layers[v])[row, col])
        z = float(result.z[v][row, col])
        why = ("no data" if np.isnan(x) else
               f"outside the evidence ({r.x[0]:g} to {r.x[-1]:g})" if np.isnan(z) else f"z = {z:.3f}")
        lines.append(f"  {v:<24} {family.roles[v]:<10} value {x:10.4g} {r.unit:<10} -> {why}")
    if st != 0:
        return "\n".join(lines)
    lim = result.limiting
    if lim:
        zs = ", ".join(f"{v} {result.z[v][row, col]:.3f}" for v in lim)
        lines.append(f"  L = min({zs}) = {result.L[row, col]:.3f}   (set by {lim[result.driver[row, col]]})")
    else:
        lines.append("  L = 1 (no limiting variable)")
    if result.modulating:
        terms = " + ".join(f"{result.weights[v]:.3f} x {result.z[v][row, col]:.3f}" for v in result.modulating)
        lines.append(f"  M = {terms} = {result.M[row, col]:.3f}")
    else:
        lines.append("  M = 1 (no modulating variable)")
    lines.append(f"  S = L x G x M = {result.L[row, col]:.3f} x {int(g)} x {result.M[row, col]:.3f} "
                 f"= {result.S[row, col]:.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- outputs


def save_run(out_dir: Path | str, grid: Grid, manifest: Mapping[str, Any], result: Result, classes: Classes,
             extra: Mapping[str, np.ndarray] | None = None) -> list[Path]:
    """Write the arrays (npz), GeoTIFFs when rasterio is installed, and the manifest (json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"S": result.S, "L": result.L, "M": result.M, "G": result.G,
              "status": result.status.astype(np.float32), "classes": classes.classes.astype(np.float32),
              **{f"z_{v}": a for v, a in result.z.items()}, **(extra or {})}
    written = [out / "arrays.npz"]
    np.savez_compressed(written[0], **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()},
                        grid_bbox=np.array(grid.bbox), grid_resolution_deg=grid.resolution_deg)
    try:
        written += write_geotiffs(out, grid, {"suitability": [result.S, result.L, result.M],
                                              "classes": [classes.classes.astype(np.float32)],
                                              "status": [result.status.astype(np.float32)]})
    except ImportError:
        pass
    m = {**manifest, "runtime_version": __version__, "written": datetime.now(UTC).isoformat(timespec="seconds"),
         "files": [p.name for p in written] + ["manifest.json"]}
    (out / "manifest.json").write_text(json.dumps(m, indent=2, default=_json_default), encoding="utf-8")
    written.append(out / "manifest.json")
    return written


def write_geotiffs(out: Path, grid: Grid, rasters: Mapping[str, list[np.ndarray]]) -> list[Path]:
    import rasterio
    from rasterio.transform import from_origin

    minx, _, _, maxy = grid.bbox
    transform = from_origin(minx, maxy, grid.resolution_deg, grid.resolution_deg)
    paths = []
    for name, bands in rasters.items():
        p = out / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=grid.height, width=grid.width, count=len(bands),
                           dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
                           compress="deflate") as dst:
            for i, b in enumerate(bands, 1):
                dst.write(np.asarray(b, dtype=np.float32), i)
        paths.append(p)
    return paths


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
