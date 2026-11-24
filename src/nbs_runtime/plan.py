"""plan(): say what is ready and what blocks a run, without touching any raster.

A hold blocks the run (a recipe that does not load, a variable with no data source, a unit that does not
match). A warning is something to keep in mind when reading the results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .recipes import RECIPES_ROOT, ROLES_PATH, Family, RecipeError, load_family
from .runconfig import RunConfig
from .variables import BY_ID, same_unit

# above this many cells at the chosen resolution, a laptop run or Earth Engine is likely to struggle
CELL_WARNING = 20_000_000


@dataclass
class Plan:
    nbs: str
    family_name: str
    config_fingerprint: str = ""
    aoi_fingerprint: str = ""
    family: Family | None = None
    holds: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.holds

    def manifest(self) -> dict[str, Any]:
        return {"holds": self.holds, "warnings": self.warnings,
                "config_fingerprint": self.config_fingerprint, "aoi_fingerprint": self.aoi_fingerprint}


def plan(config: RunConfig | Path | str, recipes_root: Path | str = RECIPES_ROOT,
         roles_path: Path | str = ROLES_PATH) -> Plan:
    cfg = config if isinstance(config, RunConfig) else RunConfig.load(config)
    p = Plan(cfg.run.nbs, cfg.run.family, cfg.fingerprint(), cfg.aoi.fingerprint())
    p.holds += [f"config: {m}" for m in cfg.validate()]

    try:
        fam = load_family(cfg.run.nbs, cfg.run.family, recipes_root, roles_path)
    except RecipeError as e:
        p.holds.append(f"recipes: {e}")
        return p
    p.family = fam

    for r in fam.recipes:
        spec = BY_ID.get(r.variable)
        if spec is None:
            p.holds.append(f"data: no catalog entry (variables.py) for recipe variable {r.variable!r}")
        elif not same_unit(r.unit, spec.unit):
            p.holds.append(f"data: {r.variable} recipe is in {r.unit!r} but the data is in {spec.unit!r}")

    if not fam.limiting:
        p.warnings.append("no limiting variable: L = 1 everywhere")
    if not fam.modulating:
        p.warnings.append("no modulating variable: M = 1 everywhere and the weights do not matter")
    elif len(fam.modulating) < 2:
        p.warnings.append("one modulating variable: the weight sensitivity tests cannot change anything")
    ignored = [r.variable for r in fam.limiting if r.weight]
    if ignored:
        p.warnings.append(f"recipe weight ignored for limiting variable(s): {', '.join(ignored)}")

    if cfg.aoi.kind == "bbox" and not cfg.aoi.source:
        p.warnings.append("aoi: a bounding box without a recorded source is a placeholder for the real boundary")
    if cfg.eligibility.mode == "skip":
        p.warnings.append("eligibility: SKIPPED, every cell is treated as eligible land cover")
    if cfg.run.outside == "clamp":
        p.warnings.append("outside='clamp': values beyond the evidence take the score of the nearest end node")
    if cfg.run.weights != "recipe":
        p.warnings.append(f"weights: {cfg.run.weights}, not the recipe weights")
    cells = cfg.aoi.estimated_cells(cfg.grid.resolution_m)
    if cells is not None and cells > CELL_WARNING:
        p.warnings.append(f"grid: about {cells:,} cells at {cfg.grid.resolution_m} m; consider a coarser grid")
    return p
