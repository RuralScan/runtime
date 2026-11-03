"""plan(): say what is ready and what is on hold for a run, without touching any raster."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .contract import Package, fingerprint_holds, load_package, weight_holds
from .runconfig import RunConfig


@dataclass
class Plan:
    policy_id: str
    package_version: str
    package_ref: str
    config_fingerprint: str = ""
    aoi_fingerprint: str = ""
    holds: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.holds


# above this many cells at the chosen resolution, a laptop run or a single Earth Engine
# request is likely to fail; it is a warning, not a hold
CELL_WARNING = 50_000_000


def plan(config: RunConfig | Path | str, pkg: Package | None = None) -> Plan:
    cfg = config if isinstance(config, RunConfig) else RunConfig.load(config)
    pkg = pkg or load_package()
    policy = pkg.policy(cfg.run.policy_id)
    p = Plan(
        policy.policy_id, pkg.version, pkg.pin["ref"],
        config_fingerprint=cfg.fingerprint(), aoi_fingerprint=cfg.aoi.fingerprint(),
    )

    # a malformed config is a hold: no run is meaningful until it is fixed
    p.holds += [f"config: {m}" for m in cfg.validate()]
    p.holds += fingerprint_holds(pkg, policy)
    p.holds += weight_holds(policy)
    p.holds += [f"upstream: {h}" for h in policy.holds]

    # open decisions recorded as empty fields
    if not cfg.aoi.source:
        p.holds.append("target: the AOI boundary and its source are not recorded")
    elif cfg.aoi.kind == "bbox":
        p.warnings.append("aoi: a bounding box is only a placeholder for the official boundary")
    if not cfg.target.applicability_decision:
        p.holds.append("target: applicability decision (policy requires target transfer) missing")
    if not (cfg.soil.adopted and cfg.soil.decision_reference):
        p.holds.append("soil: input proposal not adopted or replaced with a decision reference")
    if not cfg.masks.legal_mask_spec:
        p.holds.append("masks: no assessed legal-exclusion specification (output stays unknown)")
    if not cfg.host_inputs.get("managed_forage_without_grazing"):
        p.warnings.append("host: grassland (class 30) stays unknown without a documented source")

    if cfg.run.mode == "exploratory":
        p.warnings.append(
            "mode: EXPLORATORY. The legal mask is not applied and the holds above stay open, "
            "so results are not policy-conformant"
        )

    # practical size of the run
    cells = cfg.aoi.estimated_cells(cfg.grid.resolution_m)
    if cells is not None and cells > CELL_WARNING:
        p.warnings.append(
            f"grid: about {cells:,} cells at {cfg.grid.resolution_m} m; consider a coarser grid"
        )
    if cfg.simulation.n_draws < 1000:
        p.warnings.append("simulation: fewer than 1,000 draws gives noisy percentiles")
    return p
