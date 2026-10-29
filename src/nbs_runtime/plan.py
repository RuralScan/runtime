"""plan(): say what is ready and what is on hold for a run, without touching any raster."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .contract import Package, fingerprint_holds, load_package, weight_holds


@dataclass
class Plan:
    policy_id: str
    package_version: str
    package_ref: str
    holds: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.holds


def plan(config_path: Path, pkg: Package | None = None) -> Plan:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    pkg = pkg or load_package()
    policy = pkg.policy(cfg["run"]["policy_id"])
    p = Plan(policy.policy_id, pkg.version, pkg.pin["ref"])

    p.holds += fingerprint_holds(pkg, policy)
    p.holds += weight_holds(policy)
    p.holds += [f"upstream: {h}" for h in policy.holds]

    t, s, m = cfg["target"], cfg["soil"], cfg["masks"]
    if not t.get("aoi_source"):
        p.holds.append("target: exact AOI boundary and its source are not recorded")
    if not t.get("applicability_decision"):
        p.holds.append("target: applicability decision (policy requires target transfer) missing")
    if not (s.get("adopted") and s.get("decision_reference")):
        p.holds.append("soil: input proposal not adopted or replaced with a decision reference")
    if not m.get("legal_mask_spec"):
        p.holds.append("masks: no assessed legal-exclusion specification (output stays unknown)")
    if not cfg.get("host_inputs", {}).get("managed_forage_without_grazing"):
        p.warnings.append("host: grassland (class 30) stays unknown without a documented source")
    return p
