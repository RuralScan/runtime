"""Run configuration: everything a person decides before a run, in one place.

A run is defined by a TOML file, or by a dict (the notebook's first cell). Unknown keys are an
error, so a typo cannot silently fall back to a default. Empty decisions stay empty: the plan
reports them as holds instead of guessing.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .aoi import AOI

SCALE_MODES = ("independent", "shared")
RUN_MODES = ("policy", "exploratory")


@dataclass(frozen=True)
class RunSpec:
    nbs_id: str = "agroforestry"
    system: str = "planted_silvoarable"
    policy_id: str = "planted_terrain_soil_v1"
    mode: str = "exploratory"       # "policy" needs every decision recorded; "exploratory" does not


@dataclass(frozen=True)
class GridSpec:
    resolution_m: int = 250
    scenario: str = "baseline"


@dataclass(frozen=True)
class TargetSpec:
    country: str = ""
    applicability_decision: str = ""


@dataclass(frozen=True)
class SoilSpec:
    input_proposal: str = "planted_topsoil_0_30_v1"
    adopted: bool = False
    decision_reference: str = ""


@dataclass(frozen=True)
class MaskSpec:
    legal_mask_spec: str = ""


@dataclass(frozen=True)
class SimulationSpec:
    n_draws: int = 5000
    seed: int = 20261103
    scale_mode: str = "independent"
    prior: str = "uniform"
    weight_concentration: float = 0.0   # 0 keeps the policy weights fixed


@dataclass(frozen=True)
class DataSpec:
    order: tuple[str, ...] = ("local", "hf", "gee")
    hf_repo: str = "dagudelo30/nbs-ruralscan-data"


@dataclass(frozen=True)
class RunConfig:
    run: RunSpec = field(default_factory=RunSpec)
    aoi: AOI = field(default_factory=AOI)
    grid: GridSpec = field(default_factory=GridSpec)
    target: TargetSpec = field(default_factory=TargetSpec)
    soil: SoilSpec = field(default_factory=SoilSpec)
    masks: MaskSpec = field(default_factory=MaskSpec)
    host_inputs: dict[str, str] = field(default_factory=dict)
    simulation: SimulationSpec = field(default_factory=SimulationSpec)
    data: DataSpec = field(default_factory=DataSpec)

    # ---- construction ----
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        kw: dict[str, Any] = {}
        for name, sub in (
            ("run", RunSpec), ("grid", GridSpec), ("target", TargetSpec), ("soil", SoilSpec),
            ("masks", MaskSpec), ("simulation", SimulationSpec),
        ):
            if name in d:
                kw[name] = _build(sub, d[name], name)
        if "data" in d:
            data = dict(d["data"])
            if "order" in data:
                data["order"] = tuple(data["order"])
            kw["data"] = _build(DataSpec, data, "data")
        if "aoi" in d:
            aoi = dict(d["aoi"])
            if aoi.get("bbox") is not None:
                aoi["bbox"] = tuple(float(v) for v in aoi["bbox"])
            if "notes" in aoi:
                aoi["notes"] = tuple(aoi["notes"])
            kw["aoi"] = _build(AOI, aoi, "aoi")
        if "host_inputs" in d:
            kw["host_inputs"] = {str(k): str(v) for k, v in d["host_inputs"].items()}
        return cls(**kw)

    @classmethod
    def load(cls, path: Path | str, overrides: dict[str, Any] | None = None) -> RunConfig:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(_merge(raw, overrides or {}))

    # ---- output ----
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aoi"]["bbox"] = list(self.aoi.bbox) if self.aoi.bbox else None
        d["aoi"]["notes"] = list(self.aoi.notes)
        d["data"]["order"] = list(self.data.order)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    # ---- validation (problems that make a run meaningless, not open decisions) ----
    def validate(self) -> list[str]:
        problems = list(self.aoi.validate())
        if self.run.mode not in RUN_MODES:
            problems.append(f"run.mode must be one of {RUN_MODES}")
        if self.grid.resolution_m <= 0:
            problems.append("grid.resolution_m must be positive")
        s = self.simulation
        if s.n_draws < 1:
            problems.append("simulation.n_draws must be at least 1")
        if s.scale_mode not in SCALE_MODES:
            problems.append(f"simulation.scale_mode must be one of {SCALE_MODES}")
        if s.prior != "uniform":
            problems.append("simulation.prior: only 'uniform' is implemented")
        if s.weight_concentration < 0:
            problems.append("simulation.weight_concentration must be 0 or positive")
        bad = [o for o in self.data.order if o not in ("local", "hf", "gee")]
        if bad:
            problems.append(f"data.order has unknown source(s): {bad}")
        return problems


def _build(cls: type, values: dict[str, Any], section: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in [{section}]: {sorted(unknown)}")
    return cls(**values)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
