"""Run configuration: everything a person decides before a run, in one place.

A run is defined by a TOML file (config/runs/), optionally overridden by a dict (the notebook's
parameters cell). Unknown keys are an error, so a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .aoi import AOI
from .curves import OUTSIDE
from .rasters import WORLDCOVER_CODES
from .recipes import WEIGHT_MODES

ELIGIBILITY_MODES = ("apply", "skip")
FILE_ORDER = ("local", "hf")
HF_MODES = ("stream", "download")


@dataclass(frozen=True)
class RunSpec:
    nbs: str = "agroforestry"            # folder under config/recipes
    family: str = "cross_family"         # folder under config/recipes/<nbs>
    weights: str = "recipe"              # "recipe" (renormalised among modulating) or "equal"
    outside: str = "unknown"             # beyond a recipe's nodes: "unknown" (not evaluated) or "clamp"


@dataclass(frozen=True)
class GridSpec:
    resolution_m: int = 1000


@dataclass(frozen=True)
class EligibilitySpec:
    mode: str = "apply"                  # "apply" reads land cover; "skip" treats every cell as eligible
    classes: tuple[int, ...] = (30, 40)  # ESA WorldCover classes where the NbS can be implemented
    min_share: float = 0.5               # a cell is eligible when at least this share of it is in `classes`


@dataclass(frozen=True)
class SensitivitySpec:
    n_draws: int = 300                   # Dirichlet draws of the modulating weights
    concentration: float = 50.0          # larger keeps the draws closer to the recipe weights
    seed: int = 20261124


@dataclass(frozen=True)
class DataSpec:
    order: tuple[str, ...] = ("local", "hf")    # where a dataset file is looked for, in this order
    hf_mode: str = "stream"                     # "stream" reads only the window; "download" whole files


@dataclass(frozen=True)
class RunConfig:
    run: RunSpec = field(default_factory=RunSpec)
    aoi: AOI = field(default_factory=AOI)
    grid: GridSpec = field(default_factory=GridSpec)
    eligibility: EligibilitySpec = field(default_factory=EligibilitySpec)
    sensitivity: SensitivitySpec = field(default_factory=SensitivitySpec)
    data: DataSpec = field(default_factory=DataSpec)

    # ---- construction ----
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        kw: dict[str, Any] = {}
        for name, sub in (("run", RunSpec), ("grid", GridSpec), ("sensitivity", SensitivitySpec)):
            if name in d:
                kw[name] = _build(sub, d[name], name)
        if "eligibility" in d:
            e = dict(d["eligibility"])
            if "classes" in e:
                e["classes"] = tuple(int(c) for c in e["classes"])
            kw["eligibility"] = _build(EligibilitySpec, e, "eligibility")
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
        d["eligibility"]["classes"] = list(self.eligibility.classes)
        d["data"]["order"] = list(self.data.order)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    # ---- validation (problems that make a run meaningless) ----
    def validate(self) -> list[str]:
        problems = list(self.aoi.validate())
        r, e, s = self.run, self.eligibility, self.sensitivity
        if not r.nbs or not r.family:
            problems.append("run.nbs and run.family are required")
        if r.weights not in WEIGHT_MODES:
            problems.append(f"run.weights must be one of {WEIGHT_MODES}")
        if r.outside not in OUTSIDE:
            problems.append(f"run.outside must be one of {OUTSIDE}")
        if self.grid.resolution_m <= 0:
            problems.append("grid.resolution_m must be positive")
        if e.mode not in ELIGIBILITY_MODES:
            problems.append(f"eligibility.mode must be one of {ELIGIBILITY_MODES}")
        if e.mode == "apply" and not e.classes:
            problems.append("eligibility.classes is empty: no land cover would be eligible")
        bad = [c for c in e.classes if c not in WORLDCOVER_CODES]
        if bad:
            problems.append(f"eligibility.classes has codes that are not ESA WorldCover classes: {bad}")
        if not 0 < e.min_share <= 1:
            problems.append("eligibility.min_share must be in (0, 1]")
        if s.n_draws < 1:
            problems.append("sensitivity.n_draws must be at least 1")
        if s.concentration <= 0:
            problems.append("sensitivity.concentration must be positive")
        bad = [o for o in self.data.order if o not in FILE_ORDER]
        if bad:
            problems.append(f"data.order has unknown source(s): {bad}; use {FILE_ORDER}")
        if self.data.hf_mode not in HF_MODES:
            problems.append(f"data.hf_mode must be one of {HF_MODES}")
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
