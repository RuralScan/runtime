"""Read the per-variable recipes produced by the skill, and the runtime's role table.

A recipe is one JSON file per variable, ``config/recipes/<nbs>/<family>/<variable>.json``. It carries
the suitability function (nodes ``x`` -> ``score``, interpolated linearly) and the variable's weight.

Roles are not part of the recipe. They come from ``config/roles.toml`` and follow the methodology's
rule: what a project cannot change (climate) is *limiting*, what it can manage is *modulating*.
Only modulating variables use their weight; it is renormalised among them so that it sums to 1.
"""
from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_ROOT = REPO_ROOT / "config" / "recipes"
ROLES_PATH = REPO_ROOT / "config" / "roles.toml"

ROLES = ("limiting", "modulating")
SHAPES = ("linear",)
REQUIRED = ("practice", "family", "variable", "unit", "shape", "x", "score")
WEIGHT_MODES = ("recipe", "equal")


class RecipeError(ValueError):
    """A recipe, the role table or a family is not usable."""


@dataclass(frozen=True)
class Recipe:
    practice: str
    family: str
    variable: str
    unit: str
    shape: str
    x: tuple[float, ...]
    score: tuple[float, ...]
    weight: float | None
    low: tuple[float, ...] | None
    high: tuple[float, ...] | None
    valid_range: tuple[float | None, float | None]
    papers: tuple[str, ...]
    upstream_commit: str | None
    path: Path
    sha256: str


# --------------------------------------------------------------------------- one recipe

def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _numbers(raw: dict, key: str, where: str) -> tuple[float, ...]:
    vals = raw[key]
    if not isinstance(vals, list) or not all(_is_number(v) for v in vals):
        raise RecipeError(f"{where}: '{key}' must be a list of finite numbers")
    return tuple(float(v) for v in vals)


def load_recipe(path: Path | str) -> Recipe:
    path = Path(path)
    where = str(path)
    data = path.read_bytes()
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as e:
        raise RecipeError(f"{where}: not valid JSON ({e})") from e
    if not isinstance(raw, dict):
        raise RecipeError(f"{where}: expected a JSON object")

    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise RecipeError(f"{where}: missing {', '.join(missing)}")
    for k in ("practice", "family", "variable", "unit", "shape"):
        if not isinstance(raw[k], str) or not raw[k]:
            raise RecipeError(f"{where}: '{k}' must be a non-empty string")
    if raw["shape"] not in SHAPES:
        raise RecipeError(f"{where}: shape {raw['shape']!r} is not supported (only {SHAPES})")

    # the file must sit where it says it belongs
    if path.stem != raw["variable"]:
        raise RecipeError(f"{where}: file name does not match variable {raw['variable']!r}")
    if path.parent.name != raw["family"] or path.parent.parent.name != raw["practice"]:
        raise RecipeError(
            f"{where}: folder must be <practice>/<family> = {raw['practice']}/{raw['family']}"
        )

    x = _numbers(raw, "x", where)
    score = _numbers(raw, "score", where)
    if len(x) < 2:
        raise RecipeError(f"{where}: a linear function needs at least two nodes")
    if len(score) != len(x):
        raise RecipeError(f"{where}: 'x' has {len(x)} nodes but 'score' has {len(score)}")
    if any(b <= a for a, b in pairwise(x)):
        raise RecipeError(f"{where}: 'x' must be strictly increasing")
    if any(not 0.0 <= s <= 1.0 for s in score):
        raise RecipeError(f"{where}: every score must be between 0 and 1")

    low = high = None
    for key in ("low", "high"):
        if key in raw and raw[key] is not None:
            band = _numbers(raw, key, where)
            if len(band) != len(x):
                raise RecipeError(f"{where}: '{key}' must have one value per node")
            if any(not 0.0 <= s <= 1.0 for s in band):
                raise RecipeError(f"{where}: every '{key}' value must be between 0 and 1")
            low, high = (band, high) if key == "low" else (low, band)
    if low and any(lo > s + 1e-9 for lo, s in zip(low, score, strict=True)):
        raise RecipeError(f"{where}: 'low' is above 'score' at some node")
    if high and any(hi < s - 1e-9 for hi, s in zip(high, score, strict=True)):
        raise RecipeError(f"{where}: 'high' is below 'score' at some node")

    weight = raw.get("weight")
    if isinstance(weight, list):
        raise RecipeError(f"{where}: 'weight' must be a number, not a list")
    if weight is not None:
        if not _is_number(weight) or weight < 0:
            raise RecipeError(f"{where}: 'weight' must be a number >= 0")
        weight = float(weight)

    vr = raw.get("valid_range", [None, None])
    if not (isinstance(vr, list) and len(vr) == 2 and all(v is None or _is_number(v) for v in vr)):
        raise RecipeError(f"{where}: 'valid_range' must be [min or null, max or null]")

    papers = raw.get("papers", [])
    if not isinstance(papers, list) or not all(isinstance(p, str) for p in papers):
        raise RecipeError(f"{where}: 'papers' must be a list of strings")

    return Recipe(
        practice=raw["practice"], family=raw["family"], variable=raw["variable"],
        unit=raw["unit"], shape=raw["shape"], x=x, score=score, weight=weight,
        low=low, high=high, valid_range=(vr[0], vr[1]), papers=tuple(papers),
        upstream_commit=raw.get("upstream_commit"), path=path,
        sha256=hashlib.sha256(data).hexdigest(),
    )


# --------------------------------------------------------------------------- roles

def load_roles(nbs: str, path: Path | str = ROLES_PATH) -> dict[str, str]:
    """Roles for one NbS: the [default] table, overridden by the table named after the NbS."""
    path = Path(path)
    if not path.exists():
        raise RecipeError(f"role table not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    roles = {**data.get("default", {}), **data.get(nbs, {})}
    bad = {v: r for v, r in roles.items() if r not in ROLES}
    if bad:
        raise RecipeError(f"{path}: unknown role(s) {bad}; use one of {ROLES}")
    return roles


# --------------------------------------------------------------------------- one family

@dataclass(frozen=True)
class Family:
    """All recipes of one <nbs>/<family> folder, with their roles."""

    nbs: str
    family: str
    recipes: tuple[Recipe, ...]
    roles: Mapping[str, str]

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(r.variable for r in self.recipes)

    def recipe(self, variable: str) -> Recipe:
        for r in self.recipes:
            if r.variable == variable:
                return r
        raise KeyError(variable)

    def by_role(self, role: str, drop: tuple[str, ...] = ()) -> tuple[Recipe, ...]:
        return tuple(r for r in self.recipes if self.roles[r.variable] == role and r.variable not in drop)

    @property
    def limiting(self) -> tuple[Recipe, ...]:
        return self.by_role("limiting")

    @property
    def modulating(self) -> tuple[Recipe, ...]:
        return self.by_role("modulating")

    def modulating_weights(self, mode: str = "recipe", drop: tuple[str, ...] = ()) -> dict[str, float]:
        """Weights of the modulating variables, summing to 1. 'equal' gives each the same weight."""
        if mode not in WEIGHT_MODES:
            raise ValueError(f"weight mode must be one of {WEIGHT_MODES}")
        mods = self.by_role("modulating", drop)
        if not mods:
            return {}
        if mode == "equal":
            return {r.variable: 1.0 / len(mods) for r in mods}
        total = sum(r.weight or 0.0 for r in mods)
        if total <= 0:
            raise RecipeError(f"{self.nbs}/{self.family}: the modulating weights sum to 0")
        return {r.variable: (r.weight or 0.0) / total for r in mods}

    def manifest(self) -> dict[str, Any]:
        """What a run must record about its inputs."""
        return {
            "nbs": self.nbs,
            "family": self.family,
            "recipes": {r.variable: {"file": r.path.name, "sha256": r.sha256,
                                     "upstream_commit": r.upstream_commit} for r in self.recipes},
            "roles": {v: self.roles[v] for v in self.variables},
            "weights_in_recipes": {r.variable: r.weight for r in self.recipes},
            "modulating_weights": self.modulating_weights("recipe"),
        }


def discover(root: Path | str = RECIPES_ROOT) -> dict[tuple[str, str], list[Path]]:
    """Every <nbs>/<family> folder under root and its recipe files."""
    root = Path(root)
    found: dict[tuple[str, str], list[Path]] = {}
    for p in sorted(root.glob("*/*/*.json")):
        found.setdefault((p.parent.parent.name, p.parent.name), []).append(p)
    return found


def load_family(nbs: str, family: str, root: Path | str = RECIPES_ROOT,
                roles_path: Path | str = ROLES_PATH) -> Family:
    folder = Path(root) / nbs / family
    files = sorted(folder.glob("*.json"))
    if not files:
        raise RecipeError(f"no recipes in {folder}")
    recipes = tuple(load_recipe(p) for p in files)
    roles = load_roles(nbs, roles_path)

    missing = [r.variable for r in recipes if r.variable not in roles]
    if missing:
        raise RecipeError(f"{nbs}/{family}: no role in {roles_path} for {', '.join(missing)}")
    fam = Family(nbs, family, recipes, {r.variable: roles[r.variable] for r in recipes})

    unweighted = [r.variable for r in fam.modulating if r.weight is None]
    if unweighted:
        raise RecipeError(f"{nbs}/{family}: modulating variable(s) without weight: {', '.join(unweighted)}")
    fam.modulating_weights("recipe")          # raises if they sum to 0
    return fam


def validate_all(root: Path | str = RECIPES_ROOT, roles_path: Path | str = ROLES_PATH) -> list[str]:
    """Load every family; return the problems found (empty means everything is usable)."""
    problems = []
    for nbs, family in discover(root):
        try:
            load_family(nbs, family, root, roles_path)
        except RecipeError as e:
            problems.append(str(e))
    return problems


def describe_roles(root: Path | str = RECIPES_ROOT, roles_path: Path | str = ROLES_PATH) -> dict[str, str]:
    """variable -> where it is used, e.g. "agroforestry/cross_family: limiting" (families that fail are skipped)."""
    out: dict[str, list[str]] = {}
    for nbs, family in discover(root):
        try:
            fam = load_family(nbs, family, root, roles_path)
        except RecipeError:
            continue
        w = fam.modulating_weights()
        for v in fam.variables:
            tag = f"modulating w={w[v]:.2f}" if v in w else fam.roles[v]
            out.setdefault(v, []).append(f"{nbs}/{family}: {tag}")
    return {v: "; ".join(t) for v, t in out.items()}
