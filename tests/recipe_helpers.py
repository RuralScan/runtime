"""Build small recipe trees for tests."""
from __future__ import annotations

import json
from pathlib import Path


def write_recipe(root: Path, nbs: str, family: str, variable: str, x, score, weight=None, **extra) -> Path:
    folder = root / nbs / family
    folder.mkdir(parents=True, exist_ok=True)
    data = {"practice": nbs, "family": family, "variable": variable, "unit": "u",
            "shape": "linear", "x": list(x), "score": list(score), **extra}
    if weight is not None:
        data["weight"] = weight
    path = folder / f"{variable}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_roles(path: Path, roles: dict[str, str], nbs_overrides: dict[str, dict[str, str]] | None = None) -> Path:
    lines = ["[default]"] + [f'{v} = "{r}"' for v, r in roles.items()]
    for nbs, over in (nbs_overrides or {}).items():
        lines += ["", f"[{nbs}]"] + [f'{v} = "{r}"' for v, r in over.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def toy_family(tmp: Path):
    """Two limiting variables (rain, temp) and two modulating ones (slope 0.6, ndvi 0.4 after renormalising)."""
    root = tmp / "recipes"
    write_recipe(root, "toy", "fam", "rain", [500, 1000, 2000], [0.2, 0.8, 1.0], weight=0.5)
    write_recipe(root, "toy", "fam", "temp", [10, 20, 30], [0.5, 1.0, 0.5], weight=0.0)
    write_recipe(root, "toy", "fam", "slope", [0, 10, 30], [1.0, 0.5, 0.0], weight=0.3)
    write_recipe(root, "toy", "fam", "ndvi", [0.0, 1.0], [0.0, 1.0], weight=0.2)
    roles = write_roles(tmp / "roles.toml", {"rain": "limiting", "temp": "limiting",
                                             "slope": "modulating", "ndvi": "modulating"})
    return root, roles
