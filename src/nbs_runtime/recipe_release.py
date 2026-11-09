"""Read a pinned analytical recipe from a local release store."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._recipe_store import read_generation
from ._recipe_validation import REPORT_FILES, TABLE_STEMS, validate_recipe


def _unique_fields(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate pin field: {key}")
        result[key] = value
    return result


def _read_pin(path: Path | None) -> tuple[str, str, tuple[str, ...]]:
    source = files("nbs_runtime").joinpath("recipe_pin.json") if path is None else Path(path)
    pin = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
    if not isinstance(pin, dict) or set(pin) != {"release_id", "source_revision", "practices"}:
        raise ValueError("Expected release_id, source_revision and practices in recipe pin")
    for key, pattern in (("release_id", r"[0-9a-f]{64}"), ("source_revision", r"[0-9a-f]{40}")):
        if not isinstance(pin[key], str) or not re.fullmatch(pattern, pin[key]):
            raise ValueError(f"Invalid {key} in recipe pin")
    names = pin["practices"]
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in names
        )
    ):
        raise ValueError("Invalid practice inventory in recipe pin")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate practice in recipe pin")
    return pin["release_id"], pin["source_revision"], tuple(sorted(names))


@dataclass(frozen=True)
class RecipeRelease:
    """One verified snapshot. Each accessor returns independent data."""

    release_id: str
    practices: tuple[str, ...]
    _members: Mapping[str, bytes] = field(repr=False)

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._members["provenance.json"])

    def table(self, practice: str, table: str) -> list[dict[str, Any]]:
        if practice not in self.practices or table not in TABLE_STEMS:
            raise KeyError((practice, table))
        return json.loads(self._members[f"tables/{practice}/{TABLE_STEMS[table]}.json"])

    def setting(self, practice: str, filename: str) -> bytes:
        if practice not in self.practices:
            raise KeyError(practice)
        return self._members[f"settings/{practice}/{filename}"]

    def report(self, practice: str, filename: str) -> dict[str, Any]:
        if practice not in self.practices or filename not in REPORT_FILES:
            raise KeyError((practice, filename))
        return json.loads(self._members[f"tables/{practice}/{filename}"])


def load_recipe_release(store: Path, *, pin: Path | None = None) -> RecipeRelease:
    """Load the packaged pin, or an explicitly supplied trusted pin, without following current.json."""
    release_id, revision, expected_practices = _read_pin(pin)
    members = read_generation(Path(store), release_id)
    practices = validate_recipe(members)
    if practices != expected_practices:
        raise ValueError("Release practices differ from pinned practice inventory")
    if json.loads(members["provenance.json"])["source_revision"] != revision:
        raise ValueError("Release source revision differs from pinned revision")
    return RecipeRelease(release_id, practices, MappingProxyType(members))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path, required=True, help="Local directory containing generations/"
    )
    parser.add_argument(
        "--pin", type=Path, help="Use an explicit pin file in place of the bundled pin"
    )
    args = parser.parse_args(argv)
    try:
        release = load_recipe_release(args.store, pin=args.pin)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot load pinned recipe: {exc}\n")
    print(f"Release {release.release_id}")
    print(f"Source {release.provenance['source_revision']}")
    for practice in release.practices:
        counts = " ".join(f"{table}={len(release.table(practice, table))}" for table in TABLE_STEMS)
        ready = release.report(practice, "T4_readiness.json")["execution_ready"]
        print(f"{practice}: {counts}  execution_ready={str(ready).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
