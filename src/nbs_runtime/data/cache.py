"""A per-layer cache of what was extracted from Earth Engine, so it is extracted once.

The cache holds RAW extracted arrays (slope, SoilGrids as stored, land-cover class shares). Nothing in
it depends on CIAT's package: a new package version, a different gate or another weight policy reuses
the same files and never calls Earth Engine again. Each layer is saved as soon as it is fetched, so an
interrupted run resumes where it stopped.

A cache folder is keyed by the grid (extent, resolution, shape) and by the assets and methods used. Change
any of them and a new folder is used; nothing is overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..grid import Grid

SCHEMA_VERSION = 1


class LayerCache:
    def __init__(self, root: Path | str, grid: Grid, assets: Mapping[str, str], refresh: bool = False):
        self.root = Path(root)
        self.grid = grid
        self.assets = dict(assets)
        self.refresh = refresh
        payload = {
            "schema": SCHEMA_VERSION,
            "bbox": [round(v, 9) for v in grid.bbox],
            "resolution_m": grid.resolution_m,
            "shape": list(grid.shape),
            "assets": self.assets,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self.key = hashlib.sha256(blob).hexdigest()[:16]
        self.dir = self.root / self.key
        self._payload = payload

    def _file(self, name: str) -> Path:
        return self.dir / f"{name}.npz"

    def has(self, name: str) -> bool:
        return not self.refresh and self._file(name).exists()

    def get(self, name: str) -> dict[str, np.ndarray] | None:
        """The cached arrays of a layer, or None (also None when refresh is on)."""
        if not self.has(name):
            return None
        with np.load(self._file(name)) as z:
            return {k: z[k].astype(float) for k in z.files}

    def put(self, name: str, arrays: Mapping[str, np.ndarray]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{name}.tmp.npz"
        np.savez_compressed(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
        os.replace(tmp, self._file(name))                       # never leave a half-written file
        meta_path = self.dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
            **self._payload, "created": datetime.now(UTC).isoformat(timespec="seconds"), "layers": {}}
        first = next(iter(arrays.values()))
        meta["layers"][name] = {
            "arrays": list(arrays),
            "shape": list(np.asarray(first).shape),
            "unknown_share": round(float(np.mean([np.isnan(a).mean() for a in arrays.values()])), 4),
            "saved": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def summary(self) -> dict:
        meta_path = self.dir / "meta.json"
        return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
