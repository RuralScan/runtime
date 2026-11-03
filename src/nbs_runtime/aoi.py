"""Area of interest (AOI): what a run covers, and where that boundary comes from.

Only the bounding box is resolved here. An official boundary (for example GADM through the T7
registry) is fetched later by the data adapter; until its source is recorded the plan reports a
hold, because CIAT's package asks that the target area and its source be recorded.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EARTH_RADIUS_KM = 6371.0088
KINDS = ("bbox", "geojson", "admin")


def _walk_coords(node: Any):
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
            yield float(node[0]), float(node[1])
        else:
            for child in node:
                yield from _walk_coords(child)


def bbox_of_geojson(path: Path) -> tuple[float, float, float, float]:
    """Bounds (minx, miny, maxx, maxy) of any GeoJSON Feature, FeatureCollection or geometry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = list(_walk_coords(_collect_coordinates(data)))
    if not pts:
        raise ValueError(f"no coordinates found in {path}")
    xs, ys = zip(*pts, strict=True)
    return (min(xs), min(ys), max(xs), max(ys))


def _collect_coordinates(node: Any) -> Any:
    if isinstance(node, dict):
        if "coordinates" in node:
            return node["coordinates"]
        if node.get("type") == "Feature":
            return _collect_coordinates(node.get("geometry") or {})
        if node.get("type") == "FeatureCollection":
            return [_collect_coordinates(f) for f in node.get("features", [])]
        if node.get("type") == "GeometryCollection":
            return [_collect_coordinates(g) for g in node.get("geometries", [])]
    return []


@dataclass(frozen=True)
class AOI:
    kind: str = "bbox"
    name: str = ""
    bbox: tuple[float, float, float, float] | None = None
    source: str = ""            # where the boundary comes from (empty = not recorded)
    geojson_path: str = ""      # kind = "geojson"
    admin_code: str = ""        # kind = "admin" (for example an ISO3 or GADM code)
    notes: tuple[str, ...] = field(default=(), compare=False)

    def resolved_bbox(self, base_dir: Path | None = None) -> tuple[float, float, float, float] | None:
        if self.kind == "bbox":
            return self.bbox
        if self.kind == "geojson" and self.geojson_path:
            p = Path(self.geojson_path)
            if base_dir and not p.is_absolute():
                p = base_dir / p
            return bbox_of_geojson(p) if p.exists() else None
        return None

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the AOI is well formed."""
        if self.kind not in KINDS:
            return [f"aoi.kind must be one of {KINDS}, got {self.kind!r}"]
        problems = []
        if self.kind == "bbox":
            if self.bbox is None or len(self.bbox) != 4:
                return ["aoi.bbox must be [minx, miny, maxx, maxy]"]
            minx, miny, maxx, maxy = self.bbox
            if not (-180 <= minx < maxx <= 180):
                problems.append("aoi.bbox longitudes must satisfy -180 <= minx < maxx <= 180")
            if not (-90 <= miny < maxy <= 90):
                problems.append("aoi.bbox latitudes must satisfy -90 <= miny < maxy <= 90")
        elif self.kind == "geojson" and not self.geojson_path:
            problems.append("aoi.geojson_path is required when kind = 'geojson'")
        elif self.kind == "admin" and not self.admin_code:
            problems.append("aoi.admin_code is required when kind = 'admin'")
        return problems

    def area_km2(self, base_dir: Path | None = None) -> float | None:
        """Spherical area of the bounding box (an upper bound of the real boundary)."""
        b = self.resolved_bbox(base_dir)
        if b is None:
            return None
        minx, miny, maxx, maxy = b
        dlon = math.radians(maxx - minx)
        return abs(
            EARTH_RADIUS_KM**2 * dlon * (math.sin(math.radians(maxy)) - math.sin(math.radians(miny)))
        )

    def estimated_cells(self, resolution_m: float, base_dir: Path | None = None) -> int | None:
        """Approximate grid cells of the bounding box at a given resolution."""
        b = self.resolved_bbox(base_dir)
        if b is None or resolution_m <= 0:
            return None
        minx, miny, maxx, maxy = b
        mid = math.radians((miny + maxy) / 2)
        width_m = (maxx - minx) * 111_320 * math.cos(mid)
        height_m = (maxy - miny) * 110_574
        return math.ceil(width_m / resolution_m) * math.ceil(height_m / resolution_m)

    def to_geojson(self) -> dict[str, Any]:
        if self.kind != "bbox" or self.bbox is None:
            raise ValueError("to_geojson is only defined for a bbox AOI")
        x0, y0, x1, y1 = self.bbox
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        return {
            "type": "Feature",
            "properties": {"name": self.name, "source": self.source},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }

    def fingerprint(self) -> str:
        """Stable hash of what defines the area, for run manifests and cache keys."""
        payload = {
            "kind": self.kind,
            "bbox": list(self.bbox) if self.bbox else None,
            "geojson_path": self.geojson_path,
            "admin_code": self.admin_code,
            "source": self.source,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()
