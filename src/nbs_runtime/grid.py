"""The output grid every layer is placed on, defined once from the AOI and the resolution.

Same convention as the earlier prototype (a grid in degrees, cell size = resolution_m / 111,320),
so shapes match what was already validated against real pulls: Haiti at 5 km is 47 rows by 65 columns.
The grid is north-up: row 0 is the northern edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .aoi import AOI

METRES_PER_DEGREE = 111_320


@dataclass(frozen=True)
class Grid:
    bbox: tuple[float, float, float, float]
    resolution_m: int
    resolution_deg: float
    width: int
    height: int

    @classmethod
    def from_bbox(cls, bbox: tuple[float, float, float, float], resolution_m: int) -> Grid:
        minx, miny, maxx, maxy = bbox
        res = resolution_m / METRES_PER_DEGREE
        return cls(
            bbox=bbox,
            resolution_m=resolution_m,
            resolution_deg=res,
            width=max(1, math.ceil((maxx - minx) / res)),
            height=max(1, math.ceil((maxy - miny) / res)),
        )

    @classmethod
    def from_aoi(cls, aoi: AOI, resolution_m: int) -> Grid:
        bbox = aoi.resolved_bbox()
        if bbox is None:
            raise ValueError("the AOI has no bounding box to build a grid from")
        return cls.from_bbox(bbox, resolution_m)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def crs_transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-style affine (x size, 0, x origin, 0, -y size, y origin), as Earth Engine expects."""
        minx, _, _, maxy = self.bbox
        r = self.resolution_deg
        return (r, 0.0, minx, 0.0, -r, maxy)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) of the grid as drawn, for plotting."""
        minx, _, _, maxy = self.bbox
        r = self.resolution_deg
        return (minx, minx + self.width * r, maxy - self.height * r, maxy)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Longitude (width,) and latitude (height,) of the cell centres, latitude descending."""
        minx, _, _, maxy = self.bbox
        r = self.resolution_deg
        lon = minx + (np.arange(self.width) + 0.5) * r
        lat = maxy - (np.arange(self.height) + 0.5) * r
        return lon, lat

    def cell_area_km2(self) -> np.ndarray:
        """Approximate area of each row of cells in km2 (shrinks with latitude), shape (height,)."""
        _, lat = self.cell_centres()
        side_km = self.resolution_m / 1000.0
        return side_km * side_km * np.cos(np.radians(lat))
