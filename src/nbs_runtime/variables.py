"""The variables we can look at for an area, with the group they belong to and where each one comes from.

The catalog is the single place that says what a variable is, which group and subgroup it sits in, its unit,
where it is read from and what range of values is plausible, under four headings that read naturally in a
notebook.

A recipe (config/recipes/<nbs>/<family>/<variable>.json) is matched to its data by the variable id: every
variable that has a recipe needs an entry here. The role a variable plays in the index (limiting or
modulating) is NOT written here: it is in config/roles.toml.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

GROUPS = ("Climate variables", "Soil", "Terrain and land", "Project risk")

# where a variable is read from
GEE = "gee"      # an Earth Engine asset
FILE = "file"    # a file: hf://datasets/..., s3://, https://, or a local path; a local copy wins (DataSpec.order)

HF = "hf://datasets/dagudelo30/nbs-ruralscan-data/"   # Diego's mirror of the file datasets

METRES_PER_DEGREE = 111_320                # one degree at the equator
ARCSEC_M = METRES_PER_DEGREE / 3600       # one arc-second at the equator, for datasets on an arc-second grid

LAND_COVER_LABELS = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare / sparse vegetation", 70: "Snow and ice", 80: "Permanent water",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen",
}
DRAINAGE_LABELS = {
    1: "E: excessively drained", 2: "SE: somewhat excessively", 3: "W: well drained",
    4: "MW: moderately well", 5: "I: imperfectly drained", 6: "P: poorly drained", 7: "VP: very poorly drained",
}


@dataclass(frozen=True)
class VarSpec:
    """One variable: what it is, and everything needed to load it. Swap a dataset by editing its entry."""
    id: str
    label: str
    group: str
    subgroup: str
    unit: str
    source: str                        # GEE or FILE
    path: str                          # Earth Engine asset id, or file path inside the dataset
    dataset: str                       # a name people recognise
    native_m: float                    # native resolution of the source, metres
    band: str | None = None
    scale: float = 1.0                 # stored value x scale = value in `unit`
    resample: str = "average"          # rasterio name: "average" for continuous values, "mode" for class ids
    derive: str | None = None          # named derivation step (data/variables_loader.py); None = read as is
    params: Mapping[str, Any] = field(default_factory=dict)    # extra inputs of the derivation step
    cmap: str = "viridis"
    kind: str = "continuous"           # or "categorical"
    plausible: tuple[float, float] | None = None
    class_labels: Mapping[int, str] | None = None
    cost: str = "fast"                 # "fast" or "slow" (reads 30 m or 10 m data in Earth Engine)
    note: str = ""

    @property
    def native_deg(self) -> float:
        return self.native_m / METRES_PER_DEGREE


CATALOG: tuple[VarSpec, ...] = (
    # ---------------- Climate variables ----------------
    VarSpec("annual_precipitation", "Annual precipitation", "Climate variables", "Rainfall and temperature",
            "mm/year", FILE, HF + "raw/worldclim_v2_bioclim_precip/wc2.1_30s_bio_12.tif", "WorldClim 2.1, BIO12", 1000,
            cmap="Blues", plausible=(0.0, 6000.0),
            note="1970-2000 average (WorldClim file hosted in the dataset)."),
    VarSpec("mean_annual_temperature", "Mean annual temperature", "Climate variables", "Rainfall and temperature",
            "degC", FILE, HF + "raw/worldclim_v2_bioclim_temp/wc2.1_30s_bio_1.tif", "WorldClim 2.1, BIO1", 1000,
            cmap="coolwarm", plausible=(-15.0, 35.0),
            note="1970-2000 average (WorldClim file hosted in the dataset)."),
    VarSpec("aridity_index", "Aridity index", "Climate variables", "Dryness",
            "dimensionless", FILE, HF + "raw/global_aridity_index_v3/ai_v31_yr.tif", "Global Aridity Index v3.1", 1000,
            scale=0.0001, cmap="RdYlBu", plausible=(0.0, 3.0),
            note="Stored as integers x 10,000; multiplied by 0.0001 here. Low values are dry, high are humid."),
    # ---------------- Soil ----------------
    VarSpec("soil_ph", "Soil pH, 0-30 cm", "Soil", "Fertility", "pH units", GEE,
            "projects/soilgrids-isric/phh2o_mean", "SoilGrids 2.0", 250, scale=0.1,
            derive="soilgrids_0_30", params={"variable": "phh2o"}, cmap="RdYlGn", plausible=(3.0, 10.0),
            note="Three depths, converted once (pH x 10 to pH) and weighted by layer thickness."),
    VarSpec("soil_organic_carbon", "Soil organic carbon, 0-30 cm", "Soil", "Fertility", "g/kg", GEE,
            "projects/soilgrids-isric/soc_mean", "SoilGrids 2.0", 250, scale=0.1,
            derive="soilgrids_0_30", params={"variable": "soc"}, cmap="copper", plausible=(0.0, 600.0),
            note="Three depths, converted once (dg/kg x 0.1 to g/kg) and weighted by layer thickness."),
    VarSpec("soil_drainage", "Soil drainage class", "Soil", "Water", "class 1-7", FILE,
            HF + "raw/hwsd_v2/HWSD2.bil", "HWSD v2", 1000, resample="mode",
            derive="hwsd_drainage", params={"sqlite": HF + "raw/hwsd_v2/HWSD2.sqlite"}, cmap="YlGnBu",
            kind="categorical", plausible=(1.0, 7.0), class_labels=DRAINAGE_LABELS,
            note="The raster holds soil-unit ids; the drainage class comes from the attribute database, "
                 "taking the dominant soil of each unit."),
    # ---------------- Terrain and land ----------------
    VarSpec("slope", "Slope", "Terrain and land", "Topography", "degrees", GEE, "USGS/SRTMGL1_003", "SRTM",
            ARCSEC_M, band="elevation", derive="slope", cmap="terrain", plausible=(0.0, 90.0), cost="slow",
            note="Derived at the native 30 m scale and only then averaged to the grid."),
    VarSpec("elevation", "Elevation", "Terrain and land", "Topography", "m", GEE, "USGS/SRTMGL1_003", "SRTM",
            ARCSEC_M, band="elevation", cmap="terrain", plausible=(-450.0, 9000.0),
            note="Mean of the 30 m elevation over each grid cell. Shares the asset with slope."),
    VarSpec("land_cover", "Land cover (dominant class)", "Terrain and land", "Land cover and vegetation",
            "class", GEE, "ESA/WorldCover/v200", "ESA WorldCover 2021", 10, band="Map", derive="class_shares",
            cmap="tab20", kind="categorical", class_labels=LAND_COVER_LABELS, cost="slow",
            note="Shows the class with the largest share in each cell. The host gate uses the share of each "
                 "class, not the dominant one."),
    VarSpec("tree_canopy_cover", "Tree canopy cover", "Terrain and land", "Land cover and vegetation",
            "percent", GEE, "UMD/hansen/global_forest_change_2023_v1_11", "Hansen Global Forest Change (treecover2000)",
            ARCSEC_M, band="treecover2000", cmap="Greens", plausible=(0.0, 100.0), cost="slow",
            note="Canopy cover in the year 2000. Newer versions of the asset exist."),
    VarSpec("ndvi", "NDVI (annual median)", "Terrain and land", "Land cover and vegetation", "unitless", GEE,
            "MODIS/061/MOD13Q1", "MODIS Terra vegetation indices, 16-day, 250 m", 250, band="NDVI", scale=0.0001,
            derive="annual_median", params={"year": 2023}, cmap="YlGn", plausible=(-1.0, 1.0),
            note="Median of the 16-day composites of one year (set in params), stored x 10,000 and scaled here."),
    # ---------------- Project risk ----------------
    VarSpec("accessibility_travel_time", "Travel time to the nearest city", "Project risk", "Access and energy",
            "minutes", GEE, "Oxford/MAP/accessibility_to_cities_2015_v1_0", "Oxford MAP accessibility 2015",
            30 * ARCSEC_M, band="accessibility", cmap="magma", plausible=(0.0, 20000.0),
            note="Raw minutes. Any transformation to a 0-1 scale belongs to the function that uses it."),
    VarSpec("distance_to_road", "Distance to the nearest road", "Project risk", "Access and energy", "km",
            FILE, HF + "raw/osm_road_network/gis_osm_roads_free_1.shp", "OpenStreetMap roads (Geofabrik)", 100,
            derive="distance_to_lines", cmap="YlOrRd", plausible=(0.0, 500.0),
            note="Straight-line distance to the nearest road of any class, in a local metric projection. "
                 "The file covers Haiti and the Dominican Republic."),
    VarSpec("electrification_index", "Night-time light (electrification proxy)", "Project risk", "Access and energy",
            "nW/cm2/sr", GEE, "NOAA/VIIRS/DNB/ANNUAL_V21", "VIIRS annual night-time lights", 15 * ARCSEC_M,
            band="average", derive="annual_composite", params={"year": None}, cmap="inferno",
            plausible=(0.0, 500.0),
            note="Most rural land is near 0 and cities stand out. By default it uses the latest year the asset has; "
                 "set VIIRS_YEAR to choose another."),
    VarSpec("protected_area_status", "Share of the cell inside a protected area", "Project risk",
            "Legal and protection", "share 0-1", GEE, "WCMC/WDPA/current/polygons",
            "WDPA (World Database on Protected Areas)", 30, derive="polygon_share", cmap="Purples",
            plausible=(0.0, 1.0), cost="slow",
            note="Context only: a protected-area category alone does not establish a legal exclusion."),
)

BY_ID: dict[str, VarSpec] = {v.id: v for v in CATALOG}


def by_group() -> dict[str, dict[str, list[VarSpec]]]:
    """{group: {subgroup: [variables]}} in catalog order."""
    out: dict[str, dict[str, list[VarSpec]]] = {g: {} for g in GROUPS}
    for v in CATALOG:
        out[v.group].setdefault(v.subgroup, []).append(v)
    return out


def select(groups: tuple[str, ...] | list[str] | None = None, skip: tuple[str, ...] | list[str] = (),
           include_slow: bool = True) -> list[VarSpec]:
    """Variables of the chosen groups (all by default), without the ones in `skip`."""
    chosen = [v for v in CATALOG if (groups is None or v.group in groups) and v.id not in skip]
    return chosen if include_slow else [v for v in chosen if v.cost != "slow"]


_UNIT_ALIASES = {
    "mm/yr": "mm/year", "mm year-1": "mm/year", "mm per year": "mm/year",
    "deg": "degrees", "degree": "degrees", "°": "degrees",
    "metres": "m", "meters": "m",
    "dimensionless": "unitless", "-": "unitless", "": "unitless", "index": "unitless",
    "°c": "degc", "celsius": "degc",
}


def same_unit(a: str, b: str) -> bool:
    """Whether a recipe's unit and a catalog unit mean the same thing (a few common spellings accepted).

    A mismatch matters: a slope function in percent applied to a slope in degrees is silently wrong.
    """
    norm = lambda u: _UNIT_ALIASES.get(u.strip().lower(), u.strip().lower())
    return norm(a) == norm(b)
