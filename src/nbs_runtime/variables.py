"""The variables we can look at for an area, with the group they belong to and where each one comes from.

The catalog is the single place that says what a variable is, which group and subgroup it sits in, its unit,
where it is read from and what range of values is plausible. Groups follow how CIAT organises the variables
(its `group_id`: climatic, edaphic, topographic, land use, socio-economic, governance, infrastructure),
collected under four headings that read naturally in a notebook.

The role a variable plays in CIAT's package (a score, a gate, a mask, an operational scenario) is NOT written
here: it is read from the pinned package at run time (`contract_roles`), because those roles change when
CIAT publishes a new version.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

GROUPS = ("Climate variables", "Soil", "Terrain and land", "Project risk")

# where a variable is read from
HF_RASTER = "hf_raster"        # a global raster in the Hugging Face dataset (or a local copy of it)
HWSD = "hwsd"                  # HWSD v2: soil-unit raster + attribute database
OSM_ROADS = "osm_roads"        # road lines in the Hugging Face dataset, turned into a distance
GEE_CACHED = "gee_cached"      # Earth Engine layers shared with the index calculation (cached)
GEE_IMAGE = "gee_image"        # a single Earth Engine image, aggregated to the grid

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
    id: str
    label: str
    group: str
    subgroup: str
    unit: str
    source: str
    dataset: str                       # a name people recognise
    native_m: int                      # native resolution of the source, metres
    detail: Mapping[str, Any] = field(default_factory=dict)
    cmap: str = "viridis"
    kind: str = "continuous"           # or "categorical"
    plausible: tuple[float, float] | None = None
    class_labels: Mapping[int, str] | None = None
    cost: str = "fast"                 # "fast" or "slow" (reads 30 m or 10 m data in Earth Engine)
    note: str = ""


CATALOG: tuple[VarSpec, ...] = (
    # ---------------- Climate variables ----------------
    VarSpec("annual_precipitation", "Annual precipitation", "Climate variables", "Rainfall and temperature",
            "mm/year", HF_RASTER, "WorldClim 2.1, BIO12", 1000,
            {"rel": "raw/worldclim_v2_bioclim_precip/wc2.1_30s_bio_12.tif", "scale": 1.0}, "Blues",
            plausible=(0.0, 6000.0),
            note="1970-2000 average. CIAT's default binding for this variable points to CHELSA; this is the "
                 "WorldClim file hosted in the dataset. Confirm which one the policy should use."),
    VarSpec("mean_annual_temperature", "Mean annual temperature", "Climate variables", "Rainfall and temperature",
            "degC", HF_RASTER, "WorldClim 2.1, BIO1", 1000,
            {"rel": "raw/worldclim_v2_bioclim_temp/wc2.1_30s_bio_1.tif", "scale": 1.0}, "coolwarm",
            plausible=(-15.0, 35.0),
            note="1970-2000 average. CIAT's default binding points to CHELSA; this is the WorldClim file."),
    VarSpec("aridity_index", "Aridity index", "Climate variables", "Dryness",
            "dimensionless", HF_RASTER, "Global Aridity Index v3.1", 1000,
            {"rel": "raw/global_aridity_index_v3/ai_v31_yr.tif", "scale": 0.0001}, "RdYlBu",
            plausible=(0.0, 3.0),
            note="Stored as integers x 10,000; multiplied by 0.0001 here. Low values are dry, high are humid."),
    # ---------------- Soil ----------------
    VarSpec("soil_ph", "Soil pH, 0-30 cm", "Soil", "Fertility", "pH units", GEE_CACHED, "SoilGrids 2.0", 250,
            {"layers": ("phh2o_0-5", "phh2o_5-15", "phh2o_15-30")}, "RdYlGn", plausible=(3.0, 10.0),
            note="Three depths, converted once (pH x 10 to pH) and weighted by layer thickness."),
    VarSpec("soil_organic_carbon", "Soil organic carbon, 0-30 cm", "Soil", "Fertility", "g/kg", GEE_CACHED,
            "SoilGrids 2.0", 250, {"layers": ("soc_0-5", "soc_5-15", "soc_15-30")}, "copper",
            plausible=(0.0, 600.0),
            note="Three depths, converted once (dg/kg x 0.1 to g/kg) and weighted by layer thickness."),
    VarSpec("soil_drainage", "Soil drainage class", "Soil", "Water", "class 1-7", HWSD, "HWSD v2", 1000,
            {"rel": "raw/hwsd_v2/HWSD2.bil", "sqlite": "raw/hwsd_v2/HWSD2.sqlite"}, "YlGnBu",
            kind="categorical", plausible=(1.0, 7.0), class_labels=DRAINAGE_LABELS,
            note="The raster holds soil-unit ids; the drainage class comes from the attribute database, "
                 "taking the dominant soil of each unit."),
    # ---------------- Terrain and land ----------------
    VarSpec("slope", "Slope", "Terrain and land", "Topography", "degrees", GEE_CACHED, "SRTM", 30,
            {"layers": ("slope",)}, "terrain", plausible=(0.0, 90.0), cost="slow",
            note="Derived at the native 30 m scale and only then averaged to the grid."),
    VarSpec("land_cover", "Land cover (dominant class)", "Terrain and land", "Land cover and vegetation",
            "class", GEE_CACHED, "ESA WorldCover 2021", 10, {"layers": ("worldcover_shares",)}, "tab20",
            kind="categorical", class_labels=LAND_COVER_LABELS, cost="slow",
            note="Shows the class with the largest share in each cell. The host gate uses the share of each "
                 "class, not the dominant one."),
    VarSpec("tree_canopy_cover", "Tree canopy cover", "Terrain and land", "Land cover and vegetation",
            "percent", GEE_IMAGE, "Hansen Global Forest Change (treecover2000)", 30,
            {"asset": "UMD/hansen/global_forest_change_2023_v1_11", "band": "treecover2000"}, "Greens",
            plausible=(0.0, 100.0), cost="slow",
            note="Canopy cover in the year 2000. Newer versions of the asset exist; this is the one in CIAT's registry."),
    # ---------------- Project risk ----------------
    VarSpec("accessibility_travel_time", "Travel time to the nearest city", "Project risk", "Access and energy",
            "minutes", GEE_IMAGE, "Oxford MAP accessibility 2015", 1000,
            {"asset": "Oxford/MAP/accessibility_to_cities_2015_v1_0", "band": "accessibility"}, "magma",
            plausible=(0.0, 20000.0),
            note="Raw minutes. Any transformation to a 0-1 scale belongs to the function that uses it."),
    VarSpec("distance_to_road", "Distance to the nearest road", "Project risk", "Access and energy", "km",
            OSM_ROADS, "OpenStreetMap roads (Geofabrik)", 100,
            {"rel": "raw/osm_road_network/gis_osm_roads_free_1.shp"}, "YlOrRd", plausible=(0.0, 500.0),
            note="Straight-line distance to the nearest road of any class, in a local metric projection. "
                 "The file covers Haiti and the Dominican Republic."),
    VarSpec("electrification_index", "Night-time light (electrification proxy)", "Project risk", "Access and energy",
            "nW/cm2/sr", GEE_IMAGE, "VIIRS annual night-time lights", 500,
            {"asset": "NOAA/VIIRS/DNB/ANNUAL_V21", "band": "average", "year": None}, "inferno",
            plausible=(0.0, 500.0),
            note="Most rural land is near 0 and cities stand out. By default it uses the latest year the asset has; "
                 "set VIIRS_YEAR to choose another."),
    VarSpec("protected_area_status", "Share of the cell inside a protected area", "Project risk",
            "Legal and protection", "share 0-1", GEE_IMAGE, "WDPA (World Database on Protected Areas)", 30,
            {"asset": "WCMC/WDPA/current/polygons"}, "Purples", plausible=(0.0, 1.0), cost="slow",
            note="Context only. CIAT states that a protected-area category alone does not establish a legal "
                 "exclusion: that needs an assessed rule."),
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


def contract_roles(pkg) -> dict[str, list[str]]:
    """The role(s) each variable has in the pinned CIAT package, read at run time (never copied here)."""
    roles: dict[str, set[str]] = defaultdict(set)
    for row in pkg.rows.values():
        roles[row["variable"]].add(row["role"])
    return {k: sorted(v) for k, v in roles.items()}


def policy_variables(pkg, policy_id: str) -> set[str]:
    """The variables the approved policy actually uses: its scored criteria and its gates."""
    pol = pkg.policy(policy_id)
    ids = [c.rule_id for c in pol.criteria] + list(pol.gate_rule_ids)
    return {pkg.rows[i]["variable"] for i in ids}


ROLE_TEXT = {
    "suitability_score": "scores the index",
    "eligibility_gate": "host gate",
    "hard_exclusion": "legal mask",
    "operational_scenario": "operational (project risk)",
    "reference_only": "evidence only",
}


def role_text(var_id: str, roles: Mapping[str, list[str]], policy_vars: set[str] | None = None) -> str:
    """What a variable does in CIAT's package. With `policy_vars`, a score that the approved policy does not use
    is said so, instead of being called a score of the index."""
    if var_id not in roles:
        return "not in the package"
    parts = []
    for r in roles[var_id]:
        if r == "suitability_score" and policy_vars is not None:
            parts.append("scores the index (first-pass policy)" if var_id in policy_vars
                         else "a score in the package, not in the first-pass policy")
        else:
            parts.append(ROLE_TEXT.get(r, r))
    return ", ".join(parts)
