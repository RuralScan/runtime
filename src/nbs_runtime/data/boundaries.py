"""Official boundaries from the World Bank Official Boundaries dataset (GeoParquet, CC BY 4.0).

Each level's file (59, 88 or 175 MB) is one row group, so it can't be read in part: it is downloaded
whole into cache/boundaries/ once and filtered there.
"""
from __future__ import annotations

import urllib.request

from .gee import REPO_ROOT

URL = ("https://datacatalogfiles.worldbank.org/ddh-published/0038272/3/DR0096331/"
       "World%20Bank%20Official%20Boundaries%20%28Parquet%29/World%20Bank%20Official%20Boundaries%20-%20Admin%20{}.parquet")
SOURCE = "World Bank Official Boundaries (catalog DR0096331)"
CODE_COLUMNS = ("ISO_A3", "ADM1CD_c", "ADM2CD_c")  # level 0, 1, 2; e.g. HTI, HTI002, HTI002001
CACHE_DIR = REPO_ROOT / "cache" / "boundaries"


def admin(code: str, level: int = 0):
    """Admin units of one level inside `code`, as a GeoDataFrame in EPSG:4326.

    `code` can be any coarser or equal level: admin("KEN", 2) is every district of Kenya,
    admin("KEN001", 2) those of one region. Some ISO3 codes have several Admin 0 rows (ESP, GBR,
    AUS, BES: mainland plus territories)."""
    import geopandas as gpd

    path = CACHE_DIR / f"wb_admin{level}.parquet"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp, _ = urllib.request.urlretrieve(URL.format(level), path.with_suffix(".tmp"))
        tmp.replace(path)                                    # never leave a half-written file
    gdf = gpd.read_parquet(path, filters=[[(col, "==", code.upper())] for col in CODE_COLUMNS[:level + 1]])
    if gdf.empty:
        raise ValueError(f"no admin {level} units inside {code!r}")
    return gdf
