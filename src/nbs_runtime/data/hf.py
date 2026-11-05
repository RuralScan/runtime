"""Reading the layers hosted in the Hugging Face dataset (or a local copy of it).

A file is looked for in the order the caller gives ("local", then "hf"). A local copy keeps the same folder layout
as the dataset (`raw/<dataset folder>/<file>`), under any root you list in `local_roots`.

Large global rasters are not downloaded whole. They are opened over HTTP with GDAL's /vsicurl/ and only the window
that covers the area is read (`stream`). If streaming does not work on a network, `download` fetches the whole file
once into the Hugging Face cache. Small files (the HWSD attribute database, the road shapefile) are always downloaded.

Nothing here decides an analytical rule: it finds a file, reads a window of it, and converts a soil-unit id or a
set of road lines into a value per cell.
"""
from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..grid import Grid

DEFAULT_REPO = "dagudelo30/nbs-ruralscan-data"
ORIGINS = ("local", "hf-stream", "hf-download")


class FileNotAvailable(FileNotFoundError):
    """A file was not found in any of the places it was looked for."""


def hf_token() -> str | None:
    """A Hugging Face token from the HF_TOKEN environment variable or the .env file, if there is one."""
    from .gee import REPO_ROOT, read_env_file

    token = os.environ.get("HF_TOKEN") or read_env_file(REPO_ROOT / ".env").get("HF_TOKEN")
    return token.strip() if token and token.strip() else None


def hf_url(rel: str, repo: str = DEFAULT_REPO) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}"


def find_local(rel: str, local_roots: Sequence[Path | str]) -> Path | None:
    """A local copy of a dataset file. Tries `<root>/<rel>` and, for `raw/...`, also `<root>/...` without `raw`."""
    variants = [Path(rel)]
    if rel.startswith("raw/"):
        variants.append(Path(rel[len("raw/"):]))
    for root in local_roots:
        for v in variants:
            p = Path(root) / v
            if p.exists():
                return p
    return None


def download(rel: str, repo: str = DEFAULT_REPO) -> Path:
    """Download one file of the dataset into the Hugging Face cache (once) and return its path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is needed to download from the dataset: "
                          "python -m pip install huggingface_hub") from exc
    return Path(hf_hub_download(repo_id=repo, filename=rel, repo_type="dataset"))


def resolve_file(rel: str, order: Sequence[str] = ("local", "hf"), local_roots: Sequence[Path | str] = (),
                 repo: str = DEFAULT_REPO, siblings: Sequence[str] = ()) -> tuple[Path, str]:
    """A file on disk, local first and then downloaded. `siblings` are companion files kept next to it.

    Returns (path, origin) with origin "local" or "hf-download".
    """
    problems = []
    for where in order:
        if where == "local":
            found = find_local(rel, local_roots)
            if found is not None:
                for s in siblings:                                           # a shapefile needs .shx, .dbf, ...
                    if find_local(s, local_roots) is None:
                        problems.append(f"local copy of {rel} is missing its companion file {s}")
                        found = None
                        break
                if found is not None:
                    return found, "local"
            else:
                problems.append(f"{rel} not found under {[str(r) for r in local_roots] or 'any local root'}")
        elif where == "hf":
            try:
                path = download(rel, repo)
                for s in siblings:
                    download(s, repo)
                return path, "hf-download"
            except (ImportError, OSError) as exc:
                problems.append(f"could not download {rel} from {repo}: {exc}")
    raise FileNotAvailable("; ".join(problems) or f"{rel}: no place to look in (order={tuple(order)})")


def raster_source(rel: str, order: Sequence[str] = ("local", "hf"), local_roots: Sequence[Path | str] = (),
                  repo: str = DEFAULT_REPO, mode: str = "stream") -> tuple[str, str]:
    """What rasterio should open for a global raster, and where it came from.

    Local first. Otherwise `stream` opens the file over HTTP and reads only the window needed, `download`
    fetches it once into the Hugging Face cache.
    """
    if mode not in ("stream", "download"):
        raise ValueError("mode must be 'stream' or 'download'")
    problems = []
    for where in order:
        if where == "local":
            found = find_local(rel, local_roots)
            if found is not None:
                return str(found), "local"
            problems.append(f"{rel} not found locally")
        elif where == "hf":
            if mode == "stream":
                return "/vsicurl/" + hf_url(rel, repo), "hf-stream"
            try:
                return str(download(rel, repo)), "hf-download"
            except (ImportError, OSError) as exc:
                problems.append(f"could not download {rel}: {exc}")
    raise FileNotAvailable("; ".join(problems))


def check_access(repo: str = DEFAULT_REPO, rel: str = "raw/hwsd_v2/HWSD2.hdr", timeout: float = 20.0) -> str:
    """A quick check that the dataset is reachable from this machine. Returns a one-line verdict."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(hf_url(rel, repo), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"reachable: HTTP {r.status} for {rel}"
    except urllib.error.HTTPError as exc:
        return f"the server answered HTTP {exc.code} for {rel}: check the repo name and that the file exists"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"not reachable from this machine ({exc}); try HF_MODE = 'download' or check the network/proxy"


# ------------------------------------------------------------------ rasters
def read_raster_to_grid(source: str, grid: Grid, resampling: str = "average", scale: float = 1.0,
                        nodata: Sequence[float] = ()) -> np.ndarray:
    """The values of a raster over the grid, as a (rows, columns) array with NaN where there is no data.

    Only the part of the raster under the grid is read. `average` is for continuous values (it averages the
    fine cells under each grid cell), `mode` and `nearest` keep whole class codes. `scale` multiplies the
    result once (for example 0.0001 for the aridity index).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    methods = {"average": Resampling.average, "mode": Resampling.mode, "nearest": Resampling.nearest,
               "bilinear": Resampling.bilinear}
    if resampling not in methods:
        raise ValueError(f"resampling must be one of {sorted(methods)}")
    minx, _, _, maxy = grid.bbox
    r = grid.resolution_deg
    dst_transform = Affine(r, 0.0, minx, 0.0, -r, maxy)
    dst = np.full(grid.shape, np.nan, dtype=np.float32)
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.bil,.hdr,.stx",
           "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2"}
    token = hf_token()
    if token and source.startswith("/vsicurl/"):                       # a private dataset
        env["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"
    with rasterio.Env(**env), rasterio.open(source) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1), destination=dst, dst_transform=dst_transform, dst_crs="EPSG:4326",
            src_crs=src.crs or "EPSG:4326", src_nodata=src_nodata, dst_nodata=np.nan, resampling=methods[resampling],
        )
    out = dst.astype(float)
    for value in nodata:
        out[np.isclose(out, value)] = np.nan
    if src_nodata is not None and not np.isnan(src_nodata):
        out[np.isclose(out, src_nodata)] = np.nan
    return out * scale


# ------------------------------------------------------------------ HWSD v2 drainage
def hwsd_drainage_lookup(sqlite_path: Path | str) -> dict[int, float]:
    """{soil-unit id: drainage code} taking the dominant soil (largest SHARE) of each unit.

    HWSD2_SMU.DRAINAGE stores a text symbol; the numeric code is in D_DRAINAGE. In the official file the columns of
    that table are labelled the other way round (the one called SYMBOL holds the number), so the two columns are
    read by what they contain, not by their names.
    """
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM D_DRAINAGE")
        pairs = cur.fetchall()
        cur.execute("SELECT HWSD2_SMU_ID, DRAINAGE, SHARE FROM HWSD2_SMU")
        smu_rows = cur.fetchall()
    finally:
        con.close()

    def number(x: Any) -> float | None:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    symbol_to_code: dict[Any, float] = {}
    for a, b, *_ in pairs:
        if number(b) is not None and number(a) is None:
            symbol_to_code[a] = float(b)
        elif number(a) is not None and number(b) is None:
            symbol_to_code[b] = float(a)
    best: dict[int, float] = {}
    out: dict[int, float] = {}
    for smu_id, symbol, share in smu_rows:
        code = symbol_to_code.get(symbol)
        if code is None or smu_id is None:
            continue
        s = share or 0
        if int(smu_id) not in best or s > best[int(smu_id)]:
            best[int(smu_id)] = s
            out[int(smu_id)] = code
    return out


def drainage_from_smu(smu: np.ndarray, lookup: dict[int, float]) -> np.ndarray:
    """Replace each soil-unit id by its drainage code. Ids with no entry stay unknown (NaN)."""
    out = np.full(smu.shape, np.nan)
    valid = np.isfinite(smu) & (smu >= 0)
    if not valid.any() or not lookup:
        return out
    ids = smu[valid].astype(np.int64)
    table = np.full(int(max(ids.max(), max(lookup))) + 1, np.nan)
    for k, v in lookup.items():
        table[k] = v
    out[valid] = table[ids]
    return out


# ------------------------------------------------------------------ roads
def utm_epsg(lon: float, lat: float) -> int:
    """The WGS 84 / UTM zone covering a point (north 326xx, south 327xx)."""
    zone = math.floor((lon + 180.0) / 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def road_distance_km(shapefile: Path | str, grid: Grid, pad_deg: float = 0.5,
                     fclasses: Sequence[str] | None = None) -> np.ndarray:
    """Distance from each cell centre to the nearest road line, in km, measured in a local metric projection.

    Roads up to `pad_deg` outside the grid are used, so cells at the edge see the roads just beyond it.
    `fclasses` keeps only some road classes (for example ["primary", "secondary"]); None keeps all.
    """
    import geopandas as gpd
    import shapely
    from shapely.strtree import STRtree

    minx, miny, maxx, maxy = grid.bbox
    roads = gpd.read_file(str(shapefile), bbox=(minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg))
    if fclasses is not None and "fclass" in roads.columns:
        roads = roads[roads["fclass"].isin(list(fclasses))]
    if roads.empty:
        raise ValueError(f"no road lines near this area in {shapefile}: check the file covers it")
    epsg = utm_epsg((minx + maxx) / 2, (miny + maxy) / 2)
    roads_m = roads.to_crs(epsg=epsg)
    lon, lat = grid.cell_centres()
    xs, ys = np.meshgrid(lon, lat)
    pts = gpd.GeoSeries(gpd.points_from_xy(xs.ravel(), ys.ravel()), crs="EPSG:4326").to_crs(epsg=epsg)
    tree = STRtree(roads_m.geometry.values)
    _, dist = tree.query_nearest(shapely.points(pts.x.values, pts.y.values), return_distance=True, all_matches=False)
    return (np.asarray(dist, dtype=float) / 1000.0).reshape(grid.shape)
