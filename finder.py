#!/usr/bin/env python3
"""
Sentinel-2 RGB Image Finder
Downloads all Sentinel-2 L2A TCI (true-color RGB) images for a field shapefile,
clips each image to the field boundary, and saves as GeoTIFF.

Usage:
    python finder.py field.zip
    python finder.py field.zip --cloud-cover 5 --output my_images --workers 4
"""

import os
import sys
import time
import threading
import zipfile
import tempfile
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import shapefile as pyshp
import pyproj
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform
from dotenv import load_dotenv

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    HAS_BOTO = True
except ImportError:
    HAS_BOTO = False

load_dotenv()

TOKEN_URL    = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_URL    = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"
S3_ENDPOINT  = "https://eodata.dataspace.copernicus.eu"
S3_BUCKET    = "eodata"

_print_lock = threading.Lock()

def log(msg="", end="\n"):
    with _print_lock:
        print(msg, end=end, flush=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TokenManager:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self._token = None
        self._fetched_at = 0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if time.time() - self._fetched_at > 540:
                self._refresh()
        return self._token

    def _refresh(self):
        resp = requests.post(TOKEN_URL, data={
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }, timeout=30)
        if resp.status_code != 200:
            log(f"\nAuth failed ({resp.status_code}): {resp.text}")
            sys.exit(1)
        self._token = resp.json()["access_token"]
        self._fetched_at = time.time()

    def headers(self):
        return {"Authorization": f"Bearer {self.get()}"}


# ---------------------------------------------------------------------------
# Shapefile loading
# ---------------------------------------------------------------------------

def _to_wgs84(geom, src_crs):
    """Reproject a shapely geometry to WGS84 (EPSG:4326) if needed."""
    wgs84 = pyproj.CRS("EPSG:4326")
    if src_crs and not src_crs.equals(wgs84):
        transformer = pyproj.Transformer.from_crs(src_crs, wgs84, always_xy=True)
        geom = transform(transformer.transform, geom)
    return geom


def _load_geometry_shapefile(shp_path):
    """Read a loose .shp (assumes .shx/.dbf/.prj siblings)."""
    shp_path = Path(shp_path)
    prj_path = shp_path.with_suffix(".prj")
    src_crs  = pyproj.CRS.from_wkt(prj_path.read_text()) if prj_path.exists() else None

    with pyshp.Reader(str(shp_path)) as sf:
        geoms = [shape(s.__geo_interface__) for s in sf.shapes() if s.shapeType != 0]
    if not geoms:
        raise ValueError("Shapefile contains no features.")
    return _to_wgs84(unary_union(geoms), src_crs)


def _load_geometry_zipped_shapefile(zip_path):
    """Extract a zipped shapefile bundle and read the first .shp inside."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise ValueError("No .shp file found inside the zip.")
        return _load_geometry_shapefile(shp_files[0])


def _load_geometry_geopackage(gpkg_path):
    """Read the first polygon layer of a GeoPackage (or first layer if none are polygons)."""
    try:
        import pyogrio
    except ImportError as e:
        raise RuntimeError(
            "GeoPackage support needs the 'pyogrio' package. Install with: pip install pyogrio"
        ) from e

    gpkg_path   = str(gpkg_path)
    layers_info = pyogrio.list_layers(gpkg_path)   # ndarray of [name, geom_type] rows
    if len(layers_info) == 0:
        raise ValueError("GeoPackage contains no layers.")

    # Prefer first polygon layer; otherwise the first layer
    chosen = str(layers_info[0][0])
    for row in layers_info:
        name, gtype = str(row[0]), str(row[1] or "").lower()
        if "polygon" in gtype:
            chosen = name
            break
    if len(layers_info) > 1:
        log(f"  GeoPackage has {len(layers_info)} layers; using '{chosen}'")

    info    = pyogrio.read_info(gpkg_path, layer=chosen)
    crs_str = info.get("crs")
    src_crs = pyproj.CRS.from_user_input(crs_str) if crs_str else None

    # pyogrio.raw.read returns (meta, fids, geometry_arr, field_data_tuple)
    # geometry_arr contains shapely geometries when shapely is installed, else WKB bytes
    _, _, geom_arr, _ = pyogrio.raw.read(gpkg_path, layer=chosen, columns=[])
    if geom_arr is None or len(geom_arr) == 0:
        raise ValueError(f"GeoPackage layer '{chosen}' contains no features.")

    if isinstance(geom_arr[0], (bytes, bytearray)):
        from shapely import wkb as shp_wkb
        geoms = [shp_wkb.loads(g) for g in geom_arr if g]
    else:
        geoms = [g for g in geom_arr if g is not None]

    if not geoms:
        raise ValueError(f"GeoPackage layer '{chosen}' has no valid geometries.")
    return _to_wgs84(unary_union(geoms), src_crs)


def load_geometry(input_path):
    """Returns (full_geom_wgs84, convex_hull_wgs84).
    Accepts a zipped shapefile (.zip), a loose shapefile (.shp), or a GeoPackage (.gpkg)."""
    p = Path(input_path)
    suffix = p.suffix.lower()

    if suffix == ".zip":
        geom = _load_geometry_zipped_shapefile(p)
    elif suffix == ".shp":
        geom = _load_geometry_shapefile(p)
    elif suffix == ".gpkg":
        geom = _load_geometry_geopackage(p)
    else:
        raise ValueError(
            f"Unsupported input format '{suffix}'. Use .zip (zipped shapefile), "
            f".shp (loose shapefile), or .gpkg (GeoPackage)."
        )

    return geom, geom.convex_hull


# ---------------------------------------------------------------------------
# Clip + image outputs
# ---------------------------------------------------------------------------

def clip_to_boundary(src_path, clip_geom_wgs84, out_path, fill_nodata=None):
    """Clip raster to boundary and save as GeoTIFF. Deletes src_path when done.
    fill_nodata: if provided, outside-boundary pixels are filled with this value
    and written as the nodata tag (useful for DEMs where the source has no nodata)."""
    with rasterio.open(src_path) as src:
        geom_proj  = transform_geom("EPSG:4326", src.crs.to_string(), mapping(clip_geom_wgs84))
        nodata_fill = fill_nodata if fill_nodata is not None else src.nodata
        out_image, out_transform = rio_mask(src, [geom_proj], crop=True,
                                            nodata=nodata_fill, filled=True)
        meta = src.meta.copy()
        meta.update({
            "driver":    "GTiff",
            "height":    out_image.shape[1],
            "width":     out_image.shape[2],
            "transform": out_transform,
        })
        if fill_nodata is not None:
            meta["nodata"] = fill_nodata

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_image)

    src_path.unlink()


def compute_ndwi(b03_path, b08_path, out_path):
    """Compute NDWI = (Green - NIR) / (Green + NIR), save colored RGB GeoTIFF."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    with rasterio.open(b03_path) as src:
        b03  = src.read(1).astype("float32")
        meta = src.meta.copy()
    with rasterio.open(b08_path) as src:
        b08 = src.read(1).astype("float32")

    denom       = b03 + b08
    ndwi        = np.where(denom == 0, np.nan, (b03 - b08) / denom)
    valid       = ndwi[~np.isnan(ndwi)]
    nodata_mask = np.isnan(ndwi)

    vmin = float(np.percentile(valid, 2))  if valid.size else -0.5
    vmax = float(np.percentile(valid, 98)) if valid.size else  0.5
    half = max(abs(vmin), abs(vmax))
    vmin, vmax = -half, half

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap("RdYlBu")(norm(np.nan_to_num(ndwi)))
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[nodata_mask] = 0

    meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    with rasterio.open(out_path, "w", **meta) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


def compute_ndvi(b08_path, b04_path, out_path):
    """Compute NDVI = (NIR - Red) / (NIR + Red), save coloured RGB GeoTIFF.
    Green = healthy vegetation, Yellow = moderate, Red = bare soil / stressed / water."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    with rasterio.open(b08_path) as src:
        b08  = src.read(1).astype("float32")
        meta = src.meta.copy()
    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype("float32")

    denom       = b08 + b04
    ndvi        = np.where(denom == 0, np.nan, (b08 - b04) / denom)
    valid       = ndvi[~np.isnan(ndvi)]
    nodata_mask = np.isnan(ndvi)

    vmin = float(np.percentile(valid, 2))  if valid.size else -0.2
    vmax = float(np.percentile(valid, 98)) if valid.size else  0.9

    # Save raw float32 NDVI alongside the colorized RGB (for historical aggregation)
    raw_meta = meta.copy()
    raw_meta.update({"count": 1, "dtype": "float32", "nodata": float("nan")})
    raw_out_path = out_path.with_name(out_path.stem + "_raw.tif")
    with rasterio.open(raw_out_path, "w", **raw_meta) as dst:
        dst.write(ndvi.astype("float32"), 1)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap("RdYlGn")(norm(np.nan_to_num(ndvi)))
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[nodata_mask] = 0

    meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    with rasterio.open(out_path, "w", **meta) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


def _date_from_ndvi_filename(path):
    """Extract date from filename like T11UQT_20180713T183019_NDVI_raw.tif → date(2018, 7, 13)."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return datetime.strptime(parts[1][:8], "%Y%m%d").date()
    except ValueError:
        return None


def _in_md_window(d, smonth, sday, emonth, eday):
    """Year-agnostic month/day window check; handles year-boundary wrap."""
    md, start, end = (d.month, d.day), (smonth, sday), (emonth, eday)
    if start <= end:
        return start <= md <= end
    return md >= start or md <= end


def compute_ndvi_historical(ndvi_dir, start_month, start_day, end_month, end_day,
                            cancel_event=None, progress_callback=None):
    """Aggregate all *_NDVI_raw.tif files in ndvi_dir whose date falls within
    [start_md, end_md] (year-agnostic). Writes mean + std composites + previews.
    Returns dict with counts and outputs, or {"needs_rerun": True} if no raw files."""
    raws = sorted(ndvi_dir.glob("*_NDVI_raw.tif"))
    if not raws:
        old = list(ndvi_dir.glob("*_NDVI.tif"))
        return {"needs_rerun": True, "old_colorized": len(old)}

    matched = []
    for p in raws:
        d = _date_from_ndvi_filename(p)
        if d is not None and _in_md_window(d, start_month, start_day, end_month, end_day):
            matched.append(p)
    log(f"  [hist] scanned {len(raws)} raw NDVI files, {len(matched)} match window "
        f"{start_month:02d}/{start_day:02d}–{end_month:02d}/{end_day:02d}")

    if len(matched) < 2:
        log("  [hist] need at least 2 matching files for a meaningful average")
        return {"matched": len(matched), "used": 0}

    with rasterio.open(matched[0]) as ref:
        ref_shape, ref_tr, ref_crs = ref.shape, ref.transform, ref.crs
        meta_template = ref.meta.copy()

    H, W  = ref_shape
    stack = np.full((len(matched), H, W), np.nan, dtype="float32")
    used  = 0
    for i, p in enumerate(matched):
        if cancel_event and cancel_event.is_set():
            return {"cancelled": True}
        if progress_callback:
            progress_callback(i, len(matched))
        with rasterio.open(p) as src:
            if src.shape != ref_shape or src.transform != ref_tr or src.crs != ref_crs:
                log(f"  [hist] skip {p.name} (shape/transform mismatch)")
                continue
            stack[used] = src.read(1).astype("float32")
            used += 1
    stack = stack[:used]

    if progress_callback:
        progress_callback(len(matched), len(matched))

    if used < 2:
        log("  [hist] after alignment filter, fewer than 2 valid files")
        return {"matched": len(matched), "used": used}

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN pixels
        mean = np.nanmean(stack, axis=0)
        std  = np.nanstd(stack, axis=0)
    nodata_mask = np.isnan(mean)

    mean_raw     = ndvi_dir / "NDVI_historical_mean_raw.tif"
    mean_color   = ndvi_dir / "NDVI_historical_mean.tif"
    mean_preview = ndvi_dir / "NDVI_historical_mean_preview.png"
    std_raw      = ndvi_dir / "NDVI_historical_std_raw.tif"
    std_color    = ndvi_dir / "NDVI_historical_std.tif"
    std_preview  = ndvi_dir / "NDVI_historical_std_preview.png"

    raw_meta = meta_template.copy()
    raw_meta.update({"count": 1, "dtype": "float32", "nodata": float("nan")})
    for arr, p in [(mean, mean_raw), (std, std_raw)]:
        with rasterio.open(p, "w", **raw_meta) as dst:
            dst.write(arr.astype("float32"), 1)

    valid_m = mean[~nodata_mask]
    lo_m = float(np.percentile(valid_m, 2))  if valid_m.size else -0.2
    hi_m = float(np.percentile(valid_m, 98)) if valid_m.size else  0.9
    rgb_m = _colorize(mean, "RdYlGn", lo_m, hi_m, nodata_mask)
    _save_rgb_tif(rgb_m, matched[0], mean_color)
    _make_preview_png(mean, nodata_mask, "RdYlGn", lo_m, hi_m,
                      f"Historical mean NDVI  ({start_month:02d}/{start_day:02d} – "
                      f"{end_month:02d}/{end_day:02d}, n={used})",
                      "NDVI", mean_preview)

    valid_s = std[~nodata_mask]
    hi_s = max(float(np.percentile(valid_s, 98)) if valid_s.size else 0.1, 0.01)
    rgb_s = _colorize(std, "viridis", 0.0, hi_s, nodata_mask)
    _save_rgb_tif(rgb_s, matched[0], std_color)
    _make_preview_png(std, nodata_mask, "viridis", 0.0, hi_s,
                      f"NDVI year-to-year variability  (std, n={used})",
                      "std dev", std_preview)

    log(f"  [hist] mean NDVI: {lo_m:.3f} – {hi_m:.3f}")
    log(f"  [hist] wrote 6 files to {ndvi_dir}")
    return {"matched": len(matched), "used": used,
            "outputs": [mean_raw, mean_color, mean_preview,
                        std_raw,  std_color,  std_preview]}


def compute_false_color(b08_path, b04_path, b03_path, out_path):
    """False color composite: R=B08 (NIR), G=B04 (Red), B=B03 (Green).
    Healthy vegetation appears bright red; bare soil, water, and objects stand out."""
    with rasterio.open(b08_path) as src:
        meta = src.meta.copy()
        b08 = src.read(1).astype("float32")
    with rasterio.open(b04_path) as src:
        b04 = src.read(1).astype("float32")
    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype("float32")

    def _stretch(band):
        valid = band[band > 0]
        if valid.size == 0:
            return np.zeros_like(band, dtype=np.uint8)
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        return (np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

    rgb = np.stack([_stretch(b08), _stretch(b04), _stretch(b03)], axis=2)

    meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    with rasterio.open(out_path, "w", **meta) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


# ---------------------------------------------------------------------------
# DEM — cascade: NRCan HRDEM (1m) → OpenTopography LiDAR → Copernicus GLO-30 (30m)
# ---------------------------------------------------------------------------

NRCAN_WCS    = "https://datacube.services.geo.ca/ows/elevation"
OT_CATALOG   = "https://portal.opentopography.org/API/otCatalog"
OT_GLOBALDEM = "https://portal.opentopography.org/API/globaldem"
GLO30_BASE   = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


def _hillshade(dem, cell_size, azimuth=315, altitude=45):
    dy, dx  = np.gradient(dem, cell_size)
    slope   = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect  = np.arctan2(-dx, dy)
    az_rad  = np.deg2rad(360 - azimuth + 90)
    alt_rad = np.deg2rad(altitude)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1)


def _save_rgb_tif(rgb, ref_path, out_path):
    with rasterio.open(ref_path) as src:
        meta = src.meta.copy()
    meta.update({"count": 3, "dtype": "uint8", "nodata": None})
    with rasterio.open(out_path, "w", **meta) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


def _colorize(data, cmap_name, vmin=None, vmax=None, nodata_mask=None):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    vmin = vmin if vmin is not None else np.nanmin(data)
    vmax = vmax if vmax is not None else np.nanmax(data)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap(cmap_name)(norm(np.nan_to_num(data)))
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    if nodata_mask is not None:
        rgb[nodata_mask] = 0
    return rgb


def _try_opentopo_lidar(minx, miny, maxx, maxy, tmp_path, api_key):
    try:
        cat_resp = requests.get(OT_CATALOG, params={
            "productFormat": "PointCloud",
            "minx": minx, "miny": miny,
            "maxx": maxx, "maxy": maxy,
            "detail": "true",
            "outputFormat": "json",
            "API_Key": api_key,
        }, timeout=30)
        if cat_resp.status_code != 200:
            return None
        datasets = cat_resp.json().get("Datasets", [])
        if not datasets:
            return None

        def _res(ds):
            try:
                return float(ds.get("pointDensity") or ds.get("resolution") or 999)
            except Exception:
                return 999

        best       = min(datasets, key=_res)
        short_name = best.get("shortName") or best.get("datasetName")
        res_label  = f"{_res(best):.1f}pt/m² LiDAR"
        log(f"  [dem] OpenTopography dataset found: {short_name} ({res_label})")

        dl_resp = requests.get("https://portal.opentopography.org/API/lidar", params={
            "datasetName": short_name,
            "south": miny, "north": maxy,
            "west":  minx, "east":  maxx,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }, timeout=300, stream=True)
        if dl_resp.status_code != 200:
            return None
        ct = dl_resp.headers.get("Content-Type", "")
        if "tiff" not in ct and "octet-stream" not in ct:
            return None
        with open(tmp_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=262144):
                f.write(chunk)
        with rasterio.open(tmp_path) as src:
            if src.count > 0:
                return res_label
        tmp_path.unlink()
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
    return None


def _try_opentopo_global(minx, miny, maxx, maxy, tmp_path, api_key):
    try:
        resp = requests.get(OT_GLOBALDEM, params={
            "demtype": "NASADEM",
            "south": miny, "north": maxy,
            "west":  minx, "east":  maxx,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }, timeout=120, stream=True)
        if resp.status_code != 200:
            return None
        ct = resp.headers.get("Content-Type", "")
        if "tiff" not in ct and "octet-stream" not in ct:
            return None
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=262144):
                f.write(chunk)
        with rasterio.open(tmp_path) as src:
            if src.read(1).size > 0:
                return "30m (NASADEM)"
        tmp_path.unlink()
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
    return None


def _try_nrcan(minx, miny, maxx, maxy, tmp_path):
    for cov_id, label in [("dtm_1m", "1m"), ("dtm_2m", "2m")]:
        try:
            params = [
                ("SERVICE",       "WCS"),
                ("VERSION",       "2.0.1"),
                ("REQUEST",       "GetCoverage"),
                ("COVERAGEID",    cov_id),
                ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
                ("OUTPUTCRS",     "http://www.opengis.net/def/crs/EPSG/0/4326"),
                ("SUBSET",        f"Long({minx},{maxx})"),
                ("SUBSET",        f"Lat({miny},{maxy})"),
                ("FORMAT",        "image/tiff"),
            ]
            resp = requests.get(NRCAN_WCS, params=params, timeout=180, stream=True)
            ct = resp.headers.get("Content-Type", "")
            if resp.status_code != 200:
                log(f"  [dem] NRCan {cov_id}: HTTP {resp.status_code}")
                continue
            if "tiff" not in ct and "octet-stream" not in ct:
                log(f"  [dem] NRCan {cov_id}: unexpected content-type '{ct}' (no coverage here)")
                continue
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=262144):
                    f.write(chunk)
            with rasterio.open(tmp_path) as src:
                data = src.read(1)
                nd   = src.nodata
                if data.size > 0 and not np.all(data == nd):
                    return label
            log(f"  [dem] NRCan {cov_id}: returned all nodata (no coverage here)")
            tmp_path.unlink()
        except Exception as e:
            log(f"  [dem] NRCan {cov_id}: error — {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    return None


def _try_terrain_tiles(minx, miny, maxx, maxy, tmp_path):
    """AWS Terrain Tiles (Terrarium format) — free, global, ~3-6m at mid-latitudes. No key needed."""
    import math
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    zoom = 15  # ~3m/pixel at 51°N

    def _deg2tile(lat, lon, z):
        n = 2 ** z
        x = int((lon + 180) / 360 * n)
        lat_r = math.radians(lat)
        y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
        return x, y

    def _tile2deg(x, y, z):
        n = 2 ** z
        lon = x / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        return lat, lon

    try:
        tx0, ty0 = _deg2tile(maxy, minx, zoom)
        tx1, ty1 = _deg2tile(miny, maxx, zoom)
        tx0, tx1 = min(tx0, tx1), max(tx0, tx1)
        ty0, ty1 = min(ty0, ty1), max(ty0, ty1)

        rows = []
        for ty in range(ty0, ty1 + 1):
            cols = []
            for tx in range(tx0, tx1 + 1):
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png"
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    return None
                with MemoryFile(resp.content) as mf:
                    with mf.open() as ds:
                        rgb = ds.read()
                r = rgb[0].astype(np.float32)
                g = rgb[1].astype(np.float32)
                b = rgb[2].astype(np.float32)
                cols.append(r * 256 + g + b / 256 - 32768)
            rows.append(np.concatenate(cols, axis=1))

        elev = np.concatenate(rows, axis=0)

        top_lat,  left_lon  = _tile2deg(tx0,     ty0,     zoom)
        bot_lat,  right_lon = _tile2deg(tx1 + 1, ty1 + 1, zoom)
        h, w = elev.shape
        tr   = from_bounds(left_lon, bot_lat, right_lon, top_lat, w, h)
        res_m = abs(right_lon - left_lon) / w * 111320 * math.cos(math.radians((top_lat + bot_lat) / 2))

        # Mark sea-level void pixels (R=G=B=0 → -32768) as nodata
        elev[elev < -500] = -9999.0

        with rasterio.open(tmp_path, "w", driver="GTiff",
                           height=h, width=w, count=1, dtype="float32",
                           crs="EPSG:4326", transform=tr, nodata=-9999.0) as dst:
            dst.write(elev, 1)

        return f"~{res_m:.0f}m (AWS terrain tiles z{zoom})"

    except Exception as e:
        log(f"  [dem] terrain tiles error: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
    return None


def _download_glo30(minx, miny, maxx, maxy, tmp_dir):
    lat_tiles  = range(int(np.floor(miny)), int(np.ceil(maxy)))
    lon_tiles  = range(int(np.floor(minx)), int(np.ceil(maxx)))
    tile_paths = []
    for lat in lat_tiles:
        for lon in lon_tiles:
            ns   = "N" if lat >= 0 else "S"
            ew   = "E" if lon >= 0 else "W"
            name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            url  = f"{GLO30_BASE}/{name}/{name}.tif"
            p    = tmp_dir / f"_glo30_{lat}_{lon}.tif"
            log(f"  [dem] GLO-30 tile ({lat},{lon})...")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(p, "wb") as f:
                for chunk in resp.iter_content(chunk_size=262144):
                    f.write(chunk)
            tile_paths.append(p)

    if len(tile_paths) == 1:
        return tile_paths[0]

    datasets = [rasterio.open(p) for p in tile_paths]
    merged, tr = rio_merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({"height": merged.shape[1], "width": merged.shape[2], "transform": tr})
    for ds in datasets:
        ds.close()
    merged_path = tmp_dir / "_glo30_merged.tif"
    with rasterio.open(merged_path, "w", **meta) as dst:
        dst.write(merged)
    for p in tile_paths:
        p.unlink()
    return merged_path


def _upsample_dem(raw_path, target_m=5.0):
    """Bilinearly upsample raw DEM to target_m resolution if coarser. Overwrites raw_path."""
    import math
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    with rasterio.open(raw_path) as src:
        cell_m = abs(src.transform[0])
        if cell_m <= target_m * 1.1:
            return
        scale    = cell_m / target_m
        new_w    = math.ceil(src.width  * scale)
        new_h    = math.ceil(src.height * scale)
        new_tr   = from_bounds(*src.bounds, new_w, new_h)
        src_data = src.read(1)
        src_meta = src.meta.copy()
        src_tr   = src.transform
        src_crs  = src.crs
        src_nd   = src.nodata

    dst_data = np.empty((new_h, new_w), dtype="float32")
    reproject(
        source=src_data, destination=dst_data,
        src_transform=src_tr, src_crs=src_crs,
        dst_transform=new_tr, dst_crs=src_crs,
        resampling=Resampling.bilinear,
        src_nodata=src_nd, dst_nodata=src_nd,
    )
    src_meta.update({"width": new_w, "height": new_h, "transform": new_tr, "dtype": "float32"})
    tmp = raw_path.parent / "_upsample_tmp.tif"
    with rasterio.open(tmp, "w", **src_meta) as dst:
        dst.write(dst_data, 1)
    tmp.replace(raw_path)
    log(f"  [dem] upsampled {cell_m:.0f}m → {target_m:.0f}m (bilinear)")


def _smooth_dem(dem, nodata_mask, sigma_m, cell_m):
    """Gaussian smooth DEM, ignoring nodata. sigma_m in metres."""
    try:
        from scipy.ndimage import gaussian_filter
        sigma_px = max(1.0, sigma_m / cell_m)
        filled   = np.where(nodata_mask, np.nanmean(dem[~nodata_mask]), dem)
        smoothed = gaussian_filter(filled.astype("float64"), sigma=sigma_px)
        return np.where(nodata_mask, np.nan, smoothed.astype("float32"))
    except ImportError:
        # scipy not available — fall back to a simple 5×5 box filter via cumsum
        k = 5
        pad  = np.pad(np.nan_to_num(dem), k // 2, mode="edge")
        cs   = np.cumsum(np.cumsum(pad, axis=0), axis=1)
        box  = (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / k ** 2
        return np.where(nodata_mask, np.nan, box.astype("float32"))


def _make_preview_png(dem_data, nodata_mask, cmap_name, lo, hi,
                      title_str, unit_label, out_path):
    """Save a PNG of dem_data with a labelled colorbar on the right side."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive, thread-safe, no window
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if hi <= lo:
        hi = lo + 0.001

    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85))   # nodata → light grey

    display = np.where(nodata_mask, np.nan, dem_data.astype("float64"))
    norm    = mcolors.Normalize(vmin=lo, vmax=hi)

    fig, (ax_map, ax_cb) = plt.subplots(
        1, 2,
        figsize=(8, 6),
        gridspec_kw={"width_ratios": [20, 1]},
        facecolor="white",
    )
    im = ax_map.imshow(display, cmap=cmap, norm=norm,
                       interpolation="nearest", origin="upper")
    ax_map.set_title(title_str, fontsize=11, pad=8)
    ax_map.axis("off")

    cb = fig.colorbar(im, cax=ax_cb, orientation="vertical")
    cb.set_label(unit_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ticks = np.linspace(lo, hi, num=6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{v:.1f}" for v in ticks])

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_visuals(raw_path, elevation_path, slope_path):
    import math
    import matplotlib.cm as cm

    with rasterio.open(raw_path) as src:
        dem    = src.read(1).astype("float32")
        cell_m = abs(src.transform[0])
        nd     = src.nodata
        bounds = src.bounds

    # Mask nodata + decode artefacts
    nodata_mask = (np.isnan(dem) if nd is None else (dem == nd))
    nodata_mask = nodata_mask | (dem < -500) | (dem > 9000)

    valid = dem[~nodata_mask]
    if valid.size == 0:
        log("  [dem] WARNING: no valid elevation pixels — skipping visuals")
        return

    log(f"  [dem] elevation {valid.min():.1f}m – {valid.max():.1f}m  "
        f"range {valid.max()-valid.min():.1f}m  std {valid.std():.2f}m")

    # Resolve actual cell size in metres for slope gradient.
    # GLO-30 is EPSG:4326 so transform[0] is in degrees — convert properly.
    if cell_m < 0.01:
        lat_c        = (bounds.top + bounds.bottom) / 2.0
        cell_m_slope = cell_m * (math.pi / 180.0) * 6_371_000 * math.cos(math.radians(lat_c))
        log(f"  [dem] cell {cell_m:.6f}° ≈ {cell_m_slope:.1f}m at {lat_c:.2f}°N")
    else:
        cell_m_slope = cell_m

    def _to_rgb(data, cmap_name, lo, hi):
        if hi <= lo: hi = lo + 0.001
        norm = np.clip((data - lo) / (hi - lo), 0, 1)
        rgba = cm.get_cmap(cmap_name)(norm)
        rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
        rgb[nodata_mask] = 0
        return rgb

    # ------------------------------------------------------------------
    # 1. Elevation — Blue = lowest, Red = highest
    # ------------------------------------------------------------------
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    log(f"  [dem] colour scale: {lo:.1f}m (blue) → {hi:.1f}m (red)")
    rgb = _to_rgb(dem, "RdYlBu_r", lo, hi)
    _save_rgb_tif(rgb, raw_path, elevation_path)
    log(f"  [dem] DEM_elevation.tif  ({elevation_path.stat().st_size/1_048_576:.1f} MB)")

    elev_preview = elevation_path.parent / "DEM_elevation_preview.png"
    _make_preview_png(dem, nodata_mask, "RdYlBu_r", lo, hi,
                      f"Elevation: {lo:.1f}m – {hi:.1f}m", "metres", elev_preview)
    log(f"  [dem] DEM_elevation_preview.png  ({elev_preview.stat().st_size/1024:.0f} KB)")

    # ------------------------------------------------------------------
    # 2. Slope — auto-scaled to 98th percentile of actual slope values
    # ------------------------------------------------------------------
    dem_f  = np.where(nodata_mask, float(valid.mean()), dem)
    dy, dx = np.gradient(dem_f, cell_m_slope)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[nodata_mask] = 0
    vm     = ~nodata_mask
    max_s  = max(float(np.percentile(slope[vm], 98)), 0.1)
    rgb    = _to_rgb(slope, "YlOrRd", 0, max_s)
    _save_rgb_tif(rgb, raw_path, slope_path)
    log(f"  [dem] DEM_slope.tif      scale 0–{max_s:.3f}°  "
        f"({slope_path.stat().st_size/1_048_576:.1f} MB)")

    slope_preview = slope_path.parent / "DEM_slope_preview.png"
    _make_preview_png(slope, nodata_mask, "YlOrRd", 0.0, max_s,
                      f"Slope: 0° – {max_s:.2f}°", "degrees", slope_preview)
    log(f"  [dem] DEM_slope_preview.png      ({slope_preview.stat().st_size/1024:.0f} KB)")


def download_dem(clip_geom, output_dir):
    raw_path       = output_dir / "DEM_elevation_raw.tif"
    elevation_path = output_dir / "DEM_elevation.tif"
    slope_path     = output_dir / "DEM_slope.tif"

    if elevation_path.exists() and slope_path.exists():
        log("  [dem] already exists, skipping")
        return

    minx, miny, maxx, maxy = clip_geom.bounds
    ot_key = os.getenv("OPENTOPO_KEY")
    tmp    = output_dir / "_tmp_dem.tif"
    source = None

    log("  [dem] trying NRCan HRDEM (1m LiDAR)...")
    res = _try_nrcan(minx, miny, maxx, maxy, tmp)
    if res:
        log(f"  [dem] NRCan HRDEM {res} ✓")
        source = tmp

    if source is None and ot_key:
        log("  [dem] trying OpenTopography local LiDAR catalog...")
        res = _try_opentopo_lidar(minx, miny, maxx, maxy, tmp, ot_key)
        if res:
            log(f"  [dem] OpenTopography LiDAR {res} ✓")
            source = tmp

    if source is None:
        log("  [dem] falling back to Copernicus GLO-30 (30m)...")
        source = _download_glo30(minx, miny, maxx, maxy, output_dir)

    log("  [dem] clipping to boundary...")
    clip_to_boundary(source, clip_geom, raw_path, fill_nodata=-9999.0)

    log("  [dem] generating elevation + slope images...")
    _make_visuals(raw_path, elevation_path, slope_path)
    log("  [dem] Open *_preview.png for a labelled colour scale.")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _get_with_retry(url, params, timeout=120, retries=5):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            log(f"\n  Search HTTP {resp.status_code}, retrying ({attempt+1}/{retries})...")
        except requests.exceptions.Timeout:
            log(f"\n  Search timed out, retrying ({attempt+1}/{retries})...")
        except requests.exceptions.ConnectionError:
            log(f"\n  Connection error, retrying ({attempt+1}/{retries})...")
        time.sleep(2 ** attempt)
    log("\nSearch failed after all retries.")
    sys.exit(1)


def search_products(geom_wkt, cloud_max=10.0):
    filter_str = (
        "Collection/Name eq 'SENTINEL-2' "
        "and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value lt {cloud_max}) "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{geom_wkt}') "
        f"and Online eq true"
    )

    products  = []
    skip      = 0
    page_size = 100

    while True:
        resp = _get_with_retry(
            f"{ODATA_URL}/Products",
            params={
                "$filter":  filter_str,
                "$orderby": "ContentDate/Start asc",
                "$top":     page_size,
                "$skip":    skip,
                "$expand":  "Attributes",
            },
        )
        batch = resp.json().get("value", [])
        products.extend(batch)
        log(f"  {len(products)} online products found so far...", end="\r")
        if len(batch) < page_size:
            break
        skip += page_size

    log()
    return products


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _base_name(product_name):
    return product_name.removesuffix(".SAFE")


def _out_stem(product_name):
    parts = _base_name(product_name).split("_")
    return f"{parts[5]}_{parts[2]}"


def tci_filename_from_product(product_name):
    parts = _base_name(product_name).split("_")
    return f"{parts[5]}_{parts[2]}_TCI_10m.jp2"


def band_filename_from_product(product_name, band):
    parts = _base_name(product_name).split("_")
    return f"{parts[5]}_{parts[2]}_{band}_10m.jp2"


def sensing_date_parts(product_name):
    dt = _base_name(product_name).split("_")[2]
    return dt[:4], dt[4:6], dt[6:8]


# ---------------------------------------------------------------------------
# S3 (optional)
# ---------------------------------------------------------------------------

def make_s3_client():
    key    = os.getenv("CDSE_S3_KEY")
    secret = os.getenv("CDSE_S3_SECRET")
    if not (key and secret):
        return None
    if not HAS_BOTO:
        log("Warning: CDSE_S3_KEY/SECRET set but boto3 not installed. Falling back to HTTPS.")
        return None
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def find_granule_s3(s3_client, product_name):
    year, month, day = sensing_date_parts(product_name)
    base   = _base_name(product_name)
    prefix = f"Sentinel-2/MSI/L2A/{year}/{month}/{day}/{base}.SAFE/GRANULE/"
    resp   = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/")
    prefixes = resp.get("CommonPrefixes", [])
    if not prefixes:
        return None
    return prefixes[0]["Prefix"].rstrip("/").split("/")[-1]


def download_s3(s3_client, product_name, granule_name, filename, out_path):
    year, month, day = sensing_date_parts(product_name)
    base = _base_name(product_name)
    key  = (
        f"Sentinel-2/MSI/L2A/{year}/{month}/{day}/{base}.SAFE"
        f"/GRANULE/{granule_name}/IMG_DATA/R10m/{filename}"
    )
    cfg = TransferConfig(multipart_threshold=20 * 1024 * 1024, max_concurrency=4)
    s3_client.download_file(S3_BUCKET, key, str(out_path), Config=cfg)


# ---------------------------------------------------------------------------
# HTTPS
# ---------------------------------------------------------------------------

def _nodes_get(url, token_mgr):
    resp = requests.get(url, headers=token_mgr.headers(), timeout=60)
    if resp.status_code == 401:
        token_mgr.get()
        resp = requests.get(url, headers=token_mgr.headers(), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", data.get("value", []))


def find_granule_https(product_id, product_name, token_mgr):
    root_nodes = _nodes_get(f"{DOWNLOAD_URL}/Products({product_id})/Nodes", token_mgr)
    if not root_nodes:
        return None

    safe_folder = root_nodes[0]["Name"]

    for attempt in range(4):
        try:
            granules = _nodes_get(
                f"{DOWNLOAD_URL}/Products({product_id})/Nodes({safe_folder})/Nodes(GRANULE)/Nodes",
                token_mgr,
            )
            if granules:
                return granules[0]["Name"]
            return None
        except requests.HTTPError as e:
            log(f"  [granule lookup] HTTP {e.response.status_code}, retry {attempt+1}/4")
        except requests.exceptions.Timeout:
            log(f"  [granule lookup] timeout, retry {attempt+1}/4")
        time.sleep(2 ** attempt)
    return None


def download_https(product_id, product_name, granule_name, filename, token_mgr, out_path):
    base = _base_name(product_name)
    url  = (
        f"{DOWNLOAD_URL}/Products({product_id})"
        f"/Nodes({base}.SAFE)"
        f"/Nodes(GRANULE)"
        f"/Nodes({granule_name})"
        f"/Nodes(IMG_DATA)"
        f"/Nodes(R10m)"
        f"/Nodes({filename})/$value"
    )
    for attempt in range(6):
        with requests.get(url, headers=token_mgr.headers(), stream=True, timeout=300) as resp:
            if resp.status_code == 401:
                token_mgr.get()
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", min(15 * (2 ** attempt), 120)))
                log(f"  [429] rate limited — waiting {wait}s...  ({filename})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=262144):
                    f.write(chunk)
            return
    raise RuntimeError(f"Download failed after 6 attempts (persistent 429): {filename}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _download_clip(pid, name, granule_name, jp2_filename, token_mgr, s3_client, clip_geom, tif_path):
    jp2_path = tif_path.parent / jp2_filename
    if s3_client:
        download_s3(s3_client, name, granule_name, jp2_filename, jp2_path)
    else:
        download_https(pid, name, granule_name, jp2_filename, token_mgr, jp2_path)
    clip_to_boundary(jp2_path, clip_geom, tif_path)
    return tif_path


def process_product(product, token_mgr, s3_client, clip_geom,
                    rgb_dir, ndwi_dir, ndvi_dir, fc_dir,
                    index, total,
                    do_rgb, do_ndwi, do_ndvi, do_false_color,
                    cancel_event=None):
    if cancel_event and cancel_event.is_set():
        return "cancelled"

    pid   = product["Id"]
    name  = product["Name"]
    date  = (product.get("ContentDate") or {}).get("Start", "?")[:10]
    cloud = next(
        (round(a["Value"], 1) for a in product.get("Attributes", []) if a["Name"] == "cloudCover"),
        "?",
    )

    stem     = _out_stem(name)
    tci_tif  = rgb_dir  / f"{stem}_TCI.tif"        if do_rgb         else None
    ndwi_tif = ndwi_dir / f"{stem}_NDWI.tif"       if do_ndwi        else None
    ndvi_tif = ndvi_dir / f"{stem}_NDVI.tif"        if do_ndvi        else None
    fc_tif   = fc_dir   / f"{stem}_FalseColor.tif"  if do_false_color else None

    need_rgb  = do_rgb         and tci_tif  and not tci_tif.exists()
    need_ndwi = do_ndwi        and ndwi_tif and not ndwi_tif.exists()
    need_ndvi = do_ndvi        and ndvi_tif and not ndvi_tif.exists()
    need_fc   = do_false_color and fc_tif   and not fc_tif.exists()

    if not need_rgb and not need_ndwi and not need_ndvi and not need_fc:
        log(f"[{index:>4}/{total}]  {date}  [skip] already exists")
        return "skipped"

    log(f"[{index:>4}/{total}]  {date}  cloud={cloud}%  {name}")

    work_dir = next(d for d in [rgb_dir, ndwi_dir, ndvi_dir, fc_dir] if d is not None)
    tmp_dir  = work_dir / f"_tmp_{pid[:8]}"
    tmp_dir.mkdir(exist_ok=True)

    try:
        if s3_client:
            granule_name = find_granule_s3(s3_client, name)
            if not granule_name:
                log("  [warn] S3 granule lookup failed, trying HTTPS")
                granule_name = find_granule_https(pid, name, token_mgr)
        else:
            granule_name = find_granule_https(pid, name, token_mgr)

        if not granule_name:
            log("  [warn] granule not found, skipping")
            return "failed"

        if cancel_event and cancel_event.is_set():
            return "cancelled"

        # Determine which individual bands are needed (shared across indices)
        needed_bands = set()
        if need_ndwi: needed_bands |= {"B03", "B08"}
        if need_ndvi: needed_bands |= {"B04", "B08"}
        if need_fc:   needed_bands |= {"B03", "B04", "B08"}

        # Download TCI + all bands in parallel — each is independent
        def _fetch_band(band):
            log(f"  [band] downloading {band}...")
            tif = tmp_dir / f"{band}.tif"
            _download_clip(pid, name, granule_name, band_filename_from_product(name, band),
                           token_mgr, s3_client, clip_geom, tif)
            return band, tif

        def _fetch_tci():
            log("  [rgb] downloading TCI...")
            _download_clip(pid, name, granule_name, tci_filename_from_product(name),
                           token_mgr, s3_client, clip_geom, tci_tif)
            return "TCI", tci_tif

        band_tifs = {}
        n_workers = min(2, len(needed_bands) + (1 if need_rgb else 0))
        if n_workers > 0:
            with ThreadPoolExecutor(max_workers=n_workers) as band_pool:
                dl_futures = {}
                for band in needed_bands:
                    dl_futures[band_pool.submit(_fetch_band, band)] = band
                if need_rgb:
                    dl_futures[band_pool.submit(_fetch_tci)] = "TCI"

                for fut in as_completed(dl_futures):
                    if cancel_event and cancel_event.is_set():
                        return "cancelled"
                    key, tif = fut.result()
                    if key == "TCI":
                        log(f"  [rgb] {tci_tif.name}  ({tci_tif.stat().st_size/1_048_576:.1f} MB)")
                    else:
                        band_tifs[key] = tif

        if cancel_event and cancel_event.is_set():
            return "cancelled"

        # NDWI
        if need_ndwi:
            compute_ndwi(band_tifs["B03"], band_tifs["B08"], ndwi_tif)
            log(f"  [ndwi] {ndwi_tif.name}  ({ndwi_tif.stat().st_size/1_048_576:.1f} MB)")

        # NDVI
        if need_ndvi:
            compute_ndvi(band_tifs["B08"], band_tifs["B04"], ndvi_tif)
            log(f"  [ndvi] {ndvi_tif.name}  ({ndvi_tif.stat().st_size/1_048_576:.1f} MB)")

        # False Color
        if need_fc:
            compute_false_color(band_tifs["B08"], band_tifs["B04"], band_tifs["B03"], fc_tif)
            log(f"  [fc] {fc_tif.name}  ({fc_tif.stat().st_size/1_048_576:.1f} MB)")

        return "ok"

    except Exception as e:
        log(f"  [error] {e}")
        for p in [tci_tif, ndwi_tif, ndvi_tif, fc_tif]:
            if p and p.exists():
                p.unlink()
        return "failed"
    finally:
        if tmp_dir.exists():
            for f in tmp_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                tmp_dir.rmdir()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(shapefile_zip, output, cloud_max=10.0, workers=4,
        do_rgb=True, do_ndwi=False, do_ndvi=False, do_dem=False, do_false_color=False,
        cancel_event=None, progress_callback=None):
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    if not username or not password:
        log("Error: CDSE_USERNAME and CDSE_PASSWORD must be set in the .env file.")
        sys.exit(1)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir  = output_dir / "RGB"
    ndwi_dir = output_dir / "NDWI"
    ndvi_dir = output_dir / "NDVI"
    dem_dir  = output_dir / "DEM"
    fc_dir   = output_dir / "FalseColor"
    if do_rgb:         rgb_dir.mkdir(exist_ok=True)
    if do_ndwi:        ndwi_dir.mkdir(exist_ok=True)
    if do_ndvi:        ndvi_dir.mkdir(exist_ok=True)
    if do_dem:         dem_dir.mkdir(exist_ok=True)
    if do_false_color: fc_dir.mkdir(exist_ok=True)

    log(f"\nLoading shapefile: {shapefile_zip}")
    clip_geom, hull_geom = load_geometry(shapefile_zip)
    log(f"Bounds (lon/lat): {tuple(round(x, 5) for x in clip_geom.bounds)}")

    if do_dem:
        log("\nDownloading elevation data...")
        download_dem(clip_geom, dem_dir)

    if not (do_rgb or do_ndwi or do_ndvi or do_false_color):
        log("\nDone.")
        if do_dem: log(f"  DEM → {dem_dir.resolve()}")
        return

    log(f"\nSearching for S2 L2A images with <{cloud_max}% cloud cover (all time)...")
    products = search_products(hull_geom.wkt, cloud_max=cloud_max)
    log(f"Total matching scenes: {len(products)}\n")

    if progress_callback:
        progress_callback(0, len(products))

    if not products:
        log("No products found. Try relaxing the cloud cover threshold.")
        return

    token_mgr = TokenManager(username, password)
    token_mgr.get()

    s3_client = make_s3_client()
    if s3_client:
        log(f"Mode: S3 direct  |  Workers: {workers}")
    else:
        log(f"Mode: HTTPS      |  Workers: {workers}")
        log("  (Add CDSE_S3_KEY + CDSE_S3_SECRET to .env for faster S3 downloads)")

    what = " + ".join(filter(None, [
        "RGB"        if do_rgb         else None,
        "NDWI"       if do_ndwi        else None,
        "NDVI"       if do_ndvi        else None,
        "FalseColor" if do_false_color else None,
    ]))
    log(f"\nDownloading and clipping [{what}] to '{output_dir}/'...\n")

    ok = skipped = failed = cancelled = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_product, p, token_mgr, s3_client, clip_geom,
                rgb_dir, ndwi_dir, ndvi_dir, fc_dir,
                i, len(products),
                do_rgb, do_ndwi, do_ndvi, do_false_color,
                cancel_event,
            ): p
            for i, p in enumerate(products, 1)
        }
        done_count = 0
        for future in as_completed(futures):
            result = future.result()
            done_count += 1
            if progress_callback:
                progress_callback(done_count, len(products))
            if result == "ok":
                ok += 1
            elif result == "skipped":
                skipped += 1
            elif result == "cancelled":
                cancelled += 1
            else:
                failed += 1
            if cancel_event and cancel_event.is_set():
                log("\nCancelled by user.")
                break

    log(f"\nDone.")
    log(f"  Completed : {ok}")
    log(f"  Skipped   : {skipped}  (already existed)")
    if cancelled:
        log(f"  Cancelled : {cancelled}")
    log(f"  Failed    : {failed}")
    if do_rgb:         log(f"  RGB        → {rgb_dir.resolve()}")
    if do_ndwi:        log(f"  NDWI       → {ndwi_dir.resolve()}")
    if do_ndvi:        log(f"  NDVI       → {ndvi_dir.resolve()}")
    if do_false_color: log(f"  FalseColor → {fc_dir.resolve()}")
    if do_dem:         log(f"  DEM        → {dem_dir.resolve()}")
    log("\nAll files are GeoTIFF (.tif) — open directly in QGIS.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and clip Sentinel-2 L2A TCI images to a field boundary."
    )
    parser.add_argument("shapefile_zip", help="Path to the zipped shapefile (.zip)")
    parser.add_argument("--cloud-cover", type=float, default=10.0,
                        help="Maximum cloud cover %% (default: 10)")
    parser.add_argument("--output", default="output",
                        help="Output folder (default: output/)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4)")
    args = parser.parse_args()
    run(args.shapefile_zip, args.output, args.cloud_cover, args.workers)


if __name__ == "__main__":
    main()
