"""Rain radar map for the app's Map tab, every 15 minutes, where governments publish radar openly.

  USA      NOAA MRMS (Multi-Radar Multi-Sensor) merged reflectivity, 1 km, every 2 minutes, on AWS:
           mainland USA, Alaska, Hawaii, Puerto Rico and the Caribbean, Guam. US government data, free for
           any use (NOAA open data, attribution requested).
  Europe   EUMETNET OPERA CIRRUS maximum reflectivity composite, 1 km, every 5 minutes, about 160 radars in 25
           countries, from the open radar data 24 hour store (no account needed). CC BY 4.0, credit "EUMETNET OPERA".

Nowhere else is there open radar (India's IMD radar needs IMD's permission), so the app says "no radar here"
outside these areas and shows the satellite clouds instead.

The frame is web map tiles 2..7 (about 1.2 km a pixel at the equator at 7) over the radar areas only, packed
into one file and uploaded as the layer "radar" (wbwx/maptiles.py).

    python -m wbwx.radar --out build/radar [--time 2026-11-10T14:15Z] [--publish]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests

from . import maptiles
from .maptiles import Box

log = logging.getLogger("wbwx.radar")

LAYER = "radar"
MIN_ZOOM, MAX_ZOOM = 2, 7
STEP = dt.timedelta(minutes=15)
#: A source's scan is used for a frame when it is at most this far from the frame time.
MATCH = dt.timedelta(minutes=8)

_http = requests.Session()
_http.headers["User-Agent"] = "WeatherBridge-weather-job (+https://www.lorjova.com/weatherbridge/)"


def _get(url: str, tries: int = 4, **kw) -> requests.Response:
    for attempt in range(tries):
        try:
            r = _http.get(url, timeout=90, **kw)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise IOError(url)


def _s3_list(base: str, prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        t = _get(base, params=params).text
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", t)]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
        if not m:
            return out
        token = m.group(1)


@dataclass
class Grid:
    """Reflectivity (dBZ) on a grid, NaN outside radar coverage, -inf where the radars see no rain."""

    name: str
    box: Box
    sample: Callable[[np.ndarray, np.ndarray], np.ndarray]
    time: dt.datetime


# --- NOAA MRMS ---------------------------------------------------------------------------------------------

MRMS_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/"
MRMS_PRODUCT = "MergedReflectivityQCComposite_00.50"
MRMS_REGIONS = ["CONUS", "ALASKA", "HAWAII", "CARIB", "GUAM"]
_MRMS_TIME = re.compile(r"_(\d{8}-\d{6})\.grib2\.gz$")


def mrms(region: str, t: dt.datetime) -> Grid:
    import eccodes

    keys = []
    for day in {t - MATCH, t + MATCH}:
        keys += [k for k, _ in _s3_list(MRMS_BASE, f"{region}/{MRMS_PRODUCT}/{day:%Y%m%d}/")]
    best = None
    for k in set(keys):
        m = _MRMS_TIME.search(k)
        if not m:
            continue
        kt = dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=dt.timezone.utc)
        if abs(kt - t) <= MATCH and (best is None or abs(kt - t) < abs(best[1] - t)):
            best = (k, kt)
    if not best:
        raise FileNotFoundError(f"MRMS {region} near {t:%H:%M}")
    raw = gzip.decompress(_get(MRMS_BASE + best[0]).content)
    h = eccodes.codes_new_from_message(raw)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        lat1 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
        lon1 = eccodes.codes_get(h, "longitudeOfLastGridPointInDegrees")
        north_first = eccodes.codes_get(h, "jScansPositively") == 0
        vals = eccodes.codes_get_values(h).astype(np.float32).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    if not north_first:
        vals = vals[::-1]
        lat0, lat1 = lat1, lat0
    vals[vals <= -999] = np.nan  # outside the radars' reach
    vals[(vals < 0) & np.isfinite(vals)] = -np.inf  # covered, no echo (-99 is "no data at this point": treat as dry)
    lon0 = ((lon0 + 180) % 360) - 180
    lon1 = ((lon1 + 180) % 360) - 180
    dlat = (lat0 - lat1) / (nj - 1)
    span = (lon1 - lon0) % 360
    dlon = span / (ni - 1)

    def sample(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        col = np.rint(((lon - lon0) % 360) / dlon).astype(np.int64)
        row = np.rint((lat0 - lat) / dlat).astype(np.int64)
        ok = (col >= 0) & (col < ni) & (row >= 0) & (row < nj)
        out = np.full(lon.shape, np.nan, dtype=np.float32)
        out[ok] = vals[row[ok], col[ok]]
        return out

    west, east = lon0, lon0 + span
    box = Box(lat1, west if east <= 180 else -180, lat0, east if east <= 180 else 180)
    return Grid(f"MRMS {region}", box, sample, best[1])


# --- EUMETNET OPERA ----------------------------------------------------------------------------------------

OPERA_BASE = "https://s3.waw3-1.cloudferro.com/openradar-24h/"


def opera_keys(t: dt.datetime) -> list[tuple[str, int]]:
    """OPERA maximum reflectivity composites of the day of t, e.g. 2026/11/10/OPERA/COMP/OPERA@20261110T1505@0@DBZH.h5
    (CIRRUS, 1 km, every 5 minutes). The same folder holds rain-rate (RATE) and accumulation (ACRR) composites."""
    return [(k, s) for k, s in _s3_list(OPERA_BASE, f"{t:%Y/%m/%d}/OPERA/COMP/") if k.endswith("@DBZH.h5")]


def _key_time(k: str) -> dt.datetime | None:
    m = re.search(r"(20\d{2})(\d{2})(\d{2})[T_-]?(\d{2})(\d{2})", k.rsplit("/", 1)[-1])
    if not m:
        return None
    return dt.datetime(*map(int, m.groups()), tzinfo=dt.timezone.utc)


def opera(t: dt.datetime) -> Grid:
    import h5py
    from pyproj import Transformer

    best = None
    for k, _ in opera_keys(t):
        kt = _key_time(k)
        if kt and abs(kt - t) <= MATCH and (best is None or abs(kt - t) < abs(best[1] - t)):
            best = (k, kt)
    if not best:
        raise FileNotFoundError(f"OPERA near {t:%H:%M}")
    f = h5py.File(io.BytesIO(_get(OPERA_BASE + best[0]).content), "r")
    where = f["where"].attrs
    proj = where["projdef"].decode() if isinstance(where["projdef"], bytes) else str(where["projdef"])
    xsize, ysize = int(where["xsize"]), int(where["ysize"])
    xscale, yscale = float(where["xscale"]), float(where["yscale"])
    to_xy = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    ul_x, ul_y = to_xy.transform(float(where["UL_lon"]), float(where["UL_lat"]))
    ds = None
    for name in sorted(f):
        if not name.startswith("dataset"):
            continue
        for dn in sorted(f[name]):
            if dn.startswith("data") and "what" in f[name][dn]:
                q = f[name][dn]["what"].attrs.get("quantity", b"")
                q = q.decode() if isinstance(q, bytes) else str(q)
                if q in ("DBZH", "TH"):
                    ds = f[name][dn]
                    break
        if ds is not None:
            break
    if ds is None:
        raise ValueError(f"no reflectivity in {best[0]}")
    what = ds["what"].attrs
    raw = ds["data"][()]
    gain, offset = float(what.get("gain", 1.0)), float(what.get("offset", 0.0))
    nodata, undetect = float(what.get("nodata", np.nan)), float(what.get("undetect", np.nan))
    vals = (raw.astype(np.float32) * gain + offset).astype(np.float32)
    vals[raw == nodata] = np.nan
    vals[raw == undetect] = -np.inf
    lats = [float(where[f"{c}_lat"]) for c in ("LL", "UL", "UR", "LR")]
    lons = [float(where[f"{c}_lon"]) for c in ("LL", "UL", "UR", "LR")]

    def sample(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x, y = to_xy.transform(lon, lat)
        col = np.floor((x - ul_x) / xscale).astype(np.int64)
        row = np.floor((ul_y - y) / yscale).astype(np.int64)
        ok = np.isfinite(x) & (col >= 0) & (col < xsize) & (row >= 0) & (row < ysize)
        out = np.full(lon.shape, np.nan, dtype=np.float32)
        out[ok] = vals[row[ok], col[ok]]
        return out

    return Grid("OPERA", Box(min(lats), min(lons), max(lats), max(lons)), sample, best[1])


# --- Drawing -----------------------------------------------------------------------------------------------

#: Reflectivity colours (dBZ): light rain pale blue, then green, yellow, orange, red; hail and extreme pink.
DBZ_STOPS = [
    (10.0, (130, 205, 255, 130)),
    (18.0, (40, 170, 120, 185)),
    (28.0, (20, 150, 40, 205)),
    (35.0, (250, 225, 0, 225)),
    (42.0, (255, 140, 0, 235)),
    (50.0, (225, 20, 20, 245)),
    (58.0, (190, 0, 170, 250)),
    (65.0, (255, 160, 255, 255)),
]


def colour_for(grids: list[Grid]) -> Callable[[np.ndarray, np.ndarray], np.ndarray | None]:
    def colour(lon: np.ndarray, lat: np.ndarray) -> np.ndarray | None:
        dbz = np.full(lon.shape, np.nan, dtype=np.float32)
        for g in grids:
            v = g.sample(lon, lat)
            empty = np.isnan(dbz)
            dbz[empty] = v[empty]
            both = ~empty & ~np.isnan(v)
            dbz[both] = np.fmax(dbz[both], v[both])
        if not np.isfinite(dbz).any():
            return None
        return maptiles.ramp(dbz, DBZ_STOPS)

    return colour


def frame_time(now: dt.datetime) -> dt.datetime:
    """The newest quarter hour that the slower source (OPERA, about 15 minutes late) has published."""
    t = now - dt.timedelta(minutes=18)
    return t.replace(minute=t.minute - t.minute % 15, second=0, microsecond=0)


CREDIT = "NOAA MRMS; EUMETNET OPERA (CC BY 4.0)"


def build(t: dt.datetime) -> tuple[bytes, list[str]]:
    grids, used = [], []
    for name, fetch in [*[(f"MRMS {r}", (lambda r=r: mrms(r, t))) for r in MRMS_REGIONS], ("OPERA", lambda: opera(t))]:
        t0 = time.time()
        try:
            g = fetch()
            grids.append(g)
            used.append(name)
            log.info("%s: scan %s, read in %.0fs", name, g.time.strftime("%H:%M"), time.time() - t0)
        except Exception as e:
            log.warning("%s left out: %s", name, e)
    if not grids:
        raise RuntimeError("no radar source available")
    t0 = time.time()
    tiles = maptiles.render(colour_for(grids), range(MIN_ZOOM, MAX_ZOOM + 1), [g.box for g in grids], quality=80)
    log.info("drew %d tiles in %.0fs", len(tiles), time.time() - t0)
    return maptiles.pack(LAYER, t.strftime("%Y-%m-%dT%H:%MZ"), tiles, MIN_ZOOM, MAX_ZOOM, CREDIT), used


def main() -> None:
    from . import publish

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/radar"))
    ap.add_argument("--time", help="frame time, e.g. 2026-11-10T14:15Z (default: newest)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    t = dt.datetime.strptime(a.time, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc) if a.time else frame_time(dt.datetime.now(dt.timezone.utc))
    if a.publish and publish.sky_frame_live(LAYER, t):
        log.info("frame %s is already live", t)
        return
    data, used = build(t)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frame.wbpk").write_bytes(data)
    log.info("frame %s: %.1f MB from %s", t, len(data) / 1e6, ", ".join(used))
    if a.publish:
        publish.upload_sky(LAYER, t, data)


if __name__ == "__main__":
    main()
