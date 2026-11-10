"""World satellite cloud map (infrared, day and night) for the app's Map tab, once an hour.

Three weather satellites, each looking at its own side of the Earth, joined into one picture. Every place is
taken from the satellite that sees it most directly:

  Himawari-9  (140.7 E)                JMA via NOAA on AWS, band 13  East Asia, Australia, West Pacific
  GOES-East   (75.2 W)                 NOAA on AWS, ABI band 13      the Americas, Atlantic
  GOES-West   (137 W)                  NOAA on AWS, ABI band 13      East Pacific, Alaska, Hawaii

Licences: NOAA and JMA data via NOAA's open data programme (free, attribution requested).
EUMETSAT Meteosat is not used: the Data Store licence offered to our account forbids commercial use and
redistribution (BenK, 2026-11-10). Without it, India east of about 67 E comes from Himawari at a slant; Europe,
most of Africa, the Middle East and the western Arabian Sea are left out (GOES-East reaches only West Africa).

The frame is web map tiles 0..5 (about 5 km a pixel at the equator) packed into one file and uploaded as the
layer "satellite" (wbwx/maptiles.py, API docs/WEATHERBRIDGE_WEATHER.md in the lorjova repo).

    python -m wbwx.satellite --out build/satellite [--time 2026-11-10T14:00Z] [--publish]
"""
from __future__ import annotations

import argparse
import bz2
import datetime as dt
import glob
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from . import maptiles
from .maptiles import TILE_PX, WORLD

log = logging.getLogger("wbwx.satellite")

MAX_ZOOM = 5
LAYER = "satellite"
#: Furthest angle from the point under a satellite that is still used (beyond it the view is too slanted).
MAX_VIEW_DEG = 74.0
#: How long after the hour all satellites' full-disk images are on AWS.
FRAME_LAG = dt.timedelta(minutes=30)

_http = requests.Session()
_http.headers["User-Agent"] = "WeatherBridge-weather-job (+https://www.lorjova.com/weatherbridge/)"


@dataclass
class View:
    name: str
    sub_lon: float
    bt: np.ndarray  # brightness temperature, K, NaN where there is none
    area: object  # pyresample AreaDefinition


# --- Downloads ---------------------------------------------------------------------------------------------


def _s3_keys(bucket: str, prefix: str) -> list[str]:
    import re

    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = _http.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def _download(url: str, path: Path, tries: int = 4) -> Path:
    for attempt in range(tries):
        try:
            with _http.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return path
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    return path


def goes(sat: int, t: dt.datetime, work: Path) -> list[Path]:
    """The full-disk band 13 image (CMIP, brightness temperature) that starts in the minute of t."""
    prefix = f"ABI-L2-CMIPF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/OR_ABI-L2-CMIPF-M6C13_G{sat}_s{t:%Y}{t.timetuple().tm_yday:03d}{t:%H%M}"
    keys = _s3_keys(f"noaa-goes{sat}", prefix)
    if not keys:
        raise FileNotFoundError(f"GOES-{sat} {t:%H:%M}")
    return [_download(f"https://noaa-goes{sat}.s3.amazonaws.com/{keys[0]}", work / Path(keys[0]).name)]


def himawari(t: dt.datetime, work: Path) -> list[Path]:
    """The ten full-disk band 13 segments (2 km), unpacked from bz2 for the reader."""
    prefix = f"AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/HS_H09_{t:%Y%m%d_%H%M}_B13_FLDK_R20_S"
    keys = sorted(_s3_keys("noaa-himawari9", prefix))
    if len(keys) < 10:
        raise FileNotFoundError(f"Himawari-9 {t:%H:%M}: {len(keys)} of 10 segments")
    out = []
    for k in keys:
        p = _download(f"https://noaa-himawari9.s3.amazonaws.com/{k}", work / Path(k).name)
        raw = p.with_suffix("")  # .DAT
        raw.write_bytes(bz2.decompress(p.read_bytes()))
        p.unlink()
        out.append(raw)
    return out


# --- Reading -----------------------------------------------------------------------------------------------

READERS = {
    "goes-east": ("abi_l2_nc", "C13", -75.2),
    "goes-west": ("abi_l2_nc", "C13", -137.0),
    "himawari": ("ahi_hsd", "B13", 140.7),
}


def read(name: str, files: list[Path]) -> View:
    from satpy import Scene

    reader, band, sub_lon = READERS[name]
    scn = Scene(reader=reader, filenames=[str(f) for f in files])
    scn.load([band])
    d = scn[band]
    bt = np.asarray(d.values, dtype=np.float32)
    bt[(bt < 150) | (bt > 350)] = np.nan
    # The real position of the satellite (it drifts a little), when the file says.
    sub = d.attrs.get("orbital_parameters", {}).get("projection_longitude", sub_lon)
    return View(name, float(sub), bt, d.attrs["area"])


# --- Drawing -----------------------------------------------------------------------------------------------


def view_angle(lon: np.ndarray, lat: np.ndarray, sub_lon: float) -> np.ndarray:
    """Angle at the Earth's centre between each place and the point under the satellite, degrees."""
    dl = np.radians(((lon - sub_lon) + 180) % 360 - 180)
    return np.degrees(np.arccos(np.clip(np.cos(np.radians(lat)) * np.cos(dl), -1, 1)))


def composite(views: list[View], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Brightness temperature at each position from the satellite that sees it most directly."""
    best = np.full(lon.shape, np.nan, dtype=np.float32)
    best_angle = np.full(lon.shape, np.inf, dtype=np.float32)
    for v in views:
        ang = view_angle(lon, lat, v.sub_lon)
        use = ang < np.minimum(best_angle, MAX_VIEW_DEG)
        if not use.any():
            continue
        cols, rows = v.area.get_array_indices_from_lonlat(lon[use], lat[use])
        cols, rows = np.ma.filled(cols, -1), np.ma.filled(rows, -1)
        ok = (cols >= 0) & (rows >= 0) & (cols < v.bt.shape[1]) & (rows < v.bt.shape[0])
        val = np.full(cols.shape, np.nan, dtype=np.float32)
        val[ok] = v.bt[rows[ok], cols[ok]]
        good = np.isfinite(val)
        idx = np.flatnonzero(use)[good]
        best.flat[idx] = val[good]
        best_angle.flat[idx] = ang.flat[idx]
    return best


#: Infrared colour scale on minus the temperature: warm ground and sea are clear, low cloud is faint grey,
#: high cloud white, and the very cold tops of thunderstorms and cyclones turn blue, purple and pink.
IR_STOPS = [
    (-272.0, (200, 205, 212, 0)),
    (-262.0, (212, 216, 222, 90)),
    (-250.0, (226, 229, 233, 160)),
    (-238.0, (238, 240, 242, 205)),
    (-223.0, (250, 250, 252, 232)),
    (-213.0, (255, 255, 255, 242)),
    (-208.0, (160, 215, 255, 248)),
    (-200.0, (70, 140, 255, 255)),
    (-192.0, (160, 80, 240, 255)),
    (-185.0, (255, 80, 190, 255)),
]


def colour(bt: np.ndarray) -> np.ndarray:
    return maptiles.ramp(-bt, IR_STOPS)


def world_canvas(views: list[View], z: int = MAX_ZOOM) -> np.ndarray:
    """Brightness temperature on the whole Web Mercator world at zoom z, (2^z * 256) square."""
    n = 2 ** z
    size = n * TILE_PX
    out = np.full((size, size), np.nan, dtype=np.float32)
    for ty in range(n):
        # One row of tiles at a time keeps memory small.
        lon, lat = maptiles.pixel_lonlat(z, 0, ty)
        lon_row = (np.arange(size) + 0.5) / size * 360 - 180
        lon2 = np.broadcast_to(lon_row, (TILE_PX, size))
        lat2 = np.broadcast_to(lat[:, :1], (TILE_PX, size))
        if np.abs(lat2).min() > 81:
            continue  # beyond every satellite's view
        out[ty * TILE_PX : (ty + 1) * TILE_PX] = composite(views, np.ascontiguousarray(lon2), np.ascontiguousarray(lat2))
    return out


def halve(a: np.ndarray) -> np.ndarray:
    """Mean of each 2 x 2 block, ignoring gaps."""
    h, w = a.shape
    b = a.reshape(h // 2, 2, w // 2, 2)
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def tiles_from_canvas(canvas: np.ndarray, max_zoom: int = MAX_ZOOM, quality: int = 70) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    c = canvas
    for z in range(max_zoom, -1, -1):
        n = 2 ** z
        for ty in range(n):
            for tx in range(n):
                block = c[ty * TILE_PX : (ty + 1) * TILE_PX, tx * TILE_PX : (tx + 1) * TILE_PX]
                if not np.isfinite(block).any():
                    continue
                rgba = colour(block)
                if not rgba[..., 3].any():
                    continue
                out[f"{z}/{tx}/{ty}"] = maptiles.encode_webp(rgba, quality)
        if z:
            c = halve(c)
    return out


CREDIT = "NOAA GOES, JMA Himawari (via NOAA)"


def frame_time(now: dt.datetime) -> dt.datetime:
    """The newest full hour whose images are all on AWS."""
    t = now - FRAME_LAG
    return t.replace(minute=0, second=0, microsecond=0)


def build(t: dt.datetime, work: Path) -> tuple[bytes, list[str]]:
    work.mkdir(parents=True, exist_ok=True)
    jobs = {
        "goes-east": lambda: goes(19, t, work),
        "goes-west": lambda: goes(18, t, work),
        "himawari": lambda: himawari(t, work),
    }
    views, used = [], []
    for name, fetch in jobs.items():
        t0 = time.time()
        try:
            files = fetch()
            views.append(read(name, files))
            used.append(name)
            log.info("%s: read in %.0fs", name, time.time() - t0)
        except Exception as e:  # one missing satellite leaves its part of the map empty, not the whole map
            log.warning("%s left out: %s", name, e)
        finally:
            for f in glob.glob(str(work / "*")):
                os.remove(f)
    if len(views) < 2:
        raise RuntimeError(f"only {used} available, not publishing")
    t0 = time.time()
    canvas = world_canvas(views)
    tiles = tiles_from_canvas(canvas)
    log.info("drew %d tiles in %.0fs", len(tiles), time.time() - t0)
    iso = t.strftime("%Y-%m-%dT%H:%MZ")
    return maptiles.pack(LAYER, iso, tiles, 0, MAX_ZOOM, CREDIT), used


def main() -> None:
    from . import publish

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/satellite"))
    ap.add_argument("--time", help="frame time, e.g. 2026-11-10T14:00Z (default: newest full hour on AWS)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    now = dt.datetime.now(dt.timezone.utc)
    t = dt.datetime.strptime(a.time, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc) if a.time else frame_time(now)
    if a.publish and publish.sky_frame_live(LAYER, t):
        log.info("frame %s is already live", t)
        return
    data, used = build(t, a.out / "work")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frame.wbpk").write_bytes(data)
    log.info("frame %s: %.1f MB from %s", t, len(data) / 1e6, ", ".join(used))
    if a.publish:
        publish.upload_sky(LAYER, t, data)


if __name__ == "__main__":
    main()
