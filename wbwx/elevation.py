"""The real ground height map the API uses to correct temperatures in mountain and valley towns.

Source: NOAA ETOPO 2022, 60 arc-second surface elevation (public domain). Built once and uploaded as
the static dataset "elevation": for every point of the 0.25 degree grid, 10 x 10 sub-points spaced
0.025 degrees (about 2.8 km) across its cell, one variable each (e00 .. e99, e{i}{j} with i going
south and j going east). Sea and anything below sea level is 0. Tiles with no land are left out.

    python -m wbwx.elevation --out build/elevation [--source FILE.nc] [--publish]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

from .tiles import RES, TILE, TILE_COLS, TILE_ROWS, encode, tile_id

log = logging.getLogger("wbwx.elevation")

ETOPO_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
SUB = 10
STEP = RES / SUB
# A fixed "cycle" (2022-01-01, the ETOPO 2022 release year): the map does not change between runs.
CYCLE = 1_640_995_200


def sub_positions() -> tuple[np.ndarray, np.ndarray]:
    """Latitudes of every grid row's sub-rows (rows 0..719, north to south) and longitudes of every column's sub-columns."""
    rows = np.arange(TILE_ROWS * TILE)
    cols = np.arange(TILE_COLS * TILE)
    k = np.arange(SUB)
    lat = (90 - rows * RES)[:, None] + (SUB / 2 - 0.5 - k)[None, :] * STEP  # [row][i]
    lon = (cols * RES)[:, None] + (k - SUB / 2 + 0.5)[None, :] * STEP  # [col][j]
    lon = (lon + 180) % 360 - 180
    return np.clip(lat, -89.999, 89.999), lon


def sample(z: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray) -> np.ndarray:
    """Heights at every sub-point, int16 [row][i][col][j], from a regular lat/lon grid (nearest cell)."""
    lat, lon = sub_positions()
    dlat = float(lat_axis[1] - lat_axis[0])
    dlon = float(lon_axis[1] - lon_axis[0])
    ri = np.clip(np.rint((lat - lat_axis[0]) / dlat).astype(np.int64), 0, len(lat_axis) - 1)
    ci = np.rint((lon - lon_axis[0]) / dlon).astype(np.int64) % len(lon_axis)
    out = np.empty((lat.shape[0], SUB, lon.shape[0], SUB), dtype=np.int16)
    flat_c = ci.reshape(-1)
    for r in range(lat.shape[0]):
        rows = z[ri[r]]  # [i][all lon]
        h = rows[:, flat_c].reshape(SUB, lon.shape[0], SUB)
        out[r] = np.clip(np.rint(np.maximum(h, 0)), 0, 9000).astype(np.int16)
    return out


def build(heights: np.ndarray, out: Path) -> Path:
    """Cuts the [row][i][col][j] heights into tiles shaped [100 variables][1 step][400 points] and writes a manifest."""
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    tiles = []
    for R in range(TILE_ROWS):
        for C in range(TILE_COLS):
            block = heights[R * TILE : (R + 1) * TILE, :, C * TILE : (C + 1) * TILE, :]  # [r][i][c][j]
            if not block.any():
                continue
            v = block.transpose(1, 3, 0, 2).reshape(SUB * SUB, 1, TILE * TILE)  # [i*10+j][step][r*20+c]
            tid = tile_id(R, C)
            (out / "tiles" / f"{tid}.bin").write_bytes(encode(np.ascontiguousarray(v)))
            tiles.append(tid)
    manifest = {
        "formatVersion": 1,
        "dataset": "elevation",
        "title": "Ground height (NOAA ETOPO 2022, 60 arc-seconds)",
        "licence": "NOAA National Centers for Environmental Information, ETOPO 2022: public domain",
        "cycle": CYCLE,
        "steps": [0],
        "vars": [{"name": f"e{i}{j}", "scale": 1} for i in range(SUB) for j in range(SUB)],
        "tiles": tiles,
        "processedAt": int(time.time()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    log.info("elevation: %d tiles", len(tiles))
    return out


def read_etopo(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        return f["z"][:], f["lat"][:], f["lon"][:]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/elevation")
    ap.add_argument("--source", help="a downloaded ETOPO 2022 60s surface .nc (default: download it)")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--force", action="store_true", help="upload even if the map is already live")
    a = ap.parse_args(argv)
    if a.publish and not a.force:
        from . import publish

        if publish.is_live("elevation", CYCLE):
            log.info("elevation map is already live, nothing to do")
            return 0
    out = Path(a.out)
    src = Path(a.source) if a.source else out / "etopo.nc"
    if not src.exists():
        out.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", ETOPO_URL)
        urllib.request.urlretrieve(ETOPO_URL, src)
    z, lat, lon = read_etopo(src)
    log.info("ETOPO %s, lat %.4f..%.4f, lon %.4f..%.4f", z.shape, lat[0], lat[-1], lon[0], lon[-1])
    build(sample(z, lat, lon), out)
    if a.publish:
        from . import publish

        publish.upload(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
