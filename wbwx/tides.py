"""The tide constants map the API uses for tide heights (sea_level_height_msl in the marine answers).

Source: EOT20, the global empirical ocean tide model from multi-mission satellite altimetry by DGFI-TUM
(Hart-Davis et al. 2021, https://doi.org/10.17882/79489), licence CC BY 4.0. 17 constituents on a 0.125
degree grid, amplitude in cm and Greenwich phase lag in degrees.

Built once and uploaded as the static dataset "tides": for every sea point of the 0.25 degree grid, two
variables per constituent, {c}_re = A cos(g) and {c}_im = A sin(g) in millimetres. The API adds the
astronomical arguments and nodal corrections (lorjova backend, weather/tide.ts). A grid point that falls
on land in EOT20 takes the mean of the EOT20 sea cells next to it, so coastal towns still get a tide.

    python -m wbwx.tides --out build/tides [--source EOT20.zip] [--publish]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .tiles import MISSING, RES, TILE, TILE_COLS, TILE_ROWS, encode, quantize, tile_id

log = logging.getLogger("wbwx.tides")

EOT20_URL = "https://www.seanoe.org/data/00683/79489/data/85762.zip"
# Same order as the API (weather/tide.ts TIDE_CONSTITUENTS).
CONSTITUENTS = ("2n2", "j1", "k1", "k2", "m2", "m4", "mf", "mm", "n2", "o1", "p1", "q1", "s1", "s2", "sa", "ssa", "t2")
# A fixed "cycle" (2021-01-01, the EOT20 release year): the map does not change between runs.
CYCLE = 1_609_459_200
VARS = [f"{c}_{part}" for c in CONSTITUENTS for part in ("re", "im")]


def grid_positions() -> tuple[np.ndarray, np.ndarray]:
    """Latitudes of the API grid rows 0..719 (90N southwards) and longitudes of columns 0..1439 (0E eastwards)."""
    return 90 - np.arange(TILE_ROWS * TILE) * RES, np.arange(TILE_COLS * TILE) * RES


def to_grid(amp_cm: np.ndarray, phase_deg: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One constituent on the API grid: (re, im) in mm, NaN where there is no sea within one EOT20 cell."""
    ok = np.isfinite(amp_cm) & np.isfinite(phase_deg)
    g = np.radians(np.where(ok, phase_deg, 0))
    re = np.where(ok, amp_cm * np.cos(g) * 10, 0.0)
    im = np.where(ok, amp_cm * np.sin(g) * 10, 0.0)
    lat, lon = grid_positions()
    dlat = float(lat_axis[1] - lat_axis[0])
    dlon = float(lon_axis[1] - lon_axis[0])
    ri = np.rint((lat - lat_axis[0]) / dlat).astype(np.int64)
    ci = np.rint(((lon - lon_axis[0]) % 360) / dlon).astype(np.int64) % len(lon_axis)
    ny = len(lat_axis)
    out_re = np.full((len(lat), len(lon)), np.nan)
    out_im = np.full_like(out_re, np.nan)
    sum_re = np.zeros_like(out_re)
    sum_im = np.zeros_like(out_re)
    count = np.zeros_like(out_re)
    for dr in (-1, 0, 1):
        r = np.clip(ri + dr, 0, ny - 1)
        for dc in (-1, 0, 1):
            c = (ci + dc) % len(lon_axis)
            o = ok[np.ix_(r, c)]
            sum_re += np.where(o, re[np.ix_(r, c)], 0)
            sum_im += np.where(o, im[np.ix_(r, c)], 0)
            count += o
    centre = ok[np.ix_(np.clip(ri, 0, ny - 1), ci)] & (ri >= 0)[:, None] & (ri < ny)[:, None]
    near = count > 0
    out_re[near] = sum_re[near] / count[near]
    out_im[near] = sum_im[near] / count[near]
    # Where EOT20 has the point itself, use it exactly.
    rc = np.clip(ri, 0, ny - 1)
    out_re[centre] = re[np.ix_(rc, ci)][centre]
    out_im[centre] = im[np.ix_(rc, ci)][centre]
    return out_re, out_im


def build(fields: dict[str, np.ndarray], out: Path) -> Path:
    """fields: {var: [720 rows][1440 cols] in mm}. Writes tiles [34 vars][1 step][400 points] and a manifest."""
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    q = np.stack([quantize(fields[v], 1) for v in VARS])  # [var][row][col]
    tiles = []
    for R in range(TILE_ROWS):
        for C in range(TILE_COLS):
            block = q[:, R * TILE : (R + 1) * TILE, C * TILE : (C + 1) * TILE]
            if (block == MISSING).all():
                continue
            tid = tile_id(R, C)
            (out / "tiles" / f"{tid}.bin").write_bytes(encode(np.ascontiguousarray(block.reshape(len(VARS), 1, TILE * TILE))))
            tiles.append(tid)
    manifest = {
        "formatVersion": 1,
        "dataset": "tides",
        "title": "Ocean tide constants (EOT20, DGFI-TUM)",
        "licence": "EOT20 by DGFI-TUM (Hart-Davis et al. 2021, doi:10.17882/79489), CC BY 4.0",
        "cycle": CYCLE,
        "steps": [0],
        "vars": [{"name": v, "scale": 1} for v in VARS],
        "tiles": tiles,
        "processedAt": int(time.time()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    log.info("tides: %d tiles", len(tiles))
    return out


def _read_nc(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """amplitude, phase, lat, lon from one EOT20 file (netCDF4/HDF5, or classic netCDF for some files)."""
    import io

    if data[:4] == b"\x89HDF":
        import h5py

        f = h5py.File(io.BytesIO(data), "r")  # netCDF4
        get = lambda name: f[name]  # noqa: E731
        has = lambda name: name in f  # noqa: E731
    elif data[:3] == b"CDF":
        from scipy.io import netcdf_file

        f = netcdf_file(io.BytesIO(data), "r", mmap=False)
        get = lambda name: f.variables[name]  # noqa: E731
        has = lambda name: name in f.variables  # noqa: E731
    else:
        raise ValueError(f"not a netCDF file (starts with {data[:8]!r}, {len(data)} bytes)")
    try:
        def var(*names: str) -> np.ndarray:
            name = next(n for n in names if has(n))
            d = get(name)
            a = np.array(d[:], dtype=np.float64)
            attrs = d.attrs if hasattr(d, "attrs") else d._attributes
            for attr in ("_FillValue", "missing_value"):
                if attr in attrs:
                    a[a == float(np.asarray(attrs[attr]).ravel()[0])] = np.nan
            if "scale_factor" in attrs:
                a *= float(np.asarray(attrs["scale_factor"]).ravel()[0])
            a[np.abs(a) > 1e10] = np.nan
            return a

        amp = var("amplitude", "amp", "Amplitude")
        pha = var("phase", "pha", "Phase")
        lat = var("lat", "latitude")
        lon = var("lon", "longitude")
    finally:
        f.close()
    if amp.shape != (len(lat), len(lon)):  # stored as [lon][lat]
        amp, pha = amp.T, pha.T
    return amp, pha, lat, lon


def _nc_members(z: zipfile.ZipFile, depth: int = 0):
    """(name, bytes reader) for every .nc file in the zip, also inside zips within it."""
    for n in z.namelist():
        low = n.lower()
        base = low.rsplit("/", 1)[-1]
        if "__macosx" in low or base.startswith("._"):  # macOS metadata files, not data
            continue
        if low.endswith(".nc"):
            yield n, (lambda zz=z, nn=n: zz.read(nn))
        elif low.endswith(".zip") and depth < 2:
            import io

            yield from _nc_members(zipfile.ZipFile(io.BytesIO(z.read(n))), depth + 1)


def read_eot20(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as z:
        members = list(_nc_members(z))
        names = {}
        for n, reader in members:
            m = re.search(r"([0-9a-z]+)_ocean_eot20\.nc$", n.lower())
            if m and "load" not in n.lower():
                names[m.group(1)] = reader
        missing = [c for c in CONSTITUENTS if c not in names]
        if missing:
            listing = [n for n, _ in members][:40] or z.namelist()[:40]
            raise SystemExit(f"EOT20 zip lacks {missing}; files: {listing}")
        for c in CONSTITUENTS:
            amp, pha, lat, lon = _read_nc(names[c]())
            log.info("%s: %s, max amplitude %.0f cm", c, amp.shape, np.nanmax(amp))
            fields[f"{c}_re"], fields[f"{c}_im"] = to_grid(amp, pha, lat, lon)
    return fields


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/tides")
    ap.add_argument("--source", help="a downloaded EOT20 zip (default: download it, about 2 GB)")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--force", action="store_true", help="upload even if the map is already live")
    a = ap.parse_args(argv)
    if a.publish and not a.force:
        from . import publish

        if publish.is_live("tides", CYCLE):
            log.info("tide map is already live, nothing to do")
            return 0
    out = Path(a.out)
    src = Path(a.source) if a.source else out / "eot20.zip"
    if not src.exists():
        out.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", EOT20_URL)
        urllib.request.urlretrieve(EOT20_URL, src)
    build(read_eot20(src), out)
    if a.publish:
        from . import publish

        publish.upload(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
