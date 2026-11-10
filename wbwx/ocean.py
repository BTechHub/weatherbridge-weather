"""Ocean currents and sea temperature for the marine answers, from Copernicus Marine.

Source: Global Ocean Physics Analysis and Forecast (GLOBAL_ANALYSISFORECAST_PHY_001_024), hourly surface
fields at 1/12 degree, dataset cmems_mod_glo_phy_anfc_0.083deg_PT1H-m. Generated using E.U. Copernicus
Marine Service Information (https://doi.org/10.48670/moi-00016). Needs a free Copernicus Marine account:
WB_CMEMS_USERNAME and WB_CMEMS_PASSWORD.

Uploaded once a day as the dataset "ocean": every 3 hours from yesterday 00 UTC to the end of the forecast
(about 10 days), on the 0.25 degree grid, the mean of the 3 x 3 Copernicus cells around each point:
  current_u, current_v  surface current towards east / north, m/s (scale 100)
  sst                   sea surface temperature, deg C (scale 100)
The total current (with tides) is used when the dataset has it (utotal, vtotal), else uo, vo.

    python -m wbwx.ocean --out build/ocean [--publish] [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from .tiles import MISSING, RES, TILE, TILE_COLS, TILE_ROWS, encode, quantize, tile_id

log = logging.getLogger("wbwx.ocean")

DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
STEP_HOURS = 3
MAX_DAYS = 11
VARS = (("current_u", 100), ("current_v", 100), ("sst", 100))


def grid_positions() -> tuple[np.ndarray, np.ndarray]:
    """Latitudes of the API grid rows 0..719 (90N southwards) and longitudes of columns 0..1439 (0E eastwards)."""
    return 90 - np.arange(TILE_ROWS * TILE) * RES, np.arange(TILE_COLS * TILE) * RES


class Regrid:
    """Mean of the source cells within half a source cell of each API grid point's 3 x 3 neighbourhood."""

    def __init__(self, lat_axis: np.ndarray, lon_axis: np.ndarray):
        lat, lon = grid_positions()
        dlat = float(lat_axis[1] - lat_axis[0])
        dlon = float(lon_axis[1] - lon_axis[0])
        ri = np.rint((lat - lat_axis[0]) / dlat).astype(np.int64)
        self.inside = (ri >= 0) & (ri < len(lat_axis))
        self.ri = np.clip(ri, 0, len(lat_axis) - 1)
        self.ci = np.rint(((lon - lon_axis[0]) % 360) / dlon).astype(np.int64) % len(lon_axis)
        self.ny, self.nx = len(lat_axis), len(lon_axis)

    def __call__(self, a: np.ndarray) -> np.ndarray:
        total = np.zeros((len(self.ri), len(self.ci)))
        count = np.zeros_like(total)
        for dr in (-1, 0, 1):
            r = np.clip(self.ri + dr, 0, self.ny - 1)
            for dc in (-1, 0, 1):
                v = a[np.ix_(r, (self.ci + dc) % self.nx)]
                ok = np.isfinite(v)
                total += np.where(ok, v, 0)
                count += ok
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        out[~self.inside, :] = np.nan
        return out


def build(fields: dict[str, np.ndarray], cycle: int, steps: list[int], out: Path, title: str) -> Path:
    """fields: {var: int16 [step][720 rows][1440 cols]}. Writes tiles [3 vars][steps][400 points] and a manifest."""
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    q = np.stack([fields[name] for name, _ in VARS])  # [var][step][row][col]
    tiles = []
    for R in range(TILE_ROWS):
        for C in range(TILE_COLS):
            block = q[:, :, R * TILE : (R + 1) * TILE, C * TILE : (C + 1) * TILE]
            if (block == MISSING).all():
                continue
            tid = tile_id(R, C)
            (out / "tiles" / f"{tid}.bin").write_bytes(encode(np.ascontiguousarray(block.reshape(len(VARS), len(steps), TILE * TILE))))
            tiles.append(tid)
    manifest = {
        "formatVersion": 1,
        "dataset": "ocean",
        "title": title,
        "licence": "Generated using E.U. Copernicus Marine Service Information (https://doi.org/10.48670/moi-00016)",
        "cycle": cycle,
        "steps": steps,
        "vars": [{"name": n, "scale": s} for n, s in VARS],
        "tiles": tiles,
        "processedAt": int(time.time()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    log.info("ocean: %d tiles, %d steps", len(tiles), len(steps))
    return out


def _open(start: str, end: str):
    import copernicusmarine

    user, password = os.environ.get("WB_CMEMS_USERNAME"), os.environ.get("WB_CMEMS_PASSWORD")
    if not user or not password:
        raise SystemExit("WB_CMEMS_USERNAME and WB_CMEMS_PASSWORD are not set")
    kw = dict(dataset_id=DATASET_ID, start_datetime=start, end_datetime=end, username=user, password=password)
    ds = copernicusmarine.open_dataset(**kw)
    log.info("dataset %s: variables %s, dims %s", DATASET_ID, list(ds.data_vars), dict(ds.sizes))
    return ds


def fetch(out: Path, force: bool, publish_it: bool) -> int:
    today = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - dt.timedelta(days=1)
    ds = _open(start.strftime("%Y-%m-%dT00:00:00"), (start + dt.timedelta(days=MAX_DAYS)).strftime("%Y-%m-%dT23:59:59"))
    u_name, v_name = ("utotal", "vtotal") if "utotal" in ds and "vtotal" in ds else ("uo", "vo")
    names = {"current_u": u_name, "current_v": v_name, "sst": "thetao"}
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    times = ds["time"].values.astype("datetime64[s]").astype(np.int64)
    first = int(times[0])
    # Every STEP_HOURS from the first time stamp (hourly means are stamped at the half hour, which is fine).
    pick = [i for i, t in enumerate(times) if (int(t) - first) % (STEP_HOURS * 3600) == 0]
    steps = [(int(times[i]) - first) // 3600 for i in pick]
    if publish_it and not force:
        from . import publish

        if publish.is_live("ocean", first):
            log.info("ocean run %s is already live, nothing to do", dt.datetime.fromtimestamp(first, dt.timezone.utc))
            return 0
    regrid = Regrid(ds[lat_name].values, ds[lon_name].values)
    fields = {name: np.full((len(pick), TILE_ROWS * TILE, TILE_COLS * TILE), MISSING, dtype=np.int16) for name, _ in VARS}
    scales = dict(VARS)
    t0 = time.time()
    for k, i in enumerate(pick):
        for name, src in names.items():
            da = ds[src].isel(time=i)
            if "depth" in da.dims:
                da = da.isel(depth=0)
            a = np.asarray(da.values, dtype=np.float64)
            if a.shape != (ds.sizes[lat_name], ds.sizes[lon_name]):
                a = a.T
            fields[name][k] = quantize(regrid(a), scales[name])
        log.info("step %d/%d (+%d h) done, %.0fs", k + 1, len(pick), steps[k], time.time() - t0)
    title = f"Copernicus Marine global ocean physics forecast ({u_name}/{v_name}, thetao)"
    build(fields, first, steps, out, title)
    if publish_it:
        from . import publish

        publish.upload(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/ocean")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--force", action="store_true", help="upload even if this run is already live")
    a = ap.parse_args(argv)
    return fetch(Path(a.out), a.force, a.publish)


if __name__ == "__main__":
    raise SystemExit(main())
