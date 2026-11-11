"""Rain clouds now over India from Himawari-9 cloud tops, every 10 minutes.

Forecast models only say rain *might* fall. This map says whether rain clouds are there right now.
Rain needs tall clouds, and tall clouds have cold tops. Himawari's infrared band 13 (10.4 um)
measures the cloud-top temperature: where nothing within 25 km is colder than 0 C, it is not
raining (except drizzle from low cloud). Light rain cannot be measured this way, so the API only
uses the map to rule rain out, or to say that tall cold clouds are near.

India is near the edge of Himawari's view (it sits at 140.7 E), so clouds appear a little away
from their real place (up to about 25 km for tall ones). The API looks 25 km around each point.

Area: 66 E to 100 E, 0 N to 37 N (India, Sri Lanka, Bangladesh, the Bay of Bengal and the eastern
Arabian Sea), 0.04 degree cells. Licence: JMA Himawari via NOAA's open data programme (free,
attribution requested). INSAT (MOSDAC) is not used: it needs registration and approval.

File (upload body): b"WBNOW1\\n", one line of JSON header, then gzip of uint8 [ny][nx], each the
cloud-top temperature as K - 150 (1..255), 0 where there is no picture. Row 0 is the north edge.

    python -m wbwx.rainnow --out build/rainnow [--time 2026-11-11T11:50Z] [--publish]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import satellite

log = logging.getLogger("wbwx.rainnow")

LAT_N, LAT_S, LON_W, LON_E = 37.0, 0.0, 66.0, 100.0
RES = 0.04
NY = int(round((LAT_N - LAT_S) / RES)) + 1
NX = int(round((LON_E - LON_W) / RES)) + 1
#: Segments of the full disk (10 bands from north to south) that hold 0 N to 37 N.
SEGMENTS = (2, 3, 4, 5, 6)
#: Scans every 10 minutes; all segments of a scan are on AWS about 13 minutes after it starts.
SCAN_LAG = dt.timedelta(minutes=14)
MAGIC = b"WBNOW1\n"
CREDIT = "JMA Himawari-9 (via NOAA open data)"


def scan_times(now: dt.datetime, back: int = 6) -> list[dt.datetime]:
    t = now - SCAN_LAG
    t = t.replace(minute=t.minute - t.minute % 10, second=0, microsecond=0)
    return [t - dt.timedelta(minutes=10 * i) for i in range(back)]


def download(t: dt.datetime, work: Path) -> list[Path]:
    prefix = f"AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/HS_H09_{t:%Y%m%d_%H%M}_B13_FLDK_R20_S"
    keys = sorted(k for k in satellite._s3_keys("noaa-himawari9", prefix) if int(k[-12:-10]) in SEGMENTS)
    if len(keys) < len(SEGMENTS):
        raise FileNotFoundError(f"Himawari-9 {t:%H:%M}: {len(keys)} of {len(SEGMENTS)} segments")
    import bz2

    out = []
    for k in keys:
        p = satellite._download(f"https://noaa-himawari9.s3.amazonaws.com/{k}", work / Path(k).name)
        raw = p.with_suffix("")
        raw.write_bytes(bz2.decompress(p.read_bytes()))
        p.unlink()
        out.append(raw)
    return out


def grid_lonlat() -> tuple[np.ndarray, np.ndarray]:
    lat = LAT_N - np.arange(NY) * RES
    lon = LON_W + np.arange(NX) * RES
    lon2, lat2 = np.meshgrid(lon, lat)
    return lon2, lat2


def to_grid(view: satellite.View) -> np.ndarray:
    """Cloud-top temperature (K) on the 0.04 degree grid, NaN beyond the usable view."""
    lon, lat = grid_lonlat()
    return satellite.composite([view], lon, lat)


def encode(bt: np.ndarray, t: dt.datetime, sub_lon: float) -> bytes:
    q = np.where(np.isfinite(bt), np.clip(np.rint(bt - 150), 1, 255), 0).astype(np.uint8)
    head = {
        "time": t.strftime("%Y-%m-%dT%H:%MZ"),
        "latN": LAT_N,
        "lonW": LON_W,
        "res": RES,
        "ny": NY,
        "nx": NX,
        "satellite": "himawari-9",
        "subLon": round(sub_lon, 2),
        "credit": CREDIT,
    }
    return MAGIC + json.dumps(head).encode() + b"\n" + gzip.compress(q.tobytes(), compresslevel=6, mtime=0)


def decode(data: bytes) -> tuple[dict, np.ndarray]:
    assert data.startswith(MAGIC)
    rest = data[len(MAGIC) :]
    nl = rest.index(b"\n")
    head = json.loads(rest[:nl])
    q = np.frombuffer(gzip.decompress(rest[nl + 1 :]), np.uint8).reshape(head["ny"], head["nx"])
    return head, np.where(q == 0, np.nan, q.astype(np.float32) + 150)


def build(t: dt.datetime, work: Path) -> bytes:
    work.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        files = download(t, work)
        view = satellite.read("himawari", files)
        bt = to_grid(view)
        log.info("Himawari %s: %.0f%% of the area seen, %.0fs", t, 100 * np.isfinite(bt).mean(), time.time() - t0)
        return encode(bt, t, view.sub_lon)
    finally:
        for f in work.glob("*"):
            f.unlink()


def main() -> None:
    from . import publish

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/rainnow"))
    ap.add_argument("--time", help="scan time, e.g. 2026-11-11T11:50Z (default: the newest on AWS)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    now = dt.datetime.now(dt.timezone.utc)
    times = [dt.datetime.strptime(a.time, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)] if a.time else scan_times(now)
    for t in times:
        if a.publish and publish.rain_now_live(t):
            log.info("scan %s is already live", t)
            return
        try:
            data = build(t, a.out / "work")
        except FileNotFoundError as e:
            log.info("%s", e)
            continue
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "rain-now.wbnow").write_bytes(data)
        log.info("scan %s: %.0f KB", t, len(data) / 1e3)
        if a.publish:
            publish.upload_rain_now(t, data)
        return
    raise SystemExit("no complete Himawari scan in the last hour")


if __name__ == "__main__":
    main()
