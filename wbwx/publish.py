"""Uploads a built run to api.lorjova.com (routes in the lorjova API, docs/WEATHERBRIDGE_WEATHER.md)."""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger("wbwx")


def _base() -> tuple[str, dict]:
    url = os.environ["WB_WEATHER_INGEST_URL"].rstrip("/")  # e.g. https://api.lorjova.com/weatherbridge/ingest/weather
    token = os.environ["WB_WEATHER_INGEST_TOKEN"]
    return url, {"Authorization": f"Bearer {token}", "User-Agent": "WeatherBridge-weather-job"}


def _call(method: str, url: str, headers: dict, tries: int = 5, **kw) -> requests.Response:
    for attempt in range(tries):
        try:
            r = requests.request(method, url, headers=headers, timeout=120, **kw)
            if r.status_code < 500 and r.status_code != 429:
                if r.status_code >= 400:
                    raise RuntimeError(f"{method} {url}: {r.status_code} {r.text[:300]}")
                return r
        except requests.RequestException:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"{method} {url} kept failing")


def is_live(dataset: str, cycle: int) -> bool:
    url, h = _base()
    status = _call("GET", f"{url}/status", h).json()
    return any(r.get("cycle", 0) >= cycle for r in status.get("datasets", {}).get(dataset, []))


def upload(run_dir: Path, workers: int = 6) -> None:
    url, h = _base()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_id = _call("POST", f"{url}/runs", h, json={"manifest": manifest}).json()["runId"]
    log.info("upload %s run %s: %d tiles", manifest["dataset"], run_id, len(manifest["tiles"]))
    oh = {**h, "Content-Type": "application/octet-stream"}

    def put(tid: str) -> None:
        _call("PUT", f"{url}/runs/{run_id}/tiles/{tid}", oh, data=(run_dir / "tiles" / f"{tid}.bin").read_bytes())

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, manifest["tiles"]))
    res = _call("POST", f"{url}/runs/{run_id}/finish", h, json={"tiles": len(manifest["tiles"])}).json()
    log.info("upload done in %.0fs: %s", time.time() - t0, res)


def upload_places(path: Path) -> None:
    """The place list goes up in one request; the API swaps it in at once."""
    url, h = _base()
    base = url.rsplit("/weather", 1)[0]  # .../weatherbridge/ingest
    res = _call("PUT", f"{base}/places", {**h, "Content-Type": "application/octet-stream"}, data=path.read_bytes()).json()
    log.info("places upload: %s", res)


# --- Map layers (satellite, radar): one packed frame per upload, sent in parts of at most 6 MB. ---

SKY_PART_BYTES = 6 * 1024 * 1024


def _sky_base() -> tuple[str, dict]:
    url, h = _base()
    return url.rsplit("/weather", 1)[0] + "/sky", h  # .../weatherbridge/ingest/sky


def frame_id(t) -> str:
    return t.strftime("%Y%m%d%H%M")


def sky_frame_live(layer: str, t) -> bool:
    url, h = _sky_base()
    frames = _call("GET", f"{url}/status", h).json().get("layers", {}).get(layer, [])
    return any(f.get("id") == frame_id(t) for f in frames)


def upload_sky(layer: str, t, data: bytes) -> None:
    import hashlib

    url, h = _sky_base()
    fid = frame_id(t)
    oh = {**h, "Content-Type": "application/octet-stream"}
    parts = [data[i : i + SKY_PART_BYTES] for i in range(0, len(data), SKY_PART_BYTES)]
    t0 = time.time()
    for n, part in enumerate(parts):
        _call("PUT", f"{url}/{layer}/{fid}/parts/{n}", oh, data=part)
    res = _call(
        "POST", f"{url}/{layer}/{fid}/finish", h, json={"parts": len(parts), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    ).json()
    log.info("%s frame %s: %.1f MB in %d parts, %.0fs: %s", layer, fid, len(data) / 1e6, len(parts), time.time() - t0, res)
