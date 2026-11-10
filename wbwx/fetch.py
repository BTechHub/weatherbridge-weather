"""Byte-range downloads of single GRIB2 fields using the .idx inventory next to each file."""
from __future__ import annotations

import time
from dataclasses import dataclass

import eccodes
import numpy as np
import requests

from .tiles import NX, NY

_session = requests.Session()
_session.headers["User-Agent"] = "WeatherBridge-weather-job (+https://www.lorjova.com/weatherbridge/)"


def get(url: str, headers: dict | None = None, tries: int = 4) -> requests.Response:
    for attempt in range(tries):
        try:
            r = _session.get(url, headers=headers, timeout=60)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                raise FileNotFoundError(url)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** (attempt + 1))
    raise IOError(f"GET {url} failed")


def exists(url: str) -> bool:
    try:
        return _session.head(url, timeout=30).status_code == 200
    except requests.RequestException:
        return False


@dataclass
class Entry:
    start: int
    end: int | None  # inclusive, None for the last message
    text: str  # e.g. "d=2026111000:TMP:2 m above ground:3 hour fcst:"


def inventory(url: str) -> list[Entry]:
    lines = [l for l in get(url + ".idx").text.splitlines() if l.strip()]
    starts = [int(l.split(":")[1]) for l in lines]
    out = []
    for i, l in enumerate(lines):
        end = starts[i + 1] - 1 if i + 1 < len(lines) else None
        out.append(Entry(starts[i], end, l.split(":", 2)[2]))
    return out


def pick(inv: list[Entry], name_level: str, kind: str = "inst") -> Entry | None:
    """name_level like "TMP:2 m above ground". kind: inst (not an average or accumulation), acc, any."""
    for e in inv:
        parts = e.text.split(":")
        if f"{parts[1]}:{parts[2]}" != name_level:
            continue
        t = parts[3]
        if kind == "inst" and (" ave " in f" {t} " or " acc " in f" {t} "):
            continue
        if kind == "acc" and " acc " not in f" {t} ":
            continue
        return e
    return None


def field(url: str, e: Entry) -> tuple[np.ndarray, dict]:
    """Downloads and decodes one message to float32 [NY][NX] with NaN for missing."""
    rng = f"bytes={e.start}-{'' if e.end is None else e.end}"
    msg = get(url, headers={"Range": rng}).content
    h = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        if (ni, nj) != (NX, NY) or eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees") != 90.0:
            raise ValueError(f"unexpected grid {ni}x{nj} in {url}")
        miss = eccodes.codes_get(h, "missingValue")
        info = {"stepRange": eccodes.codes_get(h, "stepRange", ktype=str)}
        v = eccodes.codes_get_values(h).astype(np.float32)
        if eccodes.codes_get(h, "bitmapPresent"):
            v[v == miss] = np.nan
        return v.reshape(NY, NX), info
    finally:
        eccodes.codes_release(h)
