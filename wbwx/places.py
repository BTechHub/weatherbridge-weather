"""Place search data for the app, from GeoNames (CC BY 4.0, https://www.geonames.org).

Builds places.json.gz (format in the API's weather/places.ts) from the GeoNames dumps and uploads it:
cities500 (every place with 500+ people, about 200,000), admin1 names and country names.

    python -m wbwx.places --out build/places [--publish]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import time
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("wbwx.places")

DUMP = "https://download.geonames.org/export/dump/"
MAX_ALT_NAMES = 40
MAX_ALT_LEN = 60


def fetch(name: str) -> bytes:
    req = urllib.request.Request(DUMP + name, headers={"User-Agent": "WeatherBridge-places-job"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def countries(text: str) -> dict[str, list]:
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) > 16 and f[16].isdigit():
            out[f[0]] = [f[4], int(f[16])]
    return out


def admin1(text: str) -> dict[str, tuple[str, int]]:
    out = {}
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) >= 4 and f[3].isdigit():
            out[f[0]] = (f[1], int(f[3]))
    return out


def places(text: str, adm1: dict[str, tuple[str, int]]) -> list[list]:
    rows = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 19:
            continue
        gid, name, ascii_, alts, lat, lon, fclass, fcode, cc, _cc2, a1 = f[:11]
        pop, elev, dem, tz = f[14], f[15], f[16], f[17]
        a = adm1.get(f"{cc}.{a1}")
        alt = []
        seen = {name.lower()}
        for n in [ascii_, *alts.split(",")]:
            n = n.strip()
            if n and len(n) <= MAX_ALT_LEN and n.lower() not in seen and not n.startswith("http"):
                seen.add(n.lower())
                alt.append(n)
            if len(alt) >= MAX_ALT_NAMES:
                break
        height = int(elev) if elev.lstrip("-").isdigit() else int(dem) if dem.lstrip("-").isdigit() and dem != "-9999" else None
        rows.append([int(gid), name, round(float(lat), 5), round(float(lon), 5), height, fcode, cc, a[0] if a else None, a[1] if a else None, tz, int(pop or 0), alt])
    return rows


def build(out: Path, cities: str, adm1_text: str, country_text: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "formatVersion": 1,
        "source": "GeoNames cities500, www.geonames.org, CC BY 4.0",
        "builtAt": int(time.time()),
        "countries": countries(country_text),
        "places": places(cities, admin1(adm1_text)),
    }
    path = out / "places.json.gz"
    path.write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), compresslevel=9, mtime=0))
    log.info("places: %d rows, %.1f MB", len(data["places"]), path.stat().st_size / 1e6)
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="build/places")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    with zipfile.ZipFile(io.BytesIO(fetch("cities500.zip"))) as z:
        cities = z.read("cities500.txt").decode("utf-8")
    path = build(Path(a.out), cities, fetch("admin1CodesASCII.txt").decode("utf-8"), fetch("countryInfo.txt").decode("utf-8"))
    if a.publish:
        from . import publish

        publish.upload_places(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
