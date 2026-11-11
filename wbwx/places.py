"""Place search data for the app, from GeoNames (CC BY 4.0, https://www.geonames.org).

Builds places.json.gz (format in the API's weather/places.ts) from the GeoNames dumps and uploads it:
cities500 (every place with 500+ people, about 200,000), admin1 names and country names.
For the countries in ALL_TOWNS_COUNTRIES (the US), every populated place is added from the country's full dump,
so small towns and villages under 500 people are found too.

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
# Big cities have hundreds of other names (old names like Bombay or Madras, and every language's spelling).
# The first 40 alone dropped many of them, so cities this big keep up to MAX_ALT_NAMES_BIG.
BIG_CITY_POPULATION = 100_000
MAX_ALT_NAMES_BIG = 400
MAX_ALT_LEN = 60
# Countries where every town and village is listed, not only places with 500+ people.
ALL_TOWNS_COUNTRIES = ("US",)
# Populated places that are gone or not a place to live: historical, abandoned, destroyed, religious centres.
SKIP_CODES = {"PPLH", "PPLQ", "PPLW", "PPLCH", "PPLR"}


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
        limit = MAX_ALT_NAMES_BIG if int(pop or 0) >= BIG_CITY_POPULATION else MAX_ALT_NAMES
        alt = []
        seen = {name.lower()}
        for n in [ascii_, *alts.split(",")]:
            n = n.strip()
            if n and len(n) <= MAX_ALT_LEN and n.lower() not in seen and not n.startswith("http"):
                seen.add(n.lower())
                alt.append(n)
            if len(alt) >= limit:
                break
        height = int(elev) if elev.lstrip("-").isdigit() else int(dem) if dem.lstrip("-").isdigit() and dem != "-9999" else None
        rows.append([int(gid), name, round(float(lat), 5), round(float(lon), 5), height, fcode, cc, a[0] if a else None, a[1] if a else None, tz, int(pop or 0), alt])
    return rows


def towns(text: str) -> str:
    """The populated places (feature class P) of a full country dump, in the cities500 line format."""
    keep = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) >= 19 and f[6] == "P" and f[7] not in SKIP_CODES:
            keep.append(line)
    return "\n".join(keep)


def build(out: Path, cities: str, adm1_text: str, country_text: str, extra: list[str] | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    adm1 = admin1(adm1_text)
    rows = places(cities, adm1)
    seen = {r[0] for r in rows}
    for text in extra or []:
        for r in places(text, adm1):
            if r[0] not in seen:
                seen.add(r[0])
                rows.append(r)
    data = {
        "formatVersion": 1,
        "source": "GeoNames cities500 and all US towns, www.geonames.org, CC BY 4.0",
        "builtAt": int(time.time()),
        "countries": countries(country_text),
        "places": rows,
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
    extra = []
    for cc in ALL_TOWNS_COUNTRIES:
        with zipfile.ZipFile(io.BytesIO(fetch(f"{cc}.zip"))) as z:
            extra.append(towns(z.read(f"{cc}.txt").decode("utf-8")))
        log.info("places: %s full list read", cc)
    path = build(Path(a.out), cities, fetch("admin1CodesASCII.txt").decode("utf-8"), fetch("countryInfo.txt").decode("utf-8"), extra)
    if a.publish:
        from . import publish

        publish.upload_places(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
