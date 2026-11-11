"""Builds one dataset run: download every step, derive fields, cut tiles, write a manifest.

    python -m wbwx.job gfs --out build            # newest complete cycle, worldwide
    python -m wbwx.job gfswave --out build --publish
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import blend, fetch
from .models import DATASETS, Dataset
from .tiles import MISSING, NX, NY, TILE, TILE_COLS, TILE_ROWS, all_missing, encode, quantize, tile_id

log = logging.getLogger("wbwx")
FORMAT_VERSION = 1
WET_MM_PER_H = 0.1
POP_RADIUS = 3  # cells each side: a 7 x 7 neighbourhood, about 175 km


def candidate_cycles(now: dt.datetime, back_hours: int = 30) -> list[dt.datetime]:
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 6)
    return [base - dt.timedelta(hours=6 * i) for i in range(back_hours // 6 + 1)]


def newest_complete(ds: Dataset, now: dt.datetime, steps: tuple[int, ...]) -> dt.datetime | None:
    for c in candidate_cycles(now):
        if fetch.exists(ds.url(c.strftime("%Y%m%d"), c.hour, steps[-1]) + ".idx"):
            return c
    return None


_ACC = re.compile(r"(\d+)-(\d+) hour acc")


def precipitation_rates(acc: np.ndarray, ranges: list[tuple[int, int] | None], steps: tuple[int, ...]) -> np.ndarray:
    """APCP comes as accumulations that restart every few hours. Returns mm/h over each step's interval."""
    out = np.full(acc.shape, np.nan, dtype=np.float32)
    prev_range: tuple[int, int] | None = None
    for i, r in enumerate(ranges):
        if r is None:
            prev_range = None
            continue
        a, b = r
        if prev_range and prev_range[0] == a:  # same bucket as the previous step: take the difference
            amount = acc[i] - acc[i - 1]
            hours = b - prev_range[1]
        else:  # a new bucket starts at a
            amount = acc[i]
            hours = b - a
        if hours > 0:
            out[i] = np.maximum(amount, 0) / hours
        prev_range = r
    return out


# Blended fields (names in wbwx/blend.py) and the GFS variables they replace.
BLEND_TO_GFS = {"cloud": "cloud_cover", "u": "wind_u_10m", "v": "wind_v_10m", "gust": "wind_gusts_10m"}


def physical(q: np.ndarray, scale: float) -> np.ndarray:
    q = np.asarray(q)
    return np.where(q == MISSING, np.nan, q / scale).astype(np.float32)


def wet_fraction(rate: np.ndarray | None) -> np.ndarray | None:
    """Share of the cells around each point with at least WET_MM_PER_H (None when the step has no rain data)."""
    if rate is None or not np.isfinite(rate).any():
        return None
    return box_mean((np.nan_to_num(rate) >= WET_MM_PER_H).astype(np.float32), POP_RADIUS)


def box_mean(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 box; longitude wraps around, latitude edges are clamped."""
    p = np.pad(x, ((r, r), (0, 0)), mode="edge")
    p = np.pad(p, ((0, 0), (r, r)), mode="wrap")
    c = np.cumsum(np.cumsum(p, axis=0, dtype=np.float32), axis=1, dtype=np.float32)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return s / (n * n)


def build(ds: Dataset, cycle: dt.datetime, work: Path, steps: tuple[int, ...], workers: int = 16) -> Path:
    day, hh = cycle.strftime("%Y%m%d"), cycle.hour
    run_dir = work / ds.name / cycle.strftime("%Y%m%d%H")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "tiles").mkdir(parents=True)
    scratch = work / "scratch" / ds.name
    scratch.mkdir(parents=True, exist_ok=True)
    T = len(steps)
    scale_of = {v.name: v.scale for v in ds.vars}
    src_vars = [v for v in ds.vars if v.source]
    store = {v.name: np.lib.format.open_memmap(scratch / f"{v.name}.npy", "w+", np.int16, (T, NY, NX)) for v in ds.vars}
    for m in store.values():
        m[:] = MISSING
    acc_var = next((v for v in src_vars if v.kind == "acc"), None)
    acc = np.lib.format.open_memmap(scratch / "acc.npy", "w+", np.float32, (T, NY, NX)) if acc_var else None
    acc_ranges: list[tuple[int, int] | None] = [None] * T
    missing_fields: list[str] = []

    def one_step(i: int) -> None:
        url = ds.url(day, hh, steps[i])
        inv = fetch.inventory(url)
        for v in src_vars:
            e = fetch.pick(inv, v.source, v.kind)
            if e is None:
                missing_fields.append(f"f{steps[i]:03d}:{v.name}")
                continue
            values, _ = fetch.field(url, e)
            if v.kind == "acc":
                m = _ACC.search(e.text)
                if m:
                    acc[i] = values
                    acc_ranges[i] = (int(m.group(1)), int(m.group(2)))
                continue
            store[v.name][i] = quantize(v.convert(values), v.scale)

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(one_step, range(T)), 1):
            if n % 20 == 0:
                log.info("%s %s: %d/%d steps (%.0fs)", ds.name, cycle, n, T, time.time() - t0)

    sources: list[dict] = []
    if acc_var:
        rates = precipitation_rates(acc, acc_ranges, steps)
        loaded = blend.load_all(cycle, steps, scratch / "blend") if ds.blend else []
        sources = [lm.info() for lm in loaded]
        wet: list[np.ndarray | None] = [None] * T
        weights = {"gfs": blend.WEIGHTS["gfs"], **{lm.model.name: blend.WEIGHTS[lm.model.name] for lm in loaded}}
        gaps = blend.intervals(steps)
        for i in range(T):
            per: dict = {"gfs": rates[i]}
            if loaded:
                gfs = {"precip": rates[i].copy(), **{b: physical(store[n][i], scale_of[n]) for b, n in BLEND_TO_GFS.items()}}
                mixed, per = blend.mix_step(gfs, loaded, steps[i], gaps[i])
                rates[i] = mixed["precip"]
                for b, n in BLEND_TO_GFS.items():
                    store[n][i] = quantize(mixed[b], scale_of[n])
            fractions = {m: wet_fraction(r) for m, r in per.items()}
            if any(f is not None for f in fractions.values()):
                wet[i] = blend.weighted(fractions, weights)
        if loaded:
            log.info("%s %s: mixed with %s", ds.name, cycle, ", ".join(f"{x['model']} {dt.datetime.fromtimestamp(x['run'], dt.timezone.utc):%d %H}z" for x in sources))
        for i in range(T):
            store[acc_var.name][i] = quantize(rates[i], acc_var.scale)
        if "precipitation_probability" in store:
            ens = blend.load_ens(cycle, steps, scratch / "ens", WET_MM_PER_H) if ds.blend else None
            if ens:
                sources.append(ens.info())
            for i in range(T):
                around = [w for w in wet[max(0, i - 1) : i + 2] if w is not None]
                if wet[i] is not None and around:
                    near = np.mean(around, axis=0)
                    share = blend.ens_pop_at(ens, steps[i], gaps[i]) if ens else None
                    store["precipitation_probability"][i] = quantize(100 * blend.mix_pop(near, share), 1)
        del rates
    if ds.derive:
        scale = {v.name: v.scale for v in ds.vars}
        inputs = [v.name for v in ds.vars if v.source]
        for i in range(T):
            phys = {}
            for n in inputs:
                q = np.asarray(store[n][i])
                phys[n] = np.where(q == MISSING, np.nan, q / scale[n]).astype(np.float32)
            for n, values in ds.derive(phys).items():
                store[n][i] = quantize(values, scale[n])
    if ds.zero_if_missing:
        ref = store[ds.reference]
        for i in range(T):
            sea = np.asarray(ref[i]) != MISSING
            for n in ds.zero_if_missing:
                layer = store[n][i]
                layer[sea & (layer == MISSING)] = 0
    if "land" in store:  # the land mask does not change: fill steps where it was not sent
        land = store["land"]
        have = [i for i in range(T) if (land[i] != MISSING).any()]
        if have:
            ref = np.array(land[have[0]])
            for i in range(T):
                if i not in have:
                    land[i] = ref
    sea_only = [v.name for v in ds.vars if v.sea_only]
    if sea_only and "land" in store:
        for i in range(T):
            on_land = np.asarray(store["land"][i]) > 0
            for n in sea_only:
                store[n][i][on_land] = MISSING
    for m in store.values():
        m.flush()

    kept = [v for v in ds.vars if v.stored]
    names = [v.name for v in kept]
    tiles: list[str] = []
    for R in range(TILE_ROWS):
        band = np.stack([np.asarray(store[n][:, R * TILE : (R + 1) * TILE, :]) for n in names])  # V,T,20,NX
        for C in range(TILE_COLS):
            block = band[:, :, :, C * TILE : (C + 1) * TILE].reshape(len(names), T, TILE * TILE)
            if ds.skip_empty_tiles and all_missing(block):
                continue
            tid = tile_id(R, C)
            (run_dir / "tiles" / f"{tid}.bin").write_bytes(encode(np.ascontiguousarray(block)))
            tiles.append(tid)
    shutil.rmtree(scratch, ignore_errors=True)

    size = sum(p.stat().st_size for p in (run_dir / "tiles").iterdir())
    manifest = {
        "formatVersion": FORMAT_VERSION,
        "dataset": ds.name,
        "title": ds.title + "".join(f" + {x['title']}" for x in sources),
        "licence": "; ".join(dict.fromkeys([ds.licence] + [x["licence"] for x in sources])),
        "sources": sources,
        "cycle": int(cycle.timestamp()),
        "steps": list(steps),
        "vars": [{"name": v.name, "scale": v.scale, **({"circular": True} if v.circular else {})} for v in kept],
        "grid": {"res": 0.25, "ny": NY, "nx": NX, "lat0": 90.0, "lon0": 0.0, "tile": TILE},
        "tiles": tiles,
        "missingFields": sorted(missing_fields)[:200],
        "processedAt": int(time.time()),
        "bytes": size,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    log.info("%s %s: %d tiles, %.1f MB in %.0fs", ds.name, cycle, len(tiles), size / 1e6, time.time() - t0)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--out", default="build")
    ap.add_argument("--cycle", help="YYYYMMDDHH (default: newest complete cycle)")
    ap.add_argument("--max-step", type=int, help="testing: stop at this forecast hour")
    ap.add_argument("--publish", action="store_true", help="upload to WB_WEATHER_INGEST_URL")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="publish again even if this cycle is already live (after a fix)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = DATASETS[a.dataset]
    steps = tuple(s for s in ds.steps if a.max_step is None or s <= a.max_step)
    now = dt.datetime.now(dt.timezone.utc)
    cycle = dt.datetime.strptime(a.cycle, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc) if a.cycle else newest_complete(ds, now, steps)
    if cycle is None:
        log.error("no complete %s cycle in the last 30 hours", ds.name)
        return 2
    if a.publish:
        from . import publish

        if not a.force and publish.is_live(ds.name, int(cycle.timestamp())):
            log.info("%s %s is already live, nothing to do", ds.name, cycle)
            return 0
    run_dir = build(ds, cycle, Path(a.out), steps, a.workers)
    if a.publish:
        from . import publish

        publish.upload(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
