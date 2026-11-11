"""Blends two more free models into the GFS forecast: ECMWF IFS open data and DWD ICON Global.

GFS on its own over-forecasts light showers in the tropics (for example Kerala), so rain, rain chance,
cloud and wind are taken from a weighted mix of the three models. Everything else (temperature,
humidity, pressure, visibility, CAPE, rain/snow type) stays GFS. The output is still the "gfs"
dataset with the same variables, so the API and the app need no change.

- ECMWF IFS 0.25 degree open data (CC BY 4.0), from ECMWF's own bucket on AWS (ecmwf-forecasts).
  Credit: "Contains modified ECMWF open data (CC BY 4.0)" / "Copyright ECMWF".
- DWD ICON Global, 13 km icosahedral grid, from opendata.dwd.de (reuse with source note under
  GeoNutzV). Credit: "Source: Deutscher Wetterdienst (DWD), ICON". Moved onto the GFS 0.25 grid here.

Each model is matched by valid time, so model runs need not start at the same hour. A model that is
missing (download failed, run too old, past its last step) simply drops out and the others' weights
are scaled up; with no extra model the result is plain GFS, as before.

    python -m wbwx.blend check --lat 8.52 --lon 76.94      # each model and the mix at one place, next 48 h
"""
from __future__ import annotations

import argparse
import bz2
import concurrent.futures as cf
import datetime as dt
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import eccodes
import numpy as np

from . import fetch
from .tiles import NX, NY, RES

log = logging.getLogger("wbwx")

# Share of each model in the mix. ECMWF scores best on rain and wind worldwide; GFS gets the
# smallest share because of its tropical drizzle bias.
WEIGHTS = {"gfs": 0.25, "ecmwf": 0.45, "icon": 0.30}
MAX_RUN_AGE_H = 24  # an extra model run older than this (against the GFS cycle) is not used

ECMWF_BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
DWD_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"

# Fields of each model, under common names: precip = accumulated mm since the run started (kind acc);
# the rest are values at the step (kind inst). Unit conversion to GFS units is applied on download.
BLEND_VARS = ("precip", "cloud", "u", "v", "gust")


@dataclass
class Model:
    name: str
    title: str
    licence: str
    credit: str
    params: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]]  # our name -> (model param, convert)
    steps_for: Callable[[int], tuple[int, ...]]  # run hour -> forecast steps published
    workers: int = 6

    def run_is_complete(self, run: dt.datetime) -> bool:
        raise NotImplementedError

    def load(self, run: dt.datetime, step: int, names: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError


def ecmwf_steps(hour: int) -> tuple[int, ...]:
    short = tuple(range(0, 145, 3))
    return short if hour in (6, 18) else short + tuple(range(150, 361, 6))


class Ecmwf(Model):
    def _url(self, run: dt.datetime, step: int) -> str:
        d, h = run.strftime("%Y%m%d"), run.hour
        return f"{ECMWF_BASE}/{d}/{h:02d}z/ifs/0p25/oper/{d}{h:02d}0000-{step}h-oper-fc"

    def run_is_complete(self, run: dt.datetime) -> bool:
        return fetch.exists(self._url(run, self.steps_for(run.hour)[-1]) + ".index")

    def load(self, run: dt.datetime, step: int, names: list[str]) -> dict[str, np.ndarray]:
        url = self._url(run, step)
        index = [json.loads(line) for line in fetch.get(url + ".index", tries=8).text.splitlines() if line.strip()]
        out = {}
        for n in names:
            param, convert = self.params[n]
            e = next((x for x in index if x.get("param") == param and x.get("levtype") == "sfc"), None)
            if e is None:
                if step == 0 and n in ("precip", "gust"):  # not sent at step 0 (precipitation is 0 there)
                    out[n] = np.zeros((NY, NX), np.float32)
                continue
            rng = f"bytes={e['_offset']}-{e['_offset'] + e['_length'] - 1}"
            v = decode_regular(fetch.get(url + ".grib2", headers={"Range": rng}, tries=8).content)
            out[n] = convert(v)
        return out


def icon_steps(hour: int) -> tuple[int, ...]:
    # Hourly to 78 h, then 3-hourly to 180 h (120 h for 06 and 18 UTC); 3-hourly is enough here.
    return tuple(range(0, (181 if hour in (0, 12) else 121), 3))


class Icon(Model):
    _grid: "IconGrid | None" = None
    _grid_lock = threading.Lock()

    def _url(self, run: dt.datetime, step: int | None, param: str) -> str:
        r, h = run.strftime("%Y%m%d%H"), run.hour
        if step is None:
            return f"{DWD_BASE}/{h:02d}/{param.lower()}/icon_global_icosahedral_time-invariant_{r}_{param}.grib2.bz2"
        return f"{DWD_BASE}/{h:02d}/{param.lower()}/icon_global_icosahedral_single-level_{r}_{step:03d}_{param}.grib2.bz2"

    def run_is_complete(self, run: dt.datetime) -> bool:
        last = self.steps_for(run.hour)[-1]
        return all(fetch.exists(self._url(run, last, p)) for p, _ in self.params.values())

    def grid(self, run: dt.datetime) -> "IconGrid":
        with self._grid_lock:
            if self._grid is None:
                lat = decode_any(bz2.decompress(fetch.get(self._url(run, None, "CLAT")).content))
                lon = decode_any(bz2.decompress(fetch.get(self._url(run, None, "CLON")).content))
                self._grid = IconGrid(lat, lon)
            return self._grid

    def load(self, run: dt.datetime, step: int, names: list[str]) -> dict[str, np.ndarray]:
        g = self.grid(run)
        out = {}
        for n in names:
            param, convert = self.params[n]
            try:
                raw = bz2.decompress(fetch.get(self._url(run, step, param)).content)
            except FileNotFoundError:
                if step == 0 and n == "precip":
                    out[n] = np.zeros((NY, NX), np.float32)
                continue
            out[n] = convert(g.regrid(decode_any(raw)))
        return out


def _f(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


ECMWF = Ecmwf(
    name="ecmwf",
    title="ECMWF IFS 0.25 degree open data",
    licence="ECMWF open data, CC BY 4.0",
    credit="Contains modified ECMWF open data (CC BY 4.0), copyright ECMWF",
    params={
        "precip": ("tp", lambda v: _f(v * 1000)),  # m -> mm
        "cloud": ("tcc", lambda v: _f(v * 100)),  # 0..1 -> %
        "u": ("10u", _f),
        "v": ("10v", _f),
        "gust": ("10fg", _f),
    },
    steps_for=ecmwf_steps,
    workers=4,  # the bucket answers "SlowDown" to many parallel requests
)

ICON = Icon(
    name="icon",
    title="DWD ICON Global 13 km",
    licence="Deutscher Wetterdienst open data (GeoNutzV, source note required)",
    credit="Source: Deutscher Wetterdienst (DWD), ICON model data, modified",
    params={
        "precip": ("TOT_PREC", _f),  # kg/m2 = mm, from the start of the run
        "cloud": ("CLCT", _f),  # %
        "u": ("U_10M", _f),
        "v": ("V_10M", _f),
    },
    steps_for=icon_steps,
    workers=8,
)

MODELS = (ECMWF, ICON)


# --- GRIB decoding and grids ---


def decode_any(msg: bytes) -> np.ndarray:
    h = eccodes.codes_new_from_message(msg)
    try:
        v = eccodes.codes_get_values(h).astype(np.float32)
        if eccodes.codes_get(h, "bitmapPresent"):
            v[v == eccodes.codes_get(h, "missingValue")] = np.nan
        return v
    finally:
        eccodes.codes_release(h)


def decode_regular(msg: bytes) -> np.ndarray:
    """A 0.25 degree global lat-lon field to the GFS layout (90N first, 0E first)."""
    h = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        if (ni, nj) != (NX, NY) or eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees") != 90.0:
            raise ValueError(f"unexpected grid {ni}x{nj}")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees") % 360
        v = eccodes.codes_get_values(h).astype(np.float32).reshape(NY, NX)
        if eccodes.codes_get(h, "bitmapPresent"):
            v[v == eccodes.codes_get(h, "missingValue")] = np.nan
    finally:
        eccodes.codes_release(h)
    return np.roll(v, int(round(lon0 / RES)), axis=1)  # ECMWF starts at 180W: column 0 must be 0E


class IconGrid:
    """ICON cells to the 0.25 degree grid: mean of the cells inside each box, nearest cell where none is."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:  # some files give radians
            lat, lon = np.degrees(lat), np.degrees(lon)
        j = np.clip(np.rint((90 - lat) / RES).astype(np.int64), 0, NY - 1)
        i = np.rint((lon % 360) / RES).astype(np.int64) % NX
        self.box = j * NX + i
        self.count = np.bincount(self.box, minlength=NY * NX).astype(np.float32)
        empty = np.flatnonzero(self.count == 0)
        self.empty = empty
        self.fill_from = np.zeros(0, np.int64)
        if len(empty):
            from scipy.spatial import cKDTree

            tree = cKDTree(_xyz(lat, lon))
            elat = 90 - (empty // NX) * RES
            elon = (empty % NX) * RES
            self.fill_from = tree.query(_xyz(elat, elon))[1]

    def regrid(self, values: np.ndarray) -> np.ndarray:
        ok = np.isfinite(values)
        s = np.bincount(self.box[ok], weights=values[ok], minlength=NY * NX)
        c = np.bincount(self.box[ok], minlength=NY * NX)
        out = np.full(NY * NX, np.nan, np.float32)
        has = c > 0
        out[has] = s[has] / c[has]
        if len(self.empty):
            out[self.empty] = values[self.fill_from]
        return out.reshape(NY, NX)


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


# --- Time matching ---


def interp_inst(src_steps: list[float], fields: Callable[[int], np.ndarray], t: float) -> np.ndarray | None:
    """Value at t hours from the run start, linear between the two steps around it."""
    if not src_steps or t < src_steps[0] or t > src_steps[-1]:
        return None
    k = int(np.searchsorted(src_steps, t))
    if src_steps[k] == t:
        return np.asarray(fields(k), dtype=np.float32)
    a, b = src_steps[k - 1], src_steps[k]
    w = (t - a) / (b - a)
    return (1 - w) * np.asarray(fields(k - 1), dtype=np.float32) + w * np.asarray(fields(k), dtype=np.float32)


def rate_between(src_steps: list[float], acc: Callable[[int], np.ndarray], t1: float, t2: float) -> np.ndarray | None:
    """mm/h between t1 and t2 (hours from run start) from totals since the start. Between two steps the
    rain is spread evenly, so a 3-hourly model gives each hour inside its interval the same rate."""
    a1, a2 = interp_inst(src_steps, acc, t1), interp_inst(src_steps, acc, t2)
    if a1 is None or a2 is None or t2 <= t1:
        return None
    return np.maximum(a2 - a1, 0) / (t2 - t1)


def weighted(values: dict[str, np.ndarray | None], weights: dict[str, float]) -> np.ndarray:
    """Weighted mean per cell over the models that have a value there."""
    s = w = None
    for m, v in values.items():
        if v is None:
            continue
        ok = np.isfinite(v)
        part = np.where(ok, v, 0) * weights[m]
        wt = ok * np.float32(weights[m])
        s = part if s is None else s + part
        w = wt if w is None else w + wt
    if s is None:
        return np.full((NY, NX), np.nan, np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, s / np.where(w > 0, w, 1), np.nan).astype(np.float32)


def blend_wind(u: dict, v: dict, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Speed is the weighted mean of each model's speed; direction is that of the weighted mean vector.
    (Averaging u and v alone would make the wind weaker wherever the models disagree on direction.)"""
    speed = weighted({m: None if u[m] is None or v[m] is None else np.hypot(u[m], v[m]) for m in u}, weights)
    mu, mv = weighted(u, weights), weighted(v, weights)
    norm = np.hypot(mu, mv)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(norm > 1e-3, speed / np.where(norm > 1e-3, norm, 1), 0)
    return (mu * k).astype(np.float32), (mv * k).astype(np.float32)


# --- Loading a model run onto disk ---


@dataclass
class Loaded:
    model: Model
    run: dt.datetime
    offset_h: float  # GFS cycle minus this run, in hours: GFS step s is this run's step s + offset_h
    steps: list[float]
    data: dict[str, np.ndarray] = field(default_factory=dict)  # name -> [step][NY][NX] float32 (memmap)
    good: dict[str, list[int]] = field(default_factory=dict)  # name -> indexes of the steps that downloaded

    def info(self) -> dict:
        return {"model": self.model.name, "title": self.model.title, "run": int(self.run.timestamp()), "licence": self.model.licence}


def pick_run(model: Model, cycle: dt.datetime, now: dt.datetime) -> dt.datetime | None:
    """Newest complete run no more than MAX_RUN_AGE_H older than the GFS cycle (a newer one is fine too)."""
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 6)
    for i in range(6):
        run = base - dt.timedelta(hours=6 * i)
        if (cycle - run).total_seconds() / 3600 > MAX_RUN_AGE_H:
            break
        if model.run_is_complete(run):
            return run
    return None


def load_model(model: Model, run: dt.datetime, cycle: dt.datetime, need_h: tuple[float, float], scratch: Path) -> Loaded:
    """Downloads the steps covering GFS hours need_h (inclusive) into memmaps on disk."""
    off = (cycle - run).total_seconds() / 3600
    all_steps = model.steps_for(run.hour)
    lo, hi = need_h[0] + off, need_h[1] + off
    first = max([s for s in all_steps if s <= lo], default=all_steps[0])
    last = min([s for s in all_steps if s >= hi], default=all_steps[-1])
    steps = [s for s in all_steps if first <= s <= last]
    names = list(model.params)
    scratch.mkdir(parents=True, exist_ok=True)
    out = Loaded(model, run, off, [float(s) for s in steps])
    for n in names:
        m = np.lib.format.open_memmap(scratch / f"{model.name}_{n}.npy", "w+", np.float32, (len(steps), NY, NX))
        m[:] = np.nan
        out.data[n] = m
    t0 = time.time()

    def one(k: int) -> None:
        for n, values in model.load(run, steps[k], names).items():
            out.data[n][k] = values

    with cf.ThreadPoolExecutor(model.workers) as pool:
        futures = [pool.submit(one, k) for k in range(len(steps))]
        failed = 0
        for f in futures:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - one bad step only removes that step
                failed += 1
                log.warning("%s %s: a step failed: %s", model.name, run, e)
    for n in names:
        out.good[n] = [k for k in range(len(steps)) if np.isfinite(out.data[n][k]).any()]
    if failed > len(steps) // 4:
        raise RuntimeError(f"{model.name}: {failed} of {len(steps)} steps failed")
    log.info("%s run %s: %d steps loaded in %.0fs", model.name, run, len(steps), time.time() - t0)
    return out


def at_step(lm: Loaded, name: str, gfs_step: float, interval_h: float | None) -> np.ndarray | None:
    """This model's value for one GFS step: a rate over the interval for precip, else the value at the step."""
    if name not in lm.data:
        return None
    arr, good = lm.data[name], lm.good[name]
    steps = [lm.steps[k] for k in good]
    get = lambda i: arr[good[i]]  # noqa: E731
    t = gfs_step + lm.offset_h
    if name == "precip":
        if not interval_h:
            return None
        return rate_between(steps, get, t - interval_h, t)
    return interp_inst(steps, get, t)


def load_all(cycle: dt.datetime, steps: tuple[int, ...], scratch: Path, now: dt.datetime | None = None) -> list[Loaded]:
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    for model in MODELS:
        try:
            run = pick_run(model, cycle, now)
            if run is None:
                log.warning("%s: no complete run within %d h of %s, left out", model.name, MAX_RUN_AGE_H, cycle)
                continue
            out.append(load_model(model, run, cycle, (max(0, steps[0] - 3), steps[-1]), scratch / model.name))
        except Exception as e:  # noqa: BLE001 - a model that fails is left out; GFS alone still goes out
            log.warning("%s left out: %s", model.name, e)
    return out


def intervals(steps: tuple[int, ...]) -> list[float | None]:
    return [None if i == 0 else float(steps[i] - steps[i - 1]) for i in range(len(steps))]


def mix_step(gfs: dict[str, np.ndarray], loaded: list[Loaded], step: int, interval_h: float | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | None]]:
    """The mix for one GFS step. gfs has precip (mm/h), cloud, u, v, gust in physical units.
    Returns (mixed fields, each model's rain rate for the rain chance)."""
    w = {"gfs": WEIGHTS["gfs"], **{lm.model.name: WEIGHTS[lm.model.name] for lm in loaded}}
    per = {n: {"gfs": gfs.get(n)} for n in BLEND_VARS}
    for lm in loaded:
        for n in BLEND_VARS:
            per[n][lm.model.name] = at_step(lm, n, step, interval_h)
    out = {}
    out["precip"] = weighted(per["precip"], w)
    out["cloud"] = np.clip(weighted(per["cloud"], w), 0, 100)
    out["gust"] = weighted(per["gust"], w)
    out["u"], out["v"] = blend_wind(per["u"], per["v"], w)
    # Gusts are never below the mean wind.
    out["gust"] = np.where(np.isfinite(out["gust"]), np.maximum(out["gust"], np.hypot(out["u"], out["v"])), np.nan)
    return out, per["precip"]


# --- ECMWF ensemble: the rain chance ---

ENS_SHARE = 0.6  # rain chance = 60% the ensemble's share of wet members, 40% the models' wet neighbourhood


def ens_steps(hour: int) -> tuple[int, ...]:
    return ecmwf_steps(hour)


def _ens_url(run: dt.datetime, step: int) -> str:
    d, h = run.strftime("%Y%m%d"), run.hour
    return f"{ECMWF_BASE}/{d}/{h:02d}z/ifs/0p25/enfo/{d}{h:02d}0000-{step}h-enfo-ef"


def _members(run: dt.datetime, step: int, workers: int) -> dict[str, np.ndarray]:
    """Total rain since the run start (mm) of every ensemble member at one step, by member number."""
    url = _ens_url(run, step)
    index = [json.loads(line) for line in fetch.get(url + ".index", tries=8).text.splitlines() if line.strip()]
    rows = [r for r in index if r.get("param") == "tp" and r.get("levtype") == "sfc"]

    def one(r: dict) -> tuple[str, np.ndarray]:
        rng = f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"
        return str(r.get("number", "0")), decode_regular(fetch.get(url + ".grib2", headers={"Range": rng}, tries=8).content) * 1000

    with cf.ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, rows))


def wet_share(prev: dict[str, np.ndarray], cur: dict[str, np.ndarray], hours: float, wet_mm_per_h: float) -> np.ndarray | None:
    """Share of members with at least wet_mm_per_h between two steps."""
    both = sorted(set(prev) & set(cur))
    if not both or hours <= 0:
        return None
    n = np.zeros((NY, NX), np.float32)
    for m in both:
        n += ((cur[m] - prev[m]) / hours >= wet_mm_per_h)
    return n / len(both)


def load_ens(cycle: dt.datetime, steps: tuple[int, ...], scratch: Path, wet_mm_per_h: float, now: dt.datetime | None = None) -> Loaded | None:
    """The ensemble's wet share for each interval between its steps (data["pop"][k] is for steps k-1 to k).
    Members are not kept: two steps at a time are in memory."""
    now = now or dt.datetime.now(dt.timezone.utc)
    model = ECMWF_ENS
    try:
        run = pick_run(model, cycle, now)
        if run is None:
            log.warning("ecmwf_ens: no complete run within %d h of %s, left out", MAX_RUN_AGE_H, cycle)
            return None
        off = (cycle - run).total_seconds() / 3600
        all_steps = model.steps_for(run.hour)
        lo, hi = max(0, steps[0] - 3) + off, steps[-1] + off
        first = max([s for s in all_steps if s <= lo], default=all_steps[0])
        last = min([s for s in all_steps if s >= hi], default=all_steps[-1])
        use = [s for s in all_steps if first <= s <= last]
        scratch.mkdir(parents=True, exist_ok=True)
        out = Loaded(model, run, off, [float(s) for s in use])
        pop = np.lib.format.open_memmap(scratch / "ens_pop.npy", "w+", np.float32, (len(use), NY, NX))
        pop[:] = np.nan
        out.data["pop"] = pop
        t0 = time.time()
        prev = None
        for k, s in enumerate(use):
            try:
                cur = _members(run, s, model.workers) if s > 0 else None
            except Exception as e:  # noqa: BLE001 - this interval and the next are left out
                log.warning("ecmwf_ens step %d failed: %s", s, e)
                prev = None
                continue
            if s == 0:
                cur = {}  # every member starts at 0 mm
            if prev is not None and k > 0:
                if not prev:  # step 0
                    prev = {m: np.zeros((NY, NX), np.float32) for m in cur}
                share = wet_share(prev, cur, s - use[k - 1], wet_mm_per_h)
                if share is not None:
                    pop[k] = share
            prev = cur
        out.good["pop"] = [k for k in range(len(use)) if np.isfinite(pop[k]).any()]
        if len(out.good["pop"]) < (len(use) - 1) * 3 // 4:
            log.warning("ecmwf_ens: only %d of %d intervals, left out", len(out.good["pop"]), len(use) - 1)
            return None
        log.info("ecmwf_ens run %s: %d intervals in %.0fs", run, len(out.good["pop"]), time.time() - t0)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("ecmwf_ens left out: %s", e)
        return None


def ens_pop_at(lm: Loaded, gfs_step: float, interval_h: float | None) -> np.ndarray | None:
    """The ensemble wet share for the interval that holds the end of this GFS step's interval."""
    t = gfs_step + lm.offset_h
    k = int(np.searchsorted(lm.steps, t, side="left"))
    if not interval_h or k < 1 or k >= len(lm.steps) or k not in lm.good["pop"]:
        return None
    return lm.data["pop"][k]


def mix_pop(det_fraction: np.ndarray, ens_share: np.ndarray | None) -> np.ndarray:
    """Rain chance 0..1 from the models' wet neighbourhood and, where there is one, the ensemble."""
    if ens_share is None:
        return det_fraction
    return np.where(np.isfinite(ens_share), ENS_SHARE * ens_share + (1 - ENS_SHARE) * det_fraction, det_fraction)


class EcmwfEns(Model):
    def run_is_complete(self, run: dt.datetime) -> bool:
        return fetch.exists(_ens_url(run, self.steps_for(run.hour)[-1]) + ".index")


ECMWF_ENS = EcmwfEns(
    name="ecmwf_ens",
    title="ECMWF ENS 0.25 degree open data (rain chance)",
    licence="ECMWF open data, CC BY 4.0",
    credit="Contains modified ECMWF open data (CC BY 4.0), copyright ECMWF",
    params={},
    steps_for=ens_steps,
    workers=8,
)


# --- Check at one place (runs in the blend-check workflow) ---


def check(lat: float, lon: float, hours: int) -> int:
    from . import job
    from .models import GFS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    now = dt.datetime.now(dt.timezone.utc)
    steps = tuple(s for s in GFS.steps if s <= hours)
    cycle = job.newest_complete(GFS, now, steps)
    if cycle is None:
        log.error("no GFS cycle")
        return 2
    j = int(round((90 - lat) / RES))
    i = int(round((lon % 360) / RES)) % NX
    day, hh = cycle.strftime("%Y%m%d"), cycle.hour

    # GFS at the point (whole fields are downloaded; only this point is kept).
    names = {"precip": "APCP:surface", "cloud": "TCDC:entire atmosphere", "u": "UGRD:10 m above ground", "v": "VGRD:10 m above ground", "gust": "GUST:surface"}
    acc = np.full((len(steps), 1, 1), np.nan, np.float32)
    ranges: list = [None] * len(steps)
    g = {n: [np.nan] * len(steps) for n in names}

    def one(k: int) -> None:
        url = GFS.url(day, hh, steps[k])
        inv = fetch.inventory(url)
        for n, src in names.items():
            e = fetch.pick(inv, src, "acc" if n == "precip" else "inst")
            if e is None:
                continue
            v, _ = fetch.field(url, e)
            if n == "precip":
                m = job._ACC.search(e.text)
                if m:
                    acc[k] = v[j, i]
                    ranges[k] = (int(m.group(1)), int(m.group(2)))
            else:
                g[n][k] = float(v[j, i])

    with cf.ThreadPoolExecutor(8) as pool:
        list(pool.map(one, range(len(steps))))
    g["precip"] = list(job.precipitation_rates(acc, ranges, steps).ravel())

    scratch = Path("build/blend-check")
    loaded = load_all(cycle, steps, scratch, now)
    from .job import WET_MM_PER_H

    ens = load_ens(cycle, steps, scratch / "ens", WET_MM_PER_H, now)
    print(f"\nGFS cycle {cycle:%Y-%m-%d %H} UTC; " + "; ".join(f"{lm.model.name} run {lm.run:%Y-%m-%d %H} UTC" for lm in loaded + ([ens] if ens else [])))
    print(f"Point {lat}, {lon} (grid row {j}, column {i}). Rain mm/h per model, then the mix; cloud %, wind km/h.\n")
    head = f"{'UTC':>11} {'GFS':>5} " + " ".join(f"{lm.model.name.upper():>5}" for lm in loaded) + f" {'MIX':>5} | ENS wet % | cloud GFS/mix | wind GFS/mix"
    print(head)
    iv = intervals(steps)
    tot = {"gfs": 0.0, **{lm.model.name: 0.0 for lm in loaded}, "mix": 0.0}
    for k, s in enumerate(steps):
        gfs = {n: np.full((NY, NX), np.nan, np.float32) for n in names}
        for n in names:
            gfs[n][j, i] = g[n][k]
        mixed, per = mix_step(gfs, loaded, s, iv[k])
        t = cycle + dt.timedelta(hours=s)
        vals = [per[m][j, i] if per.get(m) is not None else np.nan for m in ["gfs"] + [lm.model.name for lm in loaded]]
        for m, v in zip(["gfs"] + [lm.model.name for lm in loaded], vals):
            if np.isfinite(v) and iv[k]:
                tot[m] += float(v) * iv[k]
        if np.isfinite(mixed["precip"][j, i]) and iv[k]:
            tot["mix"] += float(mixed["precip"][j, i]) * iv[k]
        ws = lambda d: np.hypot(d["u"][j, i], d["v"][j, i]) * 3.6  # noqa: E731
        share = ens_pop_at(ens, s, iv[k]) if ens else None
        ens_txt = "   -" if share is None else f"{100 * float(share[j, i]):4.0f}"
        print(
            f"{t:%d %H:%M} " + " ".join(f"{v:5.2f}" for v in vals) + f" {mixed['precip'][j, i]:5.2f} |   {ens_txt}    |"
            f" {gfs['cloud'][j, i]:5.0f}/{mixed['cloud'][j, i]:3.0f}    | {ws(gfs):5.1f}/{ws(mixed):5.1f}"
        )
    print("\nTotal rain over these hours (mm): " + ", ".join(f"{m} {v:.1f}" for m, v in tot.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--lat", type=float, required=True)
    c.add_argument("--lon", type=float, required=True)
    c.add_argument("--hours", type=int, default=48)
    a = ap.parse_args(argv)
    return check(a.lat, a.lon, a.hours)


if __name__ == "__main__":
    sys.exit(main())
