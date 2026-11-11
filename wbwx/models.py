"""Datasets the job builds. Each one lists its source fields, how to convert them and their scale.

NOAA GFS and GFS-Wave are US government works (public domain), from the NOAA Open Data
Dissemination bucket on AWS (noaa-gfs-bdp-pds), which needs no account. The gfs dataset's rain,
rain chance, cloud and wind are a mix with ECMWF open data and DWD ICON (see wbwx/blend.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

AWS_GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

K = lambda v: v - 273.15  # noqa: E731
same = lambda v: v  # noqa: E731


@dataclass(frozen=True)
class Var:
    name: str  # name used by the API
    source: str  # "NAME:level" in the .idx inventory, or "" for a derived variable
    scale: float  # stored = round(value * scale)
    convert: Callable[[np.ndarray], np.ndarray] = same
    kind: str = "inst"
    circular: bool = False  # direction in degrees (the API interpolates it on the circle)
    sea_only: bool = False  # kept only where the land mask is 0 (saves space; used for sea temperature)
    stored: bool = True  # False: downloaded only to work out other variables, not sent to the API


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    licence: str
    vars: tuple[Var, ...]
    url: Callable[[str, int, int], str]  # (yyyymmdd, cycle hour, step) -> file url
    # Hourly to 5 days, then 3-hourly to 8.5 days: enough for 7 local days from a run up to 12 hours old.
    steps: tuple[int, ...] = field(default=tuple(range(0, 121)) + tuple(range(123, 205, 3)))
    skip_empty_tiles: bool = False
    # Partition heights the wave model leaves empty when that partition does not exist: 0 where the
    # total wave height has a value (so "no wind sea" reads as 0 m, not as no data).
    zero_if_missing: tuple[str, ...] = ()
    reference: str = ""
    # Works out derived variables for one step: (physical values by name) -> {name: values}.
    derive: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None
    # Mix rain, rain chance, cloud and wind with ECMWF and DWD ICON (wbwx/blend.py).
    blend: bool = False


def gfs_url(day: str, hh: int, step: int) -> str:
    return f"{AWS_GFS}/gfs.{day}/{hh:02d}/atmos/gfs.t{hh:02d}z.pgrb2.0p25.f{step:03d}"


def wave_url(day: str, hh: int, step: int) -> str:
    return f"{AWS_GFS}/gfs.{day}/{hh:02d}/wave/gridded/gfswave.t{hh:02d}z.global.0p25.f{step:03d}.grib2"


GFS = Dataset(
    name="gfs",
    title="NOAA GFS 0.25 degree",
    licence="NOAA/NCEP GFS, US government work (public domain)",
    url=gfs_url,
    blend=True,
    vars=(
        Var("temperature_2m", "TMP:2 m above ground", 10, K),
        Var("relative_humidity_2m", "RH:2 m above ground", 1),
        Var("wind_u_10m", "UGRD:10 m above ground", 5),
        Var("wind_v_10m", "VGRD:10 m above ground", 5),
        Var("wind_gusts_10m", "GUST:surface", 5),
        Var("pressure_msl", "PRMSL:mean sea level", 10, lambda v: v / 100),
        Var("visibility", "VIS:surface", 0.01, lambda v: np.minimum(v, 24000)),
        Var("cloud_cover", "TCDC:entire atmosphere", 1),
        Var("cape", "CAPE:surface", 0.1),
        Var("surface_temperature", "TMP:surface", 10, K, sea_only=True),
        Var("rain", "CRAIN:surface", 1),
        Var("snow", "CSNOW:surface", 1),
        Var("freezing_rain", "CFRZR:surface", 1),
        Var("land", "LAND:surface", 1),
        # The model's ground height; the API moves temperature from it to the real height (wbwx/elevation.py).
        Var("orography", "HGT:surface", 1),
        # Rate in mm/h over the interval that ends at this step, from the APCP accumulations.
        Var("precipitation", "APCP:surface", 10, kind="acc"),
        # Apparent temperature and dew point are worked out by the API from temperature, humidity and wind.
        # Derived after download (see job.precipitation_probability).
        Var("precipitation_probability", "", 1),
    ),
)

SWELL_MIN_PERIOD_S = 6.0


def _combine(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], like: np.ndarray):
    """Joins wave partitions: height sqrt(sum h^2), energy-weighted mean period and direction."""
    e_sum = np.zeros_like(like)
    t_sum = np.zeros_like(like)
    x_sum = np.zeros_like(like)
    y_sum = np.zeros_like(like)
    for hh, tt, dd in parts:
        ok = np.isfinite(hh) & np.isfinite(tt) & np.isfinite(dd) & (hh > 0)
        e = np.where(ok, hh * hh, 0)
        e_sum += e
        t_sum += np.where(ok, e * tt, 0)
        rad = np.radians(np.where(ok, dd, 0))
        x_sum += e * np.cos(rad)
        y_sum += e * np.sin(rad)
    has = e_sum > 0
    height = np.sqrt(e_sum)
    period = np.where(has, t_sum / np.where(has, e_sum, 1), np.nan)
    direction = np.where(has, np.degrees(np.arctan2(y_sum, x_sum)) % 360, np.nan)
    return height, period, direction


def wave_summary(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Wave summary fields (total, wind sea, swell) from the wave model's partitions (wind sea and up to 3 swells).

    - wave_period and wave_direction: energy-weighted means over all partitions. (The model's own
      "primary" period is the spectral peak, which reads several seconds longer than the mean.)
    - The model numbers its swells by sequence, not by type, and often files short local waves
      (under 6 s) as "swell 1". Those count as wind waves here; the swell fields combine the
      partitions of 6 s and longer.
    """
    sea = np.isfinite(f["wave_height"])
    wind = [(f["wind_wave_partition_height"], f["wind_wave_period"], f["wind_wave_direction"])]
    swell = []
    for i in (1, 2, 3):
        h, t, d = f[f"swell{i}_height"], f[f"swell{i}_period"], f[f"swell{i}_direction"]
        long_ = np.isfinite(t) & (t >= SWELL_MIN_PERIOD_S)
        swell.append((np.where(long_, h, np.nan), t, d))
        wind.append((np.where(long_, np.nan, h), t, d))
    _, mean_t, mean_d = _combine(wind + swell, f["wave_height"])
    ww_h, _, _ = _combine(wind, f["wave_height"])
    sw_h, sw_t, sw_d = _combine(swell, f["wave_height"])
    return {
        "wave_period": np.where(sea, mean_t, np.nan),
        "wave_direction": np.where(sea, mean_d, np.nan),
        "wind_wave_height": np.where(sea, ww_h, np.nan),
        "swell_wave_height": np.where(sea, sw_h, np.nan),
        "swell_wave_period": np.where(sea & (sw_h > 0), sw_t, np.nan),
        "swell_wave_direction": np.where(sea & (sw_h > 0), sw_d, np.nan),
    }


GFS_WAVE = Dataset(
    name="gfswave",
    title="NOAA GFS-Wave (WAVEWATCH III) 0.25 degree",
    licence="NOAA/NCEP GFS-Wave, US government work (public domain)",
    url=wave_url,
    skip_empty_tiles=True,
    zero_if_missing=("wind_wave_height", "swell_wave_height"),
    reference="wave_height",
    derive=wave_summary,
    vars=(
        Var("wave_height", "HTSGW:surface", 100),
        Var("wave_period", "", 10),
        Var("wave_direction", "", 1, circular=True),
        Var("wind_wave_height", "", 100),
        Var("swell_wave_height", "", 100),
        Var("swell_wave_period", "", 10),
        Var("swell_wave_direction", "", 1, circular=True),
        # Partitions, used only to work out the fields above.
        Var("wind_wave_partition_height", "WVHGT:surface", 100, stored=False),
        Var("wind_wave_period", "WVPER:surface", 10, stored=False),
        Var("wind_wave_direction", "WVDIR:surface", 1, stored=False),
        *(
            Var(f"swell{i}_{k}", f"{src}:{i} in sequence", scale, stored=False)
            for i in (1, 2, 3)
            for k, src, scale in (("height", "SWELL", 100), ("period", "SWPER", 10), ("direction", "SWDIR", 1))
        ),
    ),
)

DATASETS = {d.name: d for d in (GFS, GFS_WAVE)}
