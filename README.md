# WeatherBridge weather data job

Builds the WeatherBridge app's forecast and marine data from free government model output, and
uploads it to the WeatherBridge API (api.lorjova.com), which serves it to the app.

Runs on GitHub Actions four times a day (`.github/workflows/weather.yml`), about 4 h 35 min after each
NOAA run starts (00, 06, 12, 18 UTC). Each run ends by checking `https://api.lorjova.com/weatherbridge/v1/status`,
so a stale or failed update shows as a failed run and GitHub emails the owner.

Status (2026-11-10): NOAA GFS and GFS-Wave, worldwide, live. ECMWF, DWD ICON and Copernicus Marine come next.

## Sources

| Dataset | What it gives the app | Source | Licence |
|---|---|---|---|
| `gfs` | Temperature, humidity, wind, gusts, pressure, visibility, cloud, rain, snow, storm energy (for weather codes), sea surface temperature | NOAA GFS 0.25°, AWS bucket `noaa-gfs-bdp-pds` | US government work, public domain |
| `gfswave` | Wave, wind-wave and swell height, period and direction | NOAA GFS-Wave (WAVEWATCH III) 0.25° global, same bucket | Public domain |

The API works out apparent temperature, dew point, UV index (estimated from the sun's height and
cloud), day or night, sunrise and sunset, weather codes and the daily values itself.

Rain chance comes from one model, so it is a neighbourhood estimate: the share of grid points within
about 90 km (and the hour before and after) where the model has rain. An ensemble can replace
it later.

## How a run works

1. Find the newest model run whose last forecast hour is on AWS (`--cycle` picks one by hand).
2. Ask the API whether that run is already live; if so, stop.
3. Download only the needed fields of every forecast hour (0 to 120 hourly, then 3-hourly to 204)
   with HTTP range requests, using the `.idx` inventory next to each file.
4. Turn the precipitation totals into rates, add rain chance, cut the world into 2,592 tiles of
   5° × 5° (wave tiles over land are skipped), and upload them.

Measured on 2026-11-10 (run 2026-11-09 18 UTC, 4 cores): GFS 4 min 20 s and 780 MB, GFS-Wave
2 min and 285 MB. The API keeps 4 runs per dataset, about 4.3 GB.

## Run it

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python -m unittest
.venv/bin/python -m wbwx.job gfs --out build --max-step 6          # quick local try, no upload
WB_WEATHER_INGEST_URL=https://api.lorjova.com/weatherbridge/ingest/weather \
WB_WEATHER_INGEST_TOKEN=... .venv/bin/python -m wbwx.job gfs --out build --publish
```

The token is the API's `WEATHER_INGEST_TOKEN`. In GitHub it is the secret `WB_WEATHER_INGEST_TOKEN`.

## Tile format

`wbwx/tiles.py`: int16 values `round(value × scale)`, -32768 for no data, each step stored as the
difference from the previous one (modulo 2^16), gzip. Shape `[variable][step][400 points]`.
`manifest.json` lists the variables, scales, steps and tiles.

## Licence of the data

NOAA GFS and GFS-Wave are works of the US government and in the public domain. Credit: NOAA/NCEP.
Data accessed through the NOAA Open Data Dissemination program on AWS.
