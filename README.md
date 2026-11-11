# WeatherBridge weather data jobs

GitHub Actions jobs that build WeatherBridge's forecast data from free sources and upload it to
api.lorjova.com/weatherbridge. The API code lives in the lorjova repository
(backend/docs/WEATHERBRIDGE_WEATHER.md).

## Forecast (`weather-data`, 4 times a day)

NOAA GFS 0.25 degree is the base. Rain, rain chance, cloud and wind are a mix of three models
(`wbwx/blend.py`), matched by valid time:

| Model | Share | Used for |
| --- | --- | --- |
| ECMWF IFS 0.25 open data | 45% | rain, cloud, wind, gusts |
| DWD ICON Global 13 km | 30% | rain, cloud, wind |
| NOAA GFS 0.25 | 25% | rain, cloud, wind, gusts, and everything else |
| ECMWF ENS (50 members) | | rain chance: 60% share of wet members, 40% the three models' wet neighbourhood |

A model that cannot be downloaded drops out and the others' shares grow, so the job still publishes
GFS alone if both extra sources fail. The run's manifest lists the sources used.

`python -m wbwx.blend check --lat 8.48 --lon 76.95` prints every model at one place (also the
`blend-check` workflow, on pull requests and by hand).

## Rain clouds now (`rain-now`, every 10 minutes)

`wbwx/rainnow.py` reads Himawari-9 band 13 cloud-top temperatures over India (66-100 E, 0-37 N, 0.04 degree)
and uploads one small map (about 370 KB). The API's `/v1/rain-now` uses it to say whether tall rain clouds are
over a place now, so the app does not show "raining" from the forecast alone. JMA Himawari via NOAA open data.

Waves: NOAA GFS-Wave. Also in this repo: ocean currents and sea temperature (Copernicus Marine),
tides (EOT20), ground heights (NOAA ETOPO 2022), places (GeoNames), satellite clouds (GOES,
Himawari) and radar (MRMS, OPERA).

## Credits

- NOAA/NCEP GFS and GFS-Wave: US government work, public domain.
- ECMWF open data (IFS and ENS): CC BY 4.0. "Contains modified ECMWF open data, copyright ECMWF."
- DWD ICON: Deutscher Wetterdienst open data under GeoNutzV. "Source: Deutscher Wetterdienst (DWD), modified."
