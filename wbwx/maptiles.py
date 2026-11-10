"""Web map tiles (the usual 256 px Web Mercator z/x/y scheme) for the satellite and radar map layers.

A layer frame is drawn tile by tile: every pixel's position is worked out, each source is sampled there, and
the colour comes from the layer's colour scale. All tiles of one frame go into a single "pack" file, so the
server stores one file per frame instead of thousands (the hosting plan limits the number of files):

    b"WBPK" | u32 little-endian header length | header JSON | tile bytes...
    header: {"layer", "time" (ISO UTC), "minzoom", "maxzoom", "type": "image/webp",
             "credit", "tiles": {"z/x/y": [offset, length]}}   offsets count from the end of the header
"""
from __future__ import annotations

import io
import json
import math
import struct
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from PIL import Image

TILE_PX = 256
MAX_LAT = 85.0511287798


@dataclass(frozen=True)
class Box:
    south: float
    west: float
    north: float
    east: float


WORLD = Box(-MAX_LAT, -180, MAX_LAT, 180)


def tile_range(z: int, box: Box) -> Iterable[tuple[int, int]]:
    """Tiles x, y at zoom z that touch the box."""
    n = 2 ** z
    x0 = int(math.floor((box.west + 180) / 360 * n))
    x1 = int(math.floor((min(box.east, 179.9999) + 180) / 360 * n))
    y0 = _lat_to_tile_y(min(box.north, MAX_LAT), n)
    y1 = _lat_to_tile_y(max(box.south, -MAX_LAT), n)
    for x in range(max(0, x0), min(n - 1, x1) + 1):
        for y in range(max(0, y0), min(n - 1, y1) + 1):
            yield x, y


def _lat_to_tile_y(lat: float, n: int) -> int:
    r = math.radians(lat)
    return int(math.floor((1 - math.asinh(math.tan(r)) / math.pi) / 2 * n))


def pixel_lonlat(z: int, x: int, y: int, px: int = TILE_PX) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude (degrees) of the centre of every pixel of a tile, each px x px (row 0 = north)."""
    n = 2 ** z
    f = (np.arange(px) + 0.5) / px
    lon = (x + f) / n * 360 - 180
    merc = math.pi * (1 - 2 * (y + f) / n)
    lat = np.degrees(np.arctan(np.sinh(merc)))
    lon2, lat2 = np.meshgrid(lon, lat)
    return lon2, lat2


def encode_webp(rgba: np.ndarray, quality: int = 72) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "WEBP", quality=quality, method=4, alpha_quality=80)
    return b.getvalue()


def render(
    colour: Callable[[np.ndarray, np.ndarray], np.ndarray | None],
    zooms: Iterable[int],
    boxes: Iterable[Box],
    quality: int = 72,
) -> dict[str, bytes]:
    """Draws every tile touching any of the boxes. colour(lon, lat) gives an RGBA uint8 array or None for an
    empty tile; tiles with no visible pixel are left out (the app shows nothing there)."""
    boxes = list(boxes)
    out: dict[str, bytes] = {}
    for z in zooms:
        seen = set()
        for box in boxes:
            for x, y in tile_range(z, box):
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                lon, lat = pixel_lonlat(z, x, y)
                rgba = colour(lon, lat)
                if rgba is None or not rgba[..., 3].any():
                    continue
                out[f"{z}/{x}/{y}"] = encode_webp(rgba, quality)
    return out


def pack(layer: str, time_iso: str, tiles: dict[str, bytes], minzoom: int, maxzoom: int, credit: str) -> bytes:
    index, blobs, off = {}, [], 0
    for key in sorted(tiles):
        data = tiles[key]
        index[key] = [off, len(data)]
        blobs.append(data)
        off += len(data)
    header = json.dumps(
        {"layer": layer, "time": time_iso, "minzoom": minzoom, "maxzoom": maxzoom, "type": "image/webp", "credit": credit, "tiles": index},
        separators=(",", ":"),
    ).encode()
    return b"WBPK" + struct.pack("<I", len(header)) + header + b"".join(blobs)


def unpack(data: bytes) -> tuple[dict, dict[str, bytes]]:
    assert data[:4] == b"WBPK"
    (hl,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8 : 8 + hl])
    base = 8 + hl
    return header, {k: data[base + o : base + o + n] for k, (o, n) in header["tiles"].items()}


def ramp(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int, int]]]) -> np.ndarray:
    """Piecewise linear colour scale: stops are (value, RGBA) in rising order; below the first stop is clear."""
    v = np.asarray(values, dtype=np.float32)
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    out = np.zeros(v.shape + (4,), dtype=np.uint8)
    ok = np.isfinite(v) & (v >= xs[0])
    vv = np.clip(v[ok], xs[0], xs[-1])
    for c in range(4):
        out[..., c][ok] = np.interp(vv, xs, np.array([s[1][c] for s in stops], dtype=np.float32)).astype(np.uint8)
    return out
