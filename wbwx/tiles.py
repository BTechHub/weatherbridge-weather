"""Fixed 0.25 degree global grid cut into 5 degree tiles, and the tile file format.

Grid: 721 rows from 90N to 90S, 1440 columns from 0E eastwards (the native GFS 0.25 grid).
Tiles: 20 x 20 points, rows 0..719 (the 90S row is left out), so 36 x 72 = 2592 tiles named r{R}c{C}.

Tile file: gzip of int16 little-endian values shaped [variable][step][400 points], each value
stored as round(physical * scale), MISSING for no data. Along the step axis every value is stored
as the difference from the previous step (modulo 2^16), which makes smooth series compress far
better. The API reverses this exactly with a running sum modulo 2^16.
"""
from __future__ import annotations

import gzip

import numpy as np

RES = 0.25
NY, NX = 721, 1440
TILE = 20
TILE_ROWS, TILE_COLS = 36, 72
MISSING = -32768


def tile_id(r: int, c: int) -> str:
    return f"r{r}c{c}"


def quantize(values: np.ndarray, scale: float) -> np.ndarray:
    """Physical values to int16; NaN becomes MISSING, out-of-range values are clipped."""
    ok = np.isfinite(values)
    q = np.clip(np.rint(np.where(ok, values, 0) * scale), -32767, 32767).astype(np.int16)
    q[~ok] = MISSING
    return q


def encode(block: np.ndarray) -> bytes:
    """block: int16 [vars][steps][400]. Returns the gzip-compressed, time-delta encoded tile."""
    assert block.dtype == np.int16 and block.ndim == 3 and block.shape[2] == TILE * TILE
    u = block.view(np.uint16)
    d = u.copy()
    d[:, 1:, :] = u[:, 1:, :] - u[:, :-1, :]  # uint16 arithmetic wraps, so this is exactly reversible
    return gzip.compress(d.astype("<u2").tobytes(), compresslevel=6, mtime=0)


def decode(data: bytes, nvars: int, nsteps: int) -> np.ndarray:
    d = np.frombuffer(gzip.decompress(data), dtype="<u2").reshape(nvars, nsteps, TILE * TILE)
    return np.cumsum(d, axis=1, dtype=np.uint16).view(np.int16)


def all_missing(block: np.ndarray) -> bool:
    return bool((block == MISSING).all())
