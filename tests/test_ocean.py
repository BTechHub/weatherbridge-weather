import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wbwx import ocean
from wbwx.tiles import MISSING, TILE, decode, quantize


class Ocean(unittest.TestCase):
    def test_regrid_coast_and_poles(self):
        # Copernicus-like 1/12 degree grid from 80S, longitudes from 180W.
        lat = -80 + np.arange(2041) / 12
        lon = -180 + np.arange(4320) / 12
        a = np.tile(np.where(lon < 73.7, 0.5, np.nan), (len(lat), 1))  # sea west of 73.7E
        g = ocean.Regrid(lat, lon)(a)
        r = 298  # 15.5N
        self.assertAlmostEqual(g[r, 294], 0.5)  # 73.5E
        self.assertAlmostEqual(g[r, 295], 0.5)  # 73.75E: land, but the cell to its west is sea
        self.assertTrue(np.isnan(g[r, 297]))  # 74.25E
        self.assertTrue(np.isnan(g[719, 0]))  # 89.75S: south of the model
        self.assertAlmostEqual(g[0, 0], 0.5)  # 90N, 0E

    def test_tiles(self):
        steps = [0, 3, 6]
        f = {n: np.full((3, 720, 1440), MISSING, dtype=np.int16) for n, _ in ocean.VARS}
        f["current_u"][:, 298, 294] = quantize(np.array([0.25, 0.5, 0.75]), 100)
        f["current_v"][:, 298, 294] = 0
        f["sst"][:, 298, 294] = 2940
        with tempfile.TemporaryDirectory() as d:
            out = ocean.build(f, 1_794_182_400, steps, Path(d), "test")
            m = json.loads((out / "manifest.json").read_text())
            self.assertEqual(m["dataset"], "ocean")
            self.assertEqual(m["tiles"], ["r14c14"])
            b = decode((out / "tiles" / "r14c14.bin").read_bytes(), 3, 3)
            i = (298 % TILE) * TILE + 294 % TILE
            self.assertEqual(list(b[0, :, i]), [25, 50, 75])
            self.assertEqual(b[2, 1, i], 2940)


if __name__ == "__main__":
    unittest.main()
