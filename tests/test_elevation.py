import tempfile
import unittest
from pathlib import Path

import numpy as np

from wbwx import elevation
from wbwx.tiles import TILE, decode


class Elevation(unittest.TestCase):
    def test_sub_points_and_tiles(self):
        lat_axis = np.arange(-89.5, 90, 1.0)
        lon_axis = np.arange(-179.5, 180, 1.0)
        z = np.tile(lon_axis * 10, (len(lat_axis), 1)).astype(np.float32)  # height = 10 x longitude, sea (0) in the west
        h = elevation.sample(z, lat_axis, lon_axis)
        r, c = 298, 295  # grid point 15.5N 73.75E
        self.assertEqual(h[r, 4, c, 5], 735)  # sub-point 15.5125N 73.7625E, nearest 1-degree cell 73.5E
        self.assertEqual(h[r, 0, 1439, 0], 0)  # west of 0E: below sea level counts as 0
        with tempfile.TemporaryDirectory() as d:
            out = elevation.build(h, Path(d))
            m = __import__("json").loads((out / "manifest.json").read_text())
            self.assertEqual(len(m["vars"]), 100)
            tid = f"r{r // TILE}c{c // TILE}"
            self.assertIn(tid, m["tiles"])
            block = decode((out / "tiles" / f"{tid}.bin").read_bytes(), 100, 1)
            self.assertEqual(block[4 * 10 + 5, 0, (r % TILE) * TILE + c % TILE], 735)
            self.assertNotIn("r0c60", m["tiles"])  # all of 300E..305E is sea here: left out


if __name__ == "__main__":
    unittest.main()
