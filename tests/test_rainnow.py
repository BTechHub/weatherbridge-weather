import datetime as dt
import unittest

import numpy as np

from wbwx import rainnow


class RainNow(unittest.TestCase):
    def test_round_trip(self):
        bt = np.full((rainnow.NY, rainnow.NX), 290.0, np.float32)
        bt[10, 20] = 210.4
        bt[0, 0] = np.nan
        t = dt.datetime(2026, 11, 11, 11, 50, tzinfo=dt.timezone.utc)
        head, back = rainnow.decode(rainnow.encode(bt, t, 140.7))
        self.assertEqual(head["time"], "2026-11-11T11:50Z")
        self.assertEqual((head["ny"], head["nx"]), (926, 851))
        self.assertEqual(float(back[10, 20]), 210.0)
        self.assertTrue(np.isnan(back[0, 0]))
        self.assertEqual(float(back[5, 5]), 290.0)

    def test_scan_times_step_back_in_tens(self):
        t = rainnow.scan_times(dt.datetime(2026, 11, 11, 12, 9, tzinfo=dt.timezone.utc), 3)
        self.assertEqual([x.strftime("%H:%M") for x in t], ["11:50", "11:40", "11:30"])

    def test_grid_corners(self):
        lon, lat = rainnow.grid_lonlat()
        self.assertAlmostEqual(float(lat[0, 0]), 37.0)
        self.assertAlmostEqual(float(lat[-1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(lon[0, -1]), 100.0, places=6)
