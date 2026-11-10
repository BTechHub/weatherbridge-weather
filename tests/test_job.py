import unittest

import numpy as np

from wbwx.job import box_mean, candidate_cycles, precipitation_rates
from wbwx.tiles import MISSING, all_missing, decode, encode, quantize


class Tiles(unittest.TestCase):
    def test_round_trip_is_exact_including_missing_and_extremes(self):
        rng = np.random.default_rng(1)
        block = rng.integers(-32767, 32767, size=(3, 9, 400), dtype=np.int16)
        block[1, 4, 7] = MISSING
        block[2, :, :] = MISSING
        self.assertTrue(np.array_equal(decode(encode(block), 3, 9), block))

    def test_quantize(self):
        q = quantize(np.array([1.234, np.nan, 1e9, -1e9], dtype=np.float32), 10)
        self.assertEqual(q.tolist(), [12, MISSING, 32767, -32767])

    def test_all_missing(self):
        self.assertTrue(all_missing(np.full((1, 2, 400), MISSING, np.int16)))
        self.assertFalse(all_missing(np.zeros((1, 2, 400), np.int16)))


class Precipitation(unittest.TestCase):
    def test_buckets_restart_every_six_hours(self):
        steps = (0, 1, 2, 3, 4, 5, 6, 7, 9)
        # 0-1, 0-2 ... 0-6 accumulate; 6-7 starts a new bucket; 6-9 continues it over 2 hours.
        acc = np.array([0, 1, 3, 3, 4, 4, 6, 2, 8], dtype=np.float32).reshape(-1, 1, 1)
        ranges = [None, (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (6, 7), (6, 9)]
        r = precipitation_rates(acc, ranges, steps).ravel()
        self.assertTrue(np.isnan(r[0]))
        np.testing.assert_allclose(r[1:], [1, 2, 0, 1, 0, 2, 2, 3])

    def test_three_hour_buckets(self):
        acc = np.array([3, 6], dtype=np.float32).reshape(-1, 1, 1)
        r = precipitation_rates(acc, [(120, 123), (123, 126)], (123, 126)).ravel()
        np.testing.assert_allclose(r, [1, 2])


class Neighbourhood(unittest.TestCase):
    def test_box_mean_wraps_longitude(self):
        x = np.zeros((5, 10), np.float32)
        x[2, 0] = 9
        m = box_mean(x, 1)
        self.assertAlmostEqual(float(m[2, 9]), 1.0)  # wraps to the east edge
        self.assertAlmostEqual(float(m[2, 1]), 1.0)
        self.assertAlmostEqual(float(m[2, 5]), 0.0)


class Cycles(unittest.TestCase):
    def test_candidates_are_six_hourly_newest_first(self):
        import datetime as dt

        c = candidate_cycles(dt.datetime(2026, 11, 10, 4, 30, tzinfo=dt.timezone.utc), 12)
        self.assertEqual([x.hour for x in c], [0, 18, 12])


if __name__ == "__main__":
    unittest.main()


class Waves(unittest.TestCase):
    def test_short_swell_counts_as_wind_sea_and_long_swells_combine(self):
        from wbwx.models import wave_summary

        a = lambda *x: np.array(x, dtype=np.float32)  # noqa: E731
        f = {
            "wave_height": a(0.5, np.nan),
            "wind_wave_partition_height": a(0.25, np.nan),
            "wind_wave_period": a(2.2, np.nan),
            "wind_wave_direction": a(63, np.nan),
            "swell1_height": a(0.34, np.nan),
            "swell1_period": a(4.2, np.nan),
            "swell1_direction": a(329, np.nan),
            "swell2_height": a(0.16, np.nan),
            "swell2_period": a(12.4, np.nan),
            "swell2_direction": a(209, np.nan),
            "swell3_height": a(0.13, np.nan),
            "swell3_period": a(8.0, np.nan),
            "swell3_direction": a(172, np.nan),
        }
        w = wave_summary(f)
        self.assertAlmostEqual(float(w["wind_wave_height"][0]), (0.25**2 + 0.34**2) ** 0.5, places=4)
        self.assertAlmostEqual(float(w["swell_wave_height"][0]), (0.16**2 + 0.13**2) ** 0.5, places=4)
        self.assertTrue(10 < float(w["swell_wave_period"][0]) < 11)
        self.assertTrue(185 < float(w["swell_wave_direction"][0]) < 200)
        self.assertTrue(np.isnan(w["wave_period"][1]))  # land stays empty
