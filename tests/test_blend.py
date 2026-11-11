import datetime as dt
import unittest

import numpy as np

from wbwx import blend
from wbwx.tiles import NX, NY


def full(x):
    return np.full((NY, NX), x, np.float32)


class Time(unittest.TestCase):
    def test_inst_is_linear_between_steps(self):
        f = [np.array([0.0]), np.array([30.0])]
        self.assertAlmostEqual(float(blend.interp_inst([0, 3], lambda k: f[k], 1)[0]), 10.0)
        self.assertIsNone(blend.interp_inst([0, 3], lambda k: f[k], 4))

    def test_three_hourly_rain_spreads_evenly_over_each_hour(self):
        acc = [np.array([0.0]), np.array([3.0]), np.array([9.0])]  # 1 mm/h then 2 mm/h
        get = lambda k: acc[k]  # noqa: E731
        self.assertAlmostEqual(float(blend.rate_between([0, 3, 6], get, 1, 2)[0]), 1.0)
        self.assertAlmostEqual(float(blend.rate_between([0, 3, 6], get, 3, 4)[0]), 2.0)
        self.assertAlmostEqual(float(blend.rate_between([0, 3, 6], get, 2, 5)[0]), (1 + 2 + 2) / 3)
        self.assertIsNone(blend.rate_between([0, 3, 6], get, 5, 7))  # past the last step

    def test_run_offset_matches_valid_time(self):
        # A model run 6 h older than the GFS cycle: GFS step 1 is the model's hour 7.
        lm = blend.Loaded(blend.ECMWF, dt.datetime(2026, 11, 11, 0, tzinfo=dt.timezone.utc), 6.0, [6.0, 9.0])
        lm.data["precip"] = np.stack([full(0), full(6)])  # 2 mm/h from 6 to 9
        lm.data["cloud"] = np.stack([full(10), full(40)])
        lm.good = {"precip": [0, 1], "cloud": [0, 1]}
        self.assertAlmostEqual(float(blend.at_step(lm, "precip", 1, 1.0)[0, 0]), 2.0)
        self.assertAlmostEqual(float(blend.at_step(lm, "cloud", 1, 1.0)[0, 0]), 20.0)
        self.assertIsNone(blend.at_step(lm, "precip", 0, None))


class Mixing(unittest.TestCase):
    def test_weights_scale_up_when_a_model_is_missing(self):
        w = {"a": 0.25, "b": 0.75}
        a = np.array([4.0, 4.0, np.nan])
        b = np.array([0.0, np.nan, np.nan])
        m = blend.weighted({"a": a, "b": b}, w)
        np.testing.assert_allclose(m[:2], [1.0, 4.0])
        self.assertTrue(np.isnan(m[2]))
        self.assertEqual(blend.weighted({"a": None}, w).shape, (NY, NX))

    def test_wind_keeps_speed_when_directions_differ(self):
        u = {"a": np.array([10.0]), "b": np.array([0.0])}
        v = {"a": np.array([0.0]), "b": np.array([10.0])}
        mu, mv = blend.blend_wind(u, v, {"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(float(np.hypot(mu, mv)[0]), 10.0, places=4)
        self.assertAlmostEqual(float(mu[0]), float(mv[0]), places=4)  # from the north-east-ish middle

    def test_gfs_drizzle_is_cut_by_dry_models(self):
        lm = blend.Loaded(blend.ECMWF, dt.datetime(2026, 11, 11, 0, tzinfo=dt.timezone.utc), 0.0, [0.0, 3.0])
        for n, a, b in (("precip", 0, 0), ("cloud", 20, 20), ("u", 3, 3), ("v", 0, 0), ("gust", 5, 5)):
            lm.data[n] = np.stack([full(a), full(b)])
        lm.good = {n: [0, 1] for n in lm.data}
        gfs = {"precip": full(0.2), "cloud": full(90), "u": full(3), "v": full(0), "gust": full(4)}
        mixed, per = blend.mix_step(gfs, [lm], 1, 1.0)
        w = blend.WEIGHTS
        self.assertAlmostEqual(float(mixed["precip"][0, 0]), 0.2 * w["gfs"] / (w["gfs"] + w["ecmwf"]), places=5)
        self.assertLess(float(mixed["precip"][0, 0]), 0.1)  # no longer counts as rain
        self.assertAlmostEqual(float(mixed["cloud"][0, 0]), (90 * w["gfs"] + 20 * w["ecmwf"]) / (w["gfs"] + w["ecmwf"]), places=3)
        self.assertGreaterEqual(float(mixed["gust"][0, 0]), 3.0)
        self.assertIn("ecmwf", per)


class Grids(unittest.TestCase):
    def test_icon_cells_average_into_boxes_and_fill_gaps(self):
        lat = np.array([10.0, 10.05, -45.0])
        lon = np.array([77.0, 77.05, 300.0])
        g = blend.IconGrid(lat, lon)
        out = g.regrid(np.array([1.0, 3.0, 7.0], np.float32))
        j, i = int(round((90 - 10) / 0.25)), int(round(77 / 0.25))
        self.assertAlmostEqual(float(out[j, i]), 2.0)
        self.assertAlmostEqual(float(out[int(round(135 / 0.25)), int(round(300 / 0.25))]), 7.0)
        self.assertFalse(np.isnan(out).any())  # every empty box takes its nearest cell

    def test_radians_are_detected(self):
        g = blend.IconGrid(np.radians(np.array([10.0])), np.radians(np.array([77.0])))
        out = g.regrid(np.array([5.0], np.float32))
        self.assertEqual(float(out[320, 308]), 5.0)


class Steps(unittest.TestCase):
    def test_published_steps(self):
        self.assertEqual(blend.ecmwf_steps(6)[-1], 144)
        self.assertEqual(blend.ecmwf_steps(0)[-1], 360)
        self.assertEqual(blend.icon_steps(12)[-1], 180)
        self.assertEqual(blend.icon_steps(18)[-1], 120)
        self.assertEqual(blend.intervals((0, 1, 3)), [None, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()


class Ensemble(unittest.TestCase):
    def test_wet_share_and_mix(self):
        prev = {"1": full(0), "2": full(0), "3": full(0), "4": full(0)}
        cur = {"1": full(0.6), "2": full(0.1), "3": full(0), "4": full(3)}  # mm over 3 h
        share = blend.wet_share(prev, cur, 3, 0.1)
        self.assertAlmostEqual(float(share[0, 0]), 0.5)
        pop = blend.mix_pop(full(1.0), share)
        self.assertAlmostEqual(float(pop[0, 0]), blend.ENS_SHARE * 0.5 + (1 - blend.ENS_SHARE))
        self.assertEqual(float(blend.mix_pop(full(0.2), None)[0, 0]), np.float32(0.2))

    def test_interval_lookup(self):
        lm = blend.Loaded(blend.ECMWF_ENS, dt.datetime(2026, 11, 11, 0, tzinfo=dt.timezone.utc), 6.0, [6.0, 9.0, 12.0])
        lm.data["pop"] = np.stack([full(np.nan), full(0.2), full(0.7)])
        lm.good = {"pop": [1, 2]}
        self.assertAlmostEqual(float(blend.ens_pop_at(lm, 1, 1.0)[0, 0]), 0.2)  # hour 7: interval 6-9
        self.assertAlmostEqual(float(blend.ens_pop_at(lm, 3, 1.0)[0, 0]), 0.2)  # hour 9 ends that interval
        self.assertAlmostEqual(float(blend.ens_pop_at(lm, 4, 1.0)[0, 0]), 0.7)
        self.assertIsNone(blend.ens_pop_at(lm, 7, 1.0))
