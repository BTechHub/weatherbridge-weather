import datetime as dt
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from wbwx import maptiles, radar, satellite
from wbwx.maptiles import Box


class MapTiles(unittest.TestCase):
    def test_pixel_positions(self):
        lon, lat = maptiles.pixel_lonlat(0, 0, 0)
        self.assertEqual(lon.shape, (256, 256))
        self.assertAlmostEqual(lon[0, 0], -180 + 360 / 512, places=6)
        self.assertAlmostEqual(lat[0, 0], 85.05, delta=0.5)  # top row near the Mercator limit
        self.assertAlmostEqual(lat[128, 0], -lat[127, 0], places=6)  # equator between the middle rows
        # Mumbai (19.08N 72.88E) is in tile 5/22/14.
        lon, lat = maptiles.pixel_lonlat(5, 22, 14)
        self.assertTrue(lon.min() < 72.88 < lon.max() and lat.min() < 19.08 < lat.max())

    def test_tile_range(self):
        india = Box(6, 68, 36, 98)
        tiles = set(maptiles.tile_range(5, india))
        self.assertIn((22, 14), tiles)
        self.assertNotIn((0, 0), tiles)
        self.assertEqual(set(maptiles.tile_range(0, india)), {(0, 0)})
        self.assertEqual(len(list(maptiles.tile_range(2, maptiles.WORLD))), 16)

    def test_pack_round_trip(self):
        tiles = {"0/0/0": b"a", "5/22/14": b"bombay"}
        data = maptiles.pack("satellite", "2026-11-10T14:00Z", tiles, 0, 5, "credit")
        self.assertEqual(data[:4], b"WBPK")
        header, back = maptiles.unpack(data)
        self.assertEqual(back, tiles)
        self.assertEqual(header["maxzoom"], 5)
        self.assertEqual(header["type"], "image/webp")

    def test_ramp(self):
        stops = [(10.0, (0, 0, 0, 0)), (20.0, (200, 100, 50, 250))]
        out = maptiles.ramp(np.array([5.0, 10.0, 15.0, 30.0, np.nan, -np.inf]), stops)
        self.assertEqual(out[0].tolist(), [0, 0, 0, 0])  # below the scale: clear
        self.assertEqual(out[2].tolist(), [100, 50, 25, 125])
        self.assertEqual(out[3].tolist(), [200, 100, 50, 250])  # above: the top colour
        self.assertEqual(out[4].tolist(), [0, 0, 0, 0])
        self.assertEqual(out[5].tolist(), [0, 0, 0, 0])

    def test_render_skips_empty_tiles(self):
        def colour(lon, lat):
            rgba = np.zeros(lon.shape + (4,), dtype=np.uint8)
            rgba[(lon > 70) & (lon < 75) & (lat > 15) & (lat < 20)] = (255, 0, 0, 255)
            return rgba

        tiles = maptiles.render(colour, [3], [maptiles.WORLD])
        self.assertEqual(list(tiles), ["3/5/3"])
        img = Image.open(io.BytesIO(tiles["3/5/3"]))
        self.assertEqual(img.format, "WEBP")


class FakeArea:
    """A plain longitude/latitude grid standing in for a satellite's area (pyresample's method name)."""

    def __init__(self, west, north, res, shape):
        self.west, self.north, self.res, self.shape = west, north, res, shape

    def get_array_indices_from_lonlat(self, lon, lat):
        cols = np.floor((lon - self.west) / self.res).astype(int)
        rows = np.floor((self.north - lat) / self.res).astype(int)
        bad = (cols < 0) | (rows < 0) | (cols >= self.shape[1]) | (rows >= self.shape[0])
        return np.ma.masked_array(cols, bad), np.ma.masked_array(rows, bad)


class Satellite(unittest.TestCase):
    def test_each_place_comes_from_the_satellite_that_sees_it_best(self):
        area = FakeArea(-180, 90, 1.0, (180, 360))
        india = satellite.View("meteosat-iodc", 45.5, np.full((180, 360), 230.0, dtype=np.float32), area)
        japan = satellite.View("himawari", 140.7, np.full((180, 360), 260.0, dtype=np.float32), area)
        lon = np.array([[73.0, 120.0, 93.0, -100.0]])
        lat = np.array([[19.0, 30.0, 0.0, 0.0]])
        bt = satellite.composite([india, japan], lon, lat)
        self.assertEqual(bt[0, 0], 230.0)  # Mumbai: Meteosat-9
        self.assertEqual(bt[0, 1], 260.0)  # Shanghai: Himawari
        self.assertEqual(bt[0, 2], 230.0)  # 93E: 47.5 degrees from Meteosat-9, 47.7 from Himawari
        self.assertTrue(np.isnan(bt[0, 3]))  # Pacific off Mexico: neither sees it

    def test_a_gap_in_the_best_satellite_is_filled_by_the_next(self):
        area = FakeArea(-180, 90, 1.0, (180, 360))
        bt = np.full((180, 360), 230.0, dtype=np.float32)
        bt[70, 253] = np.nan  # Mumbai's pixel missing in Meteosat-9
        india = satellite.View("meteosat-iodc", 45.5, bt, area)
        japan = satellite.View("himawari", 140.7, np.full((180, 360), 260.0, dtype=np.float32), area)
        out = satellite.composite([india, japan], np.array([[73.5]]), np.array([[19.5]]))
        self.assertEqual(out[0, 0], 260.0)

    def test_colours(self):
        c = satellite.colour(np.array([300.0, 265.0, 220.0, 195.0, np.nan]))
        self.assertEqual(c[0, 3], 0)  # warm sea: clear
        self.assertTrue(0 < c[1, 3] < c[2, 3])  # low cloud fainter than high cloud
        self.assertGreater(c[3, 2], c[3, 0])  # storm tops turn blue-purple
        self.assertEqual(c[4, 3], 0)

    def test_only_open_meteosat_images(self):
        now = dt.datetime(2026, 11, 10, 15, 4, tzinfo=dt.timezone.utc)
        self.assertEqual(satellite.frame_time(now), dt.datetime(2026, 11, 10, 13, 0, tzinfo=dt.timezone.utc))
        self.assertEqual(satellite.frame_time(now.replace(minute=6)), dt.datetime(2026, 11, 10, 14, 0, tzinfo=dt.timezone.utc))
        with self.assertRaises(ValueError):
            satellite.meteosat("x", dt.datetime(2026, 11, 10, 14, 0, tzinfo=dt.timezone.utc), None, now=now)  # 64 minutes old
        with self.assertRaises(ValueError):
            satellite.meteosat("x", dt.datetime(2026, 11, 10, 13, 15, tzinfo=dt.timezone.utc), None, now=now)  # not a full hour

    def test_meteosat_is_off_unless_switched_on(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(satellite.meteosat_allowed())
        with mock.patch.dict("os.environ", {"WB_USE_METEOSAT": "1"}):
            self.assertTrue(satellite.meteosat_allowed())
        t = dt.datetime(2026, 11, 10, 14, 0, tzinfo=dt.timezone.utc)
        area = FakeArea(-180, 90, 1.0, (180, 360))
        fake = lambda name, files: satellite.View(name, 0.0, np.full((180, 360), 220.0, dtype=np.float32), area)
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(satellite, "goes", return_value=[]), mock.patch.object(satellite, "himawari", return_value=[]), \
                mock.patch.object(satellite, "meteosat") as met, mock.patch.object(satellite, "read", side_effect=fake), \
                mock.patch.object(satellite, "world_canvas", return_value=np.full((64, 64), 220.0, dtype=np.float32)), \
                mock.patch.object(satellite, "tiles_from_canvas", return_value={"0/0/0": b"x"}):
            with tempfile.TemporaryDirectory() as work:
                data, used = satellite.build(t, Path(work))
        met.assert_not_called()
        self.assertEqual(used, ["goes-east", "goes-west", "himawari"])
        header, _ = maptiles.unpack(data)
        self.assertNotIn("EUMETSAT", header["credit"])

    def test_canvas_and_tiles(self):
        area = FakeArea(-180, 90, 1.0, (180, 360))
        v = satellite.View("meteosat-iodc", 45.5, np.full((180, 360), 220.0, dtype=np.float32), area)
        canvas = satellite.world_canvas([v], z=2)
        self.assertEqual(canvas.shape, (1024, 1024))
        tiles = satellite.tiles_from_canvas(canvas, 2)
        self.assertIn("0/0/0", tiles)
        self.assertIn("2/2/1", tiles)  # 0..90E, north
        self.assertNotIn("2/0/1", tiles)  # 180W..90W: not seen


class Radar(unittest.TestCase):
    def test_frame_time(self):
        self.assertEqual(radar.frame_time(dt.datetime(2026, 11, 10, 15, 4, tzinfo=dt.timezone.utc)), dt.datetime(2026, 11, 10, 14, 45, tzinfo=dt.timezone.utc))

    def test_opera_composite(self):
        """A small file laid out like OPERA's (ODIM HDF5, Lambert azimuthal equal area)."""
        from pyproj import Transformer

        proj = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
        inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        nx, ny, scale = 380, 440, 10_000.0
        corners = {"UL": (0, 0), "UR": (nx * scale, 0), "LL": (0, -ny * scale), "LR": (nx * scale, -ny * scale)}
        data = np.full((ny, nx), -8888000.0)
        data[:20, :] = -9999000.0  # top rows: no radar
        fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        x, y = fwd.transform(2.35, 48.86)  # Paris
        data[int(-y // scale), int(x // scale)] = 45.0
        b = io.BytesIO()
        with h5py.File(b, "w") as f:
            w = f.create_group("where")
            w.attrs["projdef"] = np.bytes_(proj)
            w.attrs["xsize"], w.attrs["ysize"], w.attrs["xscale"], w.attrs["yscale"] = nx, ny, scale, scale
            for c, (cx, cy) in corners.items():
                lo, la = inv.transform(cx, cy)
                w.attrs[f"{c}_lon"], w.attrs[f"{c}_lat"] = lo, la
            d = f.create_group("dataset1/data1")
            d.create_dataset("data", data=data)
            wh = d.create_group("what")
            wh.attrs["quantity"] = np.bytes_("DBZH")
            wh.attrs["gain"], wh.attrs["offset"], wh.attrs["nodata"], wh.attrs["undetect"] = 1.0, 0.0, -9999000.0, -8888000.0
        key = "2026/11/10/OPERA/COMP/OPERA@20261110T1505@0@DBZH.h5"
        resp = mock.Mock(content=b.getvalue())
        with mock.patch.object(radar, "opera_keys", return_value=[(key, 1)]), mock.patch.object(radar, "_get", return_value=resp):
            g = radar.opera(dt.datetime(2026, 11, 10, 15, 0, tzinfo=dt.timezone.utc))
        self.assertEqual(g.time, dt.datetime(2026, 11, 10, 15, 5, tzinfo=dt.timezone.utc))
        v = g.sample(np.array([2.35, 13.4, 10.0, 80.0]), np.array([48.86, 52.52, 69.0, 20.0]))
        self.assertEqual(v[0], 45.0)  # Paris: rain
        self.assertTrue(np.isneginf(v[1]))  # Berlin: covered, dry
        self.assertTrue(np.isnan(v[3]))  # India: outside
        rgba = radar.colour_for([g])(np.array([[2.35, 13.4]]), np.array([[48.86, 52.52]]))
        self.assertGreater(rgba[0, 0, 3], 200)
        self.assertEqual(rgba[0, 1, 3], 0)

    def test_two_sources_take_the_stronger_echo(self):
        a = radar.Grid("a", Box(0, 0, 10, 10), lambda lon, lat: np.where(lon < 5, 20.0, -np.inf).astype(np.float32), dt.datetime.now())
        b = radar.Grid("b", Box(0, 0, 10, 10), lambda lon, lat: np.where(lon < 2, 40.0, np.nan).astype(np.float32), dt.datetime.now())
        rgba = radar.colour_for([a, b])(np.array([[1.0, 3.0, 7.0]]), np.array([[1.0, 1.0, 1.0]]))
        self.assertEqual(rgba[0, 0].tolist(), maptiles.ramp(np.array([40.0]), radar.DBZ_STOPS)[0].tolist())
        self.assertEqual(rgba[0, 1].tolist(), maptiles.ramp(np.array([20.0]), radar.DBZ_STOPS)[0].tolist())
        self.assertEqual(rgba[0, 2, 3], 0)


if __name__ == "__main__":
    unittest.main()
