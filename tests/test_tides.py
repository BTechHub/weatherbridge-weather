import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from wbwx import tides
from wbwx.tiles import TILE, decode


def eot_grid():
    """A coarse stand-in for EOT20: 0.125 degrees around Goa, sea west of 73.7E, land east of it."""
    lat = np.arange(-90, 90.001, 0.125)
    lon = np.arange(0, 360, 0.125)
    amp = np.full((len(lat), len(lon)), 100.0)  # cm
    pha = np.full_like(amp, 90.0)
    land = lon[None, :] > 73.7
    amp = np.where(land, np.nan, amp)
    pha = np.where(land, np.nan, pha)
    return amp, pha, lat, lon


class Tides(unittest.TestCase):
    def test_grid_sea_coast_and_land(self):
        amp, pha, lat, lon = eot_grid()
        re, im = tides.to_grid(amp, pha, lat, lon)
        r = 298  # 15.5N
        self.assertAlmostEqual(re[r, 294], 0, places=6)  # 73.5E, sea: A cos 90
        self.assertAlmostEqual(im[r, 294], 1000)  # mm
        # 73.75E is land in EOT20 but the 73.625E cell next to it is sea: the coast still gets a tide.
        self.assertAlmostEqual(im[r, 295], 1000)
        self.assertTrue(np.isnan(im[r, 297]))  # 74.25E, inland

    def test_zip_to_tiles(self):
        amp, pha, lat, lon = eot_grid()
        import h5py

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for c in tides.CONSTITUENTS:
                f = io.BytesIO()
                if c == "k2":  # classic netCDF, as some EOT20 files are
                    from scipy.io import netcdf_file

                    nc = netcdf_file(f, "w")
                    nc.createDimension("lat", len(lat))
                    nc.createDimension("lon", len(lon))
                    for name, arr, dims in (("lat", lat, ("lat",)), ("lon", lon, ("lon",)), ("amplitude", np.nan_to_num(amp * 0.1, nan=-999), ("lat", "lon")), ("phase", np.nan_to_num(pha, nan=-999), ("lat", "lon"))):
                        v = nc.createVariable(name, "f4", dims)
                        v[:] = arr
                        if name in ("amplitude", "phase"):
                            v._FillValue = np.float32(-999)
                    nc.flush()
                    z.writestr(f"EOT20/ocean_tides/{c.upper()}_ocean_eot20.nc", f.getvalue())
                    nc.close()
                    continue
                with h5py.File(f, "w") as h:
                    k = 1.0 if c == "m2" else 0.1
                    h["amplitude"] = (amp * k).astype(np.float32)
                    h["phase"] = pha.astype(np.float32)
                    h["lat"] = lat
                    h["lon"] = lon
                z.writestr(f"EOT20/ocean_tides/{c.upper()}_ocean_eot20.nc", f.getvalue())
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "eot20.zip"
            src.write_bytes(buf.getvalue())
            out = tides.build(tides.read_eot20(src), Path(d) / "out")
            m = json.loads((out / "manifest.json").read_text())
            self.assertEqual(m["dataset"], "tides")
            self.assertEqual(len(m["vars"]), 34)
            r, c = 298, 294
            tid = f"r{r // TILE}c{c // TILE}"
            block = decode((out / "tiles" / f"{tid}.bin").read_bytes(), 34, 1)
            i = (r % TILE) * TILE + c % TILE
            self.assertEqual(block[tides.VARS.index("m2_im"), 0, i], 1000)
            self.assertEqual(block[tides.VARS.index("k1_im"), 0, i], 100)
            self.assertEqual(block[tides.VARS.index("m2_re"), 0, i], 0)
            self.assertEqual(block[tides.VARS.index("k2_im"), 0, i], 100)  # from the classic netCDF file


    def test_zip_inside_zip(self):
        amp, pha, lat, lon = eot_grid()
        import h5py

        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            for c in tides.CONSTITUENTS:
                f = io.BytesIO()
                with h5py.File(f, "w") as h:
                    h["amplitude"] = amp.astype(np.float32)
                    h["phase"] = pha.astype(np.float32)
                    h["lat"] = lat
                    h["lon"] = lon
                z.writestr(f"ocean_tides/{c.upper()}_ocean_eot20.nc", f.getvalue())
                z.writestr(f"load_tides/{c.upper()}_load_eot20.nc", b"not read")
                z.writestr(f"__MACOSX/ocean_tides/._{c.upper()}_ocean_eot20.nc", b"mac metadata")
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as z:
            z.writestr("EOT20/ocean_tides.zip", inner.getvalue())
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "eot20.zip"
            src.write_bytes(outer.getvalue())
            fields = tides.read_eot20(src)
            self.assertAlmostEqual(fields["m2_im"][298, 294], 1000)


if __name__ == "__main__":
    unittest.main()
