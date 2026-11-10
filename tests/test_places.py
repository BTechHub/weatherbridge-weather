import gzip
import json
import tempfile
import unittest
from pathlib import Path

from wbwx import places

CITIES = "\t".join(["1260607", "Panaji", "Panaji", "Pandzhim,Panjim,पणजी,https://x", "15.49835", "73.82901", "P", "PPLA", "IN", "", "33", "", "", "", "114405", "", "7", "Asia/Kolkata", "2024-01-01"]) + "\n"
ADMIN1 = "IN.33\tGoa\tGoa\t1271157\n"
COUNTRY = "#ISO\tISO3\n" + "\t".join(["IN", "IND", "356", "IN", "India", "New Delhi", "3287590", "1352617328", "AS", ".in", "INR", "Rupee", "91", "", "", "en-IN,hi", "1269750", "CN", ""]) + "\n"


class Places(unittest.TestCase):
    def test_build(self):
        with tempfile.TemporaryDirectory() as d:
            data = json.loads(gzip.decompress(places.build(Path(d), CITIES, ADMIN1, COUNTRY).read_bytes()))
        self.assertEqual(data["countries"]["IN"], ["India", 1269750])
        self.assertEqual(data["places"], [[1260607, "Panaji", 15.49835, 73.82901, 7, "PPLA", "IN", "Goa", 1271157, "Asia/Kolkata", 114405, ["Pandzhim", "Panjim", "पणजी"]]])


if __name__ == "__main__":
    unittest.main()
