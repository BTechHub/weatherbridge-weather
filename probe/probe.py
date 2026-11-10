"""Temporary: learn what EUMETSAT and OPERA give us from GitHub's network (not reachable from dev machines)."""
import datetime as dt, os, re, io, requests, h5py
B = "https://s3.waw3-1.cloudferro.com/openradar-24h/"
r = requests.get(B, params={"list-type": "2", "prefix": "2026/11/10/OPERA/COMP/", "max-keys": "1000"}, timeout=30)
keys = re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", r.text)
print(len(keys)); [print(k) for k in keys[:6]]; [print(k) for k in keys[-12:]]
for k, _ in keys[-12:]:
    if k.endswith(".h5"):
        f = h5py.File(io.BytesIO(requests.get(B + k, timeout=60).content), "r")
        print("FILE", k)
        f.visititems(lambda n, o: print("  ", n, {a: o.attrs[a] for a in o.attrs if a not in ("lat", "lon")}, getattr(o, "shape", ""), getattr(o, "dtype", "")))
        for g in ["what", "where", "how"]:
            if g in f: print(g, dict(f[g].attrs))
