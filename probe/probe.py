"""Temporary: learn what EUMETSAT and OPERA give us from GitHub's network (not reachable from dev machines)."""
import datetime as dt, os, re, time, requests, h5py, io
B = "https://s3.waw3-1.cloudferro.com/openradar-24h/"
def ls(prefix, delim=True):
    r = requests.get(B, params={"list-type": "2", "prefix": prefix, **({"delimiter": "/"} if delim else {}), "max-keys": "200"}, timeout=30)
    return re.findall(r"<Prefix>([^<]*)</Prefix>", r.text)[1:], re.findall(r"<Key>([^<]*)</Key><LastModified>([^<]*)</LastModified>.*?<Size>(\d+)</Size>", r.text)
p = "2026/"
for _ in range(6):
    pre, keys = ls(p)
    print(p, "->", pre[:12], keys[:5])
    if not pre: break
    p = pre[-1]
pre, keys = ls(p, delim=False)
print("files in", p, len(keys)); [print(" ", k) for k in keys[:40]]
comp = [k for k in keys if "OPERA" in k[0] or "COMP" in k[0].upper()]
if comp or keys:
    k = (comp or keys)[-1][0]
    d = requests.get(B + k, timeout=60).content
    print("downloaded", k, len(d))
    try:
        f = h5py.File(io.BytesIO(d))
        def show(n, o):
            print("  ", n, dict(o.attrs) if len(o.attrs) < 40 else list(o.attrs)[:40], getattr(o, "shape", ""), getattr(o, "dtype", ""))
        f.visititems(show)
        for g in ["where", "what", "how"]:
            if g in f: print(g, dict(f[g].attrs))
    except Exception as e:
        print("h5 err", e, d[:200])
print("== EUMETSAT download timing")
import eumdac
tok = eumdac.AccessToken((os.environ["EUMETSAT_CONSUMER_KEY"], os.environ["EUMETSAT_CONSUMER_SECRET"]))
ds = eumdac.DataStore(tok)
col = ds.get_collection("EO:EUM:DAT:MSG:HRSEVIRI-IODC")
end = dt.datetime.utcnow() - dt.timedelta(minutes=65); start = end - dt.timedelta(hours=1)
prods = list(col.search(dtstart=start, dtend=end))
p = [x for x in prods if x.sensing_start.minute == 0][0]
print(p, p.sensing_start, p.sensing_end)
t = time.time()
nat = [e for e in p.entries if e.endswith(".nat")][0]
with p.open(entry=nat) as src, open("/tmp/x.nat", "wb") as out:
    while True:
        b = src.read(1 << 22)
        if not b: break
        out.write(b)
print("download s", time.time() - t, os.path.getsize("/tmp/x.nat"))
from satpy import Scene
t = time.time()
s = Scene(reader="seviri_l1b_native", filenames=["/tmp/x.nat"]); s.load(["IR_108"]); d = s["IR_108"]; a = d.attrs["area"]
print("read s", time.time() - t, d.shape, a.crs.to_proj4(), a.area_extent, float(d.min()), float(d.max()))
