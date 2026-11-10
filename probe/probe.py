"""Temporary: learn what EUMETSAT and OPERA give us from GitHub's network (not reachable from dev machines)."""
import datetime as dt, os, re, time, requests
B = "https://s3.waw3-1.cloudferro.com/openradar-24h/"
def ls(prefix, delim=True, n=1000):
    r = requests.get(B, params={"list-type": "2", "prefix": prefix, **({"delimiter": "/"} if delim else {}), "max-keys": str(n)}, timeout=30)
    return re.findall(r"<Prefix>([^<]*)</Prefix>", r.text)[1:], re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", r.text)
print("== OPERA tree")
def walk(p, depth):
    pre, keys = ls(p)
    print("  " * depth + p, "dirs:", [x[len(p):] for x in pre][:30], "files:", len(keys))
    if keys: print("  " * depth, "eg", keys[:3], keys[-2:])
    if depth < 5 and pre:
        walk(pre[-1], depth + 1)
walk("2026/", 0)
pre, _ = ls("2026/11/10/")
for d in pre: 
    pr2, keys = ls(d)
    print(d, [x[len(d):] for x in pr2][:20], len(keys), keys[:2])
print("== EUMETSAT ages")
import eumdac
from eumdac.errors import EumdacError
tok = eumdac.AccessToken((os.environ["EUMETSAT_CONSUMER_KEY"], os.environ["EUMETSAT_CONSUMER_SECRET"]))
ds = eumdac.DataStore(tok)
now = dt.datetime.utcnow()
for cid in ["EO:EUM:DAT:MSG:HRSEVIRI-IODC", "EO:EUM:DAT:MSG:HRSEVIRI"]:
    col = ds.get_collection(cid)
    for age in [2, 6, 30, 24 * 8]:
        e = now - dt.timedelta(hours=age); prods = list(col.search(dtstart=e - dt.timedelta(minutes=30), dtend=e))
        if not prods: print(cid, age, "none"); continue
        p = prods[0]; nat = [x for x in p.entries if x.endswith(".nat")][0]
        try:
            with p.open(entry=nat) as src: b = src.read(1000)
            print(cid, age, "h OK", p.sensing_start)
        except Exception as ex:
            print(cid, age, "h FAIL", str(ex)[:300])
r = requests.get(f"https://api.eumetsat.int/data/download/1.0.0/collections/{requests.utils.quote('EO:EUM:DAT:MSG:HRSEVIRI-IODC', safe='')}/products/{p}/entry", params={"name": nat}, headers={"Authorization": f"Bearer {tok.access_token}"}, timeout=60, stream=True)
print("raw", r.status_code, r.headers, r.raw.read(800))
