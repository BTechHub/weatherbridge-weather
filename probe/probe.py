"""Temporary: learn what EUMETSAT, OPERA and NWS give us from GitHub's network (not reachable from dev machines)."""
import datetime as dt, os, json, requests
print("== NWS")
r = requests.get("https://api.weather.gov/alerts/active?point=35.4,-97.5", headers={"User-Agent": "(WeatherBridge, lorjova.com)"}, timeout=30)
print(r.status_code, r.headers.get("content-type"), r.text[:1500])
r = requests.get("https://api.weather.gov/alerts/active?status=actual&message_type=alert", headers={"User-Agent": "(WeatherBridge, lorjova.com)"}, timeout=60)
j = r.json(); print("all active:", len(j.get("features", [])), "bytes", len(r.content), "with geometry", sum(1 for f in j["features"] if f.get("geometry")))
print("== OPERA")
for u in ["https://s3.waw3-1.cloudferro.com/openradar-24h/?list-type=2&delimiter=/&max-keys=50"]:
    r = requests.get(u, timeout=30); print(r.status_code, r.text[:3000])
print("== EUMETSAT")
import eumdac
tok = eumdac.AccessToken((os.environ["EUMETSAT_CONSUMER_KEY"], os.environ["EUMETSAT_CONSUMER_SECRET"]))
print("token ok", str(tok.expiration))
ds = eumdac.DataStore(tok)
for c in ds.collections:
    t = str(c.title)
    if any(k in t for k in ["SEVIRI", "FCI", "Meteosat"]) and "Level 1" in t or "HRSEVIRI" in str(c) or "FCI" in str(c):
        print(c, "|", t)
end = dt.datetime.utcnow(); start = end - dt.timedelta(hours=3)
for cid in ["EO:EUM:DAT:MSG:HRSEVIRI", "EO:EUM:DAT:MSG:HRSEVIRI-IODC", "EO:EUM:DAT:0662", "EO:EUM:DAT:0665"]:
    try:
        col = ds.get_collection(cid)
        prods = list(col.search(dtstart=start, dtend=end))
        print(cid, len(prods))
        for p in prods[:3]:
            print("  ", p, p.sensing_start, p.size, list(p.entries)[:6] if hasattr(p, "entries") else "")
    except Exception as e:
        print(cid, "ERR", e)
