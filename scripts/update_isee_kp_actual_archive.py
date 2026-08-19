#!/usr/bin/env python3
"""Append NOAA observed planetary Kp to a persistent archive.

ISEE can be published several days late. A persistent Kp archive lets the
ISEE hindcast verifier match delayed ISEE observations with the Kp that was
actually observed at the same UTC time.

Output:
  docs/data/ai/isee_japan/kp_actual_archive.json
"""
from __future__ import annotations

import json, math, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

UTC = timezone.utc
OUT = Path(os.environ.get(
    "SWIFTTEC_ISEE_KP_ARCHIVE",
    "docs/data/ai/isee_japan/kp_actual_archive.json",
))
URL = os.environ.get(
    "SWIFTTEC_KP_ACTUAL_URL",
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
)
KEEP_DAYS = max(30, int(os.environ.get("SWIFTTEC_KP_ACTUAL_KEEP_DAYS", "120")))

def iso(t):
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def parse_time(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(UTC)
    except Exception:
        return None

def fetch_json(url):
    req = Request(url, headers={"User-Agent":"SWIFT-TEC/8.16 ISEE-Kp-Archive"})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def parse_rows(obj):
    out=[]
    if not isinstance(obj, list):
        return out
    for row in obj:
        if isinstance(row, list) and len(row) >= 2:
            t = parse_time(row[0])
            try: kp = float(row[1])
            except Exception: continue
        elif isinstance(row, dict):
            t = parse_time(row.get("time_tag") or row.get("time") or row.get("t"))
            try: kp = float(row.get("kp_index") or row.get("kp") or row.get("Kp"))
            except Exception: continue
        else:
            continue
        if t and math.isfinite(kp):
            out.append((t, kp))
    return out

def main():
    merged = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
            for row in old.get("rows", []):
                t = parse_time(row.get("time_utc"))
                try: kp = float(row.get("kp"))
                except Exception: continue
                if t and math.isfinite(kp):
                    merged[iso(t)] = kp
        except Exception:
            pass

    new_rows = parse_rows(fetch_json(URL))
    for t, kp in new_rows:
        merged[iso(t)] = kp

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=KEEP_DAYS)
    rows = []
    for ts, kp in sorted(merged.items()):
        t = parse_time(ts)
        if t and t >= cutoff:
            rows.append({"time_utc": iso(t), "kp": round(float(kp), 3)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version":"swifttec-isee-kp-actual-archive-v1",
        "updated_utc":iso(now),
        "source":"NOAA SWPC planetary K-index",
        "keep_days":KEEP_DAYS,
        "rows":rows,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kp actual archive: {len(rows)} rows; newly fetched={len(new_rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
