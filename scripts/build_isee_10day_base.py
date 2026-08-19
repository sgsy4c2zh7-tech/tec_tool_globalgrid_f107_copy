#!/usr/bin/env python3
"""Build Japan 10-day weighted Base VTEC from ISEE archive.

Model rule
----------
For each Japan grid cell and UTC time-of-day slot:

  Base_d = VTEC_d - F(Kp_d)
  Base10 = weighted_mean(Base_d over latest 10 available days)
  Forecast = Base10 + F(Kp_forecast)

where:
  F(Kp) = k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3

The coefficients are read from:
  docs/data/ai/isee_japan/kp_grid_coefficients.json

Outputs:
  docs/data/isee_base/index.json
  docs/data/isee_base/<HHMM>.json

Each output grid is VTEC [TECU].
"""

from __future__ import annotations

import json, math, os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

TEC_ROOT = Path(os.environ.get("SWIFTTEC_ISEE_TEC_ROOT", "docs/data/isee_tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_ISEE_AI_ROOT", "docs/data/ai/isee_japan"))
OUT = Path(os.environ.get("SWIFTTEC_ISEE_BASE_ROOT", "docs/data/isee_base"))
OUT.mkdir(parents=True, exist_ok=True)

BASE_DAYS = int(os.environ.get("SWIFTTEC_ISEE_BASE_DAYS", "10"))
MIN_DAYS = int(os.environ.get("SWIFTTEC_ISEE_BASE_MIN_DAYS", "4"))
KP_URL = os.environ.get(
    "SWIFTTEC_KP_ACTUAL_URL",
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
)

UTC = timezone.utc

def iso(t):
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00","Z")

def parse_time(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(UTC)
    except Exception:
        return None

def fetch_json(url):
    req = Request(url, headers={"User-Agent":"SWIFT-TEC/8.9 ISEE-Base"})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def parse_kp(obj):
    out=[]
    if isinstance(obj,list):
        for row in obj:
            if isinstance(row,list) and len(row)>=2:
                t=parse_time(row[0])
                try: kp=float(row[1])
                except Exception: continue
                if t and math.isfinite(kp): out.append((t,kp))
            elif isinstance(row,dict):
                t=parse_time(row.get("time_tag") or row.get("time"))
                try: kp=float(row.get("kp_index") or row.get("kp"))
                except Exception: continue
                if t and math.isfinite(kp): out.append((t,kp))
    out.sort()
    return out

def kp_at(t, rows, max_h=2.0):
    if not rows: return None
    best=min(rows, key=lambda x: abs((x[0]-t).total_seconds()))
    if abs((best[0]-t).total_seconds()) > max_h*3600:
        return None
    return float(best[1])

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def coeff_for(doc, month, i, j):
    root=doc.get("coefficients_grid") or doc.get("grid_coefficients") or {}
    m=root.get(str(month)) or root.get(month) or {}
    out=[]
    for k in ("k0","k1","k2","k3"):
        try:
            v=float(m[k][i][j])
            out.append(v if math.isfinite(v) else 0.0)
        except Exception:
            out.append(0.0)
    return out

def F(coeff, kp):
    x=float(kp)-3.0
    return coeff[0]+coeff[1]*x+coeff[2]*x*x+coeff[3]*x*x*x

def weight_for_rank(rank_from_latest):
    # Latest day weight 1.0, then 0.9 ... down to 0.1
    return max(0.1, 1.0 - 0.1*rank_from_latest)

def main():
    idx_path=TEC_ROOT/"index.json"
    coeff_path=AI_ROOT/"kp_grid_coefficients.json"
    if not idx_path.exists():
        raise RuntimeError("ISEE TEC index missing")
    if not coeff_path.exists():
        raise RuntimeError("Japan AI coefficient grid missing. Train ISEE Japan AI Corrector first.")

    idx=load_json(idx_path)
    coeff=load_json(coeff_path)
    frames=idx.get("frames") or []
    if not frames:
        raise RuntimeError("ISEE TEC archive empty")

    # actual Kp
    kp_rows=parse_kp(fetch_json(KP_URL))

    # group by UTC HHMM slot
    slots=defaultdict(list)
    for f in frames:
        t=parse_time(f.get("time_utc"))
        rel=f.get("file")
        if not t or not rel: continue
        p=TEC_ROOT/rel
        if not p.exists(): continue
        slots[t.strftime("%H%M")].append((t,p))

    outputs=[]
    now=datetime.now(UTC)
    for hhmm, items in sorted(slots.items()):
        # one frame per UTC date, newest first
        by_day={}
        for t,p in sorted(items, reverse=True):
            by_day.setdefault(t.date(), (t,p))
        chosen=list(by_day.values())[:BASE_DAYS]
        if len(chosen) < MIN_DAYS:
            continue

        first=load_json(chosen[0][1])
        lat_arr=first.get("lat_arr") or []
        lon_arr=first.get("lon_arr") or []
        nlat=len(lat_arr); nlon=len(lon_arr)
        if not nlat or not nlon: continue

        sumw=[[0.0]*nlon for _ in range(nlat)]
        sumv=[[0.0]*nlon for _ in range(nlat)]
        count=[[0]*nlon for _ in range(nlat)]

        used=[]
        for rank,(t,p) in enumerate(chosen):
            doc=load_json(p)
            grid=doc.get("grid") or []
            if len(grid)!=nlat: continue
            kp=kp_at(t,kp_rows)
            if kp is None:
                continue
            w=weight_for_rank(rank)
            month=t.month
            used.append({"time_utc":iso(t),"kp":kp,"weight":w,"file":p.name})
            for i in range(nlat):
                row=grid[i] if i < len(grid) else []
                if len(row)!=nlon: continue
                for j in range(nlon):
                    try:
                        v=float(row[j])
                    except Exception:
                        continue
                    if not math.isfinite(v): continue
                    cf=coeff_for(coeff,month,i,j)
                    base=v-F(cf,kp)
                    if not math.isfinite(base): continue
                    sumv[i][j]+=base*w
                    sumw[i][j]+=w
                    count[i][j]+=1

        outgrid=[]
        for i in range(nlat):
            row=[]
            for j in range(nlon):
                if sumw[i][j] > 0:
                    row.append(round(max(0.0,sumv[i][j]/sumw[i][j]),3))
                else:
                    row.append(None)
            outgrid.append(row)

        outdoc={
            "version":"swifttec-isee-base-v1",
            "created_utc":iso(now),
            "slot_utc_hhmm":hhmm,
            "quantity":"Base VTEC",
            "units":"TECU",
            "base_days_requested":BASE_DAYS,
            "days_used":len(used),
            "weighting":"latest=1.0, then 0.9 ... floor 0.1",
            "kp_component_removed":True,
            "model_rule":"Base_d = ISEE_VTEC_d - F(Kp_d); Base10 = weighted_mean(Base_d)",
            "lat_arr":lat_arr,
            "lon_arr":lon_arr,
            "n_lat":nlat,
            "n_lon":nlon,
            "used_days":used,
            "grid":outgrid,
        }
        fname=f"{hhmm}.json"
        (OUT/fname).write_text(json.dumps(outdoc,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
        outputs.append({"slot_utc_hhmm":hhmm,"file":fname,"days_used":len(used)})

    outidx={
        "version":"swifttec-isee-base-index-v1",
        "updated_utc":iso(now),
        "quantity":"Base VTEC",
        "units":"TECU",
        "base_days":BASE_DAYS,
        "slots":outputs,
    }
    (OUT/"index.json").write_text(json.dumps(outidx,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Built {len(outputs)} UTC-slot Base VTEC grids")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
