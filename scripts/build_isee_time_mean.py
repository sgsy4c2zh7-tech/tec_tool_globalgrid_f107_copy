#!/usr/bin/env python3
"""Build ISEE Japan same-UTC-time weighted mean VTEC.

ISEE mode in v8.14 intentionally has NO Base/KpB model.

For each UTC time-of-day slot:
  ISEE_mean(tod) = weighted_mean(raw ISEE VTEC at the same UTC time over latest N days)

Forecast:
  Forecast VTEC = ISEE_mean(tod) + F(Kp_forecast)

No historical Kp is subtracted from the ISEE observation mean.

Outputs:
  docs/data/isee_mean/index.json
  docs/data/isee_mean/<HHMM>.json
"""

from __future__ import annotations

import json, math, os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TEC_ROOT = Path(os.environ.get("SWIFTTEC_ISEE_TEC_ROOT", "docs/data/isee_tec"))
OUT = Path(os.environ.get("SWIFTTEC_ISEE_MEAN_ROOT", "docs/data/isee_mean"))
OUT.mkdir(parents=True, exist_ok=True)

MEAN_DAYS = max(1, int(os.environ.get("SWIFTTEC_ISEE_MEAN_DAYS", "10")))
MIN_DAYS = max(1, int(os.environ.get("SWIFTTEC_ISEE_MEAN_MIN_DAYS", "1")))
UTC = timezone.utc

def iso(t):
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00","Z")

def parse_time(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(UTC)
    except Exception:
        return None

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def weight_for_rank(rank_from_latest):
    # newest 1.0, then 0.9 ... floor 0.1
    return max(0.1, 1.0 - 0.1 * rank_from_latest)

def main():
    idx_path = TEC_ROOT / "index.json"
    if not idx_path.exists():
        raise RuntimeError("ISEE TEC index missing")

    idx = load_json(idx_path)
    frames = idx.get("frames") or []
    if not frames:
        raise RuntimeError("ISEE TEC archive empty")

    slots = defaultdict(list)
    for f in frames:
        t = parse_time(f.get("time_utc"))
        rel = f.get("file")
        if not t or not rel:
            continue
        p = TEC_ROOT / rel
        if p.exists():
            slots[t.strftime("%H%M")].append((t, p))

    outputs = []
    now = datetime.now(UTC)

    for hhmm, items in sorted(slots.items()):
        # one frame per UTC date for this exact HHMM slot, newest first
        by_day = {}
        for t, p in sorted(items, reverse=True):
            by_day.setdefault(t.date(), (t, p))
        chosen = list(by_day.values())[:MEAN_DAYS]
        if len(chosen) < MIN_DAYS:
            continue

        first = load_json(chosen[0][1])
        lat_arr = first.get("lat_arr") or []
        lon_arr = first.get("lon_arr") or []
        nlat, nlon = len(lat_arr), len(lon_arr)
        if not nlat or not nlon:
            continue

        sumw = [[0.0] * nlon for _ in range(nlat)]
        sumv = [[0.0] * nlon for _ in range(nlat)]
        count = [[0] * nlon for _ in range(nlat)]
        used = []

        for rank, (t, p) in enumerate(chosen):
            doc = load_json(p)
            grid = doc.get("grid") or []
            if len(grid) != nlat:
                continue
            w = weight_for_rank(rank)
            used.append({"time_utc": iso(t), "weight": w, "file": p.name})

            for i in range(nlat):
                row = grid[i] if i < len(grid) else []
                if len(row) != nlon:
                    continue
                for j in range(nlon):
                    try:
                        v = float(row[j])
                    except Exception:
                        continue
                    if not math.isfinite(v):
                        continue
                    sumv[i][j] += v * w
                    sumw[i][j] += w
                    count[i][j] += 1

        outgrid = []
        for i in range(nlat):
            row = []
            for j in range(nlon):
                if sumw[i][j] > 0:
                    row.append(round(max(0.0, sumv[i][j] / sumw[i][j]), 3))
                else:
                    row.append(None)
            outgrid.append(row)

        outdoc = {
            "version": "swifttec-isee-mean-v1",
            "created_utc": iso(now),
            "slot_utc_hhmm": hhmm,
            "quantity": "ISEE Mean VTEC",
            "units": "TECU",
            "mean_days_requested": MEAN_DAYS,
            "days_used": len(used),
            "weighting": "latest=1.0, then 0.9 ... floor 0.1",
            "kp_component_removed": False,
            "model_rule": "ISEE_mean = weighted_mean(raw ISEE VTEC at same UTC time)",
            "lat_arr": lat_arr,
            "lon_arr": lon_arr,
            "n_lat": nlat,
            "n_lon": nlon,
            "used_days": used,
            "grid": outgrid,
        }
        fname = f"{hhmm}.json"
        (OUT / fname).write_text(
            json.dumps(outdoc, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        outputs.append({
            "slot_utc_hhmm": hhmm,
            "file": fname,
            "days_used": len(used),
        })

    outidx = {
        "version": "swifttec-isee-mean-index-v1",
        "updated_utc": iso(now),
        "quantity": "ISEE Mean VTEC",
        "units": "TECU",
        "mean_days": MEAN_DAYS,
        "kp_component_removed": False,
        "slots": outputs,
    }
    (OUT / "index.json").write_text(
        json.dumps(outidx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Built {len(outputs)} UTC-slot ISEE mean VTEC grids")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
