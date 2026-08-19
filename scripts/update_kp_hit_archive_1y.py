#!/usr/bin/env python3
"""Maintain a lightweight 365-day Kp-bin hit-rate archive for NOAA / Global.

This intentionally stores ONLY what is required for the Kp-bin ±5 TECU
hit-rate display:

    N
    raw_hit
    corrected_hit

for each UTC day and each KpF bin.

It does NOT store Bias / MAE / RMSE history and does NOT store every
grid-cell case row. All valid grid-cell cases are counted, then compressed
into daily counters.

Outputs
-------
docs/data/ai/kp_hit_archive_1y.json
docs/data/ai/kp_actual_archive_recent.json

Also updates:
docs/data/ai/kp_performance.json
  -> adds `kp_bins_1y_hit`
  -> adds `kp_hit_archive`
  -> keeps existing `kp_bins` untouched for short-window Bias/MAE/RMSE.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

UTC = timezone.utc

TEC_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_AI_ROOT", "docs/data/ai"))
GRID_COEFF = AI_ROOT / "kp_grid_coefficients.json"
PERF_PATH = AI_ROOT / "kp_performance.json"

HIT_ARCHIVE = AI_ROOT / "kp_hit_archive_1y.json"
KP_ARCHIVE = AI_ROOT / "kp_actual_archive_recent.json"

K_INDEX_URL = os.environ.get(
    "SWIFTTEC_KP_URL",
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
)

KEEP_DAYS = max(30, int(os.environ.get("SWIFTTEC_KP_HIT_KEEP_DAYS", "365")))
SCAN_DAYS = max(2, int(os.environ.get("SWIFTTEC_KP_HIT_SCAN_DAYS", "10")))
KP_KEEP_DAYS = max(SCAN_DAYS + 10, int(os.environ.get("SWIFTTEC_KP_ACTUAL_KEEP_DAYS", "60")))

PAIR_HOURS = float(os.environ.get("SWIFTTEC_KP_AI_PAIR_HOURS", "24"))
PAIR_TOLERANCE_MIN = int(os.environ.get("SWIFTTEC_KP_AI_PAIR_TOLERANCE_MIN", "20"))
CLIP = float(os.environ.get("SWIFTTEC_KP_AI_CLIP", "20.0"))
HIT_THRESHOLD = float(os.environ.get("SWIFTTEC_KP_HIT_THRESHOLD", "5.0"))

KP_BINS = [
    ("0-2", 0.0, 2.0),
    ("2-3", 2.0, 3.0),
    ("3-4", 3.0, 4.0),
    ("4-5", 4.0, 5.0),
    ("5-6", 5.0, 6.0),
    ("6-7", 6.0, 7.0),
    ("7+", 7.0, 99.0),
]


def iso(t: datetime) -> str:
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_maybe_gz(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return load_json(path)


def http_json(url: str):
    req = Request(url, headers={"User-Agent": "SWIFT-TEC/8.20 lightweight-Kp-hit"})
    with urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8", "replace"))


def parse_kp_json(obj):
    out = []
    if not isinstance(obj, list):
        return out
    for row in obj:
        if isinstance(row, list) and len(row) >= 2:
            t = parse_time(row[0])
            try:
                kp = float(row[1])
            except Exception:
                continue
        elif isinstance(row, dict):
            t = parse_time(row.get("time_tag") or row.get("time") or row.get("t"))
            try:
                kp = float(row.get("kp_index") or row.get("kp") or row.get("Kp"))
            except Exception:
                continue
        else:
            continue
        if t and math.isfinite(kp):
            out.append((t, kp))
    out.sort(key=lambda x: x[0])
    return out


def update_recent_kp_archive(now: datetime):
    merged = {}
    if KP_ARCHIVE.exists():
        try:
            old = load_json(KP_ARCHIVE)
            for row in old.get("rows", []):
                t = parse_time(row.get("time_utc"))
                try:
                    kp = float(row.get("kp"))
                except Exception:
                    continue
                if t and math.isfinite(kp):
                    merged[iso(t)] = kp
        except Exception:
            pass

    fresh = parse_kp_json(http_json(K_INDEX_URL))
    for t, kp in fresh:
        merged[iso(t)] = float(kp)

    cutoff = now - timedelta(days=KP_KEEP_DAYS)
    rows = []
    for ts, kp in sorted(merged.items()):
        t = parse_time(ts)
        if t and t >= cutoff:
            rows.append({"time_utc": iso(t), "kp": round(float(kp), 3)})

    KP_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    KP_ARCHIVE.write_text(
        json.dumps({
            "version": "swifttec-kp-actual-recent-v1",
            "updated_utc": iso(now),
            "keep_days": KP_KEEP_DAYS,
            "rows": rows,
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return [(parse_time(r["time_utc"]), float(r["kp"])) for r in rows]


def nearest_kp(t: datetime, rows):
    if not rows:
        return None
    times = [x[0] for x in rows]
    pos = bisect_left(times, t)
    cand = []
    if pos < len(rows):
        cand.append(rows[pos])
    if pos > 0:
        cand.append(rows[pos - 1])
    if not cand:
        return None
    best = min(cand, key=lambda x: abs((x[0] - t).total_seconds()))
    if abs((best[0] - t).total_seconds()) > 2 * 3600:
        return None
    return float(best[1])


def kp_bin_label(kp):
    x = float(kp)
    for label, lo, hi in KP_BINS:
        if lo <= x < hi:
            return label
    return "7+"


def load_frame_meta(now: datetime):
    idx_path = TEC_ROOT / "index.json"
    if not idx_path.exists():
        return []
    idx = load_json(idx_path)
    cutoff = now - timedelta(days=SCAN_DAYS + 2)
    out = []
    for f in idx.get("frames") or []:
        t = parse_time(f.get("time_utc") or f.get("time"))
        rel = f.get("file") or f.get("path")
        if not t or not rel or t < cutoff:
            continue
        p = TEC_ROOT / rel
        if p.exists():
            out.append({"time": t, "path": p, "file": rel})
    out.sort(key=lambda x: x["time"])
    return out


def nearest_meta(metas, target: datetime):
    best = None
    best_diff = float("inf")
    for m in metas:
        d = abs((m["time"] - target).total_seconds())
        if d < best_diff:
            best = m
            best_diff = d
    if best and best_diff <= PAIR_TOLERANCE_MIN * 60:
        return best
    return None


def flatten_grid(frame):
    vals = []
    for row in frame.get("grid") or []:
        for v in row:
            try:
                x = float(v)
                vals.append(x if math.isfinite(x) else float("nan"))
            except Exception:
                vals.append(float("nan"))
    return vals


def coeff_at(month_grid, i, j):
    out = []
    for k in ("k0", "k1", "k2", "k3"):
        try:
            x = float(month_grid.get(k, [])[i][j])
        except Exception:
            x = 0.0
        out.append(x if math.isfinite(x) else 0.0)
    return out


def F(cf, kp):
    x = float(kp) - 3.0
    v = cf[0] + cf[1]*x + cf[2]*x*x + cf[3]*x*x*x
    if not math.isfinite(v):
        return 0.0
    return max(-CLIP, min(CLIP, v))


def empty_count():
    return {"n": 0, "raw_hit": 0, "corrected_hit": 0}


def add_count(dst, n=0, raw_hit=0, corrected_hit=0):
    dst["n"] = int(dst.get("n", 0)) + int(n)
    dst["raw_hit"] = int(dst.get("raw_hit", 0)) + int(raw_hit)
    dst["corrected_hit"] = int(dst.get("corrected_hit", 0)) + int(corrected_hit)


def score_recent(now, metas, kp_rows, grid_doc):
    cutoff = now - timedelta(days=SCAN_DAYS)
    root = grid_doc.get("coefficients_grid") or grid_doc.get("grid_coefficients") or {}
    daily = defaultdict(lambda: {label: empty_count() for label, _, _ in KP_BINS})

    for mf in (m for m in metas if m["time"] >= cutoff):
        mb = nearest_meta(metas, mf["time"] - timedelta(hours=PAIR_HOURS))
        if not mb:
            continue

        kp_f = nearest_kp(mf["time"], kp_rows)
        kp_b = nearest_kp(mb["time"], kp_rows)
        if kp_f is None or kp_b is None:
            continue

        try:
            ff = load_json_maybe_gz(mf["path"])
            fb = load_json_maybe_gz(mb["path"])
        except Exception:
            continue

        vf = flatten_grid(ff)
        vb = flatten_grid(fb)
        lat_arr = ff.get("lat_arr") or ff.get("latArr") or []
        lon_arr = ff.get("lon_arr") or ff.get("lonArr") or []
        n_lat, n_lon = len(lat_arr), len(lon_arr)
        if not n_lat or not n_lon or len(vf) != n_lat * n_lon or len(vb) != len(vf):
            continue

        mg = root.get(str(mf["time"].month)) or {}
        if not mg:
            continue

        label = kp_bin_label(kp_f)
        day = mf["time"].date().isoformat()
        dst = daily[day][label]

        for idx, (obs, base) in enumerate(zip(vf, vb)):
            if not math.isfinite(obs) or not math.isfinite(base):
                continue

            i, j = divmod(idx, n_lon)
            cf = coeff_at(mg, i, j)

            raw_forecast = base
            corrected_forecast = base - F(cf, kp_b) + F(cf, kp_f)

            raw_err = obs - raw_forecast
            corr_err = obs - corrected_forecast
            if not math.isfinite(raw_err) or not math.isfinite(corr_err):
                continue

            dst["n"] += 1
            if abs(raw_err) <= HIT_THRESHOLD:
                dst["raw_hit"] += 1
            if abs(corr_err) <= HIT_THRESHOLD:
                dst["corrected_hit"] += 1

    return daily


def aggregate_days(days):
    totals = {label: empty_count() for label, _, _ in KP_BINS}
    total_n = 0
    total_hit = 0

    for day in days:
        for label, cnt in (day.get("b") or {}).items():
            if label not in totals:
                continue
            add_count(
                totals[label],
                cnt.get("n", 0),
                cnt.get("r", 0),
                cnt.get("c", 0),
            )

    bins = {}
    for label, _, _ in KP_BINS:
        x = totals[label]
        n = int(x["n"])
        raw = int(x["raw_hit"])
        corr = int(x["corrected_hit"])
        total_n += n
        total_hit += corr
        bins[label] = {
            "kp_bin": label,
            "sample_count": n,
            "raw_hit_count": raw,
            "corrected_hit_count": corr,
            "raw_hit_rate": None if n <= 0 else round(raw / n, 6),
            "corrected_hit_rate": None if n <= 0 else round(corr / n, 6),
            "threshold_tecu": HIT_THRESHOLD,
        }

    return bins, total_n, total_hit


def main():
    AI_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(microsecond=0)

    if not GRID_COEFF.exists():
        raise RuntimeError("kp_grid_coefficients.json missing.")
    if not PERF_PATH.exists():
        raise RuntimeError("kp_performance.json missing.")

    kp_rows = update_recent_kp_archive(now)
    metas = load_frame_meta(now)
    if len(metas) < 2:
        raise RuntimeError("Not enough TEC frames for hit-rate update.")

    grid_doc = load_json(GRID_COEFF)
    daily_new = score_recent(now, metas, kp_rows, grid_doc)

    try:
        old = load_json(HIT_ARCHIVE) if HIT_ARCHIVE.exists() else {"days": []}
    except Exception:
        old = {"days": []}

    existing = {d.get("d"): d for d in old.get("days", []) if d.get("d")}

    # Re-scan recent days and REPLACE them, never append duplicate counts.
    for day, bins in daily_new.items():
        existing[day] = {
            "d": day,
            "b": {
                label: {
                    "n": int(x["n"]),
                    "r": int(x["raw_hit"]),
                    "c": int(x["corrected_hit"]),
                }
                for label, x in bins.items()
            },
        }

    cutoff_date = (now - timedelta(days=KEEP_DAYS)).date()
    kept = []
    for day, obj in existing.items():
        try:
            dd = datetime.fromisoformat(day).date()
        except Exception:
            continue
        if dd >= cutoff_date:
            kept.append(obj)
    kept.sort(key=lambda x: x["d"])

    bins_1y, total_n, total_hit = aggregate_days(kept)

    doc = {
        "v": 1,
        "updated_utc": iso(now),
        "window_days": KEEP_DAYS,
        "threshold_tecu": HIT_THRESHOLD,
        "all_valid_grid_cases_counted": True,
        "first_date_utc": kept[0]["d"] if kept else None,
        "last_date_utc": kept[-1]["d"] if kept else None,
        "days_retained": len(kept),
        "sample_count": total_n,
        "corrected_hit_count": total_hit,
        "days": kept,
    }
    HIT_ARCHIVE.write_text(
        json.dumps(doc, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    perf = load_json(PERF_PATH)
    perf["kp_bins_1y_hit"] = bins_1y
    perf["kp_hit_archive"] = {
        "source": "kp_hit_archive_1y.json",
        "window_days": KEEP_DAYS,
        "threshold_tecu": HIT_THRESHOLD,
        "all_valid_grid_cases_counted": True,
        "days_retained": len(kept),
        "first_date_utc": doc["first_date_utc"],
        "last_date_utc": doc["last_date_utc"],
        "sample_count": total_n,
    }
    PERF_PATH.write_text(json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Lightweight 1-year Kp hit archive: days={len(kept)}, "
        f"N={total_n}, file={HIT_ARCHIVE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
