#!/usr/bin/env python3
"""Train SWIFT-TEC Kp cubic AI corrector.

This script reads archived NOAA TEC frames from docs/data/tec, groups the world into
18 regions (3 latitude bands x 6 longitude bands), and learns month-specific cubic
Kp residual coefficients:

  residual_tecu ~= k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3

The update is blended with previous coefficients to avoid overfitting one noisy day.
Outputs:
  docs/data/ai/kp_coefficients.json
  docs/data/ai/kp_performance.json
  docs/data/ai/kp_learning_history.json
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from urllib.request import Request, urlopen

UTC = timezone.utc
TEC_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_AI_ROOT", "docs/data/ai"))
K_INDEX_URL = os.environ.get("SWIFTTEC_KP_URL", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
TRAIN_DAYS = int(os.environ.get("SWIFTTEC_KP_AI_TRAIN_DAYS", "30"))
MIN_SAMPLES = int(os.environ.get("SWIFTTEC_KP_AI_MIN_SAMPLES", "18"))
BLEND_ALPHA = float(os.environ.get("SWIFTTEC_KP_AI_BLEND_ALPHA", "0.20"))
RIDGE = float(os.environ.get("SWIFTTEC_KP_AI_RIDGE", "0.25"))
HIT_THRESHOLD_TECU = float(os.environ.get("SWIFTTEC_KP_AI_HIT_THRESHOLD", "5.0"))
CORRECTION_CLIP_TECU = float(os.environ.get("SWIFTTEC_KP_AI_CLIP", "20.0"))

LAT_BANDS = [(-90.0, -30.0, "S"), (-30.0, 30.0, "EQ"), (30.0, 90.0, "N")]
LON_BANDS = [(-180.0, -120.0), (-120.0, -60.0), (-60.0, 0.0), (0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


def iso(t: datetime) -> str:
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(s: str) -> datetime | None:
    if not s:
        return None
    text = str(s).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        return dt.astimezone(UTC)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(s).replace("Z", ""), fmt).replace(tzinfo=UTC)
        except Exception:
            continue
    return None


def http_json(url: str):
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-kp-ai-corrector/1.0"})
    with urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def load_json_maybe_gz(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def region_defs():
    regions = []
    rid = 1
    for lat_min, lat_max, lat_label in LAT_BANDS:
        for lon_min, lon_max in LON_BANDS:
            regions.append({
                "id": f"R{rid:02d}",
                "label": f"{lat_label} {lon_min:g}..{lon_max:g}",
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            })
            rid += 1
    return regions


REGIONS = region_defs()


def normalize_lon(lon: float) -> float:
    while lon < -180:
        lon += 360
    while lon >= 180:
        lon -= 360
    return lon


def region_id(lat: float, lon: float) -> str:
    lon = normalize_lon(lon)
    lat_band = 0 if lat < -30 else (1 if lat < 30 else 2)
    lon_band = int(math.floor((lon + 180) / 60))
    lon_band = max(0, min(5, lon_band))
    return f"R{lat_band * 6 + lon_band + 1:02d}"


def load_tec_frames() -> list[dict]:
    idx_path = TEC_ROOT / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} not found")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    frames_meta = idx.get("frames") or []
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=TRAIN_DAYS)
    out = []
    for row in frames_meta:
        t = parse_time(row.get("time_utc"))
        if not t or t < cutoff:
            continue
        rel = row.get("file")
        if not rel:
            continue
        path = TEC_ROOT / rel
        if not path.exists():
            continue
        try:
            frame = load_json_maybe_gz(path)
            frame_time = parse_time(frame.get("time_utc")) or t
            out.append({"time": frame_time, "frame": frame, "file": rel})
        except Exception as exc:
            print(f"WARN: failed to read {path}: {exc}")
    out.sort(key=lambda x: x["time"])
    return out


def parse_kp_json(obj) -> list[tuple[datetime, float]]:
    rows = []
    if not isinstance(obj, list):
        return rows
    header = None
    if obj and isinstance(obj[0], list) and any(str(x).lower() in ("time_tag", "kp") for x in obj[0]):
        header = [str(x).lower() for x in obj[0]]
        data = obj[1:]
    else:
        data = obj
    for row in data:
        t = None
        kp = None
        if isinstance(row, dict):
            t = parse_time(row.get("time_tag") or row.get("time") or row.get("t"))
            kp = row.get("Kp") or row.get("kp") or row.get("value")
        elif isinstance(row, list):
            if header:
                try:
                    ti = header.index("time_tag")
                except ValueError:
                    ti = 0
                try:
                    ki = header.index("kp")
                except ValueError:
                    ki = 1
                if len(row) > max(ti, ki):
                    t = parse_time(row[ti])
                    kp = row[ki]
            elif len(row) >= 2:
                t = parse_time(row[0])
                kp = row[1]
        try:
            kpf = float(kp)
        except Exception:
            continue
        if t and math.isfinite(kpf):
            rows.append((t, kpf))
    rows.sort(key=lambda x: x[0])
    return rows


def kp_at(t: datetime, kp_rows: list[tuple[datetime, float]]) -> float | None:
    if not kp_rows:
        return None
    best = None
    best_diff = 10**18
    for kt, kv in kp_rows:
        diff = abs((kt - t).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best = kv
    if best is None or best_diff > 4 * 3600:
        return None
    return best


def region_means(frame: dict) -> dict[str, float]:
    lat_arr = frame.get("lat_arr") or []
    lon_arr = frame.get("lon_arr") or []
    grid = frame.get("grid") or []
    sums = defaultdict(float)
    counts = defaultdict(int)
    for i, lat in enumerate(lat_arr):
        row = grid[i] if i < len(grid) else []
        for j, lon in enumerate(lon_arr):
            try:
                v = float(row[j])
            except Exception:
                continue
            if not math.isfinite(v) or v < 0:
                continue
            rid = region_id(float(lat), float(lon))
            sums[rid] += v
            counts[rid] += 1
    return {rid: sums[rid] / counts[rid] for rid in counts if counts[rid] > 0}


def solve4(a, b):
    # Gaussian elimination for 4x4.
    n = 4
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return [0.0, 0.0, 0.0, 0.0]
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        for k in range(col, n + 1):
            m[col][k] /= div
        for r in range(n):
            if r == col:
                continue
            fac = m[r][col]
            for k in range(col, n + 1):
                m[r][k] -= fac * m[col][k]
    return [m[i][n] for i in range(n)]


def fit_cubic(records):
    # records: [(x, y)]
    xtx = [[0.0] * 4 for _ in range(4)]
    xty = [0.0] * 4
    for x, y in records:
        phi = [1.0, x, x * x, x * x * x]
        for i in range(4):
            xty[i] += phi[i] * y
            for j in range(4):
                xtx[i][j] += phi[i] * phi[j]
    for i in range(4):
        xtx[i][i] += RIDGE
    return solve4(xtx, xty)


def predict(coeffs, kp: float) -> float:
    x = kp - 3.0
    y = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x
    return max(-CORRECTION_CLIP_TECU, min(CORRECTION_CLIP_TECU, y))


def rmse(vals):
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def old_coeff_map() -> dict:
    p = AI_ROOT / "kp_coefficients.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("coefficients") or {}
    except Exception:
        return {}


def blend_coeff(old, new):
    if old is None:
        return new
    return [(1 - BLEND_ALPHA) * float(old[i]) + BLEND_ALPHA * float(new[i]) for i in range(4)]


def main() -> int:
    AI_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(microsecond=0)
    frames = load_tec_frames()
    print(f"Loaded TEC frames: {len(frames)}")
    kp_rows = parse_kp_json(http_json(K_INDEX_URL))
    print(f"Loaded Kp rows: {len(kp_rows)}")

    samples = []
    baseline_bucket = defaultdict(list)
    for item in frames:
        t = item["time"]
        kp = kp_at(t, kp_rows)
        if kp is None:
            continue
        month = t.month
        hour = t.hour
        means = region_means(item["frame"])
        for rid, val in means.items():
            baseline_bucket[(rid, month, hour)].append(val)
            samples.append({"time": t, "region_id": rid, "month": month, "hour": hour, "kp": kp, "tec": val})

    if not samples:
        print("No samples with matching Kp. Writing empty AI files.")
    baselines = {k: median(v) for k, v in baseline_bucket.items() if v}

    groups = defaultdict(list)
    perf_records = defaultdict(list)
    for s in samples:
        base = baselines.get((s["region_id"], s["month"], s["hour"]), s["tec"])
        residual = s["tec"] - base
        groups[(s["region_id"], s["month"])].append((s["kp"] - 3.0, residual))
        perf_records[(s["region_id"], s["month"])].append((s["kp"], residual))

    old = old_coeff_map()
    coeffs_out = {}
    metrics_out = {}
    updated_groups = 0
    all_raw_err = []
    all_corr_err = []

    for reg in REGIONS:
        rid = reg["id"]
        coeffs_out[rid] = {}
        metrics_out[rid] = {}
        for month in range(1, 13):
            recs = groups.get((rid, month), [])
            old_obj = (old.get(rid) or {}).get(str(month)) or {}
            old_vals = [float(old_obj.get(f"k{i}", 0.0)) for i in range(4)] if old_obj else None
            if len(recs) >= MIN_SAMPLES:
                fitted = fit_cubic(recs)
                final = blend_coeff(old_vals, fitted)
                updated_groups += 1
            else:
                final = old_vals or [0.0, 0.0, 0.0, 0.0]

            perf = perf_records.get((rid, month), [])
            raw_errors = [r for _, r in perf]
            corr_errors = [r - predict(final, kp) for kp, r in perf]
            all_raw_err.extend(raw_errors)
            all_corr_err.extend(corr_errors)
            coeffs_out[rid][str(month)] = {
                "k0": round(final[0], 6),
                "k1": round(final[1], 6),
                "k2": round(final[2], 6),
                "k3": round(final[3], 6),
                "sample_count": len(perf),
                "updated": len(recs) >= MIN_SAMPLES,
            }
            metrics_out[rid][str(month)] = {
                "sample_count": len(perf),
                "raw_rmse": None if not raw_errors else round(rmse(raw_errors), 4),
                "corrected_rmse": None if not corr_errors else round(rmse(corr_errors), 4),
                "raw_bias": None if not raw_errors else round(mean(raw_errors), 4),
                "corrected_bias": None if not corr_errors else round(mean(corr_errors), 4),
                "raw_hit_rate": 0 if not raw_errors else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in raw_errors) / len(raw_errors), 4),
                "corrected_hit_rate": 0 if not corr_errors else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in corr_errors) / len(corr_errors), 4),
            }

    coeff_doc = {
        "version": "swifttec-kp-cubic-ai-v1",
        "updated_utc": iso(now),
        "train_days": TRAIN_DAYS,
        "regions": REGIONS,
        "model": {
            "formula": "delta_tecu = k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3",
            "x": "Kp - 3",
            "blend_alpha": BLEND_ALPHA,
            "ridge": RIDGE,
            "min_samples": MIN_SAMPLES,
            "correction_clip_tecu": CORRECTION_CLIP_TECU,
        },
        "coefficients": coeffs_out,
    }
    perf_doc = {
        "version": "swifttec-kp-cubic-ai-performance-v1",
        "updated_utc": iso(now),
        "hit_threshold_tecu": HIT_THRESHOLD_TECU,
        "summary": {
            "sample_count": len(all_raw_err),
            "updated_groups": updated_groups,
            "raw_rmse": None if not all_raw_err else round(rmse(all_raw_err), 4),
            "corrected_rmse": None if not all_corr_err else round(rmse(all_corr_err), 4),
            "raw_hit_rate": 0 if not all_raw_err else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in all_raw_err) / len(all_raw_err), 4),
            "corrected_hit_rate": 0 if not all_corr_err else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in all_corr_err) / len(all_corr_err), 4),
        },
        "metrics": metrics_out,
    }

    hist_path = AI_ROOT / "kp_learning_history.json"
    try:
        hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {"version": "swifttec-kp-cubic-ai-history-v1", "runs": []}
    except Exception:
        hist = {"version": "swifttec-kp-cubic-ai-history-v1", "runs": []}
    hist.setdefault("runs", []).append({"time_utc": iso(now), **perf_doc["summary"]})
    hist["runs"] = hist["runs"][-120:]

    (AI_ROOT / "kp_coefficients.json").write_text(json.dumps(coeff_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (AI_ROOT / "kp_performance.json").write_text(json.dumps(perf_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kp AI training complete: samples={len(all_raw_err)}, updated_groups={updated_groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
