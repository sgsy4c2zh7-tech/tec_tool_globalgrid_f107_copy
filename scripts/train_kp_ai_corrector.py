#!/usr/bin/env python3
"""Train SWIFT-TEC Kp AI coefficients following the user's Base-Kp rule.

User model:
  BaseTEC(t0, cell) = ObservedTEC(t0, cell) - F_Kp(KpB(t0), cell, month)
  ForecastTEC(t1, cell) = BaseTEC(t0, cell) + F_Kp(KpF(t1), cell, month)

Therefore, for training pairs where t0 is the previous-day same UTC slot:
  ObservedTEC(t1) - ObservedTEC(t0)
    ~= F_Kp(Kp_actual(t1)) - F_Kp(Kp_actual(t0))

The AI fits F_Kp for every grid cell and month:
  F_Kp(Kp) = k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3

Outputs:
  docs/data/ai/kp_grid_coefficients.json   # full-grid coefficients used by forecast correction
  docs/data/ai/kp_coefficients.json        # 18-region aggregated coefficients for UI display
  docs/data/ai/kp_performance.json         # 18-region hit-rate/RMSE metrics
  docs/data/ai/kp_learning_history.json    # up to ~2 years of daily trend points
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

UTC = timezone.utc

TEC_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_AI_ROOT", "docs/data/ai"))
K_INDEX_URL = os.environ.get("SWIFTTEC_KP_URL", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")

TRAIN_DAYS = int(os.environ.get("SWIFTTEC_KP_AI_TRAIN_DAYS", "30"))
PAIR_HOURS = float(os.environ.get("SWIFTTEC_KP_AI_PAIR_HOURS", "24"))
PAIR_TOLERANCE_MIN = int(os.environ.get("SWIFTTEC_KP_AI_PAIR_TOLERANCE_MIN", "20"))

MIN_CELL_SAMPLES = int(os.environ.get("SWIFTTEC_KP_AI_MIN_CELL_SAMPLES", "6"))
BLEND_ALPHA = float(os.environ.get("SWIFTTEC_KP_AI_BLEND_ALPHA", "0.20"))
RIDGE = float(os.environ.get("SWIFTTEC_KP_AI_RIDGE", "0.35"))
HIT_THRESHOLD_TECU = float(os.environ.get("SWIFTTEC_KP_AI_HIT_THRESHOLD", "5.0"))
CORRECTION_CLIP_TECU = float(os.environ.get("SWIFTTEC_KP_AI_CLIP", "20.0"))
HISTORY_MAX_RUNS = int(os.environ.get("SWIFTTEC_KP_AI_HISTORY_MAX_RUNS", "760"))

LAT_BANDS = [(-90.0, -30.0, "S"), (-30.0, 30.0, "EQ"), (30.0, 90.0, "N")]
LON_BANDS = [(-180.0, -120.0), (-120.0, -60.0), (-60.0, 0.0), (0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


def iso(t: datetime) -> str:
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(s: str) -> datetime | None:
    if not s:
        return None
    text = str(s).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(UTC)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(s).replace("Z", ""), fmt).replace(tzinfo=UTC)
        except Exception:
            continue
    return None


def http_json(url: str):
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-kp-model-rule-ai/1.0"})
    with urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def load_json_maybe_gz(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame_meta() -> list[dict]:
    idx_path = TEC_ROOT / "index.json"
    if not idx_path.exists():
        return []
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    frames = idx.get("frames") or []
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=TRAIN_DAYS + 2)
    out = []
    for f in frames:
        t = parse_time(f.get("time_utc") or f.get("time"))
        if not t or t < cutoff:
            continue
        rel = f.get("file") or f.get("path")
        if not rel:
            continue
        path = TEC_ROOT / rel
        if path.exists():
            out.append({"time": t, "path": path, "file": rel})
    out.sort(key=lambda x: x["time"])
    return out


def nearest_meta(metas: list[dict], target: datetime, tolerance_min: int) -> dict | None:
    best = None
    best_diff = float("inf")
    for m in metas:
        d = abs((m["time"] - target).total_seconds())
        if d < best_diff:
            best_diff = d
            best = m
    if best and best_diff <= tolerance_min * 60:
        return best
    return None


def flatten_grid(frame: dict):
    lat_arr = frame.get("lat_arr") or frame.get("latArr") or []
    lon_arr = frame.get("lon_arr") or frame.get("lonArr") or []
    grid = frame.get("grid") or []
    vals: list[float] = []
    for row in grid:
        for v in row:
            try:
                x = float(v)
                vals.append(x if math.isfinite(x) else float("nan"))
            except Exception:
                vals.append(float("nan"))
    return lat_arr, lon_arr, vals


def parse_kp_json(obj) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    if not isinstance(obj, list):
        return rows
    for r in obj:
        if isinstance(r, list) and len(r) >= 2:
            t = parse_time(r[0])
            try:
                kp = float(r[1])
            except Exception:
                continue
            if t and math.isfinite(kp):
                rows.append((t, kp))
        elif isinstance(r, dict):
            t = parse_time(r.get("time_tag") or r.get("time") or r.get("t"))
            try:
                kp = float(r.get("kp_index") or r.get("kp") or r.get("Kp"))
            except Exception:
                continue
            if t and math.isfinite(kp):
                rows.append((t, kp))
    rows.sort(key=lambda x: x[0])
    return rows


def kp_at(t: datetime, rows: list[tuple[datetime, float]]) -> float | None:
    if not rows:
        return None
    # NOAA K-index is 3-hourly. Use nearest within 120 minutes.
    best = None
    best_diff = float("inf")
    for rt, kp in rows:
        d = abs((rt - t).total_seconds())
        if d < best_diff:
            best_diff = d
            best = kp
    if best is None or best_diff > 2 * 3600:
        return None
    return float(best)


def phi(kp: float) -> tuple[float, float, float, float]:
    x = float(kp) - 3.0
    return (1.0, x, x * x, x * x * x)


def feature_delta(kp_forecast: float, kp_base: float) -> tuple[float, float, float, float]:
    pf = phi(kp_forecast)
    pb = phi(kp_base)
    return tuple(pf[i] - pb[i] for i in range(4))


def predict(coeff: list[float] | tuple[float, float, float, float], kp: float) -> float:
    p = phi(kp)
    return sum(float(coeff[i]) * p[i] for i in range(4))


def predict_delta(coeff: list[float] | tuple[float, float, float, float], kp_f: float, kp_b: float) -> float:
    return predict(coeff, kp_f) - predict(coeff, kp_b)


def zeros(n: int) -> list[float]:
    return [0.0] * n


def intzeros(n: int) -> list[int]:
    return [0] * n


def new_acc(n: int) -> dict[str, list[float] | list[int]]:
    return {
        "count": intzeros(n),
        "xx00": zeros(n), "xx01": zeros(n), "xx02": zeros(n), "xx03": zeros(n),
        "xx11": zeros(n), "xx12": zeros(n), "xx13": zeros(n),
        "xx22": zeros(n), "xx23": zeros(n), "xx33": zeros(n),
        "xy0": zeros(n), "xy1": zeros(n), "xy2": zeros(n), "xy3": zeros(n),
    }


def solve4(a: list[list[float]], b: list[float]) -> list[float]:
    n = 4
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-10:
            return [0.0, 0.0, 0.0, 0.0]
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f == 0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_from_acc(acc: dict, idx: int) -> list[float]:
    a = [
        [acc["xx00"][idx] + RIDGE, acc["xx01"][idx], acc["xx02"][idx], acc["xx03"][idx]],
        [acc["xx01"][idx], acc["xx11"][idx] + RIDGE, acc["xx12"][idx], acc["xx13"][idx]],
        [acc["xx02"][idx], acc["xx12"][idx], acc["xx22"][idx] + RIDGE, acc["xx23"][idx]],
        [acc["xx03"][idx], acc["xx13"][idx], acc["xx23"][idx], acc["xx33"][idx] + RIDGE],
    ]
    b = [acc["xy0"][idx], acc["xy1"][idx], acc["xy2"][idx], acc["xy3"][idx]]
    vals = solve4(a, b)
    return [max(-CORRECTION_CLIP_TECU, min(CORRECTION_CLIP_TECU, float(x))) for x in vals]


def old_grid_coeffs() -> dict:
    p = AI_ROOT / "kp_grid_coefficients.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def old_cell_coeff(old: dict, month: int, name: str, i: int, j: int) -> float:
    try:
        g = old.get("coefficients_grid") or old.get("grid_coefficients") or {}
        v = g.get(str(month), {}).get(name, [])[i][j]
        x = float(v)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


def region_defs():
    regions = []
    rid = 1
    for lat_min, lat_max, lat_name in LAT_BANDS:
        for lon_min, lon_max in LON_BANDS:
            regions.append({
                "id": f"R{rid:02d}",
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "name": f"{lat_name}_{lon_min:g}_{lon_max:g}",
            })
            rid += 1
    return regions


REGIONS = region_defs()


def region_id(lat: float, lon: float) -> str:
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    for r in REGIONS:
        lat_ok = (float(lat) >= r["lat_min"] and float(lat) < r["lat_max"]) or (r["lat_max"] == 90.0 and float(lat) <= 90.0)
        lon_ok = (lon >= r["lon_min"] and lon < r["lon_max"]) or (r["lon_max"] == 180.0 and lon <= 180.0)
        if lat_ok and lon_ok:
            return r["id"]
    return "R01"


def rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def mean(vals: list[float]) -> float | None:
    return None if not vals else sum(vals) / len(vals)


def to_grid(flat, n_lat, n_lon, digits=6):
    out = []
    k = 0
    for _ in range(n_lat):
        row = []
        for _ in range(n_lon):
            v = flat[k]
            if isinstance(v, int):
                row.append(v)
            else:
                row.append(round(float(v), digits))
            k += 1
        out.append(row)
    return out



def summarize_errors(raw: list[float], corr: list[float]) -> dict:
    base = summarize_errors_at_threshold(raw, corr, HIT_THRESHOLD_TECU)
    base["threshold_tecu"] = HIT_THRESHOLD_TECU
    return base


def summarize_errors_at_threshold(raw: list[float], corr: list[float], threshold: float) -> dict:
    return {
        "sample_count": len(raw),
        "raw_rmse": None if not raw else round(rmse(raw), 4),
        "corrected_rmse": None if not corr else round(rmse(corr), 4),
        "raw_bias": None if not raw else round(mean(raw), 4),
        "corrected_bias": None if not corr else round(mean(corr), 4),
        "raw_hit_rate": 0 if not raw else round(sum(abs(e) <= threshold for e in raw) / len(raw), 4),
        "corrected_hit_rate": 0 if not corr else round(sum(abs(e) <= threshold for e in corr) / len(corr), 4),
    }


def threshold_table(raw: list[float], corr: list[float]) -> dict:
    return {
        str(int(th)): summarize_errors_at_threshold(raw, corr, float(th))
        for th in (5, 10, 15, 20)
    }


KP_BINS = [
    ("0-2", 0.0, 2.0),
    ("2-3", 2.0, 3.0),
    ("3-4", 3.0, 4.0),
    ("4-5", 4.0, 5.0),
    ("5-6", 5.0, 6.0),
    ("6-7", 6.0, 7.0),
    ("7+", 7.0, 99.0),
]


def kp_bin_label(kp: float) -> str:
    x = float(kp)
    for label, lo, hi in KP_BINS:
        if label == "7+":
            if x >= lo:
                return label
        elif x >= lo and x < hi:
            return label
    return "unknown"



def main() -> int:
    AI_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(microsecond=0)

    metas_all = load_frame_meta()
    print(f"Loaded TEC frame metadata: {len(metas_all)}")
    if len(metas_all) < 2:
        raise RuntimeError("No enough TEC archive frames. Run Fetch NOAA TEC archive first.")

    train_cutoff = now - timedelta(days=TRAIN_DAYS)
    target_metas = [m for m in metas_all if m["time"] >= train_cutoff]

    kp_rows = parse_kp_json(http_json(K_INDEX_URL))
    print(f"Loaded Kp rows: {len(kp_rows)}")
    if not kp_rows:
        raise RuntimeError("No NOAA Kp rows loaded.")

    first = load_json_maybe_gz(metas_all[0]["path"])
    lat_arr, lon_arr, _ = flatten_grid(first)
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    n_cell = n_lat * n_lon
    if not n_cell:
        raise RuntimeError("TEC grid is empty")

    cell_region = []
    for lat in lat_arr:
        for lon in lon_arr:
            cell_region.append(region_id(lat, lon))

    # Accumulate normal equations for each grid cell and target month.
    acc_by_month = {m: new_acc(n_cell) for m in range(1, 13)}
    pair_records: list[dict] = []

    for meta_f in target_metas:
        meta_b = nearest_meta(metas_all, meta_f["time"] - timedelta(hours=PAIR_HOURS), PAIR_TOLERANCE_MIN)
        if not meta_b:
            continue
        kp_f = kp_at(meta_f["time"], kp_rows)
        kp_b = kp_at(meta_b["time"], kp_rows)
        if kp_f is None or kp_b is None:
            continue

        frame_f = load_json_maybe_gz(meta_f["path"])
        frame_b = load_json_maybe_gz(meta_b["path"])
        _, _, vals_f = flatten_grid(frame_f)
        _, _, vals_b = flatten_grid(frame_b)
        if len(vals_f) != n_cell or len(vals_b) != n_cell:
            continue

        month = meta_f["time"].month
        d = feature_delta(kp_f, kp_b)
        xx = (
            d[0]*d[0], d[0]*d[1], d[0]*d[2], d[0]*d[3],
            d[1]*d[1], d[1]*d[2], d[1]*d[3],
            d[2]*d[2], d[2]*d[3], d[3]*d[3],
        )
        acc = acc_by_month[month]
        pair_records.append({"month": month, "kp_f": kp_f, "kp_b": kp_b, "vals_f": vals_f, "vals_b": vals_b})

        for idx, (vf, vb) in enumerate(zip(vals_f, vals_b)):
            if not math.isfinite(vf) or not math.isfinite(vb):
                continue
            y = vf - vb
            acc["count"][idx] += 1
            acc["xx00"][idx] += xx[0]; acc["xx01"][idx] += xx[1]; acc["xx02"][idx] += xx[2]; acc["xx03"][idx] += xx[3]
            acc["xx11"][idx] += xx[4]; acc["xx12"][idx] += xx[5]; acc["xx13"][idx] += xx[6]
            acc["xx22"][idx] += xx[7]; acc["xx23"][idx] += xx[8]; acc["xx33"][idx] += xx[9]
            acc["xy0"][idx] += d[0] * y; acc["xy1"][idx] += d[1] * y; acc["xy2"][idx] += d[2] * y; acc["xy3"][idx] += d[3] * y

    print(f"Training pairs: {len(pair_records)}")
    if not pair_records:
        raise RuntimeError("No previous-day TEC/Kp training pairs found.")

    old_grid = old_grid_coeffs()
    grid_out = {}
    updated_cells_total = 0

    for month in range(1, 13):
        acc = acc_by_month[month]
        k0, k1, k2, k3 = zeros(n_cell), zeros(n_cell), zeros(n_cell), zeros(n_cell)
        updated = intzeros(n_cell)
        for idx in range(n_cell):
            i, j = divmod(idx, n_lon)
            old_vals = [old_cell_coeff(old_grid, month, name, i, j) for name in ("k0", "k1", "k2", "k3")]
            if acc["count"][idx] >= MIN_CELL_SAMPLES:
                fitted = fit_from_acc(acc, idx)
                vals = [(1 - BLEND_ALPHA) * old_vals[a] + BLEND_ALPHA * fitted[a] for a in range(4)]
                updated[idx] = 1
                updated_cells_total += 1
            else:
                vals = old_vals
            k0[idx], k1[idx], k2[idx], k3[idx] = vals

        grid_out[str(month)] = {
            "k0": to_grid(k0, n_lat, n_lon),
            "k1": to_grid(k1, n_lat, n_lon),
            "k2": to_grid(k2, n_lat, n_lon),
            "k3": to_grid(k3, n_lat, n_lon),
            "sample_count": to_grid(acc["count"], n_lat, n_lon, digits=0),
            "updated": to_grid(updated, n_lat, n_lon, digits=0),
        }

    # Region display coefficients = sample-count weighted mean of grid-cell coefficients.
    coeffs_region = {r["id"]: {} for r in REGIONS}
    for month in range(1, 13):
        sums = {rid: [0.0, 0.0, 0.0, 0.0, 0] for rid in coeffs_region}
        mg = grid_out[str(month)]
        for idx, rid in enumerate(cell_region):
            i, j = divmod(idx, n_lon)
            cnt = int(mg["sample_count"][i][j] or 0)
            if cnt <= 0:
                continue
            sums[rid][0] += mg["k0"][i][j] * cnt
            sums[rid][1] += mg["k1"][i][j] * cnt
            sums[rid][2] += mg["k2"][i][j] * cnt
            sums[rid][3] += mg["k3"][i][j] * cnt
            sums[rid][4] += cnt
        for rid, s in sums.items():
            w = s[4]
            coeffs_region[rid][str(month)] = {
                "k0": round(s[0] / w, 6) if w else 0.0,
                "k1": round(s[1] / w, 6) if w else 0.0,
                "k2": round(s[2] / w, 6) if w else 0.0,
                "k3": round(s[3] / w, 6) if w else 0.0,
                "sample_count": int(w),
                "updated": w > 0,
            }

    # Performance:
    # raw forecast = previous-day observed TEC.
    # corrected forecast = previous-day TEC - F(KpB) + F(KpF).
    perf_raw = defaultdict(list)
    perf_corr = defaultdict(list)
    perf_raw_by_kp = defaultdict(list)
    perf_corr_by_kp = defaultdict(list)

    for rec in pair_records:
        month = rec["month"]
        kp_f = rec["kp_f"]
        kp_b = rec["kp_b"]
        mg = grid_out[str(month)]
        for idx, (vf, vb) in enumerate(zip(rec["vals_f"], rec["vals_b"])):
            if not math.isfinite(vf) or not math.isfinite(vb):
                continue
            i, j = divmod(idx, n_lon)
            cf = [mg[name][i][j] for name in ("k0", "k1", "k2", "k3")]
            raw_forecast = vb
            corrected_forecast = vb - predict(cf, kp_b) + predict(cf, kp_f)
            raw_err = vf - raw_forecast
            corr_err = vf - corrected_forecast
            key = (cell_region[idx], month)
            perf_raw[key].append(raw_err)
            perf_corr[key].append(corr_err)

            kp_label = kp_bin_label(kp_f)
            perf_raw_by_kp[kp_label].append(raw_err)
            perf_corr_by_kp[kp_label].append(corr_err)

    metrics_region = {r["id"]: {} for r in REGIONS}
    all_raw, all_corr = [], []
    for rid in metrics_region:
        for month in range(1, 13):
            raw = perf_raw.get((rid, month), [])
            corr = perf_corr.get((rid, month), [])
            metrics_region[rid][str(month)] = summarize_errors(raw, corr)
            all_raw.extend(raw)
            all_corr.extend(corr)

    coeff_doc = {
        "version": "swifttec-kp-model-rule-ai-region-v1",
        "updated_utc": iso(now),
        "model_rule": "BaseTEC = PrevObservedTEC - F(KpB); ForecastTEC = BaseTEC + F(KpF)",
        "formula": "F(Kp) = k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3",
        "train_days": TRAIN_DAYS,
        "pair_hours": PAIR_HOURS,
        "regions": REGIONS,
        "coefficients": coeffs_region,
    }

    grid_doc = {
        "version": "swifttec-kp-model-rule-ai-grid-v1",
        "updated_utc": iso(now),
        "model_rule": "BaseTEC = PrevObservedTEC - F(KpB); ForecastTEC = BaseTEC + F(KpF)",
        "formula": "F(Kp) = k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3",
        "lat_arr": lat_arr,
        "lon_arr": lon_arr,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "train_days": TRAIN_DAYS,
        "pair_hours": PAIR_HOURS,
        "pair_tolerance_min": PAIR_TOLERANCE_MIN,
        "min_cell_samples": MIN_CELL_SAMPLES,
        "blend_alpha": BLEND_ALPHA,
        "ridge": RIDGE,
        "correction_clip_tecu": CORRECTION_CLIP_TECU,
        "coefficients_grid": grid_out,
    }

    thresholds_summary = threshold_table(all_raw, all_corr)

    kp_bins_summary = {}
    for label, _, _ in KP_BINS:
        raw = perf_raw_by_kp.get(label, [])
        corr = perf_corr_by_kp.get(label, [])
        kp_bins_summary[label] = {
            "kp_bin": label,
            "thresholds": threshold_table(raw, corr),
            **summarize_errors(raw, corr),
        }

    perf_doc = {
        "version": "swifttec-kp-model-rule-ai-performance-v1",
        "updated_utc": iso(now),
        "model_rule": "raw=PrevObservedTEC, corrected=PrevObservedTEC - F(KpB) + F(KpF)",
        "hit_threshold_tecu": HIT_THRESHOLD_TECU,
        "summary": summarize_errors(all_raw, all_corr) | {
            "updated_cells": updated_cells_total,
            "training_pairs": len(pair_records),
            "thresholds": thresholds_summary,
        },
        "thresholds": thresholds_summary,
        "kp_bins": kp_bins_summary,
        "metrics": metrics_region,
    }

    hist_path = AI_ROOT / "kp_learning_history.json"
    try:
        hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {"version": "swifttec-kp-ai-history-v1", "runs": []}
    except Exception:
        hist = {"version": "swifttec-kp-ai-history-v1", "runs": []}
    hist["version"] = "swifttec-kp-model-rule-ai-history-v1"
    hist["model_rule"] = "BaseTEC = PrevObservedTEC - F(KpB); ForecastTEC = BaseTEC + F(KpF)"
    hist.setdefault("runs", []).append({"time_utc": iso(now), **perf_doc["summary"]})
    hist["runs"] = hist["runs"][-HISTORY_MAX_RUNS:]

    (AI_ROOT / "kp_grid_coefficients.json").write_text(json.dumps(grid_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (AI_ROOT / "kp_coefficients.json").write_text(json.dumps(coeff_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (AI_ROOT / "kp_performance.json").write_text(json.dumps(perf_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Kp model-rule AI training complete: pairs={len(pair_records)}, "
        f"updated_cells={updated_cells_total}, samples={len(all_raw)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
