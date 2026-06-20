#!/usr/bin/env python3
"""Train SWIFT-TEC Kp cubic AI corrector at grid-cell level.

Learning level:
  - AI learns one cubic Kp residual model for each grid cell and month.

UI/display level:
  - The browser still displays 18 regional summaries for clarity.

Formula:
  residual_tecu ~= k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3

Outputs:
  docs/data/ai/kp_grid_coefficients.json   # grid-cell coefficients used by forecast correction
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
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-kp-grid-ai-corrector/1.0"})
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


def load_frame_meta() -> list[dict]:
    idx_path = TEC_ROOT / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} not found")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=TRAIN_DAYS)
    out = []
    for row in idx.get("frames") or []:
        t = parse_time(row.get("time_utc"))
        rel = row.get("file")
        if not t or t < cutoff or not rel:
            continue
        path = TEC_ROOT / rel
        if path.exists():
            out.append({"time": t, "path": path, "file": rel})
    out.sort(key=lambda x: x["time"])
    return out


def parse_kp_json(obj) -> list[tuple[datetime, float]]:
    rows = []
    if not isinstance(obj, list):
        return rows
    header = None
    data = obj
    if obj and isinstance(obj[0], list) and any(str(x).lower() in ("time_tag", "kp") for x in obj[0]):
        header = [str(x).lower() for x in obj[0]]
        data = obj[1:]
    for row in data:
        t = None
        kp = None
        if isinstance(row, dict):
            t = parse_time(row.get("time_tag") or row.get("time") or row.get("t"))
            kp = row.get("Kp") or row.get("kp") or row.get("value")
        elif isinstance(row, list):
            if header:
                ti = header.index("time_tag") if "time_tag" in header else 0
                ki = header.index("kp") if "kp" in header else 1
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


def flatten_grid(frame: dict) -> tuple[list[float], list[float], list[float]]:
    lat_arr = [float(x) for x in frame.get("lat_arr") or []]
    lon_arr = [float(x) for x in frame.get("lon_arr") or []]
    grid = frame.get("grid") or []
    vals = []
    for i in range(len(lat_arr)):
        row = grid[i] if i < len(grid) else []
        for j in range(len(lon_arr)):
            try:
                v = float(row[j])
            except Exception:
                v = float("nan")
            vals.append(v if math.isfinite(v) and v >= 0 else float("nan"))
    return lat_arr, lon_arr, vals


def solve4(a, b):
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


def fit_from_acc(acc, idx: int) -> list[float]:
    # symmetric terms: 00,01,02,03,11,12,13,22,23,33
    a = [
        [acc["xx00"][idx] + RIDGE, acc["xx01"][idx], acc["xx02"][idx], acc["xx03"][idx]],
        [acc["xx01"][idx], acc["xx11"][idx] + RIDGE, acc["xx12"][idx], acc["xx13"][idx]],
        [acc["xx02"][idx], acc["xx12"][idx], acc["xx22"][idx] + RIDGE, acc["xx23"][idx]],
        [acc["xx03"][idx], acc["xx13"][idx], acc["xx23"][idx], acc["xx33"][idx] + RIDGE],
    ]
    b = [acc["xy0"][idx], acc["xy1"][idx], acc["xy2"][idx], acc["xy3"][idx]]
    return solve4(a, b)


def predict(coeffs, kp: float) -> float:
    x = kp - 3.0
    y = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x
    return max(-CORRECTION_CLIP_TECU, min(CORRECTION_CLIP_TECU, y))


def rmse(vals):
    return None if not vals else math.sqrt(sum(v * v for v in vals) / len(vals))


def mean(vals):
    return None if not vals else sum(vals) / len(vals)


def zeros(n: int) -> list[float]:
    return [0.0] * n


def intzeros(n: int) -> list[int]:
    return [0] * n


def new_acc(n: int) -> dict:
    keys = ["xx00", "xx01", "xx02", "xx03", "xx11", "xx12", "xx13", "xx22", "xx23", "xx33", "xy0", "xy1", "xy2", "xy3"]
    d = {k: zeros(n) for k in keys}
    d["count"] = intzeros(n)
    return d


def old_grid_coeffs() -> dict:
    p = AI_ROOT / "kp_grid_coefficients.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("coefficients_grid") or {}
    except Exception:
        return {}


def old_cell_coeff(old, month: int, name: str, i: int, j: int):
    try:
        return float(old[str(month)][name][i][j])
    except Exception:
        return 0.0


def to_grid(flat, n_lat, n_lon, digits=6):
    out = []
    k = 0
    for _ in range(n_lat):
        row = []
        for _ in range(n_lon):
            v = flat[k]
            row.append(round(float(v), digits) if isinstance(v, float) else v)
            k += 1
        out.append(row)
    return out


def main() -> int:
    AI_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(microsecond=0)
    metas = load_frame_meta()
    print(f"Loaded TEC frame metadata: {len(metas)}")
    if not metas:
        raise RuntimeError("No TEC archive frames. Run Fetch NOAA TEC archive first.")

    kp_rows = parse_kp_json(http_json(K_INDEX_URL))
    print(f"Loaded Kp rows: {len(kp_rows)}")

    first = load_json_maybe_gz(metas[0]["path"])
    lat_arr, lon_arr, vals0 = flatten_grid(first)
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    n_cell = n_lat * n_lon
    if not n_cell:
        raise RuntimeError("TEC grid is empty")

    cell_region = []
    for lat in lat_arr:
        for lon in lon_arr:
            cell_region.append(region_id(lat, lon))

    # Pass 1: quiet-time baseline proxy = mean TEC for each month/hour/grid cell.
    base_sum: dict[tuple[int, int], list[float]] = {}
    base_cnt: dict[tuple[int, int], list[int]] = {}
    for meta in metas:
        frame = load_json_maybe_gz(meta["path"])
        _, _, vals = flatten_grid(frame)
        key = (meta["time"].month, meta["time"].hour)
        if key not in base_sum:
            base_sum[key] = zeros(n_cell)
            base_cnt[key] = intzeros(n_cell)
        s, c = base_sum[key], base_cnt[key]
        for idx, v in enumerate(vals):
            if math.isfinite(v):
                s[idx] += v
                c[idx] += 1

    # Pass 2: accumulate normal equations per grid cell and month.
    acc_by_month = {m: new_acc(n_cell) for m in range(1, 13)}
    used_frames = 0
    for meta in metas:
        kp = kp_at(meta["time"], kp_rows)
        if kp is None:
            continue
        frame = load_json_maybe_gz(meta["path"])
        _, _, vals = flatten_grid(frame)
        month = meta["time"].month
        hour = meta["time"].hour
        bs, bc = base_sum.get((month, hour)), base_cnt.get((month, hour))
        if not bs or not bc:
            continue
        x = kp - 3.0
        p0, p1, p2, p3 = 1.0, x, x * x, x * x * x
        xx = (p0*p0, p0*p1, p0*p2, p0*p3, p1*p1, p1*p2, p1*p3, p2*p2, p2*p3, p3*p3)
        acc = acc_by_month[month]
        used_frames += 1
        for idx, v in enumerate(vals):
            if not math.isfinite(v) or bc[idx] <= 0:
                continue
            residual = v - (bs[idx] / bc[idx])
            acc["count"][idx] += 1
            acc["xx00"][idx] += xx[0]; acc["xx01"][idx] += xx[1]; acc["xx02"][idx] += xx[2]; acc["xx03"][idx] += xx[3]
            acc["xx11"][idx] += xx[4]; acc["xx12"][idx] += xx[5]; acc["xx13"][idx] += xx[6]
            acc["xx22"][idx] += xx[7]; acc["xx23"][idx] += xx[8]; acc["xx33"][idx] += xx[9]
            acc["xy0"][idx] += p0 * residual; acc["xy1"][idx] += p1 * residual; acc["xy2"][idx] += p2 * residual; acc["xy3"][idx] += p3 * residual

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
            w = max(1, cnt) if cnt else 0
            if not w:
                continue
            sums[rid][0] += mg["k0"][i][j] * w
            sums[rid][1] += mg["k1"][i][j] * w
            sums[rid][2] += mg["k2"][i][j] * w
            sums[rid][3] += mg["k3"][i][j] * w
            sums[rid][4] += w
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

    # Pass 3: performance by region/month using grid-cell correction.
    perf_raw = defaultdict(list)
    perf_corr = defaultdict(list)
    for meta in metas:
        kp = kp_at(meta["time"], kp_rows)
        if kp is None:
            continue
        frame = load_json_maybe_gz(meta["path"])
        _, _, vals = flatten_grid(frame)
        month = meta["time"].month
        hour = meta["time"].hour
        bs, bc = base_sum.get((month, hour)), base_cnt.get((month, hour))
        if not bs or not bc:
            continue
        mg = grid_out[str(month)]
        for idx, v in enumerate(vals):
            if not math.isfinite(v) or bc[idx] <= 0:
                continue
            residual = v - (bs[idx] / bc[idx])
            i, j = divmod(idx, n_lon)
            cf = [mg[name][i][j] for name in ("k0", "k1", "k2", "k3")]
            corr = residual - predict(cf, kp)
            key = (cell_region[idx], month)
            perf_raw[key].append(residual)
            perf_corr[key].append(corr)

    metrics_out = {r["id"]: {} for r in REGIONS}
    all_raw, all_corr = [], []
    for rid in metrics_out:
        for month in range(1, 13):
            raw = perf_raw.get((rid, month), [])
            corr = perf_corr.get((rid, month), [])
            all_raw.extend(raw)
            all_corr.extend(corr)
            metrics_out[rid][str(month)] = {
                "sample_count": len(raw),
                "raw_rmse": None if not raw else round(rmse(raw), 4),
                "corrected_rmse": None if not corr else round(rmse(corr), 4),
                "raw_bias": None if not raw else round(mean(raw), 4),
                "corrected_bias": None if not corr else round(mean(corr), 4),
                "raw_hit_rate": 0 if not raw else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in raw) / len(raw), 4),
                "corrected_hit_rate": 0 if not corr else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in corr) / len(corr), 4),
            }

    model = {
        "formula": "delta_tecu = k0 + k1*(Kp-3) + k2*(Kp-3)^2 + k3*(Kp-3)^3",
        "x": "Kp - 3",
        "learning_level": "grid_cell",
        "display_level": "18_regions",
        "blend_alpha": BLEND_ALPHA,
        "ridge": RIDGE,
        "min_cell_samples": MIN_CELL_SAMPLES,
        "correction_clip_tecu": CORRECTION_CLIP_TECU,
    }
    grid_doc = {
        "version": "swifttec-kp-cubic-ai-grid-v1",
        "updated_utc": iso(now),
        "train_days": TRAIN_DAYS,
        "used_frames": used_frames,
        "lat_arr": lat_arr,
        "lon_arr": lon_arr,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "model": model,
        "coefficients_grid": grid_out,
    }
    coeff_doc = {
        "version": "swifttec-kp-cubic-ai-region-display-v2",
        "updated_utc": iso(now),
        "train_days": TRAIN_DAYS,
        "regions": REGIONS,
        "model": model,
        "note": "Display coefficients are aggregated from grid-cell AI coefficients. Forecast correction uses kp_grid_coefficients.json when available.",
        "coefficients": coeffs_region,
    }
    perf_doc = {
        "version": "swifttec-kp-cubic-ai-performance-v2",
        "updated_utc": iso(now),
        "hit_threshold_tecu": HIT_THRESHOLD_TECU,
        "summary": {
            "sample_count": len(all_raw),
            "updated_groups": updated_cells_total,
            "updated_cells": updated_cells_total,
            "raw_rmse": None if not all_raw else round(rmse(all_raw), 4),
            "corrected_rmse": None if not all_corr else round(rmse(all_corr), 4),
            "raw_hit_rate": 0 if not all_raw else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in all_raw) / len(all_raw), 4),
            "corrected_hit_rate": 0 if not all_corr else round(sum(abs(e) <= HIT_THRESHOLD_TECU for e in all_corr) / len(all_corr), 4),
        },
        "metrics": metrics_out,
    }

    hist_path = AI_ROOT / "kp_learning_history.json"
    try:
        hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {"version": "swifttec-kp-cubic-ai-history-v2", "runs": []}
    except Exception:
        hist = {"version": "swifttec-kp-cubic-ai-history-v2", "runs": []}
    hist["version"] = "swifttec-kp-cubic-ai-history-v2"
    hist["retention_days"] = 730
    hist["learning_level"] = "grid_cell"
    hist.setdefault("runs", []).append({"time_utc": iso(now), **perf_doc["summary"]})
    hist["runs"] = hist["runs"][-HISTORY_MAX_RUNS:]

    (AI_ROOT / "kp_grid_coefficients.json").write_text(json.dumps(grid_doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (AI_ROOT / "kp_coefficients.json").write_text(json.dumps(coeff_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (AI_ROOT / "kp_performance.json").write_text(json.dumps(perf_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kp grid AI training complete: cells={n_cell}, used_frames={used_frames}, updated_cells={updated_cells_total}, samples={len(all_raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
