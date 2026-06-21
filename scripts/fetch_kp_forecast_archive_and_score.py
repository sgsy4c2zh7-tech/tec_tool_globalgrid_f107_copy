#!/usr/bin/env python3
"""Archive NOAA forecast Kp and score operational TEC hit rate by lead day.

Purpose:
- Keep model-hit-rate (actual Kp) separate from operational-hit-rate (forecast Kp).
- Archive forecast Kp daily.
- Score TEC forecast errors by lead day 1,2,3,4 when historical forecast and actual TEC are available.

Outputs:
- docs/data/ai/kp_forecast_archive.json
- docs/data/ai/operational_hit_rate.json
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

UTC = timezone.utc
TEC_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_AI_ROOT", "docs/data/ai"))
AI_ROOT.mkdir(parents=True, exist_ok=True)

KP_ACTUAL_URL = os.environ.get("SWIFTTEC_KP_ACTUAL_URL", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
KP_FORECAST_URL = os.environ.get("SWIFTTEC_KP_FORECAST_URL", "https://services.swpc.noaa.gov/text/3-day-forecast.txt")

ARCHIVE_PATH = AI_ROOT / "kp_forecast_archive.json"
OP_PATH = AI_ROOT / "operational_hit_rate.json"
MAX_FORECAST_ISSUES = int(os.environ.get("SWIFTTEC_KP_FORECAST_ARCHIVE_MAX", "240"))
SCORE_DAYS_BACK = int(os.environ.get("SWIFTTEC_OPERATIONAL_SCORE_DAYS", "90"))
PAIR_TOLERANCE_MIN = int(os.environ.get("SWIFTTEC_OPERATIONAL_TEC_TOLERANCE_MIN", "45"))
MAX_CELL_SAMPLES_PER_FRAME = int(os.environ.get("SWIFTTEC_OPERATIONAL_MAX_CELL_SAMPLES_PER_FRAME", "10000"))


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


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


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-kp-forecast-archive/1.0"})
    with urlopen(req, timeout=60) as res:
        return res.read().decode("utf-8", errors="replace")


def fetch_json(url: str):
    return json.loads(fetch_text(url))


def parse_actual_kp(obj) -> list[tuple[datetime, float]]:
    rows = []
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


def kp_at(t: datetime, rows: list[tuple[datetime, float]], max_diff_hours: float = 2.0) -> float | None:
    best = None
    best_diff = float("inf")
    for rt, kp in rows:
        d = abs((rt - t).total_seconds())
        if d < best_diff:
            best_diff = d
            best = kp
    if best is None or best_diff > max_diff_hours * 3600:
        return None
    return float(best)


def parse_3day_kp_forecast(text: str, issue: datetime) -> list[dict]:
    """Parse SWPC 3-day forecast text.

    Expected table resembles:
      NOAA Kp index forecast  Jun 21  Jun 22  Jun 23
      00-03UT        2.67      2.67      2.33
    The parser is deliberately tolerant.
    """
    lines = text.splitlines()
    header_idx = None
    header = ""
    for i, line in enumerate(lines):
        if "NOAA Kp index forecast" in line:
            header_idx = i
            header = line
            break
    if header_idx is None:
        return []

    # Dates may appear as MM-DD, Mon DD, or just columns. Try to extract month/day.
    date_tokens: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d{1,2})[-/](\d{1,2})", header):
        date_tokens.append((int(m.group(1)), int(m.group(2))))

    if not date_tokens:
        # Fallback: issue day, issue+1, issue+2
        date_tokens = []
        for k in range(3):
            d = (issue + timedelta(days=k)).date()
            date_tokens.append((d.month, d.day))

    slots: list[dict] = []
    base_year = issue.year
    prev_month = issue.month

    for line in lines[header_idx + 1: header_idx + 12]:
        m = re.search(r"(\d{2})\s*[-–]\s*(\d{2})\s*UT", line, flags=re.I)
        if not m:
            if slots and not line.strip():
                break
            continue
        start_h = int(m.group(1))
        vals = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line[m.end():])]
        for col, kp in enumerate(vals[:len(date_tokens)]):
            month, day = date_tokens[col]
            year = base_year
            # Basic year rollover handling.
            if month < prev_month - 6:
                year += 1
            try:
                t = datetime(year, month, day, start_h, tzinfo=UTC)
            except Exception:
                continue
            lead_hours = (t - issue).total_seconds() / 3600.0
            if lead_hours < -6:
                continue
            lead_day = max(0, int(math.ceil(max(0.0, lead_hours) / 24.0)))
            slots.append({
                "time_utc": iso(t),
                "kp_forecast": kp,
                "lead_hours": round(lead_hours, 2),
                "lead_day": lead_day,
            })
    slots.sort(key=lambda x: x["time_utc"])
    return slots


def load_archive() -> dict:
    if not ARCHIVE_PATH.exists():
        return {"version": "swifttec-kp-forecast-archive-v1", "forecasts": []}
    try:
        return json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": "swifttec-kp-forecast-archive-v1", "forecasts": []}


def save_archive(doc: dict) -> None:
    doc["version"] = "swifttec-kp-forecast-archive-v1"
    # De-duplicate by issue hour.
    seen = set()
    forecasts = []
    for f in sorted(doc.get("forecasts", []), key=lambda x: x.get("issue_utc", "")):
        key = f.get("issue_utc", "")[:13]
        if key in seen:
            continue
        seen.add(key)
        forecasts.append(f)
    doc["forecasts"] = forecasts[-MAX_FORECAST_ISSUES:]
    doc["updated_utc"] = iso(now_utc())
    ARCHIVE_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


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
    cutoff = now_utc() - timedelta(days=SCORE_DAYS_BACK + 6)
    out = []
    for f in frames:
        t = parse_time(f.get("time_utc") or f.get("time"))
        rel = f.get("file") or f.get("path")
        if not t or not rel or t < cutoff:
            continue
        path = TEC_ROOT / rel
        if path.exists():
            out.append({"time": t, "path": path, "file": rel})
    out.sort(key=lambda x: x["time"])
    return out


def nearest_meta(metas: list[dict], target: datetime, tolerance_min: int = PAIR_TOLERANCE_MIN) -> dict | None:
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
    vals = []
    for row in grid:
        for v in row:
            try:
                x = float(v)
                vals.append(x if math.isfinite(x) else float("nan"))
            except Exception:
                vals.append(float("nan"))
    return lat_arr, lon_arr, vals


def phi(kp: float) -> tuple[float, float, float, float]:
    x = float(kp) - 3.0
    return (1.0, x, x * x, x * x * x)


def predict(coeff: list[float], kp: float) -> float:
    p = phi(kp)
    return sum(float(coeff[i]) * p[i] for i in range(4))


def load_coeff_doc() -> dict | None:
    p = AI_ROOT / "kp_grid_coefficients.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def coeff_for_cell(doc: dict, month: int, i: int, j: int) -> list[float]:
    root = doc.get("coefficients_grid") or doc.get("grid_coefficients") or {}
    m = root.get(str(month)) or root.get(month) or {}
    out = []
    for k in ("k0", "k1", "k2", "k3"):
        try:
            v = float(m[k][i][j])
            out.append(v if math.isfinite(v) else 0.0)
        except Exception:
            out.append(0.0)
    return out


def threshold_summary(errors: list[float]) -> dict:
    out = {}
    for th in (5, 10, 15, 20):
        if not errors:
            out[str(th)] = {"sample_count": 0, "corrected_hit_rate": 0, "corrected_rmse": None, "corrected_bias": None}
        else:
            out[str(th)] = {
                "sample_count": len(errors),
                "corrected_hit_rate": round(sum(abs(e) <= th for e in errors) / len(errors), 4),
                "corrected_rmse": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 4),
                "corrected_bias": round(sum(errors) / len(errors), 4),
            }
    return out


def score_operational(doc: dict, actual_kp: list[tuple[datetime, float]]) -> dict:
    metas = load_frame_meta()
    coeff_doc = load_coeff_doc()
    updated = iso(now_utc())
    if not metas:
        return {"version": "swifttec-operational-hit-rate-v1", "updated_utc": updated, "reason": "no TEC archive", "by_lead_day": {}}
    if not coeff_doc:
        return {"version": "swifttec-operational-hit-rate-v1", "updated_utc": updated, "reason": "no kp_grid_coefficients.json", "by_lead_day": {}}
    if not actual_kp:
        return {"version": "swifttec-operational-hit-rate-v1", "updated_utc": updated, "reason": "no actual Kp", "by_lead_day": {}}

    first = load_json_maybe_gz(metas[0]["path"])
    lat_arr, lon_arr, _ = flatten_grid(first)
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    if not n_lat or not n_lon:
        return {"version": "swifttec-operational-hit-rate-v1", "updated_utc": updated, "reason": "empty TEC grid", "by_lead_day": {}}

    step = max(1, int(math.ceil((n_lat * n_lon) / MAX_CELL_SAMPLES_PER_FRAME)))
    errors_by_lead: dict[int, list[float]] = defaultdict(list)
    slot_count_by_lead: dict[int, int] = defaultdict(int)

    cutoff = now_utc() - timedelta(days=SCORE_DAYS_BACK)

    for forecast in doc.get("forecasts", []):
        issue = parse_time(forecast.get("issue_utc"))
        if not issue or issue < cutoff - timedelta(days=5):
            continue
        for slot in forecast.get("slots", []):
            t = parse_time(slot.get("time_utc"))
            if not t or t < cutoff:
                continue
            lead = int(slot.get("lead_day") or math.ceil(max(0, (t - issue).total_seconds()) / 86400.0))
            if lead not in (1, 2, 3, 4):
                continue

            try:
                kp_f = float(slot.get("kp_forecast"))
            except Exception:
                continue

            base_time = t - timedelta(days=lead)
            meta_f = nearest_meta(metas, t)
            meta_b = nearest_meta(metas, base_time)
            if not meta_f or not meta_b:
                continue
            kp_b = kp_at(base_time, actual_kp)
            if kp_b is None:
                continue

            try:
                frame_f = load_json_maybe_gz(meta_f["path"])
                frame_b = load_json_maybe_gz(meta_b["path"])
                _, _, vals_f = flatten_grid(frame_f)
                _, _, vals_b = flatten_grid(frame_b)
            except Exception:
                continue
            if len(vals_f) != n_lat * n_lon or len(vals_b) != n_lat * n_lon:
                continue

            month = t.month
            slot_count_by_lead[lead] += 1
            for idx in range(0, n_lat * n_lon, step):
                vf = vals_f[idx]
                vb = vals_b[idx]
                if not math.isfinite(vf) or not math.isfinite(vb):
                    continue
                i, j = divmod(idx, n_lon)
                cf = coeff_for_cell(coeff_doc, month, i, j)
                forecast_tec = vb - predict(cf, kp_b) + predict(cf, kp_f)
                errors_by_lead[lead].append(vf - forecast_tec)

    by_lead = {}
    for lead in (1, 2, 3, 4):
        errs = errors_by_lead.get(lead, [])
        by_lead[str(lead)] = {
            "lead_day": lead,
            "slot_count": int(slot_count_by_lead.get(lead, 0)),
            "thresholds": threshold_summary(errs),
            **threshold_summary(errs).get("5", {}),
        }

    return {
        "version": "swifttec-operational-hit-rate-v1",
        "updated_utc": updated,
        "definition": "Operational hit rate uses archived forecast Kp, not actual Kp, in ForecastTEC = BaseTEC + F(KpForecast).",
        "forecast_issue_count": len(doc.get("forecasts", [])),
        "score_days_back": SCORE_DAYS_BACK,
        "max_cell_samples_per_frame": MAX_CELL_SAMPLES_PER_FRAME,
        "by_lead_day": by_lead,
    }


def main() -> int:
    issue = now_utc()
    text = fetch_text(KP_FORECAST_URL)
    slots = parse_3day_kp_forecast(text, issue)

    archive = load_archive()
    archive.setdefault("forecasts", []).append({
        "issue_utc": iso(issue),
        "source_url": KP_FORECAST_URL,
        "horizon_note": "SWPC 3-day forecast is archived when available. Lead 4 requires a forecast source with >=4-day horizon; otherwise N remains 0.",
        "slots": slots,
    })
    save_archive(archive)

    try:
        actual_kp = parse_actual_kp(fetch_json(KP_ACTUAL_URL))
    except Exception:
        actual_kp = []

    scored = score_operational(archive, actual_kp)
    OP_PATH.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Archived forecast slots: {len(slots)}")
    print(f"Forecast issues: {len(archive.get('forecasts', []))}")
    print(f"Operational score written: {OP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
