#!/usr/bin/env python3
"""Archive NOAA forecast Kp and score operational TEC hit rate by lead day.

v8.0 changes:
- Pulls a longer Kp actual history first from SWPC json/planetary_k_index_1m.json.
- Uses SWPC noaa-planetary-k-index-forecast.json when available, with text forecast fallback.
- Cold-start Kp-history backfill is rebuilt every run so stale cold-start entries cannot leave the score at N=0.
- Scores by TEC frame time first, so 30-minute TEC frames and 3-hour Kp slots no longer leave N=0.

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
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

UTC = timezone.utc
TEC_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_AI_ROOT", "docs/data/ai"))
AI_ROOT.mkdir(parents=True, exist_ok=True)

KP_ACTUAL_URLS = [
    os.environ.get("SWIFTTEC_KP_ACTUAL_1M_URL", "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"),
    os.environ.get("SWIFTTEC_KP_ACTUAL_URL", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"),
]
KP_FORECAST_JSON_URL = os.environ.get("SWIFTTEC_KP_FORECAST_JSON_URL", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json")
KP_FORECAST_TEXT_URL = os.environ.get("SWIFTTEC_KP_FORECAST_URL", "https://services.swpc.noaa.gov/text/3-day-forecast.txt")

ARCHIVE_PATH = AI_ROOT / "kp_forecast_archive.json"
OP_PATH = AI_ROOT / "operational_hit_rate.json"
MAX_FORECAST_ISSUES = int(os.environ.get("SWIFTTEC_KP_FORECAST_ARCHIVE_MAX", "360"))
SCORE_DAYS_BACK = int(os.environ.get("SWIFTTEC_OPERATIONAL_SCORE_DAYS", "90"))
PAIR_TOLERANCE_MIN = int(os.environ.get("SWIFTTEC_OPERATIONAL_TEC_TOLERANCE_MIN", "45"))
MAX_CELL_SAMPLES_PER_FRAME = int(os.environ.get("SWIFTTEC_OPERATIONAL_MAX_CELL_SAMPLES_PER_FRAME", "10000"))
COLDSTART_DAYS = int(os.environ.get("SWIFTTEC_KP_COLDSTART_DAYS", "30"))
ENABLE_COLDSTART = os.environ.get("SWIFTTEC_KP_COLDSTART_BACKFILL", "1").lower() not in ("0", "false", "no")


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


def fetch_text(url: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-kp-forecast-archive/1.1"})
    with urlopen(req, timeout=timeout) as res:
        return res.read().decode("utf-8", errors="replace")


def fetch_json(url: str):
    return json.loads(fetch_text(url, timeout=60))


def parse_actual_kp(obj) -> list[tuple[datetime, float]]:
    rows = []
    if not isinstance(obj, list):
        return rows
    for r in obj:
        if isinstance(r, list) and len(r) >= 2:
            t = parse_time(r[0])
            kp_raw = r[1]
        elif isinstance(r, dict):
            t = parse_time(r.get("time_tag") or r.get("time") or r.get("t") or r.get("time_utc"))
            kp_raw = r.get("kp_index")
            if kp_raw is None:
                kp_raw = r.get("Kp")
            if kp_raw is None:
                kp_raw = r.get("kp")
            if kp_raw is None:
                kp_raw = r.get("k_index")
        else:
            continue
        try:
            kp = float(kp_raw)
        except Exception:
            continue
        if t and math.isfinite(kp):
            rows.append((t, kp))
    rows.sort(key=lambda x: x[0])
    # De-duplicate by exact timestamp, prefer later endpoint rows.
    dedup: dict[str, tuple[datetime, float]] = {}
    for t, kp in rows:
        dedup[iso(t)] = (t, kp)
    return sorted(dedup.values(), key=lambda x: x[0])


def fetch_actual_kp_history() -> tuple[list[tuple[datetime, float]], str, list[str]]:
    errors = []
    best_rows: list[tuple[datetime, float]] = []
    best_url = ""
    for url in KP_ACTUAL_URLS:
        try:
            rows = parse_actual_kp(fetch_json(url))
            if len(rows) > len(best_rows):
                best_rows = rows
                best_url = url
        except Exception as e:
            errors.append(f"{url}: {e}")
    return best_rows, best_url, errors


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


def parse_forecast_json_slots(obj, issue: datetime) -> list[dict]:
    slots = []
    if not isinstance(obj, list):
        return slots
    for r in obj:
        if not isinstance(r, dict):
            continue
        t = parse_time(r.get("time_tag") or r.get("time") or r.get("time_utc"))
        if not t:
            continue
        try:
            kp = float(r.get("kp") if r.get("kp") is not None else r.get("Kp"))
        except Exception:
            continue
        status = str(r.get("observed") or r.get("status") or "").lower()
        # Archive forecast-side values only. Observed history belongs in actual Kp, not forecast archive.
        if status == "observed":
            continue
        lead_hours = (t - issue).total_seconds() / 3600.0
        if lead_hours < -9:
            continue
        lead_day = max(0, int(math.ceil(max(0.0, lead_hours) / 24.0)))
        slots.append({
            "time_utc": iso(t),
            "kp_forecast": kp,
            "lead_hours": round(lead_hours, 2),
            "lead_day": lead_day,
            "forecast_source": f"swpc_json_{status or 'forecast'}",
        })
    slots.sort(key=lambda x: x["time_utc"])
    return slots


def parse_3day_kp_forecast(text: str, issue: datetime) -> list[dict]:
    """Parse SWPC 3-day forecast text."""
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

    date_tokens: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d{1,2})[-/](\d{1,2})", header):
        date_tokens.append((int(m.group(1)), int(m.group(2))))

    if not date_tokens:
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
                "forecast_source": "swpc_text_3day",
            })
    slots.sort(key=lambda x: x["time_utc"])
    return slots


def load_archive() -> dict:
    if not ARCHIVE_PATH.exists():
        return {"version": "swifttec-kp-forecast-archive-v2", "forecasts": []}
    try:
        return json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": "swifttec-kp-forecast-archive-v2", "forecasts": []}


def save_archive(doc: dict) -> None:
    doc["version"] = "swifttec-kp-forecast-archive-v2"
    seen = set()
    forecasts = []
    # Prefer real forecast over cold-start if issue hour collides.
    def priority(f):
        src = f.get("source_type") or f.get("source_url") or ""
        cold = 1 if "cold_start" in str(src) else 0
        return (f.get("issue_utc", ""), cold)
    for f in sorted(doc.get("forecasts", []), key=priority):
        key = f.get("issue_utc", "")[:13]
        if key in seen:
            continue
        seen.add(key)
        forecasts.append(f)
    doc["forecasts"] = sorted(forecasts, key=lambda x: x.get("issue_utc", ""))[-MAX_FORECAST_ISSUES:]
    doc["updated_utc"] = iso(now_utc())
    ARCHIVE_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def add_coldstart_forecasts(doc: dict, actual_kp: list[tuple[datetime, float]], days: int = COLDSTART_DAYS) -> int:
    """Rebuild clearly-labelled initial slots from Kp history every run.

    This is not a reconstruction of historical SWPC forecasts. It prevents a new install from
    staying at N=0 while real forecast issues start accumulating from GitHub Actions.
    """
    if not ENABLE_COLDSTART or not actual_kp:
        doc["cold_start_backfill"] = {"enabled": False, "reason": "disabled or no actual Kp"}
        return 0

    # Remove stale cold-start entries first.  v7.8 skipped rebuilding when one old cold-start
    # issue existed, which could leave the score at N=0 forever if that old entry did not overlap
    # the available TEC archive.
    real_forecasts = [
        f for f in doc.get("forecasts", [])
        if f.get("source_type") != "cold_start_actual_kp_history"
    ]
    doc["forecasts"] = real_forecasts

    now = now_utc()
    cutoff = now - timedelta(days=days)
    kp_rows = [(t, kp) for t, kp in actual_kp if cutoff <= t <= now - timedelta(hours=3)]
    if not kp_rows:
        doc["cold_start_backfill"] = {"enabled": True, "days": days, "issue_count_added": 0, "reason": "actual Kp rows did not overlap cold-start window"}
        return 0

    # One synthetic issue per 00Z day; each issue contains targets at +1..+4 days.
    first_issue_day = (kp_rows[0][0] - timedelta(days=4)).date()
    last_issue_day = (kp_rows[-1][0] - timedelta(days=1)).date()
    existing_real = {f.get("issue_utc", "")[:13] for f in real_forecasts}
    added = 0

    issue_day = first_issue_day
    while issue_day <= last_issue_day:
        issue = datetime(issue_day.year, issue_day.month, issue_day.day, tzinfo=UTC)
        # Do not replace a real forecast issue with a pseudo issue at the same hour.
        if iso(issue)[:13] in existing_real:
            issue_day = issue_day + timedelta(days=1)
            continue

        slots = []
        for t, kp in kp_rows:
            lead_hours = (t - issue).total_seconds() / 3600.0
            if 0 < lead_hours <= 4 * 24 + 3:
                lead_day = int(math.ceil(lead_hours / 24.0))
                if lead_day in (1, 2, 3, 4):
                    slots.append({
                        "time_utc": iso(t),
                        "kp_forecast": kp,
                        "lead_hours": round(lead_hours, 2),
                        "lead_day": lead_day,
                        "forecast_source": "cold_start_actual_kp_history",
                    })
        if slots:
            doc.setdefault("forecasts", []).append({
                "issue_utc": iso(issue),
                "source_type": "cold_start_actual_kp_history",
                "source_url": KP_ACTUAL_URLS[0],
                "horizon_note": "Cold-start backfill uses historical actual Kp as a temporary pseudo-forecast. Real operational forecast scoring uses subsequently archived predicted Kp.",
                "slots": slots,
            })
            added += 1
        issue_day = issue_day + timedelta(days=1)

    doc["cold_start_backfill"] = {
        "enabled": True,
        "days": days,
        "issue_count_added": added,
        "kp_rows_in_window": len(kp_rows),
        "note": "Initial fill only; source is actual Kp history, not historical issued forecasts.",
    }
    return added



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



def score_operational(doc: dict, actual_kp: list[tuple[datetime, float]], actual_source_url: str = "") -> dict:
    """Score operational/reference TEC hit rate by walking TEC frame times first.

    v8.0 fix:
    The older scorer walked Kp forecast slots first.  If Kp slots were 3-hourly or
    1-minute cold-start slots while TEC frames were 30-minute archive frames, the
    exact target/base pair often missed and all lead days stayed N=0.

    This version walks available TEC frames as the truth timeline, then assigns:
      - base TEC frame at target - lead_day
      - actual base Kp near that base time
      - forecast Kp near target time from archived forecast slots, or labelled
        cold-start actual-Kp reference if no real archived forecast exists yet.
    """
    metas = load_frame_meta()
    coeff_doc = load_coeff_doc()
    updated = iso(now_utc())

    diagnostics = {
        "scoring_mode": "tec_frame_driven_v8",
        "tec_frame_count": len(metas),
        "forecast_issue_count": len(doc.get("forecasts", [])),
        "actual_kp_count": len(actual_kp),
        "coefficients_available": bool(coeff_doc),
        "pair_tolerance_min": PAIR_TOLERANCE_MIN,
        "forecast_kp_match_tolerance_min": max(120, PAIR_TOLERANCE_MIN),
        "skip_counts": {},
        "seen_tec_targets_by_lead": {},
        "used_tec_targets_by_lead": {},
    }

    def bump(name: str, lead: int | None = None, n: int = 1):
        diagnostics["skip_counts"][name] = diagnostics["skip_counts.get"](name, 0) + n if False else diagnostics["skip_counts"].get(name, 0) + n
        if lead in (1, 2, 3, 4):
            key = str(lead)
            diagnostics.setdefault(f"{name}_by_lead", {})
            diagnostics[f"{name}_by_lead"][key] = diagnostics[f"{name}_by_lead"].get(key, 0) + n

    if not metas:
        return {
            "version": "swifttec-operational-hit-rate-v4",
            "updated_utc": updated,
            "reason": "TEC archiveがないため採点できません",
            "score_diagnostics": diagnostics,
            "by_lead_day": {},
        }
    if not coeff_doc:
        return {
            "version": "swifttec-operational-hit-rate-v4",
            "updated_utc": updated,
            "reason": "kp_grid_coefficients.json がないため採点できません",
            "score_diagnostics": diagnostics,
            "by_lead_day": {},
        }
    if not actual_kp:
        return {
            "version": "swifttec-operational-hit-rate-v4",
            "updated_utc": updated,
            "reason": "Kp実測履歴がないため採点できません",
            "score_diagnostics": diagnostics,
            "by_lead_day": {},
        }

    first = load_json_maybe_gz(metas[0]["path"])
    lat_arr, lon_arr, _ = flatten_grid(first)
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    if not n_lat or not n_lon:
        return {
            "version": "swifttec-operational-hit-rate-v4",
            "updated_utc": updated,
            "reason": "TEC格子が空のため採点できません",
            "score_diagnostics": diagnostics,
            "by_lead_day": {},
        }

    # Build forecast-slot index.  We keep real forecasts and cold-start pseudo forecasts
    # separate so the output can label which source actually drove the score.
    forecast_slots_by_lead: dict[int, list[dict]] = defaultdict(list)
    source_counts = Counter()
    cold_issue_count = 0
    for f in doc.get("forecasts", []):
        issue = parse_time(f.get("issue_utc"))
        src = f.get("source_type") or f.get("source_url") or "unknown"
        source_counts[src] += 1
        if "cold_start" in str(src):
            cold_issue_count += 1
        for s in f.get("slots", []):
            t = parse_time(s.get("time_utc"))
            if not t:
                continue
            try:
                kp = float(s.get("kp_forecast"))
            except Exception:
                continue
            lead = int(s.get("lead_day") or (math.ceil(max(0, (t - issue).total_seconds()) / 86400.0) if issue else 0))
            if lead in (1, 2, 3, 4) and math.isfinite(kp):
                forecast_slots_by_lead[lead].append({
                    "time": t,
                    "kp": kp,
                    "issue": issue,
                    "source": s.get("forecast_source") or src,
                    "source_type": src,
                    "is_cold": "cold_start" in str(s.get("forecast_source") or src),
                })

    for lead in (1, 2, 3, 4):
        forecast_slots_by_lead[lead].sort(key=lambda x: x["time"])

    def forecast_kp_for_target(target: datetime, lead: int) -> tuple[float | None, str, bool]:
        """Return forecast Kp for a TEC frame target time.

        Prefer a non-cold archived forecast slot near the target.  When no real
        forecast archive has accumulated yet, use actual Kp history as a labelled
        cold-start reference so the panel can populate immediately.
        """
        tol_sec = max(120, PAIR_TOLERANCE_MIN) * 60
        best = None
        best_diff = float("inf")

        # Prefer real archived forecast slots.
        for s in forecast_slots_by_lead.get(lead, []):
            if s.get("is_cold"):
                continue
            d = abs((s["time"] - target).total_seconds())
            if d < best_diff:
                best_diff = d
                best = s
        if best is not None and best_diff <= tol_sec:
            return float(best["kp"]), str(best.get("source") or "archived_forecast"), False

        # Next, allow the generated cold-start slot if it lines up.
        best = None
        best_diff = float("inf")
        for s in forecast_slots_by_lead.get(lead, []):
            if not s.get("is_cold"):
                continue
            d = abs((s["time"] - target).total_seconds())
            if d < best_diff:
                best_diff = d
                best = s
        if best is not None and best_diff <= tol_sec:
            return float(best["kp"]), str(best.get("source") or "cold_start_actual_kp_history"), True

        # Final cold-start fallback: actual Kp at target TEC frame time.
        # This is not a true past forecast; it is only an initial/reference display.
        if ENABLE_COLDSTART:
            k = kp_at(target, actual_kp, max_diff_hours=2.0)
            if k is not None:
                return float(k), "cold_start_actual_kp_history_frame_fallback", True
        return None, "forecast_kp_missing", False

    def frame_vals_cached(meta: dict, cache: dict[str, list[float]]):
        key = str(meta["path"])
        if key not in cache:
            frame = load_json_maybe_gz(meta["path"])
            _, _, vals = flatten_grid(frame)
            cache[key] = vals
        return cache[key]

    step = max(1, int(math.ceil((n_lat * n_lon) / MAX_CELL_SAMPLES_PER_FRAME)))
    errors_by_lead: dict[int, list[float]] = defaultdict(list)
    target_count_by_lead: dict[int, int] = defaultdict(int)
    seen_targets_by_lead: dict[int, int] = defaultdict(int)
    used_source_by_lead: dict[int, Counter] = defaultdict(Counter)
    cache_vals: dict[str, list[float]] = {}

    now = now_utc()
    cutoff = now - timedelta(days=SCORE_DAYS_BACK)
    # Need at least lead-day base TEC, so target frames in the current score window are enough.
    target_metas = [m for m in metas if cutoff <= m["time"] <= now - timedelta(minutes=30)]

    for meta_f in target_metas:
        target = meta_f["time"]
        for lead in (1, 2, 3, 4):
            seen_targets_by_lead[lead] += 1
            base_time = target - timedelta(days=lead)
            meta_b = nearest_meta(metas, base_time, tolerance_min=max(PAIR_TOLERANCE_MIN, 90))
            if not meta_b:
                bump("base_tec_frame_missing", lead)
                continue

            kp_b = kp_at(base_time, actual_kp, max_diff_hours=2.0)
            if kp_b is None:
                bump("base_kp_missing", lead)
                continue

            kp_f, kp_source, cold_used = forecast_kp_for_target(target, lead)
            if kp_f is None:
                bump("forecast_kp_missing_for_target_frame", lead)
                continue

            try:
                vals_f = frame_vals_cached(meta_f, cache_vals)
                vals_b = frame_vals_cached(meta_b, cache_vals)
            except Exception:
                bump("tec_frame_load_error", lead)
                continue

            if len(vals_f) != n_lat * n_lon or len(vals_b) != n_lat * n_lon:
                bump("tec_grid_size_mismatch", lead)
                continue

            month = target.month
            before = len(errors_by_lead[lead])
            for idx in range(0, n_lat * n_lon, step):
                vf = vals_f[idx]
                vb = vals_b[idx]
                if not math.isfinite(vf) or not math.isfinite(vb):
                    continue
                i, j = divmod(idx, n_lon)
                cf = coeff_for_cell(coeff_doc, month, i, j)
                forecast_tec = vb - predict(cf, kp_b) + predict(cf, kp_f)
                errors_by_lead[lead].append(vf - forecast_tec)

            if len(errors_by_lead[lead]) == before:
                bump("no_finite_tec_cells", lead)
                continue

            target_count_by_lead[lead] += 1
            used_source_by_lead[lead][kp_source] += 1

    by_lead = {}
    cold_used_in_score = False
    total_samples = 0

    for lead in (1, 2, 3, 4):
        errs = errors_by_lead.get(lead, [])
        total_samples += len(errs)
        sources = dict(used_source_by_lead.get(lead, Counter()))
        if any("cold_start" in str(k) for k in sources):
            cold_used_in_score = True
        th = threshold_summary(errs)
        by_lead[str(lead)] = {
            "lead_day": lead,
            "slot_count": int(target_count_by_lead.get(lead, 0)),
            "seen_slot_count": int(seen_targets_by_lead.get(lead, 0)),
            "forecast_sources": sources,
            "thresholds": th,
            **th.get("5", {}),
        }

    diagnostics["seen_tec_targets_by_lead"] = {str(k): int(v) for k, v in seen_targets_by_lead.items()}
    diagnostics["used_tec_targets_by_lead"] = {str(k): int(v) for k, v in target_count_by_lead.items()}
    diagnostics["total_error_samples"] = total_samples
    diagnostics["target_tec_frame_count_in_score_window"] = len(target_metas)
    diagnostics["forecast_slots_by_lead"] = {str(k): len(v) for k, v in forecast_slots_by_lead.items()}

    reason = None
    if total_samples == 0:
        sc = diagnostics["skip_counts"]
        if sc.get("base_tec_frame_missing"):
            reason = "TEC実測アーカイブがlead日数分まだ不足しているため N=0"
        elif sc.get("forecast_kp_missing_for_target_frame"):
            reason = "TEC時刻に対応するKp予報がなく、cold-start補完も使えないため N=0"
        elif sc.get("base_kp_missing"):
            reason = "Base時刻のKp実測がないため N=0"
        elif not target_metas:
            reason = "採点対象のTEC実測フレームがまだないため N=0"
        else:
            reason = "採点対象はありますが、TEC/Kp/格子の照合で有効サンプルが0です"

    return {
        "version": "swifttec-operational-hit-rate-v4",
        "updated_utc": updated,
        "definition": "Operational hit rate uses archived forecast Kp when available. During cold start, a labelled actual-Kp reference fills the panel until real forecast archives accumulate.",
        "reason": reason,
        "forecast_issue_count": len(doc.get("forecasts", [])),
        "cold_start_issue_count": cold_issue_count,
        "score_days_back": SCORE_DAYS_BACK,
        "actual_kp_source_url": actual_source_url,
        "actual_kp_count": len(actual_kp),
        "forecast_issue_source_counts": dict(source_counts),
        "cold_start_backfill_used_in_score": cold_used_in_score,
        "cold_start_note": "cold_start_actual_kp_history uses actual historical Kp and should be treated as initial/reference display, not a true past forecast reconstruction.",
        "max_cell_samples_per_frame": MAX_CELL_SAMPLES_PER_FRAME,
        "score_diagnostics": diagnostics,
        "by_lead_day": by_lead,
    }



def fetch_forecast_slots(issue: datetime) -> tuple[list[dict], str, str]:
    try:
        slots = parse_forecast_json_slots(fetch_json(KP_FORECAST_JSON_URL), issue)
        if slots:
            return slots, KP_FORECAST_JSON_URL, "swpc_json_forecast"
    except Exception:
        pass
    text = fetch_text(KP_FORECAST_TEXT_URL)
    return parse_3day_kp_forecast(text, issue), KP_FORECAST_TEXT_URL, "swpc_text_3day"


def main() -> int:
    issue = now_utc()
    actual_kp, actual_source_url, actual_errors = fetch_actual_kp_history()

    slots, forecast_url, forecast_source_type = fetch_forecast_slots(issue)

    archive = load_archive()
    archive.setdefault("forecasts", []).append({
        "issue_utc": iso(issue),
        "source_url": forecast_url,
        "source_type": forecast_source_type,
        "horizon_note": "SWPC forecast is archived when available. Lead 4 requires >=4-day forecast data or cold-start/reference backfill.",
        "slots": slots,
    })

    cold_added = add_coldstart_forecasts(archive, actual_kp, days=COLDSTART_DAYS)
    archive["actual_kp_source_url"] = actual_source_url
    archive["actual_kp_count_latest_fetch"] = len(actual_kp)
    if actual_errors:
        archive["actual_kp_fetch_errors"] = actual_errors
    archive["last_forecast_source_type"] = forecast_source_type
    archive["last_forecast_slot_count"] = len(slots)
    archive["cold_start_issue_added_latest_run"] = cold_added
    save_archive(archive)

    scored = score_operational(archive, actual_kp, actual_source_url=actual_source_url)
    OP_PATH.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Actual Kp rows: {len(actual_kp)} from {actual_source_url or 'none'}")
    print(f"Archived forecast slots: {len(slots)} from {forecast_source_type}")
    print(f"Cold-start issues added: {cold_added}")
    print(f"Forecast issues: {len(archive.get('forecasts', []))}")
    print(f"Operational score written: {OP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
