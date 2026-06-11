#!/usr/bin/env python3
"""Fetch NOAA SWPC GloTEC GeoJSON frames and maintain a 30-day archive.

Output layout:
  docs/data/tec/index.json
  docs/data/tec/YYYY-MM-DD/HHMM.json.gz

The browser add-on reads these files directly from GitHub Pages.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

NOAA_GLOTEC_INDEX_URL = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt.json"
NOAA_GLOTEC_BASE_URL = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt/"
OUT_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
KEEP_DAYS = int(os.environ.get("SWIFTTEC_KEEP_DAYS", "30"))
TARGET_LAT_STEP = float(os.environ.get("SWIFTTEC_LAT_STEP", "2.0"))
TARGET_LON_STEP = float(os.environ.get("SWIFTTEC_LON_STEP", "5.0"))
LOOKBACK_HOURS = int(os.environ.get("SWIFTTEC_LOOKBACK_HOURS", "36"))
TARGET_INTERVAL_MIN = int(os.environ.get("SWIFTTEC_TARGET_INTERVAL_MIN", "120"))
MAX_PER_RUN = int(os.environ.get("SWIFTTEC_MAX_PER_RUN", "30"))

UTC = timezone.utc


def http_json(url: str):
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-v4-archive/1.0"})
    with urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def http_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-v4-archive/1.0"})
    with urlopen(req, timeout=90) as res:
        return res.read().decode("utf-8")


def normalize_index(obj) -> list[str]:
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for key in ("files", "data", "items"):
            if isinstance(obj.get(key), list):
                items = obj[key]
                break
        else:
            items = []
            for value in obj.values():
                if isinstance(value, list):
                    items = value
                    break
    else:
        items = []

    paths: list[str] = []
    for item in items:
        if isinstance(item, str):
            p = item
        elif isinstance(item, dict):
            p = item.get("url") or item.get("href") or item.get("path") or item.get("name") or item.get("file") or item.get("filename") or ""
        else:
            p = str(item)
        if ".geojson" in p.lower():
            paths.append(p)
    return paths


def basename(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def parse_utc_from_filename(name: str) -> datetime | None:
    m = re.search(r"(\d{8})T(\d{6})Z", name, flags=re.I)
    if not m:
        return None
    y = int(m.group(1)[0:4])
    mo = int(m.group(1)[4:6])
    d = int(m.group(1)[6:8])
    hh = int(m.group(2)[0:2])
    mm = int(m.group(2)[2:4])
    ss = int(m.group(2)[4:6])
    return datetime(y, mo, d, hh, mm, ss, tzinfo=UTC)


def as_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if "/" in path:
        return urljoin("https://services.swpc.noaa.gov/", path.lstrip("/"))
    return NOAA_GLOTEC_BASE_URL + path


def round_to_target_slot(dt: datetime, interval_min: int = TARGET_INTERVAL_MIN) -> datetime:
    day = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    minutes = int((dt - day).total_seconds() // 60)
    slot = round(minutes / interval_min) * interval_min
    if slot >= 24 * 60:
        return day + timedelta(days=1)
    return day + timedelta(minutes=slot)


def build_target_slots(now: datetime) -> set[datetime]:
    start = now - timedelta(hours=LOOKBACK_HOURS)
    start_slot = round_to_target_slot(start)
    end_slot = round_to_target_slot(now)
    out: set[datetime] = set()
    t = start_slot
    while t <= end_slot:
        out.add(t)
        t += timedelta(minutes=TARGET_INTERVAL_MIN)
    return out


def value_from_properties(props: dict) -> float | None:
    for key in ("tec", "vtec", "VTEC", "TEC", "value", "Value", "grid_value", "gridValue", "tecu", "TECU"):
        if key in props:
            try:
                v = float(props[key])
                if math.isfinite(v):
                    return max(0.0, math.floor(v))
            except Exception:
                pass
    for value in props.values():
        try:
            v = float(value)
            if math.isfinite(v):
                return max(0.0, math.floor(v))
        except Exception:
            pass
    return None


def lon_norm(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def feature_centroid(feature: dict) -> tuple[float, float] | None:
    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
        lon = float(coords[0])
        lat = float(coords[1])
        return lat, lon_norm(lon)

    ring = None
    if gtype == "Polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
    elif gtype == "MultiPolygon" and isinstance(coords, list) and coords and coords[0]:
        ring = coords[0][0]

    if not isinstance(ring, list) or len(ring) < 3:
        return None

    lats: list[float] = []
    lons: list[float] = []
    for c in ring:
        if isinstance(c, list) and len(c) >= 2:
            try:
                lons.append(lon_norm(float(c[0])))
                lats.append(float(c[1]))
            except Exception:
                pass
    if not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def parse_glotec_geojson(text: str, valid_time: datetime) -> dict:
    obj = json.loads(text)
    features = obj.get("features") or []
    points: list[tuple[float, float, float]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            continue
        val = value_from_properties(props)
        if val is None:
            continue
        center = feature_centroid(feature)
        if center is None:
            continue
        lat, lon = center
        points.append((lat, lon, val))

    if not points:
        raise ValueError("No TEC points found in GloTEC GeoJSON")

    lat_arr = [round(-90.0 + i * TARGET_LAT_STEP, 6) for i in range(int(round(180 / TARGET_LAT_STEP)) + 1)]
    lon_arr = [round(-180.0 + j * TARGET_LON_STEP, 6) for j in range(int(round(360 / TARGET_LON_STEP)))]

    grid: list[list[int]] = []
    # Nearest-neighbor gridding. GloTEC native grid is already dense enough for display.
    for lat0 in lat_arr:
        row: list[int] = []
        for lon0 in lon_arr:
            best_v = 0.0
            best_d = 1.0e99
            for lat, lon, val in points:
                dlon = abs(lon - lon0)
                dlon = min(dlon, 360.0 - dlon)
                d = abs(lat - lat0) + dlon
                if d < best_d:
                    best_d = d
                    best_v = val
            row.append(int(best_v))
        grid.append(row)

    return {
        "source": "NOAA_SWPC_GLOTEC_geojson_2d_urt",
        "time_utc": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lat_step": TARGET_LAT_STEP,
        "lon_step": TARGET_LON_STEP,
        "lat_arr": lat_arr,
        "lon_arr": lon_arr,
        "n_lat": len(lat_arr),
        "n_lon": len(lon_arr),
        "grid": grid,
    }


def output_path_for_time(t: datetime) -> Path:
    day = t.strftime("%Y-%m-%d")
    hhmm = t.strftime("%H%M")
    return OUT_ROOT / day / f"{hhmm}.json.gz"


def write_frame(frame: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(raw)


def load_existing_index() -> dict:
    idx = OUT_ROOT / "index.json"
    if idx.exists():
        try:
            return json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def rebuild_index(now: datetime) -> dict:
    cutoff = now - timedelta(days=KEEP_DAYS)
    frames: list[dict] = []
    if OUT_ROOT.exists():
        for path in sorted(OUT_ROOT.glob("????-??-??/*.json.gz")):
            try:
                day = path.parent.name
                hhmm = path.stem.split(".")[0]
                t = datetime.strptime(day + hhmm, "%Y-%m-%d%H%M").replace(tzinfo=UTC)
            except Exception:
                continue
            if t < cutoff:
                try:
                    path.unlink()
                except Exception:
                    pass
                continue
            rel = path.relative_to(OUT_ROOT).as_posix()
            frames.append({"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "file": rel})

    # prune empty day dirs
    for d in OUT_ROOT.glob("????-??-??"):
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except Exception:
                pass

    return {
        "version": "swifttec-tec-archive-v1",
        "updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "keep_days": KEEP_DAYS,
        "lat_step": TARGET_LAT_STEP,
        "lon_step": TARGET_LON_STEP,
        "target_interval_minutes": TARGET_INTERVAL_MIN,
        "frames": frames,
    }


def main() -> int:
    now = datetime.now(UTC).replace(microsecond=0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Fetching NOAA GloTEC index: {NOAA_GLOTEC_INDEX_URL}")
    index_obj = http_json(NOAA_GLOTEC_INDEX_URL)
    paths = normalize_index(index_obj)
    if not paths:
        raise RuntimeError("No .geojson entries found in NOAA GloTEC index")

    entries: list[dict] = []
    for p in paths:
        fn = basename(p)
        t = parse_utc_from_filename(fn)
        if not t:
            continue
        entries.append({"time": t, "fn": fn, "url": as_url(p)})
    entries.sort(key=lambda x: x["time"])

    target_slots = build_target_slots(now)
    # Choose the nearest available file for each 2-hour slot in the lookback window.
    chosen: dict[datetime, dict] = {}
    for slot in target_slots:
        best = None
        best_diff = 10**18
        for e in entries:
            diff = abs((e["time"] - slot).total_seconds())
            if diff < best_diff:
                best = e
                best_diff = diff
        if best and best_diff <= TARGET_INTERVAL_MIN * 60 * 0.75:
            chosen[slot] = best

    # Avoid too many downloads if index contains dense data and workflow reruns manually.
    work = sorted(chosen.items(), key=lambda kv: kv[0])[-MAX_PER_RUN:]
    new_count = 0
    skipped = 0
    failed = 0

    for slot, e in work:
        out = output_path_for_time(slot)
        if out.exists() and out.stat().st_size > 100:
            skipped += 1
            continue
        try:
            print(f"Downloading {slot:%Y-%m-%dT%H:%MZ}: {e['url']}")
            txt = http_text(e["url"])
            frame = parse_glotec_geojson(txt, slot)
            frame["source_file"] = e["fn"]
            write_frame(frame, out)
            new_count += 1
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            failed += 1
            print(f"WARN: failed {slot.isoformat()} {e['url']}: {exc}", file=sys.stderr)

    idx = rebuild_index(now)
    (OUT_ROOT / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Archive update complete: new={new_count}, skipped={skipped}, failed={failed}, total={len(idx['frames'])}")
    return 0 if new_count or skipped or failed == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
