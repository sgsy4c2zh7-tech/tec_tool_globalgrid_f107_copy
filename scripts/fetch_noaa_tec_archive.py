#!/usr/bin/env python3
"""Fetch NOAA SWPC GloTEC GeoJSON frames and maintain a 30-day archive.

Output layout:
  docs/data/tec/index.json
  docs/data/tec/YYYY-MM-DD/HHMM.json.gz

This version is intentionally strict: if no TEC frame is produced, the workflow
fails instead of showing a green check with an empty docs/data directory.
"""
from __future__ import annotations

import gzip
import html
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
NOAA_GLOTEC_DIR_URL = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt/"
OUT_ROOT = Path(os.environ.get("SWIFTTEC_TEC_ROOT", "docs/data/tec"))
KEEP_DAYS = int(os.environ.get("SWIFTTEC_KEEP_DAYS", "30"))
TARGET_LAT_STEP = float(os.environ.get("SWIFTTEC_LAT_STEP", "2.0"))
TARGET_LON_STEP = float(os.environ.get("SWIFTTEC_LON_STEP", "5.0"))
LOOKBACK_HOURS = int(os.environ.get("SWIFTTEC_LOOKBACK_HOURS", "36"))
TARGET_INTERVAL_MIN = int(os.environ.get("SWIFTTEC_TARGET_INTERVAL_MIN", "120"))
MAX_PER_RUN = int(os.environ.get("SWIFTTEC_MAX_PER_RUN", "24"))
MAX_SLOT_DIFF_MIN = int(os.environ.get("SWIFTTEC_MAX_SLOT_DIFF_MIN", "90"))

UTC = timezone.utc


def http_bytes(url: str, timeout: int = 90) -> bytes:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-v4-archive/1.1"})
    with urlopen(req, timeout=timeout) as res:
        return res.read()


def http_text(url: str, timeout: int = 90) -> str:
    return http_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def http_json(url: str):
    return json.loads(http_text(url, timeout=60))


def normalize_index(obj) -> list[str]:
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = []
        for key in ("files", "data", "items", "products", "urls"):
            if isinstance(obj.get(key), list):
                items = obj[key]
                break
        if not items:
            for value in obj.values():
                if isinstance(value, list):
                    items = value
                    break
    else:
        items = []

    candidate_keys = (
        "url", "href", "path", "name", "file", "filename", "product_location",
        "productLocation", "location", "link"
    )
    paths: list[str] = []
    for item in items:
        if isinstance(item, str):
            p = item
        elif isinstance(item, dict):
            p = ""
            for key in candidate_keys:
                if item.get(key):
                    p = str(item[key])
                    break
        else:
            p = str(item)
        if ".geojson" in p.lower():
            paths.append(p)
    return paths


def scrape_directory_listing() -> list[str]:
    """Fallback for NOAA's Apache-style index page."""
    txt = http_text(NOAA_GLOTEC_DIR_URL, timeout=60)
    paths = re.findall(r'href=["\']([^"\']+\.geojson)["\']', txt, flags=re.I)
    if not paths:
        paths = re.findall(r'(glotec_icao_\d{8}T\d{6}Z\.geojson)', txt, flags=re.I)
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        p = html.unescape(p)
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


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
    if "/" in path and not path.startswith("./"):
        return urljoin("https://services.swpc.noaa.gov/", path.lstrip("/"))
    return NOAA_GLOTEC_DIR_URL + basename(path)


def floor_to_slot(dt: datetime, interval_min: int = TARGET_INTERVAL_MIN) -> datetime:
    day = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    minutes = int((dt - day).total_seconds() // 60)
    slot = (minutes // interval_min) * interval_min
    return day + timedelta(minutes=slot)


def build_target_slots(now: datetime) -> list[datetime]:
    start = floor_to_slot(now - timedelta(hours=LOOKBACK_HOURS))
    end = floor_to_slot(now)
    out: list[datetime] = []
    t = start
    while t <= end:
        out.append(t)
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
        try:
            return float(coords[1]), lon_norm(float(coords[0]))
        except Exception:
            return None

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


def nearest_index(value: float, start: float, step: float, count: int, wrap: bool = False) -> int | None:
    idx = int(round((value - start) / step))
    if wrap:
        return idx % count
    if 0 <= idx < count:
        return idx
    return None


def parse_glotec_geojson(text: str, valid_time: datetime) -> dict:
    obj = json.loads(text)
    features = obj.get("features") or []
    if not isinstance(features, list):
        raise ValueError("GeoJSON features is not a list")

    lat_arr = [round(-90.0 + i * TARGET_LAT_STEP, 6) for i in range(int(round(180 / TARGET_LAT_STEP)) + 1)]
    lon_arr = [round(-180.0 + j * TARGET_LON_STEP, 6) for j in range(int(round(360 / TARGET_LON_STEP)))]
    n_lat = len(lat_arr)
    n_lon = len(lon_arr)

    sums = [[0.0 for _ in range(n_lon)] for _ in range(n_lat)]
    counts = [[0 for _ in range(n_lon)] for _ in range(n_lat)]
    point_count = 0

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
        i = nearest_index(lat, -90.0, TARGET_LAT_STEP, n_lat, wrap=False)
        j = nearest_index(lon, -180.0, TARGET_LON_STEP, n_lon, wrap=True)
        if i is None or j is None:
            continue
        sums[i][j] += val
        counts[i][j] += 1
        point_count += 1

    if point_count == 0:
        raise ValueError("No TEC points found in GloTEC GeoJSON")

    grid: list[list[int]] = []
    for i in range(n_lat):
        row: list[int] = []
        for j in range(n_lon):
            if counts[i][j] > 0:
                row.append(int(round(sums[i][j] / counts[i][j])))
            else:
                row.append(0)
        grid.append(row)

    return {
        "source": "NOAA_SWPC_GLOTEC_geojson_2d_urt",
        "time_utc": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lat_step": TARGET_LAT_STEP,
        "lon_step": TARGET_LON_STEP,
        "lat_arr": lat_arr,
        "lon_arr": lon_arr,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "grid": grid,
    }


def output_path_for_time(t: datetime) -> Path:
    return OUT_ROOT / t.strftime("%Y-%m-%d") / f"{t:%H%M}.json.gz"


def write_frame(frame: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(raw)


def rebuild_index(now: datetime) -> dict:
    cutoff = now - timedelta(days=KEEP_DAYS)
    frames: list[dict] = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for path in sorted(OUT_ROOT.glob("????-??-??/*.json.gz")):
        try:
            day = path.parent.name
            hhmm = path.name.replace(".json.gz", "")
            t = datetime.strptime(day + hhmm, "%Y-%m-%d%H%M").replace(tzinfo=UTC)
        except Exception:
            continue
        if t < cutoff:
            try:
                path.unlink()
            except Exception:
                pass
            continue
        frames.append({"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "file": path.relative_to(OUT_ROOT).as_posix()})

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


def load_noaa_paths() -> list[str]:
    paths: list[str] = []
    try:
        print(f"Fetching NOAA GloTEC JSON index: {NOAA_GLOTEC_INDEX_URL}")
        paths = normalize_index(http_json(NOAA_GLOTEC_INDEX_URL))
        print(f"JSON index entries: {len(paths)}")
    except Exception as exc:
        print(f"WARN: JSON index failed: {exc}", file=sys.stderr)
    if not paths:
        print(f"Fetching NOAA GloTEC directory listing: {NOAA_GLOTEC_DIR_URL}")
        paths = scrape_directory_listing()
        print(f"Directory entries: {len(paths)}")
    return paths


def main() -> int:
    now = datetime.now(UTC).replace(microsecond=0)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    paths = load_noaa_paths()
    if not paths:
        raise RuntimeError("No .geojson entries found from NOAA GloTEC index or directory listing")

    entries: list[dict] = []
    for p in paths:
        fn = basename(p)
        t = parse_utc_from_filename(fn)
        if t:
            entries.append({"time": t, "fn": fn, "url": as_url(p)})
    entries.sort(key=lambda x: x["time"])
    if not entries:
        raise RuntimeError("NOAA paths were found, but no UTC timestamps could be parsed from filenames")

    print(f"Available NOAA range: {entries[0]['time']:%Y-%m-%dT%H:%MZ} .. {entries[-1]['time']:%Y-%m-%dT%H:%MZ} ({len(entries)} files)")

    chosen: dict[datetime, dict] = {}
    for slot in build_target_slots(now):
        best = min(entries, key=lambda e: abs((e["time"] - slot).total_seconds()))
        diff_min = abs((best["time"] - slot).total_seconds()) / 60.0
        if diff_min <= MAX_SLOT_DIFF_MIN:
            chosen[slot] = best

    work = sorted(chosen.items(), key=lambda kv: kv[0])[-MAX_PER_RUN:]
    print(f"Target slots selected: {len(work)}")

    new_count = 0
    skipped = 0
    failed = 0
    for slot, e in work:
        out = output_path_for_time(slot)
        if out.exists() and out.stat().st_size > 100:
            print(f"Skip existing {out}")
            skipped += 1
            continue
        try:
            print(f"Downloading {slot:%Y-%m-%dT%H:%MZ} <- {e['fn']}")
            txt = http_text(e["url"], timeout=120)
            frame = parse_glotec_geojson(txt, slot)
            frame["source_file"] = e["fn"]
            write_frame(frame, out)
            print(f"Wrote {out} ({out.stat().st_size} bytes)")
            new_count += 1
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
            failed += 1
            print(f"WARN: failed {slot.isoformat()} {e['url']}: {exc}", file=sys.stderr)

    idx = rebuild_index(now)
    (OUT_ROOT / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Archive update complete: new={new_count}, skipped={skipped}, failed={failed}, total={len(idx['frames'])}")

    if new_count + skipped == 0:
        raise RuntimeError("No TEC frames were written or found. See logs above; failing workflow instead of silently succeeding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
