#!/usr/bin/env python3
"""SWIFT-TEC v8.8 — ISEE Japan VTEC backfill fetcher.

Purpose
-------
Fetch ISEE Global GNSS *Vertical TEC* NetCDF (AGRID2, *_atec.nc), crop the
Japan area, convert/output TEC as VTEC in TECU, and maintain a rolling archive.

Key behavior
------------
1. Discover the newest actually-published ISEE hourly NetCDF, rather than
   assuming "now" exists. ISEE products can have publication latency.
2. On the first run, backfill up to the latest 10 days available at ISEE.
3. On later runs, download only hourly source files not already represented
   in docs/data/isee_tec/index.json.
4. Expand each hourly NetCDF into its native time records (normally 5-min).
5. Always write JSON grids as VTEC [TECU].
6. Keep the latest 30 days of VTEC frames for Japan-AI training and Base generation.
7. Keep source latency metadata in index.json so the UI can show how old the
   newest upstream ISEE product is.

Environment
-----------
SWIFTTEC_ISEE_BACKFILL_HOURS   default 240
SWIFTTEC_ISEE_KEEP_DAYS        default 30
SWIFTTEC_ISEE_SEARCH_DAYS      default 90
SWIFTTEC_ISEE_MAX_HOURLY_FILES default 240
SWIFTTEC_ISEE_FORCE_REFETCH    default 0
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr

BASE = "https://stdb2.isee.nagoya-u.ac.jp/GPS/shinbori/AGRID2/nc/"
OUT = Path("docs/data/isee_tec")
OUT.mkdir(parents=True, exist_ok=True)

LAT_MIN = float(os.environ.get("SWIFTTEC_ISEE_LAT_MIN", "24"))
LAT_MAX = float(os.environ.get("SWIFTTEC_ISEE_LAT_MAX", "46"))
LON_MIN = float(os.environ.get("SWIFTTEC_ISEE_LON_MIN", "122"))
LON_MAX = float(os.environ.get("SWIFTTEC_ISEE_LON_MAX", "150"))

BACKFILL_HOURS = max(24, int(os.environ.get("SWIFTTEC_ISEE_BACKFILL_HOURS", "240")))
KEEP_DAYS = max(10, int(os.environ.get("SWIFTTEC_ISEE_KEEP_DAYS", "30")))
SEARCH_DAYS = max(14, int(os.environ.get("SWIFTTEC_ISEE_SEARCH_DAYS", "90")))
MAX_HOURLY_FILES = max(1, int(os.environ.get("SWIFTTEC_ISEE_MAX_HOURLY_FILES", "240")))
FORCE_REFETCH = os.environ.get("SWIFTTEC_ISEE_FORCE_REFETCH", "0").strip().lower() in {"1", "true", "yes"}

UA = "SWIFT-TEC/8.8 ISEE-Japan-VTEC"


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def http_text(url: str, timeout: int = 45) -> str:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def http_bytes(url: str, timeout: int = 150) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def day_url(d: datetime) -> str:
    return f"{BASE}{d.year}/{d.timetuple().tm_yday:03d}/"


def parse_hour_from_name(name: str) -> datetime | None:
    m = re.search(r"(\d{10})_atec\.nc$", name, re.I)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def list_day_files(d: datetime, cache: dict[str, list[str]]) -> list[str]:
    key = d.strftime("%Y-%m-%d")
    if key in cache:
        return cache[key]
    u = day_url(d)
    try:
        html = http_text(u)
        names = sorted(set(re.findall(r'href=["\']([^"\']+_atec\.nc)["\']', html, re.I)))
    except Exception as exc:
        print(f"WARN day index failed {u}: {exc}")
        names = []
    cache[key] = names
    return names


def find_latest_available(cache: dict[str, list[str]]) -> tuple[datetime, str, str]:
    now = datetime.now(timezone.utc)
    for back in range(SEARCH_DAYS + 1):
        d = now - timedelta(days=back)
        names = list_day_files(d, cache)
        if not names:
            continue
        candidates = []
        for name in names:
            t = parse_hour_from_name(name)
            if t:
                candidates.append((t, name))
        if candidates:
            t, name = max(candidates)
            return t, urljoin(day_url(t), name), name
    raise RuntimeError(f"No recent ISEE AGRID2 *_atec.nc found in last {SEARCH_DAYS} days")


def coord_var_name(ds: xr.Dataset, kind: str) -> str | None:
    candidates = {
        "lat": ["lat", "latitude", "glat", "gdlat"],
        "lon": ["lon", "longitude", "glon", "gdlon"],
        "time": ["time", "ut", "utc", "datetime"],
    }[kind]
    names = list(ds.coords) + list(ds.variables)
    for n in names:
        if str(n).lower() in candidates:
            return n
    for n in names:
        low = str(n).lower()
        if any(c in low for c in candidates):
            return n
    return None


def coord_dim_name(ds: xr.Dataset, var_name: str | None, kind: str) -> str | None:
    if var_name and var_name in ds.variables:
        dims = list(ds[var_name].dims)
        if len(dims) == 1:
            return dims[0]

    candidates = {
        "lat": ["latitude", "lat", "glat", "gdlat"],
        "lon": ["longitude", "lon", "glon", "gdlon"],
        "time": ["time", "ut", "utc", "datetime"],
    }[kind]
    for d in ds.dims:
        if str(d).lower() in candidates:
            return d
    for d in ds.dims:
        low = str(d).lower()
        if any(c in low for c in candidates):
            return d
    return None


def tec_var_name(ds: xr.Dataset, latdim: str | None, londim: str | None) -> str:
    scored: list[tuple[int, str]] = []
    for n, da in ds.data_vars.items():
        if not np.issubdtype(da.dtype, np.number):
            continue
        low = n.lower()
        score = 0
        if low == "atec":
            score += 200
        if "tec" in low:
            score += 100
        if latdim and latdim in da.dims:
            score += 25
        if londim and londim in da.dims:
            score += 25
        score += min(10, da.ndim)
        scored.append((score, n))
    if not scored:
        raise RuntimeError("No numeric TEC-like variable found")
    return max(scored)[1]


def lon180(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return ((a + 180.0) % 360.0) - 180.0


def units_to_tecu_factor(units: str | None, var_name: str) -> tuple[float, str]:
    """Return multiplicative factor to TECU.

    ISEE AGRID2 `atec` is published from the site's Vertical TEC data product.
    We still inspect NetCDF units and convert electron/m^2 when explicitly given.
    """
    u = (units or "").strip().lower().replace(" ", "")
    if not u:
        return 1.0, "ISEE AGRID2 atec assumed TECU (official product: Vertical TEC)"
    if "tecu" in u:
        return 1.0, f"NetCDF units={units}"
    if "10^16" in u or "10**16" in u or "1e16" in u:
        return 1.0, f"NetCDF units={units}"
    if ("electron" in u or "el" in u) and ("m-2" in u or "/m2" in u or "m^-2" in u):
        return 1.0e-16, f"converted {units} -> TECU"
    # ISEE's atec is the absolute/vertical TEC product; avoid a speculative rescale.
    if var_name.lower() == "atec":
        return 1.0, f"ISEE atec treated as TECU; raw units={units}"
    raise RuntimeError(f"Unsupported TEC units for {var_name}: {units!r}")


def decode_times(ds: xr.Dataset, time_var: str | None, time_dim: str | None,
                 source_hour: datetime, n_time: int) -> list[datetime]:
    vals = None
    if time_var and time_var in ds.variables:
        vals = np.asarray(ds[time_var].values).reshape(-1)
    elif time_dim and time_dim in ds.variables:
        vals = np.asarray(ds[time_dim].values).reshape(-1)

    out: list[datetime] = []
    if vals is not None and len(vals) == n_time:
        for x in vals:
            try:
                ns = np.datetime64(x, "ns").astype("int64")
                t = datetime.fromtimestamp(float(ns) / 1e9, tz=timezone.utc)
                # Reject obviously bad decoded dates.
                if abs((t - source_hour).total_seconds()) < 3 * 3600:
                    out.append(t)
                else:
                    raise ValueError("decoded time far from source hour")
            except Exception:
                out = []
                break

    if len(out) == n_time:
        return out

    # Hourly files normally contain 12 records => 5 min. Generalize if count differs.
    if n_time <= 1:
        return [source_hour]
    step_sec = 3600.0 / n_time
    return [source_hour + timedelta(seconds=i * step_sec) for i in range(n_time)]


def process_hourly_nc(url: str, source_name: str, source_hour: datetime) -> list[dict]:
    raw = http_bytes(url)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(raw)
        tf.flush()
        ds = xr.open_dataset(tf.name, decode_times=True)

        latv = coord_var_name(ds, "lat")
        lonv = coord_var_name(ds, "lon")
        timev = coord_var_name(ds, "time")
        if not latv or not lonv:
            raise RuntimeError(f"lat/lon variables not detected; vars={list(ds.variables)}")

        latdim = coord_dim_name(ds, latv, "lat")
        londim = coord_dim_name(ds, lonv, "lon")
        timedim = coord_dim_name(ds, timev, "time")
        tecn = tec_var_name(ds, latdim, londim)
        da = ds[tecn]

        lat_dim = latdim if latdim in da.dims else next((d for d in da.dims if "lat" in d.lower()), None)
        lon_dim = londim if londim in da.dims else next((d for d in da.dims if "lon" in d.lower()), None)
        time_dim = timedim if timedim in da.dims else next((d for d in da.dims if "time" in d.lower()), None)

        if not lat_dim or not lon_dim:
            raise RuntimeError(f"TEC dims do not expose lat/lon: tec={tecn}, dims={da.dims}")

        lat = np.asarray(ds[latv].values, dtype=float).squeeze()
        lon = lon180(np.asarray(ds[lonv].values, dtype=float).squeeze())
        if lat.ndim != 1 or lon.ndim != 1:
            raise RuntimeError("Expected 1-D latitude/longitude coordinates")

        ilat = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
        ilon = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
        if not len(ilat) or not len(ilon):
            raise RuntimeError("Japan crop contains no cells")

        extra_dims = [d for d in da.dims if d not in {lat_dim, lon_dim, time_dim}]
        for d in extra_dims:
            da = da.isel({d: 0})

        if time_dim:
            da = da.transpose(time_dim, lat_dim, lon_dim)
        else:
            da = da.transpose(lat_dim, lon_dim).expand_dims({"_time": [0]})
            time_dim = "_time"

        arr = np.asarray(da.values, dtype=float)
        arr = arr[:, ilat, :][:, :, ilon]
        latc = lat[ilat]
        lonc = lon[ilon]

        if len(latc) > 1 and latc[0] > latc[-1]:
            latc = latc[::-1]
            arr = arr[:, ::-1, :]

        order_lon = np.argsort(lonc)
        lonc = lonc[order_lon]
        arr = arr[:, :, order_lon]

        factor, unit_note = units_to_tecu_factor(da.attrs.get("units"), tecn)
        arr = arr * factor

        # Plausibility/fill guard after unit conversion.
        arr[~np.isfinite(arr)] = np.nan
        arr[(arr < -0.5) | (arr > 300.0)] = np.nan
        arr[arr < 0] = 0.0

        times = decode_times(ds, timev, time_dim, source_hour, arr.shape[0])

        print(
            "Detected:",
            {
                "source": source_name,
                "tec_var": tecn,
                "tec_dims": tuple(ds[tecn].dims),
                "raw_units": ds[tecn].attrs.get("units"),
                "output_quantity": "VTEC",
                "output_units": "TECU",
                "records": len(times),
                "japan_grid": f"{len(latc)}x{len(lonc)}",
                "unit_note": unit_note,
            },
        )

        frames = []
        for i, t in enumerate(times):
            grid = arr[i]
            fname = f"{t.strftime('%Y%m%dT%H%M%SZ')}.json"
            payload = {
                "version": "swifttec-isee-japan-vtec-v2",
                "time_utc": iso(t),
                "quantity": "VTEC",
                "units": "TECU",
                "tec_definition": "Vertical Total Electron Content",
                "source_product": "ISEE Global GNSS Vertical TEC / AGRID2",
                "source_file": source_name,
                "source_url": url,
                "source_variable": tecn,
                "source_units": ds[tecn].attrs.get("units"),
                "unit_conversion_note": unit_note,
                "lat_arr": [round(float(x), 5) for x in latc],
                "lon_arr": [round(float(x), 5) for x in lonc],
                "n_lat": len(latc),
                "n_lon": len(lonc),
                # Keep `grid` for existing SWIFT-TEC compatibility.
                "grid": [
                    [None if not np.isfinite(v) else round(float(v), 3) for v in row]
                    for row in grid
                ],
            }
            (OUT / fname).write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            frames.append(
                {
                    "time_utc": payload["time_utc"],
                    "file": fname,
                    "quantity": "VTEC",
                    "units": "TECU",
                    "source_file": source_name,
                    "source_url": url,
                }
            )
        return frames


def load_old_index() -> dict:
    p = OUT / "index.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    now = datetime.now(timezone.utc)
    day_cache: dict[str, list[str]] = {}
    latest_hour, latest_url, latest_name = find_latest_available(day_cache)
    latency_h = max(0.0, (now - latest_hour).total_seconds() / 3600.0)

    print("ISEE newest published source:", latest_url)
    print(f"ISEE source latency: {latency_h:.1f} hours")
    print(f"Backfill window: {BACKFILL_HOURS} hours ending at {iso(latest_hour)}")

    old_index = load_old_index()
    old_frames = old_index.get("frames", []) if isinstance(old_index.get("frames"), list) else []
    existing_sources = defaultdict(int)
    for f in old_frames:
        sf = str(f.get("source_file") or "")
        if sf:
            existing_sources[sf] += 1

    start_hour = latest_hour - timedelta(hours=BACKFILL_HOURS - 1)
    wanted_hours = [start_hour + timedelta(hours=i) for i in range(BACKFILL_HOURS)]

    targets: list[tuple[datetime, str, str]] = []
    missing_upstream = 0
    for h in wanted_hours:
        names = list_day_files(h, day_cache)
        expected = h.strftime("%Y%m%d%H") + "_atec.nc"
        if expected not in names:
            missing_upstream += 1
            continue
        if not FORCE_REFETCH and existing_sources.get(expected, 0) >= 1:
            continue
        targets.append((h, urljoin(day_url(h), expected), expected))

    if len(targets) > MAX_HOURLY_FILES:
        # Prefer newest missing hours.
        targets = targets[-MAX_HOURLY_FILES:]

    print(
        f"Existing indexed frames={len(old_frames)} / "
        f"new hourly NetCDF to fetch={len(targets)} / "
        f"missing upstream hours={missing_upstream}"
    )

    new_frames: list[dict] = []
    failed = []
    for n, (h, url, name) in enumerate(targets, 1):
        try:
            print(f"[{n}/{len(targets)}] Fetch {url}")
            new_frames.extend(process_hourly_nc(url, name, h))
        except Exception as exc:
            failed.append({"source_file": name, "error": str(exc)})
            print(f"WARN failed {name}: {exc}")

    # Merge old/new by UTC frame time.
    by_time: dict[str, dict] = {}
    for f in old_frames + new_frames:
        t = str(f.get("time_utc") or "")
        if t:
            by_time[t] = f

    cutoff = now - timedelta(days=KEEP_DAYS)
    merged: list[dict] = []
    for key in sorted(by_time):
        try:
            t = datetime.fromisoformat(key.replace("Z", "+00:00"))
        except Exception:
            continue
        if t >= cutoff:
            # normalize legacy entries too
            f = dict(by_time[key])
            f["quantity"] = "VTEC"
            f["units"] = "TECU"
            merged.append(f)

    # Delete stale JSON frames no longer in rolling index, but never index.json.
    keep_files = {str(f.get("file")) for f in merged}
    for p in OUT.glob("*.json"):
        if p.name == "index.json":
            continue
        if p.name not in keep_files:
            try:
                p.unlink()
            except Exception:
                pass

    newest_frame_time = None
    if merged:
        newest_frame_time = datetime.fromisoformat(merged[-1]["time_utc"].replace("Z", "+00:00"))

    index = {
        "version": "swifttec-isee-japan-index-v4-daily30d-vtec",
        "updated_utc": iso(now),
        "quantity": "VTEC",
        "units": "TECU",
        "tec_definition": "Vertical Total Electron Content",
        "source_product": "ISEE Global GNSS Vertical TEC / AGRID2",
        "source_base_url": BASE,
        "latest_source_hour_utc": iso(latest_hour),
        "latest_source_file": latest_name,
        "source_latency_hours_at_fetch": round(latency_h, 2),
        "note": (
            "ISEE data can be published with latency; latest_source_hour_utc is the newest "
            "actually available upstream product, not necessarily current UTC."
        ),
        "region": {
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
        },
        "backfill_hours": BACKFILL_HOURS,
        "keep_days": KEEP_DAYS,
        "new_hourly_files_processed": len(targets),
        "new_frames_written": len(new_frames),
        "failed_hourly_files": failed,
        "frames": merged,
    }
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"DONE: new_frames={len(new_frames)}, total_frames={len(merged)}, "
        f"newest_frame={iso(newest_frame_time) if newest_frame_time else '--'}, "
        f"VTEC units=TECU"
    )
    if failed and not new_frames and not merged:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
