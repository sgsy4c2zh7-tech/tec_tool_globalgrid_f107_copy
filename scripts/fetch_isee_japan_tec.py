#!/usr/bin/env python3
"""Fetch latest ISEE AGRID2 NetCDF and create lightweight Japan TEC JSON.

The public AGRID2 directory is hourly NetCDF. This script:
- searches recent YYYY/DOY directories,
- downloads the newest *_atec.nc,
- detects lat/lon/time/TEC variables,
- crops Japan surroundings (24..46 N, 122..150 E),
- writes 5-minute frames to docs/data/isee_tec/.

v8.5 robustly handles ISEE coordinate aliases (lat/lon vars vs latitude/longitude dims).\n\nThis is intentionally independent from the NOAA/global forecast pipeline.
"""
from __future__ import annotations
import json, re, sys, tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin

import numpy as np
import xarray as xr

BASE = "https://stdb2.isee.nagoya-u.ac.jp/GPS/shinbori/AGRID2/nc/"
OUT = Path("docs/data/isee_tec")
OUT.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 24.0, 46.0
LON_MIN, LON_MAX = 122.0, 150.0
STEP_MIN = 5

def http_text(url: str, timeout=45) -> str:
    req = Request(url, headers={"User-Agent":"SWIFT-TEC/ISEE-Japan"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def http_bytes(url: str, timeout=120) -> bytes:
    req = Request(url, headers={"User-Agent":"SWIFT-TEC/ISEE-Japan"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()

def find_latest_file(max_back_days=45):
    now = datetime.now(timezone.utc)
    for back in range(max_back_days + 1):
        d = now - timedelta(days=back)
        doy = d.timetuple().tm_yday
        dir_url = f"{BASE}{d.year}/{doy:03d}/"
        try:
            html = http_text(dir_url)
        except Exception:
            continue
        names = sorted(set(re.findall(r'href=["\']([^"\']+_atec\.nc)["\']', html, re.I)))
        if names:
            return urljoin(dir_url, names[-1]), names[-1]
    raise RuntimeError("Recent ISEE AGRID2 *_atec.nc file was not found")

def coord_name(ds, kind):
    candidates = {
        "lat":["lat","latitude","glat","gdlat"],
        "lon":["lon","longitude","glon","gdlon"],
        "time":["time","ut","utc","datetime"],
    }[kind]
    for n in list(ds.coords) + list(ds.variables):
        low = n.lower()
        if low in candidates or any(low == c for c in candidates):
            return n
    for n in list(ds.coords) + list(ds.variables):
        low = n.lower()
        if any(c in low for c in candidates):
            return n
    return None

def coord_dim_name(ds, coord_var_name: str, kind: str) -> str | None:
    """Resolve a coordinate variable (e.g. lat) to the actual data dimension
    (e.g. latitude). ISEE AGRID2 uses aliases such as:
      coord variable: lat / lon
      TEC dims:       latitude / longitude / time
    """
    if coord_var_name in ds.variables:
        dims = list(ds[coord_var_name].dims)
        if len(dims) == 1:
            return dims[0]

    candidates = {
        "lat": ["latitude", "lat", "glat", "gdlat"],
        "lon": ["longitude", "lon", "glon", "gdlon"],
        "time": ["time", "ut", "utc", "datetime"],
    }[kind]

    for d in ds.dims:
        low = str(d).lower()
        if low in candidates:
            return d
    for d in ds.dims:
        low = str(d).lower()
        if any(c in low for c in candidates):
            return d
    return None


def tec_var_name(ds, latn, lonn, latdim=None, londim=None):
    scored = []
    for n, da in ds.data_vars.items():
        low = n.lower()
        if not np.issubdtype(da.dtype, np.number):
            continue
        score = 0
        if "tec" in low: score += 100
        if "atec" in low: score += 30
        if latn in da.dims or (latdim and latdim in da.dims): score += 20
        if lonn in da.dims or (londim and londim in da.dims): score += 20
        score += min(10, da.ndim)
        scored.append((score, n))
    if not scored:
        raise RuntimeError("No numeric TEC-like variable found in NetCDF")
    scored.sort(reverse=True)
    return scored[0][1]

def to_lon180(a):
    a = np.asarray(a, dtype=float)
    return ((a + 180.0) % 360.0) - 180.0

def main():
    url, source_name = find_latest_file()
    print("ISEE source:", url)
    raw = http_bytes(url)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(raw); tf.flush()
        ds = xr.open_dataset(tf.name, decode_times=True)

        latn = coord_name(ds, "lat")
        lonn = coord_name(ds, "lon")
        timen = coord_name(ds, "time")
        if not latn or not lonn:
            raise RuntimeError(f"lat/lon coordinate not detected. vars={list(ds.variables)}")

        latdim = coord_dim_name(ds, latn, "lat")
        londim = coord_dim_name(ds, lonn, "lon")
        timedim = coord_dim_name(ds, timen, "time") if timen else None

        tecn = tec_var_name(ds, latn, lonn, latdim, londim)
        print("Detected:", {
            "lat_var": latn, "lat_dim": latdim,
            "lon_var": lonn, "lon_dim": londim,
            "time_var": timen, "time_dim": timedim,
            "tec": tecn, "tec_dims": tuple(ds[tecn].dims),
        })

        lat = np.asarray(ds[latn].values, dtype=float).squeeze()
        lon0 = np.asarray(ds[lonn].values, dtype=float).squeeze()
        lon = to_lon180(lon0)
        if lat.ndim != 1 or lon.ndim != 1:
            raise RuntimeError("This first implementation expects 1-D lat/lon coordinates")

        ilat = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
        ilon = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
        if not len(ilat) or not len(ilon):
            raise RuntimeError("Japan crop has no grid cells")

        da = ds[tecn]
        # Arrange dimensions to [time?, lat, lon].
        # ISEE AGRID2 can expose coordinate vars named lat/lon while the actual
        # TEC dimensions are latitude/longitude.
        lat_dim = latdim if latdim in da.dims else (latn if latn in da.dims else None)
        lon_dim = londim if londim in da.dims else (lonn if lonn in da.dims else None)

        if lat_dim is None or lon_dim is None:
            # Last-resort semantic match against the TEC variable dimensions.
            for d in da.dims:
                dl = str(d).lower()
                if lat_dim is None and ("lat" in dl):
                    lat_dim = d
                if lon_dim is None and ("lon" in dl):
                    lon_dim = d

        if lat_dim is None or lon_dim is None:
            raise RuntimeError(
                f"TEC variable dims do not contain recognizable lat/lon dimensions: "
                f"tec={tecn}, dims={da.dims}, lat_var={latn}, lon_var={lonn}"
            )

        time_dim = timedim if timedim in da.dims else (timen if timen in da.dims else None)
        if time_dim is None:
            for d in da.dims:
                if "time" in str(d).lower():
                    time_dim = d
                    break

        extra_dims = [d for d in da.dims if d not in {lat_dim, lon_dim, time_dim}]
        for d in extra_dims:
            da = da.isel({d:0})

        if time_dim:
            da = da.transpose(time_dim, lat_dim, lon_dim)
            # Use the coordinate attached to the actual time dimension where possible.
            if timen and timen in ds.variables:
                times = np.asarray(ds[timen].values)
            elif time_dim in ds.variables:
                times = np.asarray(ds[time_dim].values)
            else:
                times = np.arange(da.sizes[time_dim])
        else:
            da = da.transpose(lat_dim, lon_dim).expand_dims({"_time":[0]})
            time_dim = "_time"
            # filename YYYYMMDDHH
            m = re.search(r"(\d{10})_atec", source_name)
            base_time = datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc) if m else datetime.now(timezone.utc)
            times = np.array([np.datetime64(base_time.replace(tzinfo=None))])

        arr = np.asarray(da.values, dtype=float)
        arr = arr[:, ilat, :][:, :, ilon]
        latc = lat[ilat]
        lonc = lon[ilon]

        # Ensure ascending axes.
        if len(latc) > 1 and latc[0] > latc[-1]:
            latc = latc[::-1]; arr = arr[:, ::-1, :]
        order_lon = np.argsort(lonc)
        lonc = lonc[order_lon]; arr = arr[:, :, order_lon]

        # Decode times and sample nearest record every 5 min.
        py_times = []
        for x in times:
            try:
                ns = np.datetime64(x, "ns").astype("int64")
                py_times.append(datetime.fromtimestamp(ns / 1e9, tz=timezone.utc))
            except Exception:
                py_times.append(datetime.now(timezone.utc))

        if len(py_times) > 1:
            chosen = []
            last = None
            for i,t in enumerate(py_times):
                if last is None or (t-last).total_seconds() >= STEP_MIN*60 - 1:
                    chosen.append(i); last=t
        else:
            chosen = [0]

        frames = []
        for i in chosen:
            grid = arr[i]
            grid = np.where(np.isfinite(grid), grid, np.nan)
            # Common fill/sentinel protection.
            grid[(grid < -1) | (grid > 500)] = np.nan
            fname = f"{py_times[i].strftime('%Y%m%dT%H%M%SZ')}.json"
            payload = {
                "version":"swifttec-isee-japan-v1",
                "time_utc":py_times[i].isoformat().replace("+00:00","Z"),
                "source_file":source_name,
                "source_url":url,
                "lat_arr":[round(float(x),5) for x in latc],
                "lon_arr":[round(float(x),5) for x in lonc],
                "n_lat":len(latc),
                "n_lon":len(lonc),
                "grid":[[None if not np.isfinite(v) else round(float(v),3) for v in row] for row in grid],
            }
            (OUT/fname).write_text(json.dumps(payload, ensure_ascii=False, separators=(",",":")), encoding="utf-8")
            frames.append({"time_utc":payload["time_utc"],"file":fname,"source_file":source_name})

        # Keep prior index entries but retain max 7 days.
        old = []
        ip = OUT/"index.json"
        if ip.exists():
            try: old = json.loads(ip.read_text(encoding="utf-8")).get("frames",[])
            except Exception: old=[]
        by_time = {str(x.get("time_utc")):x for x in old if x.get("time_utc")}
        for x in frames: by_time[x["time_utc"]] = x
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        merged = []
        for k in sorted(by_time):
            try:
                t = datetime.fromisoformat(k.replace("Z","+00:00"))
                if t >= cutoff: merged.append(by_time[k])
            except Exception: pass

        index = {
            "version":"swifttec-isee-japan-index-v1",
            "updated_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "region":{"lat_min":LAT_MIN,"lat_max":LAT_MAX,"lon_min":LON_MIN,"lon_max":LON_MAX},
            "step_minutes":STEP_MIN,
            "source":"ISEE AGRID2 NetCDF",
            "frames":merged,
        }
        ip.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(frames)} new Japan frames; index={len(merged)}")

if __name__ == "__main__":
    main()
