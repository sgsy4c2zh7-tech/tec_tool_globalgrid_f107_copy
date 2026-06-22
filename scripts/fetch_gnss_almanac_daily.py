#!/usr/bin/env python3
"""Daily GNSS TLE + GPS Almanac Health fetcher.

Outputs under docs/data/gnss:
- gps_latest.tle
- galileo_latest.tle
- glonass_latest.tle
- beidou_latest.tle
- qzss_latest.tle
- gps_yuma_current.alm
- gps_almanac_health.json

The web UI uses TLE for SGP4 propagation and the GPS almanac health map
to mark GPS PRNs active/inactive.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path("docs/data/gnss")
OUT.mkdir(parents=True, exist_ok=True)
VENDOR = Path("docs/vendor")
VENDOR.mkdir(parents=True, exist_ok=True)

SATELLITE_JS_URLS = [
    "https://unpkg.com/satellite.js/dist/satellite.min.js",
    "https://cdn.jsdelivr.net/npm/satellite.js/dist/satellite.min.js",
    "https://unpkg.com/satellite.js@5.0.0/dist/satellite.min.js",
    "https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/dist/satellite.min.js",
]

TLE_SOURCES = {
    "gps_latest.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
    "galileo_latest.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
    "glonass_latest.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle",
    "beidou_latest.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle",
    "qzss_latest.tle": "https://celestrak.org/NORAD/elements/gp.php?GROUP=qzss&FORMAT=tle",
}

YUMA_URLS = [
    # CelesTrak "Latest Yuma Almanac" link.
    "https://celestrak.org/GPS/almanac/Yuma/almanac.yuma.txt",
    # Older aliases kept as fallback.
    "https://celestrak.org/GPS/almanac/Yuma/current.al3",
    "https://celestrak.org/GPS/almanac/Yuma/current.txt",
    "https://celestrak.org/GPS/almanac/Yuma/current.alm",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_text(url: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-GNSS-daily-fetch/1.0"})
    with urlopen(req, timeout=timeout) as res:
        return res.read().decode("utf-8", errors="replace")


def looks_like_tle(text: str) -> bool:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return any(x.startswith("1 ") for x in lines) and any(x.startswith("2 ") for x in lines)


def parse_yuma_health(text: str) -> dict[str, int]:
    """Parse Yuma Health fields robustly.

    CelesTrak's latest Yuma text may be returned with or without blank lines
    between PRN blocks, so do not rely on paragraph splitting only.
    """
    health: dict[str, int] = {}

    # Most robust: pair each ID with the following Health before the next block.
    for m in re.finditer(r"(?:ID|PRN)\s*:\s*(\d+)[\s\S]{0,350}?Health\s*:\s*([0-9]+)", text, flags=re.I):
        prn = f"{int(m.group(1)):02d}"
        health[prn] = int(m.group(2))

    if health:
        return health

    # Fallback for heavily reflowed text: scan tokens and pair ID -> next Health.
    tokens = list(re.finditer(r"(?:ID|PRN|Health)\s*:\s*([0-9]+)", text, flags=re.I))
    last_prn: str | None = None
    for m in tokens:
        key = m.group(0).split(":", 1)[0].strip().lower()
        val = int(m.group(1))
        if key in ("id", "prn"):
            last_prn = f"{val:02d}"
        elif key == "health" and last_prn:
            health[last_prn] = val
            last_prn = None
    return health

def main() -> int:
    status = {"updated_utc": now_iso(), "tle": {}, "almanac": {}}

    for fname, url in TLE_SOURCES.items():
        try:
            text = fetch_text(url)
            if not looks_like_tle(text):
                raise RuntimeError("downloaded text does not look like TLE")
            (OUT / fname).write_text(text.strip() + "\n", encoding="utf-8")
            status["tle"][fname] = {"ok": True, "url": url, "bytes": len(text.encode("utf-8"))}
            print(f"OK {fname}: {len(text)} chars")
        except Exception as e:
            status["tle"][fname] = {"ok": False, "url": url, "error": str(e)}
            print(f"NG {fname}: {e}")
        time.sleep(0.8)

    yuma_text = None
    yuma_url = None
    for url in YUMA_URLS:
        try:
            yuma_text = fetch_text(url)
            if "Health" not in yuma_text:
                raise RuntimeError("no Health fields")
            yuma_url = url
            break
        except Exception as e:
            print(f"Yuma try failed {url}: {e}")
            time.sleep(0.8)

    if yuma_text and yuma_url:
        (OUT / "gps_yuma_current.alm").write_text(yuma_text, encoding="utf-8")
        h = parse_yuma_health(yuma_text)
        doc = {
            "updated_utc": now_iso(),
            "source_url": yuma_url,
            "system": "GPS",
            "health_meaning": "0 is normally healthy/usable; non-zero is treated as inactive by the UI.",
            "health_by_prn": h,
            "count": len(h),
        }
        (OUT / "gps_almanac_health.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        status["almanac"] = {"ok": True, "url": yuma_url, "health_count": len(h)}
        print(f"OK gps_yuma_current.alm health_count={len(h)}")
    else:
        status["almanac"] = {"ok": False, "error": "No Yuma almanac fetched"}

    for url in SATELLITE_JS_URLS:
        try:
            lib = fetch_text(url)
            if "twoline2satrec" not in lib or "propagate" not in lib:
                raise RuntimeError("downloaded file does not look like satellite.js")
            (VENDOR / "satellite.min.js").write_text(lib, encoding="utf-8")
            status["satellite_js"] = {"ok": True, "url": url, "bytes": len(lib.encode("utf-8"))}
            print(f"OK satellite.min.js from {url}")
            break
        except Exception as e:
            status["satellite_js"] = {"ok": False, "url": url, "error": str(e)}
            print(f"satellite.js try failed {url}: {e}")
            time.sleep(0.8)


    (OUT / "gnss_fetch_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
