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
- gnss_fetch_status.json

The web UI uses TLE for SGP4 propagation and the GPS almanac health map
to mark GPS PRNs active/inactive.

TLE source:
- GPS      : CelesTrak GROUP=gps-ops
- Galileo  : CelesTrak GROUP=galileo
- GLONASS  : CelesTrak GROUP=glo-ops
- BeiDou   : CelesTrak GROUP=beidou
- QZSS     : CelesTrak NAME=QZS
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


# ============================================================
# Output directories
# ============================================================

OUT = Path("docs/data/gnss")
OUT.mkdir(parents=True, exist_ok=True)

VENDOR = Path("docs/vendor")
VENDOR.mkdir(parents=True, exist_ok=True)


# ============================================================
# satellite.js mirrors
# ============================================================

SATELLITE_JS_URLS = [
    "https://unpkg.com/satellite.js/dist/satellite.min.js",
    "https://cdn.jsdelivr.net/npm/satellite.js/dist/satellite.min.js",
    "https://unpkg.com/satellite.js@5.0.0/dist/satellite.min.js",
    "https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/dist/satellite.min.js",
]


# ============================================================
# GNSS TLE sources
#
# NOTE:
# CelesTrak does not use GROUP=qzss here.
# QZSS is retrieved with NAME=QZS.
# ============================================================

TLE_SOURCES = {
    "gps_latest.tle": (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?GROUP=gps-ops&FORMAT=tle"
    ),

    "galileo_latest.tle": (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?GROUP=galileo&FORMAT=tle"
    ),

    "glonass_latest.tle": (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?GROUP=glo-ops&FORMAT=tle"
    ),

    "beidou_latest.tle": (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?GROUP=beidou&FORMAT=tle"
    ),

    # QZSS:
    # OLD:
    # gp.php?GROUP=qzss&FORMAT=tle
    #
    # NEW:
    # gp.php?NAME=QZS&FORMAT=tle
    "qzss_latest.tle": (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?NAME=QZS&FORMAT=tle"
    ),
}


# ============================================================
# GPS Yuma Almanac sources
# ============================================================

YUMA_URLS = [
    # CelesTrak "Latest Yuma Almanac"
    "https://celestrak.org/GPS/almanac/Yuma/almanac.yuma.txt",

    # Older aliases kept as fallback
    "https://celestrak.org/GPS/almanac/Yuma/current.al3",
    "https://celestrak.org/GPS/almanac/Yuma/current.txt",
    "https://celestrak.org/GPS/almanac/Yuma/current.alm",
]


# ============================================================
# Utility
# ============================================================

def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def fetch_text(
    url: str,
    timeout: int = 25,
) -> str:
    """Download text from URL."""

    req = Request(
        url,
        headers={
            "User-Agent": "SWIFT-TEC-GNSS-daily-fetch/1.1",
            "Accept": "text/plain,*/*",
        },
    )

    with urlopen(
        req,
        timeout=timeout,
    ) as res:

        return res.read().decode(
            "utf-8",
            errors="replace",
        )


# ============================================================
# TLE validation
# ============================================================

def count_tle_pairs(text: str) -> int:
    """Count valid Line 1 / Line 2 TLE pairs."""

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    count = 0

    for i in range(len(lines) - 1):
        if (
            lines[i].startswith("1 ")
            and lines[i + 1].startswith("2 ")
        ):
            count += 1

    return count


def looks_like_tle(text: str) -> bool:
    """Check whether downloaded text looks like valid TLE data."""

    stripped = text.strip()

    if not stripped:
        return False

    # CelesTrak no-data response
    if "No GP data found" in text:
        return False

    # Prevent saving HTML error pages as .tle
    lower = stripped.lower()

    if (
        lower.startswith("<!doctype html")
        or lower.startswith("<html")
    ):
        return False

    return count_tle_pairs(text) > 0


# ============================================================
# GPS Yuma Health parser
# ============================================================

def parse_yuma_health(
    text: str,
) -> dict[str, int]:
    """Parse Yuma Health fields robustly.

    CelesTrak's latest Yuma text may be returned with or
    without blank lines between PRN blocks.

    Therefore this parser does not rely only on paragraph
    splitting.
    """

    health: dict[str, int] = {}

    # --------------------------------------------------------
    # Primary parser
    #
    # Match:
    #
    # ID: 1
    # ...
    # Health: 0
    #
    # or
    #
    # PRN: 1
    # ...
    # Health: 0
    # --------------------------------------------------------

    for m in re.finditer(
        r"(?:ID|PRN)\s*:\s*(\d+)"
        r"[\s\S]{0,350}?"
        r"Health\s*:\s*([0-9]+)",
        text,
        flags=re.I,
    ):

        prn = f"{int(m.group(1)):02d}"

        health[prn] = int(
            m.group(2)
        )

    if health:
        return health

    # --------------------------------------------------------
    # Fallback
    #
    # Handles heavily reflowed text.
    # Pair ID/PRN with next Health token.
    # --------------------------------------------------------

    tokens = list(
        re.finditer(
            r"(?:ID|PRN|Health)"
            r"\s*:\s*([0-9]+)",
            text,
            flags=re.I,
        )
    )

    last_prn: str | None = None

    for m in tokens:

        key = (
            m.group(0)
            .split(":", 1)[0]
            .strip()
            .lower()
        )

        val = int(
            m.group(1)
        )

        if key in (
            "id",
            "prn",
        ):
            last_prn = f"{val:02d}"

        elif (
            key == "health"
            and last_prn
        ):
            health[last_prn] = val
            last_prn = None

    return health


# ============================================================
# Main
# ============================================================

def main() -> int:

    status = {
        "updated_utc": now_iso(),
        "tle": {},
        "almanac": {},
    }

    # ========================================================
    # 1. GNSS TLE
    # ========================================================

    for fname, url in TLE_SOURCES.items():

        try:

            print(
                f"Fetching {fname}: {url}"
            )

            text = fetch_text(
                url
            )

            if not looks_like_tle(
                text
            ):
                raise RuntimeError(
                    "downloaded text does not look like TLE"
                )

            tle_count = count_tle_pairs(
                text
            )

            # Write only after validation succeeds
            (
                OUT / fname
            ).write_text(
                text.strip() + "\n",
                encoding="utf-8",
            )

            status["tle"][fname] = {
                "ok": True,
                "url": url,
                "bytes": len(
                    text.encode("utf-8")
                ),
                "count": tle_count,
            }

            print(
                f"OK {fname}: "
                f"{tle_count} TLE pairs, "
                f"{len(text)} chars"
            )

        except Exception as e:

            status["tle"][fname] = {
                "ok": False,
                "url": url,
                "error": str(e),
            }

            print(
                f"NG {fname}: {e}"
            )

        # Be polite to CelesTrak
        time.sleep(0.8)

    # ========================================================
    # 2. GPS Yuma Almanac
    # ========================================================

    yuma_text: str | None = None
    yuma_url: str | None = None

    for url in YUMA_URLS:

        try:

            print(
                f"Fetching GPS Yuma Almanac: {url}"
            )

            candidate = fetch_text(
                url
            )

            if "Health" not in candidate:
                raise RuntimeError(
                    "no Health fields"
                )

            parsed_health = parse_yuma_health(
                candidate
            )

            if not parsed_health:
                raise RuntimeError(
                    "Health fields found but no PRN health values parsed"
                )

            yuma_text = candidate
            yuma_url = url

            break

        except Exception as e:

            print(
                f"Yuma try failed {url}: {e}"
            )

            time.sleep(0.8)

    # ========================================================
    # 3. Save Yuma + health JSON
    # ========================================================

    if (
        yuma_text
        and yuma_url
    ):

        (
            OUT / "gps_yuma_current.alm"
        ).write_text(
            yuma_text,
            encoding="utf-8",
        )

        h = parse_yuma_health(
            yuma_text
        )

        doc = {
            "updated_utc": now_iso(),
            "source_url": yuma_url,
            "system": "GPS",
            "health_meaning": (
                "0 is normally healthy/usable; "
                "non-zero is treated as inactive by the UI."
            ),
            "health_by_prn": h,
            "count": len(h),
        }

        (
            OUT / "gps_almanac_health.json"
        ).write_text(
            json.dumps(
                doc,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        status["almanac"] = {
            "ok": True,
            "url": yuma_url,
            "health_count": len(h),
        }

        print(
            "OK gps_yuma_current.alm "
            f"health_count={len(h)}"
        )

    else:

        status["almanac"] = {
            "ok": False,
            "error": "No Yuma almanac fetched",
        }

    # ========================================================
    # 4. satellite.js
    # ========================================================

    satellite_js_ok = False

    for url in SATELLITE_JS_URLS:

        try:

            print(
                f"Fetching satellite.js: {url}"
            )

            lib = fetch_text(
                url
            )

            if (
                "twoline2satrec" not in lib
                or "propagate" not in lib
            ):
                raise RuntimeError(
                    "downloaded file does not look like satellite.js"
                )

            (
                VENDOR / "satellite.min.js"
            ).write_text(
                lib,
                encoding="utf-8",
            )

            status["satellite_js"] = {
                "ok": True,
                "url": url,
                "bytes": len(
                    lib.encode("utf-8")
                ),
            }

            satellite_js_ok = True

            print(
                f"OK satellite.min.js from {url}"
            )

            break

        except Exception as e:

            status["satellite_js"] = {
                "ok": False,
                "url": url,
                "error": str(e),
            }

            print(
                f"satellite.js try failed {url}: {e}"
            )

            time.sleep(0.8)

    if not satellite_js_ok:

        print(
            "WARNING: all satellite.js download sources failed"
        )

    # ========================================================
    # 5. Save fetch status
    # ========================================================

    (
        OUT / "gnss_fetch_status.json"
    ).write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Summary
    # ========================================================

    tle_ok = sum(
        1
        for item in status["tle"].values()
        if item.get("ok")
    )

    print(
        f"GNSS TLE fetch complete: "
        f"{tle_ok}/{len(TLE_SOURCES)} OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
