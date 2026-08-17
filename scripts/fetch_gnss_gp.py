#!/usr/bin/env python3
"""Fetch multi-GNSS TLE files for SWIFT-TEC v4.

Outputs:
  docs/data/gnss/gps_latest.tle
  docs/data/gnss/galileo_latest.tle
  docs/data/gnss/glonass_latest.tle
  docs/data/gnss/beidou_latest.tle
  docs/data/gnss/qzss_latest.tle
  docs/data/gnss/index.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OUT_DIR = Path("docs/data/gnss")


# ============================================================
# CelesTrak GNSS TLE sources
#
# GPS / Galileo / GLONASS / BeiDou:
#   CelesTrak GROUP query
#
# QZSS:
#   CelesTrak NAME=QZS query
#   ("GROUP=qzss" is not used)
# ============================================================

SOURCES = {
    "gps": {
        "label": "GPS",
        "filename": "gps_latest.tle",
        "url": (
            "https://celestrak.org/NORAD/elements/"
            "gp.php?GROUP=gps-ops&FORMAT=tle"
        ),
    },

    "galileo": {
        "label": "Galileo",
        "filename": "galileo_latest.tle",
        "url": (
            "https://celestrak.org/NORAD/elements/"
            "gp.php?GROUP=galileo&FORMAT=tle"
        ),
    },

    "glonass": {
        "label": "GLONASS",
        "filename": "glonass_latest.tle",
        "url": (
            "https://celestrak.org/NORAD/elements/"
            "gp.php?GROUP=glo-ops&FORMAT=tle"
        ),
    },

    "beidou": {
        "label": "BeiDou",
        "filename": "beidou_latest.tle",
        "url": (
            "https://celestrak.org/NORAD/elements/"
            "gp.php?GROUP=beidou&FORMAT=tle"
        ),
    },

    # QZSS only:
    #
    # Previous:
    #   GROUP=qzss
    #
    # New:
    #   NAME=QZS
    #
    # This retrieves satellites whose names contain "QZS".
    "qzss": {
        "label": "QZSS",
        "filename": "qzss_latest.tle",
        "url": (
            "https://celestrak.org/NORAD/elements/"
            "gp.php?NAME=QZS&FORMAT=tle"
        ),
    },
}


def fetch_text(url: str) -> str:
    """Fetch TLE text from CelesTrak."""

    req = Request(
        url,
        headers={
            "User-Agent": "SWIFT-TEC-v4-gnss/1.0",
            "Accept": "text/plain,*/*",
        },
    )

    with urlopen(req, timeout=60) as res:
        return res.read().decode(
            "utf-8",
            errors="replace",
        )


def count_tle(text: str) -> int:
    """Count valid TLE Line 1 / Line 2 pairs."""

    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip()
    ]

    n = 0

    for i in range(len(lines) - 1):
        if (
            lines[i].startswith("1 ")
            and lines[i + 1].startswith("2 ")
        ):
            n += 1

    return n


def validate_tle_text(text: str) -> int:
    """Validate downloaded TLE text and return TLE count."""

    stripped = text.strip()

    if not stripped:
        raise RuntimeError("empty response")

    if "No GP data found" in text:
        raise RuntimeError("No GP data found")

    if len(stripped) < 20:
        raise RuntimeError("response too short")

    # Sometimes an HTTP endpoint can return an HTML error page.
    lower = stripped.lower()

    if (
        lower.startswith("<!doctype html")
        or lower.startswith("<html")
    ):
        raise RuntimeError(
            "CelesTrak returned HTML instead of TLE data"
        )

    n = count_tle(text)

    if n <= 0:
        raise RuntimeError(
            "no TLE pairs parsed"
        )

    return n


def main() -> int:

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    now = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # --------------------------------------------------------
    # index.json structure
    # --------------------------------------------------------

    index = {
        "version": "swifttec-gnss-gp-v1",
        "updated_utc": now,
        "sources": {},
    }

    failed: list[str] = []

    # --------------------------------------------------------
    # Fetch each GNSS constellation
    # --------------------------------------------------------

    for key, src in SOURCES.items():

        try:

            print(
                f"Fetching {src['label']}: "
                f"{src['url']}"
            )

            # Download
            text = fetch_text(
                src["url"]
            )

            # Validate
            n = validate_tle_text(
                text
            )

            # Output file
            out = (
                OUT_DIR
                / src["filename"]
            )

            # Write TLE
            out.write_text(
                text,
                encoding="utf-8",
            )

            # Store success info
            index["sources"][key] = {
                "label": src["label"],
                "file": src["filename"],
                "url": src["url"],
                "count": n,
                "ok": True,
            }

            print(
                f"  OK: {n} TLE pairs -> {out}"
            )

        except (
            HTTPError,
            URLError,
            TimeoutError,
            RuntimeError,
            OSError,
        ) as exc:

            failed.append(
                f"{key}: {exc}"
            )

            index["sources"][key] = {
                "label": src["label"],
                "file": src["filename"],
                "url": src["url"],
                "count": 0,
                "ok": False,
                "error": str(exc),
            }

            print(
                f"  FAIL: {key}: {exc}",
                file=sys.stderr,
            )

    # --------------------------------------------------------
    # Write index.json
    # --------------------------------------------------------

    index_path = (
        OUT_DIR
        / "index.json"
    )

    index_path.write_text(
        json.dumps(
            index,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    ok_count = sum(
        1
        for v in index["sources"].values()
        if v.get("ok")
    )

    # All failed
    if ok_count <= 0:

        print(
            "No GNSS TLE source succeeded.",
            file=sys.stderr,
        )

        return 1

    # Some failed
    if failed:

        print(
            "Partial failures: "
            + "; ".join(failed),
            file=sys.stderr,
        )

    # Done
    print(
        f"GNSS fetch complete: "
        f"{ok_count}/{len(SOURCES)} sources OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
