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

SOURCES = {
    "gps": {
        "label": "GPS",
        "filename": "gps_latest.tle",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
    },
    "galileo": {
        "label": "Galileo",
        "filename": "galileo_latest.tle",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
    },
    "glonass": {
        "label": "GLONASS",
        "filename": "glonass_latest.tle",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle",
    },
    "beidou": {
        "label": "BeiDou",
        "filename": "beidou_latest.tle",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle",
    },
    "qzss": {
        "label": "QZSS",
        "filename": "qzss_latest.tle",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=qzss&FORMAT=tle",
    },
}


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-v4-gnss/1.0"})
    with urlopen(req, timeout=60) as res:
        return res.read().decode("utf-8", errors="replace")


def count_tle(text: str) -> int:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = 0
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            n += 1
    return n


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    index = {
        "version": "swifttec-gnss-gp-v1",
        "updated_utc": now,
        "sources": {},
    }
    failed: list[str] = []

    for key, src in SOURCES.items():
        try:
            print(f"Fetching {src['label']}: {src['url']}")
            text = fetch_text(src["url"])
            if "No GP data found" in text or len(text.strip()) < 20:
                raise RuntimeError("empty/no GP data")
            n = count_tle(text)
            if n <= 0:
                raise RuntimeError("no TLE pairs parsed")

            out = OUT_DIR / src["filename"]
            out.write_text(text, encoding="utf-8")
            index["sources"][key] = {
                "label": src["label"],
                "file": src["filename"],
                "url": src["url"],
                "count": n,
                "ok": True,
            }
            print(f"  OK: {n} TLE pairs -> {out}")
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            failed.append(f"{key}: {exc}")
            index["sources"][key] = {
                "label": src["label"],
                "file": src["filename"],
                "url": src["url"],
                "count": 0,
                "ok": False,
                "error": str(exc),
            }
            print(f"  FAIL: {key}: {exc}", file=sys.stderr)

    (OUT_DIR / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_count = sum(1 for v in index["sources"].values() if v.get("ok"))
    if ok_count <= 0:
        print("No GNSS TLE source succeeded.", file=sys.stderr)
        return 1

    if failed:
        print("Partial failures: " + "; ".join(failed), file=sys.stderr)
    print(f"GNSS fetch complete: {ok_count}/{len(SOURCES)} sources OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
