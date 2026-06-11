#!/usr/bin/env python3
"""Fetch GNSS TLE/GP data from CelesTrak for SWIFT-TEC DOP calculations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

OUT_ROOT = Path("docs/data/gnss")
BASE = "https://celestrak.org/NORAD/elements/gp.php"
GROUPS = {
    "gps": "gps-ops",
    "galileo": "galileo",
    "glonass": "glo-ops",
    "beidou": "beidou",
    "qzss": "qzss",
}


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "SWIFT-TEC-v4-gnss/1.0"})
    with urlopen(req, timeout=90) as res:
        return res.read()


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "swifttec-gnss-gp-v1",
        "updated_utc": datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "groups": [],
    }

    for name, group in GROUPS.items():
        tle_url = f"{BASE}?GROUP={group}&FORMAT=tle"
        json_url = f"{BASE}?GROUP={group}&FORMAT=json"
        print(f"Fetching {name}: {tle_url}")
        try:
            tle = fetch(tle_url)
            (OUT_ROOT / f"{name}_latest.tle").write_bytes(tle)
            gp_json = fetch(json_url)
            (OUT_ROOT / f"{name}_latest.json").write_bytes(gp_json)
            manifest["groups"].append({
                "name": name,
                "celestrak_group": group,
                "tle_file": f"{name}_latest.tle",
                "json_file": f"{name}_latest.json",
            })
        except Exception as exc:
            print(f"WARN: {name} failed: {exc}")

    (OUT_ROOT / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
