#!/usr/bin/env python3
"""Fetch NOAA planetary Kp for SWIFT-TEC Base Kp.

Why this exists
---------------
The browser-side direct fetch to services.swpc.noaa.gov can fail because of
HTTP/CORS/network policy even though GitHub Actions can fetch NOAA normally.

This script runs in GitHub Actions and stores a same-origin static file:

  docs/data/ai/noaa_base_kp.json

The UI then loads that local file first.

Base-day rule
-------------
SWIFT-TEC NOAA uses the previous UTC day as its base TEC day, so Base Kp is
also selected from the previous UTC day.

Primary source:
  https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json

Fallback source:
  https://services.swpc.noaa.gov/json/planetary_k_index_1m.json

If the primary source does not contain enough rows for the previous UTC day,
the 1-minute product is reduced to 3-hour mean Kp values.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

UTC = timezone.utc

OUT = Path(os.environ.get(
    "SWIFTTEC_BASE_KP_OUT",
    "docs/data/ai/noaa_base_kp.json",
))

PRIMARY_URL = os.environ.get(
    "SWIFTTEC_BASE_KP_PRIMARY_URL",
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
)

FALLBACK_URL = os.environ.get(
    "SWIFTTEC_BASE_KP_FALLBACK_URL",
    "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
)


def parse_time(s):
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(text.replace("Z", ""), fmt).replace(tzinfo=UTC)
        except Exception:
            continue
    return None


def iso(t: datetime) -> str:
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_json(url: str):
    req = Request(url, headers={
        "User-Agent": "SWIFT-TEC-noaa-base-kp/1.0",
        "Accept": "application/json",
    })
    with urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8", "replace"))


def parse_rows(obj):
    rows = []
    if not isinstance(obj, list):
        return rows

    for r in obj:
        if isinstance(r, list) and len(r) >= 2:
            t = parse_time(r[0])
            raw = r[1]
        elif isinstance(r, dict):
            t = parse_time(
                r.get("time_tag")
                or r.get("time")
                or r.get("time_utc")
                or r.get("t")
            )
            raw = r.get("kp_index")
            if raw is None:
                raw = r.get("Kp")
            if raw is None:
                raw = r.get("kp")
            if raw is None:
                raw = r.get("k_index")
            if raw is None:
                raw = r.get("estimated_kp")
        else:
            continue

        try:
            kp = float(raw)
        except Exception:
            continue

        if t and math.isfinite(kp):
            rows.append((t, kp))

    dedup = {}
    for t, kp in rows:
        dedup[iso(t)] = (t, kp)
    return sorted(dedup.values(), key=lambda x: x[0])


def rows_for_day(rows, day):
    return [(t, kp) for t, kp in rows if t.date() == day]


def reduce_1m_to_3h(rows, day):
    """Reduce dense estimated Kp rows to 8 three-hour mean values."""
    buckets = defaultdict(list)
    for t, kp in rows:
        if t.date() != day or not math.isfinite(kp):
            continue
        start_hour = (t.hour // 3) * 3
        buckets[start_hour].append(float(kp))

    out = []
    for hour in range(0, 24, 3):
        vals = buckets.get(hour) or []
        if not vals:
            continue
        mean_kp = sum(vals) / len(vals)
        t = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
        out.append((t, mean_kp))
    return out


def normalize_primary(rows):
    """Keep primary official rows as-is, sorted and rounded."""
    return [(t, float(kp)) for t, kp in rows if math.isfinite(float(kp))]


def main() -> int:
    now = datetime.now(UTC)
    base_day = (now - timedelta(days=1)).date()

    primary_rows = []
    fallback_rows = []
    errors = []

    try:
        primary_rows = parse_rows(fetch_json(PRIMARY_URL))
    except Exception as e:
        errors.append(f"primary: {e}")

    selected = normalize_primary(rows_for_day(primary_rows, base_day))
    source = PRIMARY_URL

    # A normal official 3-hour day should contain around 8 rows.
    # If the previous UTC day is not sufficiently represented, use the dense
    # 1-minute product and reduce it to 3-hour means.
    if len(selected) < 6:
        try:
            fallback_rows = parse_rows(fetch_json(FALLBACK_URL))
        except Exception as e:
            errors.append(f"fallback: {e}")

        reduced = reduce_1m_to_3h(fallback_rows, base_day)
        if len(reduced) >= len(selected):
            selected = reduced
            source = FALLBACK_URL + " (3h mean fallback)"

    # Final emergency fallback: use the most recent 24h ending before today.
    # This should be rare, but keeps the local cache useful if NOAA's day
    # boundary publication is delayed.
    fallback_mode = False
    if not selected:
        merged = sorted(primary_rows + fallback_rows, key=lambda x: x[0])
        cutoff_end = datetime(now.year, now.month, now.day, tzinfo=UTC)
        cutoff_start = cutoff_end - timedelta(hours=24)
        selected = [(t, kp) for t, kp in merged if cutoff_start <= t < cutoff_end]
        fallback_mode = True

    if not selected:
        raise RuntimeError(
            "No Base Kp rows available from NOAA. " + ("; ".join(errors) if errors else "")
        )

    # Compact browser-compatible array. parseNoaaPlanetaryKIndexJson accepts
    # [time, kp] rows directly.
    browser_rows = [[iso(t), round(float(kp), 3)] for t, kp in selected]

    doc = {
        "version": "swifttec-noaa-base-kp-v1",
        "base_day_utc": base_day.isoformat(),
        "source": source,
        "fallback_mode": fallback_mode,
        "row_count": len(browser_rows),
        "first_time_utc": browser_rows[0][0],
        "last_time_utc": browser_rows[-1][0],
        "rows": browser_rows,
        "errors": errors,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Do not rewrite the file merely because the action ran again.
    # This avoids needless 30-minute commits when the previous UTC day has
    # not changed.
    new_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    old_text = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if old_text != new_text:
        OUT.write_text(new_text, encoding="utf-8")
        print(f"Updated {OUT}: day={base_day}, rows={len(browser_rows)}, source={source}")
    else:
        print(f"No Base Kp change: day={base_day}, rows={len(browser_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
