#!/usr/bin/env python3
"""Verify the SWIFT-TEC v8.14+ ISEE no-Base forecast against ISEE observations.

The verifier uses only information that existed before each target time:

  raw forecast:
    weighted mean of prior same-UTC-time ISEE VTEC (up to 10 previous days)

  Kp-corrected forecast:
    raw forecast + F(Kp_actual_at_target)

This mirrors the current ISEE forecast structure:
    ISEE same-time mean + forecast Kp correction

For historical verification, actual Kp is used as a proxy for a perfect Kp
forecast so this file measures the TEC model/Kp-response error, not the NOAA
Kp forecast product's own error.

Outputs:
  docs/data/ai/isee_japan/forecast_verification.json

The JSON includes:
- overall observed-vs-forecast Bias / MAE / RMSE
- hit rate for ±5/10/15/20 TECU
- Kp-bin error metrics
- recent frame-level actual-vs-forecast deviations
"""
from __future__ import annotations

import json, math, os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

UTC = timezone.utc
TEC_ROOT = Path(os.environ.get("SWIFTTEC_ISEE_TEC_ROOT", "docs/data/isee_tec"))
AI_ROOT = Path(os.environ.get("SWIFTTEC_ISEE_AI_ROOT", "docs/data/ai/isee_japan"))
KP_ARCHIVE = Path(os.environ.get(
    "SWIFTTEC_ISEE_KP_ARCHIVE",
    "docs/data/ai/isee_japan/kp_actual_archive.json",
))
OUT = AI_ROOT / "forecast_verification.json"

MEAN_DAYS = max(1, int(os.environ.get("SWIFTTEC_ISEE_VERIFY_MEAN_DAYS", "10")))
MIN_HISTORY_DAYS = max(1, int(os.environ.get("SWIFTTEC_ISEE_VERIFY_MIN_HISTORY_DAYS", "1")))
VERIFY_DAYS = max(1, int(os.environ.get("SWIFTTEC_ISEE_VERIFY_DAYS", "14")))
STEP_MIN = max(5, int(os.environ.get("SWIFTTEC_ISEE_VERIFY_STEP_MIN", "30")))
GRID_STRIDE = max(1, int(os.environ.get("SWIFTTEC_ISEE_VERIFY_GRID_STRIDE", "4")))
CLIP_DEFAULT = float(os.environ.get("SWIFTTEC_ISEE_VERIFY_CLIP_TECU", "20"))
THRESHOLDS = (5.0, 10.0, 15.0, 20.0)
KP_BINS = (
    ("0-2", 0.0, 2.0),
    ("2-3", 2.0, 3.0),
    ("3-4", 3.0, 4.0),
    ("4-5", 4.0, 5.0),
    ("5-6", 5.0, 6.0),
    ("6-7", 6.0, 7.0),
    ("7+", 7.0, 99.0),
)

def iso(t):
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00","Z")

def parse_time(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(UTC)
    except Exception:
        return None

def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def kp_bin_label(kp):
    x=float(kp)
    for label, lo, hi in KP_BINS:
        if lo <= x < hi:
            return label
    return "7+"

def nearest_kp(t, rows):
    if not rows:
        return None
    # 3-hourly Kp: nearest within 2h
    best=None; best_d=float("inf")
    for rt, kp in rows:
        d=abs((rt-t).total_seconds())
        if d < best_d:
            best_d=d; best=kp
    if best is None or best_d > 2*3600:
        return None
    return float(best)

def load_kp_rows():
    if not KP_ARCHIVE.exists():
        return []
    doc=load_json(KP_ARCHIVE)
    out=[]
    for row in doc.get("rows",[]):
        t=parse_time(row.get("time_utc"))
        try: kp=float(row.get("kp"))
        except Exception: continue
        if t and math.isfinite(kp):
            out.append((t,kp))
    out.sort()
    return out

class Agg:
    def __init__(self):
        self.n=0
        self.raw_sum=0.0; self.raw_abs=0.0; self.raw_sq=0.0
        self.corr_sum=0.0; self.corr_abs=0.0; self.corr_sq=0.0
        self.raw_hits={th:0 for th in THRESHOLDS}
        self.corr_hits={th:0 for th in THRESHOLDS}
    def add(self, raw_err, corr_err):
        r=np.asarray(raw_err,dtype=float).ravel()
        c=np.asarray(corr_err,dtype=float).ravel()
        mask=np.isfinite(r)&np.isfinite(c)
        if not np.any(mask): return
        r=r[mask]; c=c[mask]; n=int(r.size)
        self.n += n
        self.raw_sum += float(r.sum()); self.raw_abs += float(np.abs(r).sum()); self.raw_sq += float(np.square(r).sum())
        self.corr_sum += float(c.sum()); self.corr_abs += float(np.abs(c).sum()); self.corr_sq += float(np.square(c).sum())
        for th in THRESHOLDS:
            self.raw_hits[th] += int(np.count_nonzero(np.abs(r) <= th))
            self.corr_hits[th] += int(np.count_nonzero(np.abs(c) <= th))
    def summary(self):
        if self.n <= 0:
            return {
                "sample_count":0,
                "raw_bias":None,"corrected_bias":None,
                "raw_mae":None,"corrected_mae":None,
                "raw_rmse":None,"corrected_rmse":None,
                "raw_hit_rate":None,"corrected_hit_rate":None,
            }
        n=float(self.n)
        return {
            "sample_count":self.n,
            "raw_bias":round(self.raw_sum/n,4),
            "corrected_bias":round(self.corr_sum/n,4),
            "raw_mae":round(self.raw_abs/n,4),
            "corrected_mae":round(self.corr_abs/n,4),
            "raw_rmse":round(math.sqrt(self.raw_sq/n),4),
            "corrected_rmse":round(math.sqrt(self.corr_sq/n),4),
            "raw_hit_rate":round(self.raw_hits[5.0]/n,6),
            "corrected_hit_rate":round(self.corr_hits[5.0]/n,6),
        }
    def thresholds(self):
        s=self.summary()
        out={}
        for th in THRESHOLDS:
            out[str(int(th))]={
                "threshold_tecu":th,
                "sample_count":self.n,
                "raw_hit_rate":None if not self.n else round(self.raw_hits[th]/self.n,6),
                "corrected_hit_rate":None if not self.n else round(self.corr_hits[th]/self.n,6),
                "raw_bias":s["raw_bias"],
                "corrected_bias":s["corrected_bias"],
                "raw_mae":s["raw_mae"],
                "corrected_mae":s["corrected_mae"],
                "raw_rmse":s["raw_rmse"],
                "corrected_rmse":s["corrected_rmse"],
            }
        return out

def weight(rank):
    return max(0.1, 1.0 - 0.1*rank)

def load_frame(meta):
    doc=load_json(meta["path"])
    grid=np.asarray(doc.get("grid") or [],dtype=float)
    return doc, grid

def month_coeff_arrays(grid_doc, month, nlat, nlon):
    root=grid_doc.get("coefficients_grid") or grid_doc.get("grid_coefficients") or {}
    m=root.get(str(month)) or root.get(month) or {}
    def arr(name):
        try:
            a=np.asarray(m.get(name),dtype=float)
            if a.shape != (nlat,nlon):
                return np.zeros((nlat,nlon),dtype=float)
            return np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0)
        except Exception:
            return np.zeros((nlat,nlon),dtype=float)
    return tuple(arr(k) for k in ("k0","k1","k2","k3"))

def main():
    idx_path=TEC_ROOT/"index.json"
    if not idx_path.exists():
        raise RuntimeError("ISEE TEC index missing")
    idx=load_json(idx_path)
    frames=[]
    for f in idx.get("frames",[]):
        t=parse_time(f.get("time_utc"))
        rel=f.get("file")
        if not t or not rel: continue
        if t.minute % STEP_MIN != 0: continue
        p=TEC_ROOT/rel
        if p.exists():
            frames.append({"time":t,"path":p,"file":rel})
    frames.sort(key=lambda x:x["time"])
    if not frames:
        raise RuntimeError("No ISEE frames available for verification")

    latest=frames[-1]["time"]
    verify_cutoff=latest - timedelta(days=VERIFY_DAYS)

    kp_rows=load_kp_rows()
    if not kp_rows:
        raise RuntimeError("Kp actual archive is empty. Run update_isee_kp_actual_archive.py first.")

    grid_coeff_path=AI_ROOT/"kp_grid_coefficients.json"
    grid_doc=load_json(grid_coeff_path) if grid_coeff_path.exists() else {}
    clip=float(grid_doc.get("correction_clip_tecu") or CLIP_DEFAULT)
    clip=max(1.0,min(60.0,clip))
    ai_available=bool(grid_doc)

    by_slot=defaultdict(list)
    for m in frames:
        by_slot[m["time"].strftime("%H%M")].append(m)

    overall=Agg()
    by_kp={label:Agg() for label,_,_ in KP_BINS}
    recent=[]
    frames_scored=0
    frames_skipped_kp=0
    frames_skipped_history=0
    coeff_cache={}

    for hhmm, metas in sorted(by_slot.items()):
        metas=sorted(metas,key=lambda x:x["time"])
        if len(metas) < 2:
            continue

        # Load this time-of-day slot's daily grids once, then release.
        loaded=[]
        lat_arr=lon_arr=None
        for m in metas:
            try:
                doc,g=load_frame(m)
            except Exception:
                continue
            if g.ndim != 2 or g.size == 0:
                continue
            if lat_arr is None:
                lat_arr=doc.get("lat_arr") or []
                lon_arr=doc.get("lon_arr") or []
            loaded.append((m,doc,g))
        if len(loaded)<2:
            continue

        for ti in range(1,len(loaded)):
            meta, doc, actual_full = loaded[ti]
            t=meta["time"]
            if t < verify_cutoff:
                continue

            hist=loaded[max(0,ti-MEAN_DAYS):ti]
            if len(hist) < MIN_HISTORY_DAYS:
                frames_skipped_history += 1
                continue

            kp=nearest_kp(t,kp_rows)
            if kp is None:
                frames_skipped_kp += 1
                continue

            # Weighted mean of prior same-time observations only; no future leakage.
            sw=0.0
            mean_full=np.zeros_like(actual_full,dtype=float)
            valid_w=np.zeros_like(actual_full,dtype=float)
            for rank, (_,_,g) in enumerate(reversed(hist)):
                w=weight(rank)
                mask=np.isfinite(g)
                mean_full[mask] += g[mask]*w
                valid_w[mask] += w
                sw += w
            with np.errstate(invalid="ignore",divide="ignore"):
                raw_full=np.where(valid_w>0,mean_full/valid_w,np.nan)

            nlat,nlon=actual_full.shape
            month=t.month
            key=(month,nlat,nlon)
            if key not in coeff_cache:
                coeff_cache[key]=month_coeff_arrays(grid_doc,month,nlat,nlon)
            k0,k1,k2,k3=coeff_cache[key]
            x=float(kp)-3.0
            corr_term=k0+k1*x+k2*x*x+k3*x*x*x if ai_available else np.zeros_like(raw_full)
            corr_term=np.clip(np.nan_to_num(corr_term,nan=0.0),-clip,clip)
            forecast_full=np.clip(raw_full+corr_term,0.0,300.0)

            # Spatial sampling for speed; same stride for actual and forecast.
            actual=actual_full[::GRID_STRIDE,::GRID_STRIDE]
            raw=raw_full[::GRID_STRIDE,::GRID_STRIDE]
            forecast=forecast_full[::GRID_STRIDE,::GRID_STRIDE]
            mask=np.isfinite(actual)&np.isfinite(raw)&np.isfinite(forecast)
            if not np.any(mask):
                continue
            raw_err=actual[mask]-raw[mask]
            corr_err=actual[mask]-forecast[mask]

            overall.add(raw_err,corr_err)
            label=kp_bin_label(kp)
            by_kp[label].add(raw_err,corr_err)
            frames_scored += 1

            # Frame-level actual-vs-forecast deviation for UI.
            recent.append({
                "time_utc":iso(t),
                "kp":round(float(kp),3),
                "history_days":len(hist),
                "sample_count":int(mask.sum()),
                "observed_mean_tecu":round(float(np.mean(actual[mask])),3),
                "raw_forecast_mean_tecu":round(float(np.mean(raw[mask])),3),
                "forecast_mean_tecu":round(float(np.mean(forecast[mask])),3),
                "raw_bias_tecu":round(float(np.mean(raw_err)),3),
                "bias_tecu":round(float(np.mean(corr_err)),3),
                "raw_mae_tecu":round(float(np.mean(np.abs(raw_err))),3),
                "mae_tecu":round(float(np.mean(np.abs(corr_err))),3),
                "raw_rmse_tecu":round(float(math.sqrt(np.mean(np.square(raw_err)))),3),
                "rmse_tecu":round(float(math.sqrt(np.mean(np.square(corr_err)))),3),
            })

    kp_bins={}
    for label,_,_ in KP_BINS:
        agg=by_kp[label]
        kp_bins[label]={
            "kp_bin":label,
            "thresholds":agg.thresholds(),
            **agg.summary(),
        }

    now=datetime.now(UTC)
    summary=overall.summary()
    summary.update({
        "frames_scored":frames_scored,
        "frames_skipped_no_kp":frames_skipped_kp,
        "frames_skipped_no_history":frames_skipped_history,
        "mean_days":MEAN_DAYS,
        "min_history_days":MIN_HISTORY_DAYS,
        "verify_days":VERIFY_DAYS,
        "step_min":STEP_MIN,
        "grid_stride":GRID_STRIDE,
        "latest_isee_time_utc":iso(latest),
        "ai_coefficients_available":ai_available,
        "correction_clip_tecu":clip,
    })

    doc={
        "version":"swifttec-isee-no-base-forecast-verification-v1",
        "updated_utc":iso(now),
        "data_source":"ISEE Japan High-Res VTEC",
        "forecast_model":"same-UTC-time weighted ISEE mean + F(KpF); no Base/KpB",
        "verification_note":"Historical observed Kp is used as KpF, so this isolates TEC/Kp-response model error rather than NOAA Kp forecast-product error.",
        "error_sign":"error = observed ISEE VTEC - forecast VTEC; positive bias means forecast is too low.",
        "summary":summary,
        "thresholds":overall.thresholds(),
        "kp_bins":kp_bins,
        "recent":recent[-120:],
    }

    AI_ROOT.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf-8")
    print(
        f"ISEE verification complete: frames={frames_scored}, samples={summary.get('sample_count',0)}, "
        f"RMSE={summary.get('corrected_rmse')}, Bias={summary.get('corrected_bias')}"
    )
    return 0

if __name__=="__main__":
    raise SystemExit(main())
