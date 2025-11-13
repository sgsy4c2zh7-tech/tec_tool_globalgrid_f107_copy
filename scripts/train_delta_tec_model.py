# scripts/update_and_train.py
#
# 役割：
#  1. BoM の World0000_tec.txt を取得 → data/bom_tec/ に保存
#  2. NOAA の 27-day outlook から F10.7 / Ap を取得
#  3. （Dst / X-ray は TODO：ひとまずダミー）
#  4. data/space_weather.csv に時刻＋宇宙天気を追記
#  5. BoMグリッド & 宇宙天気ログから ΔTEC モデルを学習
#  6. swift_tec_coeffs.js を更新

from pathlib import Path
from datetime import datetime, timezone
import re
import math

import numpy as np
import pandas as pd
import requests


# ========== 設定値 ==========

REPO_ROOT = Path(__file__).resolve().parents[1]
BOM_DIR = REPO_ROOT / "data" / "bom_tec"
SPACE_WEATHER_CSV = REPO_ROOT / "data" / "space_weather.csv"
OUT_JS = REPO_ROOT / "swift_tec_coeffs.js"

BOM_TEC_URL = "https://downloads.sws.bom.gov.au/data/Satellite/World0000_tec.txt"

# NOAA 27-day outlook テキスト（F10.7 & Ap）
NOAA_27DAY_URL = "https://services.swpc.noaa.gov/text/27-day-outlook.txt"

# 平常値（偏差の基準）
F0 = 120.0
AP0 = 5.0
DST0 = 0.0

MAX_DELTA_HOURS = 3.1

# BoMファイル名パターン
BOM_FILENAME_RE = re.compile(r"World0000_tec_(\d{8})_(\d{4})\.txt")


# ========== 1. BoM TEC を取得して保存 ==========

def fetch_bom_tec_and_save():
    """
    BoM の World0000_tec.txt を取得し、ヘッダの VALID DATE/TIME から
    有効時刻を読み取ってファイル名にして保存する。
    戻り値: (dt_valid, saved_path)
    """
    BOM_DIR.mkdir(parents=True, exist_ok=True)

    resp = requests.get(BOM_TEC_URL, timeout=30)
    resp.raise_for_status()
    txt = resp.text

    # VALID DATE/TIME 行をパース
    # 例: "# VALID DATE/TIME:                 11/11/2025 00:00:00UTC"
    m = re.search(r"VALID DATE/TIME:\s*(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})UTC", txt)
    if m:
        date_str, time_str = m.groups()
        dt_valid = datetime.strptime(date_str + " " + time_str, "%d/%m/%Y %H:%M:%S")
        dt_valid = dt_valid.replace(tzinfo=timezone.utc)
    else:
        # ダメなら「今のUTC」を時刻として扱う
        dt_valid = datetime.now(timezone.utc)

    fname = f"World0000_tec_{dt_valid:%Y%m%d_%H%M}.txt"
    out_path = BOM_DIR / fname
    out_path.write_text(txt, encoding="utf-8")
    print(f"[BoM] Saved {out_path.name}")

    return dt_valid, out_path


# ========== 2. NOAA から F10.7 / Ap を取得 ==========

def fetch_noaa_f107_ap_for_date(target_date_utc: datetime):
    """
    NOAA 27-day outlook テキストから、target_date_utc に最も近い日付の
    F10.7 / Ap を取得する（1日単位）。
    戻り値: (F107, Ap)
    """
    resp = requests.get(NOAA_27DAY_URL, timeout=30)
    resp.raise_for_status()
    txt = resp.text

    # テーブル部の行をパース
    # 例の行フォーマット:
    # 2025 Nov 10     175          18          5
    pattern = re.compile(
        r"^(20\d{2}\s+\w{3}\s+\d{2})\s+(\d+)\s+(\d+)\s+(\d+)",
        re.MULTILINE
    )
    rows = pattern.findall(txt)
    if not rows:
        raise RuntimeError("NOAA 27-day outlook parse failed")

    # 日付差が最小の行を選ぶ
    target_date = target_date_utc.date()
    best = None
    best_diff = None
    for date_str, f107_str, ap_str, kp_str in rows:
        dt = datetime.strptime(date_str, "%Y %b %d").date()
        diff = abs((dt - target_date).days)
        if best is None or diff < best_diff:
            best_diff = diff
            best = (dt, float(f107_str), float(ap_str))

    if best is None:
        raise RuntimeError("No matching date in 27-day outlook")

    _, F107, Ap = best
    print(f"[NOAA] Using F10.7={F107}, Ap={Ap} for date {target_date}")
    return F107, Ap


# ========== 3. Dst / X-ray はとりあえずダミー（TODO） ==========

def fetch_dst_xray_for_time(dt_utc: datetime):
    """
    TODO: Kyoto WDC / NOAA GOES のAPIから本物の値を取る。
    ひとまずダミーとして Dst=0, Xray=0 を返す。
    """
    Dst = 0.0
    Xray = 0.0
    return Dst, Xray


# ========== 4. space_weather.csv の更新 ==========

def append_space_weather_row(dt_utc: datetime, F107: float, Ap: float, Dst: float, Xray: float):
    """
    data/space_weather.csv に1行追加する。
    既に同じ datetime_utc の行がある場合は上書きしてもよいが、
    ここでは簡単に「追記のみ」としておく。
    """
    SPACE_WEATHER_CSV.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "datetime_utc": dt_utc.isoformat().replace("+00:00", "Z"),
        "F107": F107,
        "Ap": Ap,
        "Dst": Dst,
        "Xray": Xray,
    }

    if SPACE_WEATHER_CSV.exists():
        df = pd.read_csv(SPACE_WEATHER_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    # 同じ時刻重複があれば最後のものだけ残す
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc")
    df = df.drop_duplicates(subset=["datetime_utc"], keep="last")
    df.to_csv(SPACE_WEATHER_CSV, index=False)
    print(f"[SpaceWeather] Appended {dt_utc.isoformat()} to {SPACE_WEATHER_CSV}")


def load_space_weather():
    if not SPACE_WEATHER_CSV.exists():
        raise FileNotFoundError(f"{SPACE_WEATHER_CSV} not found")
    df = pd.read_csv(SPACE_WEATHER_CSV)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


# ========== 5. BoMグリッド読み込み＆学習データ構築 ==========

def read_bom_grid(path: Path) -> np.ndarray:
    """
    BoM World0000_tec.txt を 2次元 numpy array (lat x lon) で返す。
    """
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines()

    data_lines = []
    seen_blank = False
    for line in lines:
        if not line.strip():
            seen_blank = True
            continue
        if seen_blank:
            data_lines.append(line)

    rows = []
    for line in data_lines:
        row_vals = []
        for v in line.split(","):
            v = v.strip()
            if not v:
                continue
            try:
                row_vals.append(float(v))
            except ValueError:
                pass
        if row_vals:
            rows.append(row_vals)

    if not rows:
        raise RuntimeError(f"No TEC data parsed in {path}")

    arr = np.array(rows, dtype=float)
    return arr


def parse_bom_timestamp(path: Path) -> datetime:
    m = BOM_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected BoM filename: {path.name}")
    date_str, time_str = m.groups()  # "20251113", "0030"
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return dt.replace(tzinfo=timezone.utc)


def local_solar_time_hours(ut_dt: datetime, lon_deg: float) -> float:
    ut_hours = ut_dt.hour + ut_dt.minute / 60.0 + ut_dt.second / 3600.0
    lst = (ut_hours + lon_deg / 15.0) % 24.0
    return lst


def day_weight(lst_hours: float) -> float:
    x = math.pi * (lst_hours - 12.0) / 12.0
    return max(0.0, math.cos(x))


def build_training_set_grid():
    space_df = load_space_weather()

    bom_files = sorted(BOM_DIR.glob("World0000_tec_*.txt"))
    if len(bom_files) < 2:
        raise RuntimeError(f"BoM TEC logs too few in {BOM_DIR}")

    bom_maps = []
    for f in bom_files:
        dt = parse_bom_timestamp(f)
        grid = read_bom_grid(f)
        bom_maps.append((dt, grid))

    sample_grid = bom_maps[0][1]
    n_lat, n_lon = sample_grid.shape
    lons = np.linspace(-180.0, 180.0, n_lon)

    X_rows = []
    y_rows = []

    for (dt_now, grid_now), (dt_future, grid_future) in zip(bom_maps[:-1], bom_maps[1:]):
        delta_t_hours = (dt_future - dt_now).total_seconds() / 3600.0
        if delta_t_hours <= 0 or delta_t_hours > MAX_DELTA_HOURS:
            continue

        sw_now = get_space_weather_for_time(space_df, dt_now)
        F107 = sw_now["F107"]
        Ap   = sw_now["Ap"]
        Dst  = sw_now["Dst"]
        Xray = sw_now["Xray"]

        Fdev   = F107 - F0
        Apdev  = Ap   - AP0
        Dstdev = Dst  - DST0

        n_lat, n_lon = grid_now.shape
        for j in range(n_lon):
            lon = lons[j]
            lst_now    = local_solar_time_hours(dt_now, lon)
            lst_future = local_solar_time_hours(dt_future, lon)
            D_now      = day_weight(lst_now)
            D_future   = day_weight(lst_future)

            for i in range(n_lat):
                TEC_now    = grid_now[i, j]
                TEC_future = grid_future[i, j]
                dTEC       = TEC_future - TEC_now

                X_row = [
                    1.0,
                    Fdev,
                    Fdev**2,
                    Apdev,
                    Dstdev,
                    Xray,
                    D_now,
                    D_future,
                ]
                X_rows.append(X_row)
                y_rows.append(dTEC)

    if not X_rows:
        raise RuntimeError("No training samples built")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    return X, y


# ========== 6. 回帰 & JS書き出し ==========

def get_space_weather_for_time(df: pd.DataFrame, dt: datetime) -> dict:
    idx = (df["datetime_utc"] - dt).abs().idxmin()
    row = df.loc[idx]
    return {
        "F107": float(row["F107"]),
        "Ap":   float(row["Ap"]),
        "Dst":  float(row["Dst"]),
        "Xray": float(row["Xray"]),
    }


def fit_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return beta


def write_js(coeffs: np.ndarray):
    a0, aF1, aF2, aAp, aDst, aX, aD1, aD2 = coeffs

    js = f"""// This file is auto-generated by scripts/update_and_train.py
// Do not edit by hand.

const SWIFT_TEC_COEFFS = {{
  a0:  {a0:.6e},
  aF1: {aF1:.6e},
  aF2: {aF2:.6e},
  aAp: {aAp:.6e},
  aDst:{aDst:.6e},
  aX:  {aX:.6e},
  aD1: {aD1:.6e},
  aD2: {aD2:.6e}
}};
"""
    OUT_JS.write_text(js, encoding="utf-8")
    print(f"[Model] Wrote coefficients to {OUT_JS}")


# ========== メインフロー ==========

def main():
    print("=== SWIFT-TEC update_and_train ===")

    # 1. BoM TEC を取得 & 保存
    dt_valid, path_bom = fetch_bom_tec_and_save()

    # 2. NOAA からその日用の F10.7 / Ap
    F107, Ap = fetch_noaa_f107_ap_for_date(dt_valid)

    # 3. Dst / X-ray (TODO: いまはダミー)
    Dst, Xray = fetch_dst_xray_for_time(dt_valid)

    # 4. space_weather.csv に追記
    append_space_weather_row(dt_valid, F107, Ap, Dst, Xray)

    # 5. 学習用データ構築
    space_df = load_space_weather()
    X, y = build_training_set_grid()
    print(f"[Train] Samples: {X.shape[0]}, features per sample: {X.shape[1]}")

    # 6. 回帰
    coeffs = fit_coefficients(X, y)
    print("[Train] Coefficients:", coeffs)

    # 7. JS書き出し
    write_js(coeffs)

    print("=== Done ===")


if __name__ == "__main__":
    main()
