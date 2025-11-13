# scripts/train_delta_tec_model.py
#
# BoM TECグリッド (World0000_tec_*.txt) と宇宙天気ログ (space_weather.csv) から
# ΔTEC モデルの係数を最小二乗で学習し、swift_tec_coeffs.js に出力する完全版。
#
# モデル:
#   ΔTEC = a0
#        + aF1 * (F107 - F0)
#        + aF2 * (F107 - F0)^2
#        + aAp * (Ap   - AP0)
#        + aDst* (Dst  - DST0)
#        + aX  *  Xray
#        + aD1 * D_day(t)
#        + aD2 * D_day(t+Δt)
#
# ここでは BoM グリッド (lat x lon) 全点の ΔTEC をサンプルとして使う。
# 係数は全球共通だが、昼夜D_dayは経度で変わるため、空間構造もある程度反映される。

from pathlib import Path
from datetime import datetime, timezone
import re
import math

import numpy as np
import pandas as pd


# ========== 設定値 ==========

# 平常値（偏差の基準）
F0 = 120.0   # F10.7 平常
AP0 = 5.0    # Ap 平常
DST0 = 0.0   # Dst 平常

# Δt の最大許容時間 [hours]
# （隣り合うファイル間で3時間以上空いてたら学習には使わない、など）
MAX_DELTA_HOURS = 3.1

# パス関係
REPO_ROOT = Path(__file__).resolve().parents[1]
BOM_DIR = REPO_ROOT / "data" / "bom_tec"
SPACE_WEATHER_CSV = REPO_ROOT / "data" / "space_weather.csv"
OUT_JS = REPO_ROOT / "swift_tec_coeffs.js"

# BoMファイル名から日時を抜く正規表現
# 例: World0000_tec_20251113_0030.txt
BOM_FILENAME_RE = re.compile(r"World0000_tec_(\d{8})_(\d{4})\.txt")


# ========== ユーティリティ関数 ==========

def parse_bom_timestamp(path: Path) -> datetime:
    """
    BoMファイル名からUTCのdatetimeを復元する。
    例: World0000_tec_20251113_0030.txt → 2025-11-13 00:30 UTC
    """
    m = BOM_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected BoM filename: {path.name}")
    date_str, time_str = m.groups()  # "20251113", "0030"
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return dt.replace(tzinfo=timezone.utc)


def read_bom_grid(path: Path) -> np.ndarray:
    """
    BoM World0000_tec.txt を 2次元 numpy array (lat x lon) で返す。
    ヘッダの後ろの「TEC値の行」だけを読み取る。
    """
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines()

    data_lines = []
    seen_blank = False
    for line in lines:
        # 空行の次からがデータ本体という前提
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
                # 数値でないものは無視
                pass
        if row_vals:
            rows.append(row_vals)

    if not rows:
        raise RuntimeError(f"No TEC data parsed in {path}")

    arr = np.array(rows, dtype=float)  # shape=(n_lat, n_lon)
    return arr


def load_space_weather() -> pd.DataFrame:
    """
    data/space_weather.csv を読み込み、datetime_utc列をdatetimeに変換して返す。
    CSV例:
        datetime_utc,F107,Ap,Dst,Xray
        2025-11-13T00:00:00,180,10,-20,1e-6
    """
    if not SPACE_WEATHER_CSV.exists():
        raise FileNotFoundError(f"space_weather.csv not found: {SPACE_WEATHER_CSV}")

    df = pd.read_csv(SPACE_WEATHER_CSV)
    if "datetime_utc" not in df.columns:
        raise ValueError("space_weather.csv must have 'datetime_utc' column")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def get_space_weather_for_time(df: pd.DataFrame, dt: datetime) -> dict:
    """
    指定したUTC時刻 dt に最も近い space_weather の行を返す。
    dt: timezone-aware datetime (UTC)
    """
    # 近い時刻を距離最小で選ぶ
    idx = (df["datetime_utc"] - dt).abs().idxmin()
    row = df.loc[idx]
    return {
        "F107": float(row["F107"]),
        "Ap":   float(row["Ap"]),
        "Dst":  float(row["Dst"]),
        "Xray": float(row["Xray"]),
    }


def local_solar_time_hours(ut_dt: datetime, lon_deg: float) -> float:
    """
    簡易ローカル太陽時 LST [hours] を計算する。
    LST ≒ UT[hours] + lon/15 (東経を+とする)。
    """
    ut_hours = ut_dt.hour + ut_dt.minute / 60.0 + ut_dt.second / 3600.0
    lst = (ut_hours + lon_deg / 15.0) % 24.0
    return lst


def day_weight(lst_hours: float) -> float:
    """
    D_day(l) = max(0, cos(pi * (LST-12)/12))
    昼: ~1、夜: 0 に近い連続関数。
    """
    x = math.pi * (lst_hours - 12.0) / 12.0
    return max(0.0, math.cos(x))


# ========== 学習用データセット構築（グリッド版） ==========

def build_training_set_grid():
    """
    BoM TEC グリッドログ（World0000_tec_*.txt）と宇宙天気ログから、
    ΔTEC = TEC(t+Δt) - TEC(t) を目的変数とする学習用 (X, y) を構築する。
    Xは [1, Fdev, Fdev^2, Apdev, Dstdev, Xray, D_now, D_future] の8次元。
    サンプルは「全時刻ペア × 全グリッド点」の組み合わせ。
    """
    space_df = load_space_weather()

    bom_files = sorted(BOM_DIR.glob("World0000_tec_*.txt"))
    if len(bom_files) < 2:
        raise RuntimeError(f"BoM TEC logs are too few in {BOM_DIR} (need >= 2 files).")

    # すべての BoM ファイルを読み込み（時刻＆グリッド）
    bom_maps = []
    for f in bom_files:
        dt = parse_bom_timestamp(f)
        grid = read_bom_grid(f)  # shape=(n_lat, n_lon)
        bom_maps.append((dt, grid))

    # 経度配列（BoMファイルの列数から自動判定）
    sample_grid = bom_maps[0][1]
    n_lat, n_lon = sample_grid.shape
    # -180〜180を等間隔（BoMはstep=5degで73点）
    lons = np.linspace(-180.0, 180.0, n_lon)

    X_rows = []
    y_rows = []

    for (dt_now, grid_now), (dt_future, grid_future) in zip(bom_maps[:-1], bom_maps[1:]):
        delta_t_hours = (dt_future - dt_now).total_seconds() / 3600.0

        # Δtが大きすぎる（欠損など）のは学習から除外
        if delta_t_hours <= 0 or delta_t_hours > MAX_DELTA_HOURS:
            continue

        # 宇宙天気（t）の値
        sw_now = get_space_weather_for_time(space_df, dt_now)
        F107 = sw_now["F107"]
        Ap   = sw_now["Ap"]
        Dst  = sw_now["Dst"]
        Xray = sw_now["Xray"]

        Fdev   = F107 - F0
        Apdev  = Ap   - AP0
        Dstdev = Dst  - DST0

        # 各格子点(i,j)をサンプルにする
        n_lat, n_lon = grid_now.shape
        for j in range(n_lon):
            lon = lons[j]
            # 昼夜（現在と未来）を経度依存で計算
            lst_now    = local_solar_time_hours(dt_now, lon)
            lst_future = local_solar_time_hours(dt_future, lon)
            D_now      = day_weight(lst_now)
            D_future   = day_weight(lst_future)

            for i in range(n_lat):
                TEC_now    = grid_now[i, j]
                TEC_future = grid_future[i, j]
                dTEC       = TEC_future - TEC_now

                X_row = [
                    1.0,           # 定数項
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
        raise RuntimeError("No training samples built. Check BoM logs and space_weather.csv timing.")

    X = np.array(X_rows, dtype=float)  # shape=(N, 8)
    y = np.array(y_rows, dtype=float)  # shape=(N,)
    return X, y


# ========== 回帰 & JS書き出し ==========

def fit_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    最小二乗で beta を求める。
    X: [N,8], y:[N]
    戻り値 beta: [a0, aF1, aF2, aAp, aDst, aX, aD1, aD2]
    """
    # lstsq: 最小二乗解 (正則化なし)
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return beta


def write_js(coeffs: np.ndarray):
    """
    swift_tec_coeffs.js に係数を書き出す。
    """
    a0, aF1, aF2, aAp, aDst, aX, aD1, aD2 = coeffs

    js = f"""// This file is auto-generated by scripts/train_delta_tec_model.py
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
    print("Updated", OUT_JS)


def main():
    print("=== SWIFT-TEC ΔTEC モデル学習 (グリッド版) ===")
    print(f"BoM dir: {BOM_DIR}")
    print(f"Space weather CSV: {SPACE_WEATHER_CSV}")

    X, y = build_training_set_grid()
    print(f"Samples: {X.shape[0]}  (features per sample: {X.shape[1]})")

    print("Fitting coefficients (least squares)...")
    coeffs = fit_coefficients(X, y)
    print("Coefficients:")
    print("  a0, aF1, aF2, aAp, aDst, aX, aD1, aD2 =")
    print(" ", coeffs)

    write_js(coeffs)
    print("Done.")


if __name__ == "__main__":
    main()
