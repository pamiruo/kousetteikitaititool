# logic/magireko.py
from flask import render_template, request
import pandas as pd
import numpy as np
import os

# ====== 定数 ======
COIN_MOCHI = 1.5337   # コイン持ち
JUNZOU     = 2.6     # 純増
YAME       = 10      # ヤメG
KAISHU     = 46
KOUKAN     = 51
TOKA       = 20
KAI        = 800     # 1時間あたりの回転数（G/h）

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "magireko.csv")


# ====== ユーティリティ ======
def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp932")

def _prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    for col in [
        "当選G", "●回目", "前回有利差枚", "有利フラグ", "前回AT間",
        "当選時AT間", "AT間", "当選時出玉", "出玉",
        "非当選加算", "非当選累計"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "朝一" not in df.columns:
        df["朝一"] = np.nan
    if "当非" not in df.columns:
        df["当非"] = np.nan

    if "当選時出玉" in df.columns and df["当選時出玉"].notna().any():
        payout_col = "当選時出玉"
    elif "出玉" in df.columns:
        payout_col = "出玉"
    else:
        payout_col = "出玉"
        df[payout_col] = np.nan

    if "当選時AT間" in df.columns and df["当選時AT間"].notna().any():
        at_interval_col = "当選時AT間"
    elif "AT間" in df.columns:
        at_interval_col = "AT間"
    else:
        at_interval_col = "当選時AT間"
        df[at_interval_col] = np.nan

    return df, payout_col, at_interval_col

def _is_empty(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")

def _map_suru_to_col_value(suru_input: int):
    try:
        n = int(suru_input)
    except Exception:
        return None
    return (n + 1) if n >= 0 else None

def _calc_possible_spins(remaining_minutes: int) -> int:
    return int((remaining_minutes / 60.0) * KAI)


# ====== ボーナス狙い ======
def _calc_cz(df: pd.DataFrame, payout_col: str,
             *, start_g: int, through_input: int,
             diff_min: int, diff_max: int,
             remaining_minutes: int) -> dict:

    kanou_cz = _calc_possible_spins(remaining_minutes)
    kanou_hosei_cz = kanou_cz + start_g
    suru_eq = _map_suru_to_col_value(through_input)
    tousen_jogai_val = 0 if start_g == 0 else 20
    tousen_threshold = start_g + tousen_jogai_val

    base = df.dropna(subset=["当選G"]).copy()
    base = base[_is_empty(base["朝一"])]
    base = base[base["有利フラグ"].between(0, 2)]
    base = base[base["当選G"] > tousen_threshold]
    if suru_eq is not None:
        base = base[base["●回目"] == suru_eq]
    base = base[base["前回有利差枚"].between(diff_min, diff_max)]

    all_sample = len(base)
    cz_win = base[base["当選G"] < kanou_hosei_cz]
    cz_tousen_sample = len(cz_win)
    cz_rate = (cz_tousen_sample / all_sample) if all_sample > 0 else 0.0
    cz_hatsu = float(cz_win["当選G"].mean()) - start_g if cz_tousen_sample > 0 else float(kanou_cz)
    kanou_kakutoku = max(0.0, (kanou_cz - cz_hatsu) * JUNZOU)

    cz_payout_col = "出玉" if "出玉" in df.columns else payout_col
    win_with_payout = cz_win.dropna(subset=[cz_payout_col]).copy()
    get_sum = float(win_with_payout[win_with_payout[cz_payout_col] <= kanou_kakutoku][cz_payout_col].sum())
    over_cnt = int((win_with_payout[cz_payout_col] > kanou_kakutoku).sum())
    cz_get = (get_sum + over_cnt * kanou_kakutoku) / cz_tousen_sample if cz_tousen_sample > 0 else 0.0
    seiki_get = cz_get * cz_rate

    if cz_tousen_sample > 0:
        at_win = cz_win[cz_win["当非"].astype(str) != "非"]
        at_tousen_sample = len(at_win)
        at_rate = at_tousen_sample / cz_tousen_sample
    else:
        at_tousen_sample, at_rate = 0, 0.0

    numer = (seiki_get / JUNZOU) * (3 + JUNZOU) + (cz_hatsu + at_rate * YAME) * (3 - COIN_MOCHI)
    denom = (seiki_get / JUNZOU) * 3 + (cz_hatsu + at_rate * YAME) * 3
    rate = "-" if denom <= 0 else round(numer / denom * 100, 1)

    # 結果まとめ
    result = {"機械割(%)": rate, "サンプル数": int(cz_tousen_sample),
              "平均獲得枚数": round(cz_get, 1)}

    # ✅ サンプル数チェック
    if result["サンプル数"] <= 50:
        result["機械割(%)"] = "-"
        result["平均獲得枚数"] = "-"
        result["note"] = "サンプルが50件以下のため非表示"

    return result


# ====== AT狙い ======
def _calc_at(df: pd.DataFrame, payout_col: str, at_interval_col: str,
             *, start_g: int, through_input: int,
             diff_min: int, diff_max: int,
             remaining_minutes: int) -> dict:

    kanou_at = _calc_possible_spins(remaining_minutes)
    suru_eq = _map_suru_to_col_value(through_input)
    tousen_jogai_val = 0 if start_g == 0 else 20
    tousen_threshold = start_g + tousen_jogai_val

    base = df.dropna(subset=["当選G"]).copy()
    base = base[_is_empty(base["朝一"])]
    base = base[base["有利フラグ"].between(0, 2)]
    base = base[base["当選G"] > tousen_threshold]
    if suru_eq is not None:
        base = base[base["●回目"] == suru_eq]
    base = base[base["前回有利差枚"].between(diff_min, diff_max)]

    # 前回AT間
    at_nonnull = base.dropna(subset=[at_interval_col]).copy()
    prev_at_mean = float(at_nonnull["前回AT間"].mean()) if "前回AT間" in at_nonnull.columns and not at_nonnull.empty else 0.0
    kanou_hosei_at = kanou_at + start_g + prev_at_mean

    # AT当選
    at_win = at_nonnull[at_nonnull[at_interval_col] < kanou_hosei_at]
    at_tousen_sample = len(at_win)
    all_sample = len(at_nonnull)

    at_rate = at_tousen_sample / all_sample if all_sample > 0 else 0.0
    at_tousen_g = float(at_win[at_interval_col].mean()) if at_tousen_sample > 0 else float(kanou_hosei_at)
    at_hatsu = max(0.0, at_tousen_g - prev_at_mean - start_g)

    # ✅ 平均通常GATを追加
    avg_normal_g = at_rate * at_hatsu + (1 - at_rate) * kanou_at

    kanou_kakutoku = max(0.0, (kanou_at - at_hatsu) * JUNZOU)

    # 獲得枚数
    win_with_payout = at_win.dropna(subset=[payout_col]).copy()
    get_sum = float(win_with_payout[win_with_payout[payout_col] <= kanou_kakutoku][payout_col].sum())
    over_cnt = int((win_with_payout[payout_col] > kanou_kakutoku).sum())
    at_get = (get_sum + over_cnt * kanou_kakutoku) / at_tousen_sample if at_tousen_sample > 0 else 0.0
    seiki_get = at_get * at_rate

    # 非当選
    prev_hiatari = float(at_win["非当選加算"].mean()) if ("非当選加算" in at_win.columns and at_tousen_sample > 0) else 0.0
    hiatari_rui = float(at_win["非当選累計"].mean()) if ("非当選累計" in at_win.columns and at_tousen_sample > 0) else 0.0
    hiatari_get = hiatari_rui - prev_hiatari

    # ✅ 機械割に「平均通常GAT」を使用
    numer = ((seiki_get + hiatari_get) / JUNZOU) * (3 + JUNZOU) + (avg_normal_g + YAME) * (3 - COIN_MOCHI)
    denom = ((seiki_get + hiatari_get) / JUNZOU) * 3 + (avg_normal_g + YAME) * 3
    rate = "-" if denom <= 0 else round(numer / denom * 100, 1)

    # 結果まとめ
    result = {"機械割(%)": rate, "サンプル数": int(at_tousen_sample),
              "平均獲得枚数": round(at_get, 1)}

    # ✅ サンプル数チェック
    if result["サンプル数"] <= 50:
        result["機械割(%)"] = "-"
        result["平均獲得枚数"] = "-"
        result["note"] = "サンプルが50件以下のため非表示"

    return result



# ====== 引き戻し/天国ゾーン狙い 共通 ======
def _calc_hikimodoshi_like(df: pd.DataFrame, g_threshold: int,
                           *, start_g: int, through_input: int,
                           diff_min: int, diff_max: int,
                           remaining_minutes: int) -> dict:

    kanou = _calc_possible_spins(remaining_minutes)
    suru_eq = _map_suru_to_col_value(through_input)
    tousen_jogai_val = 0 if start_g == 0 else 20
    tousen_threshold = start_g + tousen_jogai_val

    base = df.dropna(subset=["当選G"]).copy()
    base = base[_is_empty(base["朝一"])]
    base = base[base["有利フラグ"].between(0, 2)]
    base = base[base["当選G"] > tousen_threshold]
    if suru_eq is not None:
        base = base[base["●回目"] == suru_eq]
    base = base[base["前回有利差枚"].between(diff_min, diff_max)]

    all_sample = len(base)
    h_win = base[base["当選G"] <= g_threshold]
    h_tousen_sample = len(h_win)
    h_rate = h_tousen_sample / all_sample if all_sample > 0 else 0.0
    h_tousen_g = float(h_win["当選G"].mean()) if h_tousen_sample > 0 else float(g_threshold)

    # 平均G（修正版）
    h_avg_g = h_rate * (h_tousen_g - start_g) + (1 - h_rate) * max(0.0, g_threshold - start_g)

    # ★ 修正版：可能獲得引き戻し
    kanou_kakutoku = max(0.0, (kanou - (h_tousen_g - start_g)) * JUNZOU)

    payout_col = "出玉" if "出玉" in df.columns else None
    if payout_col:
        win_with_payout = h_win.dropna(subset=[payout_col]).copy()
        get_sum = float(win_with_payout[win_with_payout[payout_col] <= kanou_kakutoku][payout_col].sum())
        over_cnt = int((win_with_payout[payout_col] > kanou_kakutoku).sum())
    else:
        get_sum, over_cnt = 0.0, 0

    h_get = (get_sum + over_cnt * kanou_kakutoku) / h_tousen_sample if h_tousen_sample > 0 else 0.0
    h_avg_get = h_get * h_rate

    if h_tousen_sample > 0:
        at_win = h_win[h_win["当非"].astype(str) != "非"]
        at_tousen_sample = len(at_win)
        at_rate = at_tousen_sample / h_tousen_sample
    else:
        at_tousen_sample, at_rate = 0, 0.0

    numer = (h_avg_get / JUNZOU) * (3 + JUNZOU) + (h_avg_g + h_rate * at_rate * YAME) * (3 - COIN_MOCHI)
    denom = (h_avg_get / JUNZOU) * 3 + (h_avg_g + h_rate * at_rate * YAME) * 3
    rate = "-" if denom <= 0 else round(numer / denom * 100, 1)

    # 結果まとめ
    result = {"機械割(%)": rate, "サンプル数": int(h_tousen_sample),
              "平均獲得枚数": round(h_get, 1)}

    # ✅ サンプル数チェック
    if result["サンプル数"] <= 50:
        result["機械割(%)"] = "-"
        result["平均獲得枚数"] = "-"
        result["note"] = "サンプルが50件以下のため非表示"

    return result


# ====== Flask ハンドラ ======
def magireko_handler():
    filters = {
        "remaining": 180,
        "target": "ボーナス狙い",
        "through": 0,
        "start": 0,
        "差枚_min": -10000,
        "差枚_max": 2500,
    }
    show_result = False
    result = None

    if request.method == "POST":
        def _iget(name, default):
            v = request.form.get(name, default)
            try:
                return int(v)
            except Exception:
                return default

        filters["remaining"] = _iget("remaining", filters["remaining"])
        filters["target"]    = request.form.get("target", filters["target"])
        filters["through"]   = _iget("through", filters["through"])
        filters["start"]     = _iget("start", filters["start"])
        filters["差枚_min"]  = _iget("差枚_min", filters["差枚_min"])
        filters["差枚_max"]  = _iget("差枚_max", filters["差枚_max"])

        path = CSV_PATH if os.path.exists(CSV_PATH) else os.path.join(os.getcwd(), "data", "tokyoguru.csv")
        df = _safe_read_csv(path)
        df, payout_col, at_interval_col = _prepare_df(df)

        if filters["target"] == "ボーナス狙い":
            result = _calc_cz(df=df, payout_col=payout_col,
                              start_g=filters["start"], through_input=filters["through"],
                              diff_min=filters["差枚_min"], diff_max=filters["差枚_max"],
                              remaining_minutes=filters["remaining"])
        elif filters["target"] == "AT狙い":
            result = _calc_at(df=df, payout_col=payout_col, at_interval_col=at_interval_col,
                              start_g=filters["start"], through_input=filters["through"],
                              diff_min=filters["差枚_min"], diff_max=filters["差枚_max"],
                              remaining_minutes=filters["remaining"])
        elif filters["target"] == "引き戻し狙い":
            result = _calc_hikimodoshi_like(df=df, g_threshold=70,
                                            start_g=filters["start"], through_input=filters["through"],
                                            diff_min=filters["差枚_min"], diff_max=filters["差枚_max"],
                                            remaining_minutes=filters["remaining"])
        elif filters["target"] == "天国ゾーン狙い":
            result = _calc_hikimodoshi_like(df=df, g_threshold=140,
                                            start_g=filters["start"], through_input=filters["through"],
                                            diff_min=filters["差枚_min"], diff_max=filters["差枚_max"],
                                            remaining_minutes=filters["remaining"])
        else:
            result = {"機械割(%)": "-", "サンプル数": 0,
                      "平均獲得枚数": 0.0,
                      "AT当選サンプル": 0, "AT当選率": 0.0}

        show_result = True

    return render_template("magireko.html", filters=filters,
                           show_result=show_result, result=result)
