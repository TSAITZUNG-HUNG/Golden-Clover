# -*- coding: utf-8 -*-
"""
刮刮樂模擬程式（批次統計；停機看累計RTP；含必輸；含累積標準差）
- 必輸機率 = 1 - TARGET_RTP（例：100%→0、96%→0.04）
- 必輸卡：派彩=0；不抽組合、不計倍數線
- Excel 每列是「該批次自己的統計值（非累計）」；新增「累積標準差」欄位
- 停機條件：看「累計 RTP」是否連續 10 批落在目標 ±0.1%
- 每批完成印出：單批RTP、累計RTP、必輸比例、累計張數
"""

import numpy as np
import pandas as pd

# ===== 參數 =====
MULTIPLIERS = np.array([1, 500, 5, 5000, 1, 50, 10, 5000], dtype=np.int64)  # 倍數1~8
BET_PER_CARD = 1.0
TARGET_RTP = 0.93            # 可改：例如 0.96 → 必輸機率 0.04
TOLERANCE = 0.001            # 停機容忍：±0.1%
BATCH_SIZE = 1_000_000       # 每批 100 萬張
REQUIRED_STREAK = 10         # 連續 10 批（看「累計RTP」）
SEED = 42

# 由目標RTP推得的必輸機率（夾在 [0,1]）
MUST_LOSE_PROB = float(np.clip(1.0 - TARGET_RTP, 0.0, 1.0))

# ===== 內建 37 列（flags=8條倍數線; weight=權重）=====
SCRATCH_COMBOS = [
    ([0,0,0,1,0,0,0,1], 1),
    ([0,1,0,1,0,0,0,0], 1),
    ([0,1,0,0,0,0,0,1], 1),
    ([0,0,0,1,0,1,0,0], 0),
    ([0,0,0,0,0,1,0,1], 0),
    ([0,0,0,1,0,0,1,0], 0),
    ([0,0,0,0,0,0,1,1], 0),
    ([0,0,1,1,0,0,0,0], 0),
    ([0,0,1,0,0,0,0,1], 0),
    ([1,0,0,1,0,0,0,0], 0),
    ([1,0,0,0,0,0,0,1], 0),
    ([0,0,0,1,1,0,0,0], 0),
    ([0,0,0,0,1,0,0,1], 0),
    ([0,0,0,0,0,0,0,1], 1),
    ([0,0,0,1,0,0,0,0], 1),
    ([0,1,0,0,0,1,0,0], 10),
    ([0,1,0,0,0,0,1,0], 5),
    ([0,1,1,0,0,0,0,0], 5),
    ([1,1,0,0,0,0,0,0], 5),
    ([0,1,0,0,1,0,0,0], 5),
    ([0,1,0,0,0,0,0,0], 5),
    ([0,0,0,0,0,1,1,0], 100),
    ([0,0,1,0,0,1,0,0], 180),
    ([1,0,0,0,0,1,0,0], 140),
    ([0,0,0,0,1,1,0,0], 140),
    ([0,0,0,0,0,1,0,0], 140),
    ([0,0,1,0,0,0,1,0], 2500),
    ([1,0,0,0,0,0,1,0], 5130),
    ([0,0,0,0,1,0,1,0], 5130),
    ([0,0,0,0,0,0,1,0], 7130),
    ([1,0,1,0,0,0,0,0], 15020),
    ([0,0,1,0,1,0,0,0], 15020),
    ([0,0,1,0,0,0,0,0], 38090),
    ([1,0,0,0,1,0,0,0], 100550),
    ([1,0,0,0,0,0,0,0], 60143),
    ([0,0,0,0,1,0,0,0], 60142),
    ([0,0,0,0,0,0,0,0], 690405),
]

# ===== 工具函數 =====
def _prepare_tables():
    layout = np.array([c[0] for c in SCRATCH_COMBOS], dtype=np.int8)       # (items, 8)
    weights = np.array([c[1] for c in SCRATCH_COMBOS], dtype=np.float64)   # (items,)
    if weights.sum() <= 0:
        raise ValueError("權重總和需大於 0")
    return layout, weights

def _weighted_sample_indices(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """以權重分佈抽樣 n 次（向量化）。"""
    prob = weights / weights.sum()
    cdf = np.cumsum(prob)
    u = rng.random(n)
    return np.searchsorted(cdf, u, side="right").astype(np.int32)

def _combine_running_stats(n_a, mean_a, M2_a, n_b, mean_b, var_b_sample):
    """
    Chan/Welford 合併樣本方差統計：
    n:樣本數, mean:平均, M2:sum of squared deviations。var_b_sample 為樣本方差（ddof=1）。
    """
    if n_b == 0:
        return n_a, mean_a, M2_a
    if n_a == 0:
        return n_b, mean_b, var_b_sample * (n_b - 1)
    M2_b = var_b_sample * (n_b - 1)
    delta = mean_b - mean_a
    n = n_a + n_b
    mean = mean_a + delta * (n_b / n)
    M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b / n)
    return n, mean, M2

# ===== 主流程（輸出每批統計；停機看累計RTP；含必輸；累積標準差） =====
def simulate_batches_batchwise() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    layout, weights = _prepare_tables()
    items = layout.shape[0]
    item_payout = layout.dot(MULTIPLIERS).astype(np.int64)  # 各組合派彩倍數總和

    rows = []

    # 停機用累計 RTP
    streak = 0
    batch_id = 1
    cum_total_pay = 0.0
    cum_total_win = 0.0
    cum_cards = 0

    # 累積標準差的 running stats（以「單卡派彩」為樣本）
    run_n = 0
    run_mean = 0.0
    run_M2 = 0.0

    while True:
        # --- 遊戲前「必輸」判斷 ---
        must_lose_mask = rng.random(BATCH_SIZE) < MUST_LOSE_PROB
        n_play = int((~must_lose_mask).sum())

        # 預設整批派彩=0
        pays = np.zeros(BATCH_SIZE, dtype=np.float64)

        # 僅對「非必輸」的卡去抽組合 & 給派彩；倍數線出現只計非必輸
        if n_play > 0:
            idx_play = _weighted_sample_indices(weights, n_play, rng)         # (n_play,)
            pays[~must_lose_mask] = item_payout[idx_play].astype(np.float64)

            binc = np.bincount(idx_play, minlength=items).astype(np.int64)    # 每組合抽到次數
            line_counts_batch = (binc[:, None] * layout).sum(axis=0).astype(np.int64)
        else:
            line_counts_batch = np.zeros(8, dtype=np.int64)

        # --- 本批統計（輸出到 Excel 的就是這些「批次數值」）---
        batch_total_pay = BATCH_SIZE * BET_PER_CARD
        batch_total_win = float(pays.sum())
        batch_rtp = batch_total_win / batch_total_pay
        batch_std = float(pays.std(ddof=1)) if BATCH_SIZE > 1 else 0.0
        batch_win_cards = int((pays > 0).sum())
        batch_win_rate = batch_win_cards / BATCH_SIZE

        # --- 累積統計（只用來停機 & 累積標準差）---
        cum_cards += BATCH_SIZE
        cum_total_pay += batch_total_pay
        cum_total_win += batch_total_win
        cum_rtp = cum_total_win / cum_total_pay

        # 累積標準差（Chan/Welford 合併）
        run_n, run_mean, run_M2 = _combine_running_stats(
            run_n, run_mean, run_M2,
            BATCH_SIZE, float(pays.mean()), (batch_std ** 2)
        )
        cum_std = (np.sqrt(run_M2 / (run_n - 1)) if run_n > 1 else 0.0)

        # --- 寫一列（批次值 + 累積標準差）---
        row = {
            "驗證次數": batch_id,
            "Total Pay": float(batch_total_pay),
            "Total Win": float(batch_total_win),
            "RTP": float(batch_rtp),
            "標準差": float(batch_std),
            "累積標準差": float(cum_std),
            "中獎卡數": int(batch_win_cards),
            "中獎率": float(batch_win_rate),
        }
        for j in range(8):
            row[f"倍數{j+1}"] = int(line_counts_batch[j])
        rows.append(row)

        # --- 進度輸出（顯示單批&累計）---
        print(
            f"批次 {batch_id} 完成：單批RTP={batch_rtp:.6f}，"
            f"累計RTP={cum_rtp:.6f}，已模擬 {cum_cards:_} 張，"
            f"本批必輸比例={must_lose_mask.mean():.4f}"
        )

        # --- 停止條件：看「累計 RTP」是否連續 10 批在目標範圍 ---
        if abs(cum_rtp - TARGET_RTP) <= TOLERANCE:
            streak += 1
        else:
            streak = 0
        if streak >= REQUIRED_STREAK:
            break

        batch_id += 1

    return pd.DataFrame(rows)

def main(output_path: str = "刮刮樂_RTP驗證結果.xlsx"):
    df = simulate_batches_batchwise()
    ordered_cols = ["驗證次數", "Total Pay", "Total Win", "RTP", "標準差", "累積標準差", "中獎卡數", "中獎率"] + [f"倍數{i}" for i in range(1, 9)]
    df = df[ordered_cols]
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="RTP驗證")
    print(f"已輸出：{output_path}")
    print(df.tail(5).to_string(index=False))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="刮刮樂_RTP驗證結果.xlsx", help="輸出 Excel 檔名")
    args = p.parse_args()
    main(args.out)
