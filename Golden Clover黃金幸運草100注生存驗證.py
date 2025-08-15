# -*- coding: utf-8 -*-
"""
驗證2：100注生存驗證
- 以每100注為一個單位（Unit），總共 10,000 個單位 → 1,000,000 注
- 每單位計算該100注的RTP，並以0.05步進做分佈區間（含頂端 ≥2）
- 遊戲邏輯：沿用前述刮刮樂（8條倍數、37組合+權重）
- 目標RTP → 必輸機率 = 1 - TARGET_RTP（本驗證預設 1.0 → 必輸=0）
- 輸出Excel：RTP範圍 / 數量 / 佔比例 + 底部 總押 / 總贏 / RTP
"""

import numpy as np
import pandas as pd

# ===== 參數 =====
MULTIPLIERS = np.array([1, 500, 5, 5000, 1, 50, 10, 5000], dtype=np.int64)  # 倍數1~8
BET_PER_CARD = 1.0

TARGET_RTP = 0.93             # 本驗證只需 1.0（可改為其他值）
SEED = 42

UNIT_SIZE = 100               # 每單位 100 注
NUM_UNITS = 10_000            # 單位數（→ 總注數 = 100 * 10,000 = 1,000,000）
TOTAL_BETS = UNIT_SIZE * NUM_UNITS

# 必輸機率（夾在 [0,1]）
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

def _format_bin_label(lower: float, upper: float) -> str:
    """格式：<upper, ≥ lower；兩位小數；頂端bin另處理。"""
    return f"<{upper:.2f}, ≥ {lower:.2f}"

# ===== 主流程：100注生存驗證 =====
def simulate_survival_100() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    layout, weights = _prepare_tables()
    item_payout = layout.dot(MULTIPLIERS).astype(np.int64)   # 各組合派彩倍數總和

    # --- 先產生所有 1,000,000 注的派彩（向量化） ---
    # 1) 必輸判斷
    must_lose_mask = rng.random(TOTAL_BETS) < MUST_LOSE_PROB
    n_play = int((~must_lose_mask).sum())

    pays = np.zeros(TOTAL_BETS, dtype=np.float64)
    if n_play > 0:
        idx_play = _weighted_sample_indices(weights, n_play, rng)
        pays[~must_lose_mask] = item_payout[idx_play].astype(np.float64)

    # 2) 轉成 10,000 x 100，取得每單位RTP
    unit_win = pays.reshape(NUM_UNITS, UNIT_SIZE).sum(axis=1)
    unit_rtp = unit_win / (UNIT_SIZE * BET_PER_CARD)   # 每單位 RTP

    # --- 分佈（步進 0.05），頂端為 ≥ 2 ---
    # 先統計 [0, 2) 的 0.05 區間
    edges = np.arange(0.0, 2.0 + 1e-9, 0.05)  # 0, 0.05, ..., 2.0
    counts_lt2, _ = np.histogram(unit_rtp, bins=edges)  # 40 個 bin: [0,0.05),...,[1.95,2.0)
    count_ge2 = int((unit_rtp >= 2.0).sum())

    # 組成表格（照需求的順序由上往下：≥2，接著 <2, ≥1.95，...，<0.05, ≥0）
    labels = []
    counts = []

    labels.append("≥2")
    counts.append(count_ge2)

    # 依序加入 <2, ≥1.95 ... 到 <0.05, ≥0
    # counts_lt2 對應 [0,0.05), [0.05,0.10), ..., [1.95,2.0)；我們要反向輸出
    uppers = np.arange(2.0, 0.0, -0.05)  # 2.00, 1.95, ..., 0.05
    for i, upper in enumerate(uppers, start=1):
        lower = upper - 0.05
        # histogram bins: index 0 -> [0,0.05), index 39 -> [1.95,2.0)
        bin_idx = int(round((upper - 0.05) / 0.05))  #  1.95→39, 0.05→1
        labels.append(_format_bin_label(lower, upper))
        counts.append(int(counts_lt2[bin_idx]))

    # 佔比例
    proportions = [c / NUM_UNITS for c in counts]

    # 組成 DataFrame
    df = pd.DataFrame({
        "RTP範圍": labels,
        "數量": counts,
        "佔比例": proportions,
    })

    # 底部附上總押 / 總贏 / RTP
    total_pay = float(TOTAL_BETS * BET_PER_CARD)
    total_win = float(pays.sum())
    overall_rtp = total_win / total_pay if total_pay > 0 else 0.0

    # 用於回傳與輸出
    summary = {
        "總押": total_pay,
        "總贏": total_win,
        "RTP": overall_rtp,
    }
    return df, summary

def main(output_xlsx: str = "100注生存驗證_分佈.xlsx"):
    df, summary = simulate_survival_100()

    # 輸出 Excel
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="100注生存驗證")
        # 在同一工作表底部加上總結
        # 找到最後一列
        start_row = len(df) + 2
        ws = writer.sheets["100注生存驗證"]
        ws.write(start_row + 0, 0, "總押")
        ws.write(start_row + 0, 1, summary["總押"])
        ws.write(start_row + 1, 0, "總贏")
        ws.write(start_row + 1, 1, summary["總贏"])
        ws.write(start_row + 2, 0, "RTP")
        ws.write(start_row + 2, 1, summary["RTP"])

    # 同時印出到終端
    print(df.to_string(index=False))
    print("\n--- 總結 ---")
    print(f"總押：{summary['總押']:.0f}")
    print(f"總贏：{summary['總贏']:.6f}")
    print(f"RTP ：{summary['RTP']:.6f}")
    print(f"\n已輸出：{output_xlsx}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="100注生存驗證_分佈.xlsx", help="輸出 Excel 檔名")
    args = p.parse_args()
    main(args.out)
