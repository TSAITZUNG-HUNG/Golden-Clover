# -*- coding: utf-8 -*-
"""
批次 Runs Test（事件 = 派彩「倍數總和 ≥ 門檻」）
- 一次驗 4 種 TARGET_RTP：1.0、0.965、0.95、0.93
- 必輸機率 = 1 − TARGET_RTP（必輸不抽組合；事件=0）
- Excel 輸出：
  * RunTest結果：每個 RTP 一列（R/E/Var/Z/p、事件率、非必輸數、門檻等）
  * 事件目標對照：37 項目的派彩倍數總和、權重、是否屬於事件（≥門檻）
用法：
  python run_test_payout_ge_threshold.py \
      --out run_test_payout_ge10_multi_target.xlsx \
      --draws 1000000 \
      --targets 1.0,0.965,0.95,0.93 \
      --threshold 10 \
      --seed 42
"""

import math
import numpy as np
import pandas as pd


# ===== 遊戲定義 =====
# 倍數1~8
MULTIPLIERS = np.array([1, 500, 5, 5000, 1, 50, 10, 5000], dtype=np.int64)

# 37 個項目：8 條倍數線 flag + 權重
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


# ===== 基本工具 =====
def must_lose_prob(target_rtp: float) -> float:
    """必輸機率 = 1 − RTP。"""
    return float(np.clip(1.0 - target_rtp, 0.0, 1.0))


def weighted_sample_indices(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """根據權重抽樣 n 次（inverse CDF）。"""
    prob = weights / weights.sum()
    cdf = np.cumsum(prob)
    u = rng.random(n)
    return np.searchsorted(cdf, u, side="right").astype(np.int32)


def runs_test_wald_wolfowitz(seq01: np.ndarray):
    """
    Wald–Wolfowitz Runs Test（0/1 序列）
    回傳 dict: R, E, Var, Z, p(two-sided), note
    """
    n = len(seq01)
    n1 = int(seq01.sum())
    n0 = n - n1
    if n == 0:
        return {"R": None, "E": None, "Var": None, "Z": None, "p": None, "note": "序列長度為 0"}
    if n1 == 0 or n0 == 0:
        return {"R": None, "E": None, "Var": None, "Z": None, "p": 1.0, "note": "全為同一類（無法檢定）"}

    R = 1 + int((seq01[1:] != seq01[:-1]).sum())
    E = 1 + 2 * n1 * n0 / (n1 + n0)
    Var = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    Z = (R - E) / math.sqrt(Var) if Var > 0 else 0.0
    p_two = math.erfc(abs(Z) / math.sqrt(2.0))
    return {"R": R, "E": E, "Var": Var, "Z": Z, "p": p_two, "note": ""}


# ===== 單一 RTP：模擬 + Run Test =====
def simulate_and_test_payout_threshold(total_draws: int,
                                       target_rtp: float,
                                       seed: int,
                                       payout_threshold: int = 10):
    """
    事件定義：抽到的「派彩倍數總和 >= payout_threshold」。
    必輸先判斷；必輸不抽組合、事件=0。
    """
    rng = np.random.default_rng(seed)
    layout  = np.array([f for f, _ in SCRATCH_COMBOS], dtype=np.int8)     # (37, 8)
    weights = np.array([w for _, w in SCRATCH_COMBOS], dtype=np.float64)  # (37,)
    item_payout = layout.dot(MULTIPLIERS).astype(np.int64)                # (37,)

    event_mask = (item_payout >= payout_threshold)  # (37,)

    # 必輸
    mlp = must_lose_prob(target_rtp)
    must_lose = rng.random(total_draws) < mlp
    play_positions = np.flatnonzero(~must_lose)     # 只有這些位置會抽組合
    n_play = play_positions.size

    # 形成 0/1 事件序列
    events = np.zeros(total_draws, dtype=np.uint8)
    if n_play > 0:
        idx = weighted_sample_indices(weights, n_play, rng)               # 0..36
        events[play_positions] = event_mask[idx].astype(np.uint8)

    # Run Test
    res = runs_test_wald_wolfowitz(events)

    # 基本資料
    meta = {
        "TARGET_RTP": target_rtp,
        "必輸機率": mlp,
        "總注數": total_draws,
        "非必輸數": int(n_play),
        "事件次數": int(events.sum()),
        "事件率": float(events.mean()),
        "event_mode": "payout_threshold",
        "threshold": int(payout_threshold),
        "亂數種子": seed,
    }

    # 事件目標對照表
    target_info = pd.DataFrame({
        "項目ID": np.arange(1, layout.shape[0] + 1),
        "派彩倍數總和": item_payout,
        "權重": weights.astype(int),
        "屬於事件?(>=threshold)": event_mask,
    })

    return res, meta, target_info


# ===== 主流程：一次驗 4 種 RTP，輸出 Excel =====
def main(out_xlsx: str = "Golden Clover黃金幸運草RunTest統計驗證.xlsx",
         total_draws: int = 1_000_000,
         targets: list[float] = [1.0, 0.965, 0.95, 0.93],
         threshold: int = 10,
         base_seed: int = 42):
    rows = []
    target_info_df = None

    # 每個 RTP 用不同 seed（seed, seed+1, ...）
    for i, t in enumerate(targets):
        res, meta, info = simulate_and_test_payout_threshold(
            total_draws=total_draws,
            target_rtp=t,
            seed=base_seed + i,
            payout_threshold=threshold,
        )
        rows.append({
            **meta,
            "R": res["R"],
            "E[R]": res["E"],
            "Var[R]": res["Var"],
            "Z": res["Z"],
            "p(雙尾)": res["p"],
        })
        target_info_df = info  # 事件定義與 RTP 無關，任取一次即可

    result_df = pd.DataFrame(rows)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="RunTest結果")
        target_info_df.to_excel(writer, index=False, sheet_name="事件目標對照")

    # 同步列印摘要
    print("=== Runs Test（事件=倍數總和≥{}） ===".format(threshold))
    print(result_df.to_string(index=False))
    print(f"\n已輸出：{out_xlsx}")


# ===== CLI =====
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="Golden Clover黃金幸運草RunTest統計驗證.xlsx", help="輸出 Excel 檔名")
    p.add_argument("--draws", type=int, default=1_000_000, help="總注數（預設100萬）")
    p.add_argument("--targets", type=str, default="1.0,0.965,0.95,0.93", help="目標RTP列表（逗號分隔）")
    p.add_argument("--threshold", type=int, default=10, help="事件門檻：派彩倍數總和 ≥ threshold")
    p.add_argument("--seed", type=int, default=42, help="基礎亂數種子（各RTP使用 seed, seed+1, ...）")
    args = p.parse_args()

    targets = [float(x) for x in args.targets.split(",")]
    main(out_xlsx=args.out,
         total_draws=args.draws,
         targets=targets,
         threshold=args.threshold,
         base_seed=args.seed)
