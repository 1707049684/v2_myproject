# -*- coding: utf-8 -*-
"""a0910 · random_split：只跑 6 个 Ours 策略 + X-DER(mem=5000)，合成一张表。

包含方法（无 EWC/DER++/C-LoRA）：
  Base / Ours-Ablated / Ours (Dynamic DNA) / Ours (LoRA) / Full Replay Oracle /
  Naive FT (NFT)  —— 由 run_experiment 一次产出（alpha=0.1, ΔK=auto_new_concepts(0.34)）；
  X-DER (mem=5000) —— 由 run_xder 产出，G-NCDM 同骨干、buf 无泄漏预测，逐行可比。

口径与主实验完全一致（mode='buf'，forward_using_buf）；X-DER 内部自训 Task1 Base（同 seed/
alpha/数据，与 run_experiment 的 Base 一致），只输出 X-DER 行，不重复 Base。

产物：incremental_result/ours_xder_a0910_random_split.{csv,md}
a0910 题量大(17746)，务必 GPU 服务器。
运行：cd GNCDM/experiments && python run_ours_xder_a0910_random_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from run_incremental_math1 import set_seed, run_experiment, SAVE_DIR
from run_incremental_a0910 import auto_new_concepts
from run_xder import run_xder, COLS

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.1
BUFFER_SIZE = 5000


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，CPU 会很慢，建议 GPU 服务器。")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    new_concepts = auto_new_concepts(np.load(Q_path), 0.34)
    rnd = os.path.join(DATA_DIR, "new_random_split")
    tr, va, te = (os.path.join(rnd, f) for f in ("train.csv", "valid.csv", "test.csv"))

    # ---- 1) 6 个 Ours 策略（run_strategies=None → 全跑：Base/Ablated/DNA/LoRA/Oracle/NFT）----
    set_seed(42)
    ours = run_experiment(
        "a0910_random_split",
        "buf",
        tr,
        va,
        te,
        Q_path,
        device,
        n_user=N_USER,
        n_item_total=N_ITEM,
        n_know_total=N_KNOW,
        new_concepts=new_concepts,
        alpha=ALPHA,
    )

    # ---- 2) X-DER (mem=5000)，同骨干同口径 ----
    xder = run_xder(
        split_name="a0910_random_split",
        ds_name="a0910",
        train_path=tr,
        valid_path=va,
        test_path=te,
        Q_path=Q_path,
        device=device,
        n_user=N_USER,
        n_item_total=N_ITEM,
        n_know_total=N_KNOW,
        new_concepts=new_concepts,
        alpha=ALPHA,
        buffer_size=BUFFER_SIZE,
    )

    # ---- 3) 合并成一张表（run_experiment 用 'Model' 键 → 统一改名 'Method'）----
    rows = [{"Method": r["Model"], **{c: r[c] for c in COLS[1:]}} for r in ours]
    rows.append({c: xder[c] for c in COLS})
    df = pd.DataFrame(rows, columns=COLS)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_csv = os.path.join(SAVE_DIR, "ours_xder_a0910_random_split.csv")
    df.to_csv(out_csv, index=False)

    def _fmt(x):
        return x if isinstance(x, str) else f"{x:.4f}"

    lines = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([str(r["Method"])] + [_fmt(r[c]) for c in COLS[1:]]) + " |")
    note = (
        f"\n*口径*：a0910 random_split，G-NCDM 骨干，buf 无泄漏预测（forward_using_buf）。"
        f"alpha={ALPHA}，ΔK=auto_new_concepts(0.34)，set_seed(42)。\n"
        f"*方法*：6 Ours（Base/Ablated/DNA/LoRA/Oracle/NFT）+ X-DER(mem={BUFFER_SIZE})；"
        f"未含 EWC/DER++/C-LoRA。\n"
        "*TMD*：均在 G-NCDM 概念 θ 空间（前 K_old 列），可逐行对比。\n"
    )
    with open(
        os.path.join(SAVE_DIR, "ours_xder_a0910_random_split.md"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(lines) + "\n" + note)

    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print(f"\n完成：{out_csv}")
    print("       incremental_result/ours_xder_a0910_random_split.md")


if __name__ == "__main__":
    main()
