# -*- coding: utf-8 -*-
"""XES3G5M · random_split · 只跑「X-DER + 三大 CL 基线」（Ours 6 法已另跑）。

同骨架的 6 Ours 已经跑完，产物在 incremental_result/incremental_results_xes_random_split.csv
（列首为 Model）。本脚本不再重跑那 6 个（最贵的部分），只补：
  1) 三大 CL 基线 EWC / DER++ / C-LoRA —— clbase.run_one() 跑完后读 Ours CSV 合成
     all_methods_xes_random_split.{csv,md}（Ours 6 + 3 基线 = 9 行）；
  2) X-DER（同 G-NCDM 骨干, buf 无泄漏口径）—— append 进上面的总表（→ 10 行）。

口径与主入口 run_incremental_xes_random_split.py 一致：
  ΔK=auto_new_concepts(0.34)（→689 新概念占 82%、新题占 34%），alpha=0.20。
固定超参（不扫 lambda，单值定档）：
  EWC (lambda=10000) | DER++ (mem=5000) | C-LoRA (lambda=10000) | X-DER (mem=5000)

前置：incremental_result/incremental_results_xes_random_split.csv 必须存在
（即 6 Ours 已跑过；否则 clbase.run_one 合表时会因找不到 Ours CSV 报错）。

产物：
  incremental_result/all_methods_xes_random_split.{csv,md}  —— 10 行总表（Ours6 + EWC/DER++/C-LoRA + X-DER）
  incremental_result/xder_xes_random_split.{csv,md}         —— X-DER 单行原件（run_xder 写出）
  incremental_result/{ewc,clora}_lambda_sweep_xes_random_split.csv —— λ 扫描原件

运行：cd GNCDM/experiments && python run_remaining_baselines_xes_random_split.py
交互量 ~1M、KC 837，建议 GPU 服务器。
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

from run_incremental_a0910 import auto_new_concepts
from run_incremental_math1 import SAVE_DIR
from run_xder import run_xder, COLS
import cl_baselines_random_split as clbase

DATA_DIR = os.path.join(gncdm_dir, "data")
PREFIX = "xes"
N_USER, N_ITEM, N_KNOW = 3402, 7056, 837
ALPHA = 0.20  # 与主入口一致（sweep 实扫 DNA mean(valid AUC) 在 0.20 见顶）

# 固定超参（不扫 lambda，单值定档）：
#   EWC (lambda=10000) | DER++ (mem=5000) | C-LoRA (lambda=10000) | X-DER (mem=5000)
MEM = 5000  # DER++ / X-DER buffer_size
EWC_LAMBDA = 10000
CLORA_LAMBDA = 10000

OURS_CSV = os.path.join(SAVE_DIR, f"incremental_results_{PREFIX}_random_split.csv")


def append_xder_to_all_methods(xder_row):
    """把 X-DER 单行并进 all_methods 总表（csv + md），列与 COLS 一致。"""
    base = os.path.join(SAVE_DIR, f"all_methods_{PREFIX}_random_split")
    df = pd.read_csv(base + ".csv")
    df = df[df["Method"] != xder_row["Method"]]  # 防重跑重复 append
    df = pd.concat([df, pd.DataFrame([{k: xder_row[k] for k in COLS}])], ignore_index=True)
    df = df[COLS]
    df.to_csv(base + ".csv", index=False)

    def _fmt(x):
        if isinstance(x, str):
            return x
        return "-" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x:.4f}"

    lines = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in df.to_dict("records"):
        lines.append("| " + " | ".join([str(r["Method"])] + [_fmt(r[c]) for c in COLS[1:]]) + " |")
    note = (
        f"\n*口径*：{PREFIX} random_split（test 用户与训练共享，预测口径）。Ours/X-DER 走 "
        "G-NCDM 骨干 buf 无泄漏预测，CL 基线（CognitiveBackbone）直接预测；均无自信息，可逐行对比。\n"
        "*pos_rate≈0.79 偏高*：ACC 平凡基线就有 0.79，主指标看 AUC。\n"
        "*RD 红线*：Ours/X-DER 行 RD 同在 G-NCDM 概念 θ 空间可互比；EWC/DER++/C-LoRA 行 RD 在 "
        "embedding 空间，量级不可与之直接比，仅看是否>0。骨干不同，勿称纯策略胜出。\n"
    )
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)
    print(f"\n>>> X-DER 行已并入 {base}.csv / .md")


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ XES3G5M 子集交互量大(~1M),建议 GPU 服务器。")

    if not os.path.isfile(OURS_CSV):
        sys.exit(
            f"✗ 找不到 Ours 中间表：{OURS_CSV}\n"
            "  需先跑完同骨架 6 Ours（run_incremental_xes_random_split.py 的 step 1，"
            "或单独产出该 CSV）再运行本脚本。"
        )

    Q_path = os.path.join(DATA_DIR, f"{PREFIX}_Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, 0.34)
    tr, va, te = (
        os.path.join(DATA_DIR, f"{PREFIX}_{s}_0.8_0.1_0.1.csv") for s in ("train", "valid", "test")
    )

    # 1) 三大 CL 基线（EWC/DER++/C-LoRA），固定超参（不扫 lambda），读 Ours CSV 合成 all_methods（9 行）
    clbase.MEM_SIZE = MEM
    clbase.EWC_LAMBDA_SWEEP = [EWC_LAMBDA]
    clbase.CLORA_LAMBDA_SWEEP = [CLORA_LAMBDA]
    clbase.run_one(
        {
            "name": PREFIX,
            "train": tr,
            "valid": va,
            "test": te,
            "Q": Q_path,
            "n_item": N_ITEM,
            "n_know": N_KNOW,
            "new_concepts": "auto",
            "ours_csv": f"incremental_results_{PREFIX}_random_split.csv",
        },
        device,
    )

    # 2) X-DER（同 G-NCDM 骨干, buf 口径），append 进 all_methods（→ 10 行）
    xder_row = run_xder(
        split_name=f"{PREFIX}_random_split",
        ds_name=PREFIX,
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
        buffer_size=MEM,
    )
    append_xder_to_all_methods(xder_row)

    print(
        f"\n完成：all_methods_{PREFIX}_random_split.csv（Ours 6 + EWC/DER++/C-LoRA + X-DER = 10 行）"
    )


if __name__ == "__main__":
    main()
