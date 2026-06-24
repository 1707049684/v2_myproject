# -*- coding: utf-8 -*-
"""XES3G5M（子采样至 ~1M 交互）· random_split 十方法总表
(6 Ours + EWC/DER++/C-LoRA + X-DER)。

数据：GNCDM/data/xes_*（3402 users × 7056 questions × 837 KC，稀疏 Q ≈1.16 KC/题，
KC 来自 question_level 的 concepts 路由；score=responses 原生 0/1，pos_rate≈0.793 偏高
→ 主指标看 AUC，ACC 平凡基线就有 0.79）。由 XES3G5M 全量按学生累加子采样到 ~1M 交互。
test 用户与训练共享 → buf 预测口径（论文 RQ2）。一个脚本跑全部模型：
  run_experiment(buf) 出 6 Ours → cl_baselines_random_split.run_one() 跑 3 CL 基线并合表
  → run_xder() 跑 X-DER 并把该行 append 进 all_methods 总表。
ΔK 用 auto_new_concepts(0.34)（→689 新概念占 82%、新题占比 34%）。
alpha 暂取 0.10（占位：新概念占比 82%>a0910 67% → 按"占比大→alpha 小"规律取 0.1，未实扫；
有 GPU 后用 _core/sweep_xes_random_alpha.py 同口径 sweep 确认再改）。
产物：incremental_result/all_methods_xes_random_split.{csv,md}（已含 X-DER 行）
     + incremental_result/xder_xes_random_split.{csv,md}（X-DER 单行原件）。
交互量 ~1M、KC 837，建议 GPU。
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
import cl_baselines_random_split as clbase

DATA_DIR = os.path.join(gncdm_dir, "data")
PREFIX = "xes"
N_USER, N_ITEM, N_KNOW = 3402, 7056, 837
ALPHA = 0.10  # 占位：新概念占比 82%>a0910 67% → 规律取 0.1，未实扫，有 GPU 后 sweep 确认


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
        "*TMD 红线*：Ours/X-DER 行 TMD 同在 G-NCDM 概念 θ 空间可互比；EWC/DER++/C-LoRA 行 TMD 在 "
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

    Q_path = os.path.join(DATA_DIR, f"{PREFIX}_Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, 0.34)
    tr, va, te = (
        os.path.join(DATA_DIR, f"{PREFIX}_{s}_0.8_0.1_0.1.csv") for s in ("train", "valid", "test")
    )

    # 1) 6 Ours（buf 口径）
    set_seed(42)
    run_experiment(
        "xes_random_split",
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

    # 2) 3 CL 基线（EWC/DER++/C-LoRA）并写 all_methods 总表
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

    # 3) X-DER（同骨干 buf 口径），并 append 进 all_methods 总表
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
    )
    append_xder_to_all_methods(xder_row)

    print("\n完成：incremental_result/all_methods_xes_random_split.csv（含 X-DER 行）")


if __name__ == "__main__":
    main()
