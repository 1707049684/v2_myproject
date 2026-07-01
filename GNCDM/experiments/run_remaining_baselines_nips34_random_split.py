# -*- coding: utf-8 -*-
"""NIPS34 · random_split · 只跑 X-DER（Ours 6 法已另跑、结果见 Ours CSV）。

固定超参（不扫 lambda）：
  X-DER  (mem=5000)
口径与主入口一致：random_split / buf 无泄漏预测，ΔK=auto_new_concepts(0.34)，alpha=0.20。

产物：
  incremental_result/baselines_nips34_random_split.{csv,md}        —— X-DER 单行（始终写）
  incremental_result/all_methods_nips34_random_split.{csv,md}      —— 若 Ours CSV 存在则合成 7 行总表
Ours CSV 默认取 incremental_result/incremental_results_nips34_random_split.csv（列首为 Model）；
把你那份部分结果放到该路径（或改下面的 OURS_CSV）即可自动合表。

运行：cd GNCDM/experiments && python run_remaining_baselines_nips34_random_split.py
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
from run_xder import run_xder, COLS
import cl_baselines_random_split as clbase

DATA_DIR = os.path.join(gncdm_dir, "data")
SAVE_DIR = clbase.SAVE_DIR
PREFIX = "nips34"
N_USER, N_ITEM, N_KNOW = 4918, 948, 57
ALPHA = 0.20

# 固定超参
# EWC_LAMBDA = 10000
# CLORA_LAMBDA = 10000
MEM = 5000  # X-DER buffer_size

# Ours 6 法结果（列首为 Model）；放到此路径即自动合成 all_methods 总表
OURS_CSV = os.path.join(SAVE_DIR, f"incremental_results_{PREFIX}_random_split.csv")


def _fmt(x):
    if isinstance(x, str):
        return x
    return "-" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x:.4f}"


def _write_table(rows, base_path, note):
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(base_path + ".csv", index=False)
    lines = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([str(r["Method"])] + [_fmt(r[c]) for c in COLS[1:]]) + " |")
    with open(base_path + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)
    print(f">>> 写入 {base_path}.csv / .md")


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ NIPS34 交互量大(1.38M),建议 GPU 服务器。")

    # pin λ 扫描为单值（DER++/X-DER 的 mem 由 cl_baselines.MEM_SIZE=5000 提供）
    # clbase.EWC_LAMBDA_SWEEP = [EWC_LAMBDA]
    # clbase.CLORA_LAMBDA_SWEEP = [CLORA_LAMBDA]
    # clbase.MEM_SIZE = MEM

    Q_path = os.path.join(DATA_DIR, f"{PREFIX}_Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, 0.34)
    tr, va, te = (
        os.path.join(DATA_DIR, f"{PREFIX}_{s}_0.8_0.1_0.1.csv") for s in ("train", "valid", "test")
    )

    cfg = {
        "name": PREFIX,
        "train": tr,
        "valid": va,
        "test": te,
        "Q": Q_path,
        "n_item": N_ITEM,
        "n_know": N_KNOW,
        "new_concepts": "auto",
    }

    # ── 三个 CL 基线（固定 λ）────────────────────────────────────────────
    # meta = clbase.load_random(cfg)
    # ewc_row = clbase.run_ewc(meta, device)[0]  # 单值 λ → 单行
    # der_row = clbase.run_der(meta, device)
    # clora_row = clbase.run_clora(meta, device)[0]

    # ── X-DER（同 G-NCDM 骨干, buf 口径）─────────────────────────────────
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
        n_epoch=15,  # ← 修改为 15 epochs
    )

    baseline_rows = [
        # {k: ewc_row[k] for k in COLS},
        # {k: der_row[k] for k in COLS},
        # {k: clora_row[k] for k in COLS},
        {k: xder_row[k] for k in COLS},
    ]
    note_base = (
        f"\n*口径*：{PREFIX} random_split，固定超参 X-DER mem={MEM}；alpha={ALPHA}。\n"
        "*RD*：X-DER 在 G-NCDM 概念 θ 空间。\n"
    )
    _write_table(
        baseline_rows, os.path.join(SAVE_DIR, f"baselines_{PREFIX}_random_split"), note_base
    )

    # ── 若 Ours CSV 存在 → 合成 10 行 all_methods 总表 ───────────────────
    if os.path.isfile(OURS_CSV):
        ours = pd.read_csv(OURS_CSV).rename(columns={"Model": "Method", "TMD": "RD"})
        ours_rows = ours.to_dict("records")
        all_rows = ours_rows + baseline_rows
        note_all = (
            f"\n*口径*：{PREFIX} random_split（test 用户与训练共享，预测口径）。Ours/X-DER 走 G-NCDM "
            "骨干 buf 无泄漏预测；均无自信息，可逐行对比。\n"
            f"*固定超参*：X-DER mem={MEM}。\n"
            "*RD 红线*：Ours/X-DER 行 RD 同在 G-NCDM 概念 θ 空间可互比。\n"
        )
        _write_table(
            all_rows, os.path.join(SAVE_DIR, f"all_methods_{PREFIX}_random_split"), note_all
        )
        print(
            f"\n完成：all_methods_{PREFIX}_random_split.csv（Ours 6 + X-DER = {len(all_rows)} 行）"
        )
    else:
        print(
            f"\n⚠️ 未找到 Ours CSV：{OURS_CSV}\n"
            f"   已写出 X-DER 单行表 baselines_{PREFIX}_random_split.csv。\n"
            "   把你的 Ours 部分结果（列首 Model）放到上述路径后重跑即可自动合成 all_methods 总表。"
        )


if __name__ == "__main__":
    main()
