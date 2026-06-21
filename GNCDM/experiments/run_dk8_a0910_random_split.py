# -*- coding: utf-8 -*-
"""a0910 · random_split · ΔK=8 补跑（DNA vs LoRA）

补跑 verify_lora_vs_dna_scaling.py 的 deltak 扫描中缺失的 dk=8 点。
产物格式与已有的 dk16/dk32/dk64 结果一致（每个 alpha×seed 一个独立 CSV）：
    incremental_result/incremental_results_a0910_dk8_f1.0_a{alpha}_s{seed}.csv

使用最冷门的 8 个概念作为 ΔK（与原脚本 select_cold_concepts 一致）。
仅跑 Ours(Dynamic DNA) 与 Ours(LoRA) 两种策略（Base 为扩展基座仍需跑）。
运行：cd GNCDM/experiments && python run_dk8_a0910_random_split.py
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

from run_incremental_math1 import SAVE_DIR, run_experiment, set_seed

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123

DELTA_K = 8
FRAC = 1.0
ALPHAS = [0.1, 0.9]
SEEDS = [1, 7, 42]

DNA_KEY = "Ours (Dynamic DNA)"
LORA_KEY = "Ours (LoRA)"


def select_cold_concepts(Q, n):
    """最冷门的 n 个概念（按被多少题使用升序），与原 verify_lora_vs_dna_scaling 一致。"""
    freq = (Q > 0).sum(axis=0)
    order = np.argsort(freq)
    return sorted(int(k) for k in order[:n])


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，建议 GPU 服务器。")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = select_cold_concepts(Q, DELTA_K)
    print(f"ΔK={DELTA_K} 最冷门概念: {new_concepts}")

    rnd = os.path.join(DATA_DIR, "new_random_split")
    train_path = os.path.join(rnd, "train.csv")
    valid_path = os.path.join(rnd, "valid.csv")
    test_path = os.path.join(rnd, "test.csv")

    summary_rows = []

    for alpha in ALPHAS:
        for seed in SEEDS:
            split_name = f"a0910_dk{DELTA_K}_f{FRAC}_a{alpha}_s{seed}"
            out_path = os.path.join(SAVE_DIR, f"incremental_results_{split_name}.csv")
            if os.path.exists(out_path):
                print(f"[跳过] {split_name} 已存在")
                existing = pd.read_csv(out_path)
                summary_rows.append((split_name, alpha, seed, existing))
                continue

            print(f"\n>>> ΔK={DELTA_K} frac={FRAC} alpha={alpha} seed={seed}")
            set_seed(seed)
            run_experiment(
                split_name,
                "buf",
                train_path,
                valid_path,
                test_path,
                Q_path,
                device,
                n_user=N_USER,
                n_item_total=N_ITEM,
                n_know_total=N_KNOW,
                new_concepts=new_concepts,
                alpha=alpha,
                run_strategies={DNA_KEY, LORA_KEY},
            )
            df = pd.read_csv(out_path)
            summary_rows.append((split_name, alpha, seed, df))

    # 汇总打印
    print("\n" + "=" * 70)
    print(f"ΔK={DELTA_K} 汇总（DNA vs LoRA · AUC_new）")
    print("=" * 70)
    print(f"{'split':>36} {'method':>22} {'AUC_new':>8}")
    for split_name, alpha, seed, df in summary_rows:
        for key in (DNA_KEY, LORA_KEY):
            sub = df[df["Model"] == key] if "Model" in df.columns else df[df["method"] == key]
            if len(sub):
                auc = sub.iloc[0].get("AUC_new", sub.iloc[0].get("auc_new", float("nan")))
                print(f"  {split_name:>36} {key:>22} {float(auc):.4f}")
    print("=" * 70)
    print(f"\n完成：所有结果在 {SAVE_DIR}/incremental_results_a0910_dk{DELTA_K}_*.csv")


if __name__ == "__main__":
    main()
