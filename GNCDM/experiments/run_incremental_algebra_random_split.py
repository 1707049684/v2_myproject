# -*- coding: utf-8 -*-
"""KDD Cup Algebra 2005-2006 · random_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。

数据：GNCDM/data/algebra0506kc_*（574 users × 1206 problems × 112 KC，
score=整题全部 step 首答全对；KDD Cup 2010 Algebra 05-06 经题级聚合）。
test 用户与训练共享 → buf 预测口径（论文 RQ2）。run_experiment(buf) 出 6 Ours →
cl_baselines_random_split.run_one() 跑 3 基线直接预测并合并。ΔK 用 auto_new_concepts(0.34)。
alpha 由 sweep_algebra_random_alpha.py 全扫 0.1~0.95、按 DNA mean(valid AUC) 选定。
产物：incremental_result/all_methods_algebra_random_split.{csv,md}
algebra 规模小（1206 题），CPU 也能跑（需 avalanche 给 EWC/DER）。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts
import cl_baselines_random_split as clbase

DATA_DIR = os.path.join(gncdm_dir, "data")
PREFIX = "algebra0506kc"
N_USER, N_ITEM, N_KNOW = 574, 1206, 112
ALPHA = 0.30  # sweep_algebra_random_alpha.py 全扫 0.1~0.95，DNA mean(valid AUC) 0.30 见顶(selAUC=0.8304)


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, f"{PREFIX}_Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, 0.34)
    tr, va, te = (
        os.path.join(DATA_DIR, f"{PREFIX}_{s}_0.8_0.1_0.1.csv") for s in ("train", "valid", "test")
    )

    set_seed(42)
    run_experiment(
        "algebra_random_split",
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
    clbase.run_one(
        {
            "name": "algebra",
            "train": tr,
            "valid": va,
            "test": te,
            "Q": Q_path,
            "n_item": N_ITEM,
            "n_know": N_KNOW,
            "new_concepts": "auto",
            "ours_csv": "incremental_results_algebra_random_split.csv",
        },
        device,
    )
    print("\n完成：incremental_result/all_methods_algebra_random_split.csv")


if __name__ == "__main__":
    main()
