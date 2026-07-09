# -*- coding: utf-8 -*-
"""junyi · random_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。alpha=0.1。

topic 级共享概念，ReliCD/QCCDM 对齐稠密版 1000×712×39（人均 ~204 作答）。
test 用户与训练共享 → 预测口径。alpha 由 sweep_junyi_random_alpha.py 全扫
0.1~0.95、按 DNA mean(valid ACC) 选定（0.1 见顶 sel_DNA_validACC=0.7704）。
run_experiment(buf) 出 6 Ours → cl_baselines_random_split.run_one() 跑 3 基线并合并。
维度从文件读；ΔK 用 auto_new_concepts(0.34)。
产物：incremental_result/all_methods_junyi_random_split.{csv,md}（需 avalanche）。
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

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts
import cl_baselines_random_split as clbase

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "junyi")
ALPHA = 0.1


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    n_item, n_know = int(Q.shape[0]), int(Q.shape[1])
    rnd = os.path.join(DATA_DIR, "new_random_split")
    tr, va, te = (os.path.join(rnd, f) for f in ("train.csv", "valid.csv", "test.csv"))
    n_user = max(int(pd.read_csv(f)["user_id"].max()) + 1 for f in (tr, va, te))
    new_concepts = auto_new_concepts(Q, 0.34)
    print(f"dims: n_user={n_user} n_item={n_item} n_know={n_know}")

    set_seed(42)
    run_experiment(
        "junyi_random_split",
        "buf",
        tr,
        va,
        te,
        Q_path,
        device,
        n_user=n_user,
        n_item_total=n_item,
        n_know_total=n_know,
        new_concepts=new_concepts,
        alpha=ALPHA,
    )
    clbase.run_one(
        {
            "name": "junyi",
            "train": tr,
            "valid": va,
            "test": te,
            "Q": Q_path,
            "n_item": n_item,
            "n_know": n_know,
            "new_concepts": "auto",
            "ours_csv": "incremental_results_junyi_random_split.csv",
        },
        device,
    )
    print("\n完成：incremental_result/all_methods_junyi_random_split.csv")


if __name__ == "__main__":
    main()
