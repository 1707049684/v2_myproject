# -*- coding: utf-8 -*-
"""math1 · random_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。alpha=0.20。

test 用户与训练共享 → 预测口径。run_experiment(buf) 出 6 Ours →
cl_baselines_random_split.run_one() 跑 3 基线直接预测并合并。
产物：incremental_result/all_methods_math1_random_split.{csv,md}
运行（需 avalanche）：cd GNCDM/experiments && python run_incremental_math1_random_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_incremental_math1 import set_seed, run_experiment
import cl_baselines_random_split as clbase

DATA_DIR = os.path.join(gncdm_dir, "data")
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHA = 0.20


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, "math1_Q_matrix.npy")
    tr = os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv")
    va = os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv")
    te = os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv")

    set_seed(42)
    run_experiment(
        "math1_random_split",
        "buf",
        tr,
        va,
        te,
        Q_path,
        device,
        n_user=4209,
        n_item_total=20,
        n_know_total=11,
        new_concepts=NEW_CONCEPTS,
        alpha=ALPHA,
    )
    clbase.run_one(
        {
            "name": "math1",
            "train": tr,
            "valid": va,
            "test": te,
            "Q": Q_path,
            "n_item": 20,
            "n_know": 11,
            "new_concepts": NEW_CONCEPTS,
            "ours_csv": "incremental_results_math1_random_split.csv",
        },
        device,
    )
    print("\n完成：incremental_result/all_methods_math1_random_split.csv")


if __name__ == "__main__":
    main()
