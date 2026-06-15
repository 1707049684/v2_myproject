# -*- coding: utf-8 -*-
"""X-DER 基线 · math1 · random_split（G-NCDM 骨干, alpha=0.20）。

只跑 X-DER 一种方法,产出 incremental_result/xder_math1_random_split.{csv,md}（单行,列同 all_methods）。
不改 all_methods 主表;合并交给调用方/人工。
运行：cd GNCDM/experiments && python run_xder_math1_random_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_xder import run_xder

DATA_DIR = os.path.join(gncdm_dir, "data")
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHA = 0.20


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    run_xder(
        split_name="math1_random_split",
        ds_name="math1",
        train_path=os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"),
        valid_path=os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"),
        test_path=os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv"),
        Q_path=os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
        device=device,
        n_user=4209,
        n_item_total=20,
        n_know_total=11,
        new_concepts=NEW_CONCEPTS,
        alpha=ALPHA,
    )
    print("\n完成：incremental_result/xder_math1_random_split.csv")


if __name__ == "__main__":
    main()
