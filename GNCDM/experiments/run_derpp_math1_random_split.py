# -*- coding: utf-8 -*-
"""DER++ 基线 · Math1 · random_split（G-NCDM 骨干, alpha=0.20, ΔK={0,1,3,6}）。

只跑 DER++ 一种方法，产出 incremental_result/derpp_math1_random_split.{csv,md}（单行，列同 all_methods）。
运行：cd GNCDM/experiments && python run_derpp_math1_random_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_derpp import run_derpp

DATA_DIR = os.path.join(gncdm_dir, "data")
N_USER, N_ITEM, N_KNOW = 4209, 20, 11
ALPHA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    run_derpp(
        split_name="math1_random_split",
        ds_name="math1",
        train_path=os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"),
        valid_path=os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"),
        test_path=os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv"),
        Q_path=os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
        device=device,
        n_user=N_USER,
        n_item_total=N_ITEM,
        n_know_total=N_KNOW,
        new_concepts=NEW_CONCEPTS,
        alpha=ALPHA,
    )
    print("\n完成：incremental_result/derpp_math1_random_split.csv")


if __name__ == "__main__":
    main()
