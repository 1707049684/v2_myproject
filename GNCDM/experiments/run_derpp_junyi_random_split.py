# -*- coding: utf-8 -*-
"""DER++ 基线 · junyi · random_split（G-NCDM 骨干, alpha=0.1, ΔK=auto_new_concepts(0.34)）。

只跑 DER++ 一种方法，产出 incremental_result/derpp_junyi_random_split.{csv,md}（单行，列同 all_methods）。
运行：cd GNCDM/experiments && python run_derpp_junyi_random_split.py
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

from run_derpp import run_derpp
from run_incremental_a0910 import auto_new_concepts

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "junyi")
ALPHA = 0.1


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    rnd = os.path.join(DATA_DIR, "new_random_split")
    n_item, n_know = int(Q.shape[0]), int(Q.shape[1])
    n_user = max(
        int(pd.read_csv(os.path.join(rnd, f))["user_id"].max()) + 1
        for f in ("train.csv", "valid.csv", "test.csv")
    )
    new_concepts = auto_new_concepts(Q, 0.34)

    run_derpp(
        split_name="junyi_random_split",
        ds_name="junyi",
        train_path=os.path.join(rnd, "train.csv"),
        valid_path=os.path.join(rnd, "valid.csv"),
        test_path=os.path.join(rnd, "test.csv"),
        Q_path=Q_path,
        device=device,
        n_user=n_user,
        n_item_total=n_item,
        n_know_total=n_know,
        new_concepts=new_concepts,
        alpha=ALPHA,
    )
    print("\n完成：incremental_result/derpp_junyi_random_split.csv")


if __name__ == "__main__":
    main()
