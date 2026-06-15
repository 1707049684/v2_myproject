# -*- coding: utf-8 -*-
"""X-DER 基线 · junyi · random_split（G-NCDM 骨干, alpha=0.1, ΔK=auto_new_concepts(0.34)）。

稠密版 1000×712×39,维度从文件读。只跑 X-DER 一种方法,
产出 incremental_result/xder_junyi_random_split.{csv,md}（单行,列同 all_methods）。
运行：cd GNCDM/experiments && python run_xder_junyi_random_split.py
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

from run_xder import run_xder
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
    n_item, n_know = int(Q.shape[0]), int(Q.shape[1])
    rnd = os.path.join(DATA_DIR, "new_random_split")
    tr, va, te = (os.path.join(rnd, f) for f in ("train.csv", "valid.csv", "test.csv"))
    n_user = max(int(pd.read_csv(f)["user_id"].max()) + 1 for f in (tr, va, te))
    new_concepts = auto_new_concepts(Q, 0.34)
    print(f"dims: n_user={n_user} n_item={n_item} n_know={n_know}")

    run_xder(
        split_name="junyi_random_split",
        ds_name="junyi",
        train_path=tr,
        valid_path=va,
        test_path=te,
        Q_path=Q_path,
        device=device,
        n_user=n_user,
        n_item_total=n_item,
        n_know_total=n_know,
        new_concepts=new_concepts,
        alpha=ALPHA,
    )
    print("\n完成：incremental_result/xder_junyi_random_split.csv")


if __name__ == "__main__":
    main()
