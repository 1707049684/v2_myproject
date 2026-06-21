# -*- coding: utf-8 -*-
"""X-DER 基线 · a0910 · user_split（G-NCDM 骨干, alpha=0.6, ΔK=auto_new_concepts(0.34)）。

只跑 X-DER 一种方法，产出 incremental_result/xder_a0910_user_split.{csv,md}（单行，列同 all_methods）。
评测口径：support/query 留出（frac=0.5, seed=7），与 eval_all_methods_user_split 完全一致。
a0910 题量大(17746)，务必 GPU 服务器。
运行：cd GNCDM/experiments && python run_xder_a0910_user_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from run_xder import run_xder_user_split
from run_incremental_a0910 import auto_new_concepts

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.6  # a0910 user_split 最优 alpha


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，建议 GPU 服务器。")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    new_concepts = auto_new_concepts(np.load(Q_path), 0.34)
    usr = os.path.join(DATA_DIR, "new_user_split")

    run_xder_user_split(
        split_name="a0910_user_split",
        ds_name="a0910",
        train_path=os.path.join(usr, "train.csv"),
        valid_path=os.path.join(usr, "valid.csv"),
        test_path=os.path.join(usr, "test.csv"),
        Q_path=Q_path,
        device=device,
        n_user=N_USER,
        n_item_total=N_ITEM,
        n_know_total=N_KNOW,
        new_concepts=new_concepts,
        alpha=ALPHA,
    )
    print("\n完成：incremental_result/xder_a0910_user_split.csv")


if __name__ == "__main__":
    main()
