# -*- coding: utf-8 -*-
"""junyi · user_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。alpha=0.6。

topic 级共享概念（1000×712×39 稠密版）。用户互斥 → 统一 support/query 冷启动口径。
eval_all_methods_user_split.run_one() 在同一份 support/query 上一次跑完 6 Ours + 3 基线。
维度从文件读；ΔK 用 auto_new_concepts(0.34)。
产物：incremental_result/all_methods_junyi_user_split.{csv,md}（需 avalanche）。
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
from eval_all_methods_user_split import run_one as user_split_all_methods

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "junyi")
ALPHA = 0.6


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    n_item, n_know = int(Q.shape[0]), int(Q.shape[1])
    usr = os.path.join(DATA_DIR, "new_user_split")
    tr, va, te = (os.path.join(usr, f) for f in ("train.csv", "valid.csv", "test.csv"))
    n_user = max(int(pd.read_csv(f)["user_id"].max()) + 1 for f in (tr, va, te))
    print(f"dims: n_user={n_user} n_item={n_item} n_know={n_know}")

    user_split_all_methods(
        "junyi_user_split",
        {
            "train": tr,
            "valid": va,
            "test": te,
            "Q": Q_path,
            "n_user": n_user,
            "n_item": n_item,
            "n_know": n_know,
            "new_concepts": auto_new_concepts(Q, 0.34),
            "alpha": ALPHA,
        },
        device,
    )
    print("\n完成：incremental_result/all_methods_junyi_user_split.csv")


if __name__ == "__main__":
    main()
