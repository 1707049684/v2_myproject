# -*- coding: utf-8 -*-
"""a0910 · user_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。alpha=0.6。

用户互斥 → 统一 support/query 冷启动口径。eval_all_methods_user_split.run_one()
在同一份 support/query 上一次跑完 6 Ours + 3 基线。ΔK 用 auto_new_concepts(0.34)。
产物：incremental_result/all_methods_a0910_user_split.{csv,md}
a0910 题量大(17746)，务必 GPU 服务器（需 avalanche）。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from run_incremental_a0910 import auto_new_concepts
from eval_all_methods_user_split import run_one as user_split_all_methods

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.6


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，建议 GPU 服务器。")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    usr = os.path.join(DATA_DIR, "new_user_split")

    user_split_all_methods(
        "a0910_user_split",
        {
            "train": os.path.join(usr, "train.csv"),
            "valid": os.path.join(usr, "valid.csv"),
            "test": os.path.join(usr, "test.csv"),
            "Q": Q_path,
            "n_user": N_USER,
            "n_item": N_ITEM,
            "n_know": N_KNOW,
            "new_concepts": auto_new_concepts(Q, 0.34),
            "alpha": ALPHA,
        },
        device,
    )
    print("\n完成：incremental_result/all_methods_a0910_user_split.csv")


if __name__ == "__main__":
    main()
