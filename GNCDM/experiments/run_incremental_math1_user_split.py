# -*- coding: utf-8 -*-
"""math1 · user_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。alpha=0.70。

用户互斥 → 统一 support/query 冷启动口径。eval_all_methods_user_split.run_one()
在同一份 support/query 上一次跑完 6 Ours + 3 基线。
产物：incremental_result/all_methods_math1_user_split.{csv,md}
运行（需 avalanche）：cd GNCDM/experiments && python run_incremental_math1_user_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_all_methods_user_split import run_one as user_split_all_methods

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(gncdm_dir, "data")
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHA = 0.70


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    user_split_all_methods(
        "math1_user_split",
        {
            "train": os.path.join(repo_root, "data", "math1", "user_split", "train.csv"),
            "valid": os.path.join(repo_root, "data", "math1", "user_split", "valid.csv"),
            "test": os.path.join(repo_root, "data", "math1", "user_split", "test.csv"),
            "Q": os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
            "n_user": 4209,
            "n_item": 20,
            "n_know": 11,
            "new_concepts": NEW_CONCEPTS,
            "alpha": ALPHA,
        },
        device,
    )
    print("\n完成：incremental_result/all_methods_math1_user_split.csv")


if __name__ == "__main__":
    main()
