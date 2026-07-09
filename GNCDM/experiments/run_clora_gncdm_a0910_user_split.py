# -*- coding: utf-8 -*-
"""G-NCDM + C-LoRA · a0910 · user_split（support/query 评测，alpha=0.6）。

薄入口：调 gncdm_clora_baseline.run_user_split()。
a0910 题量大(17746)，务必 GPU。

运行：cd GNCDM/experiments && python run_clora_gncdm_a0910_user_split.py
产物：incremental_result/clora_gncdm_lambda_sweep_a0910_user_split.csv
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(HERE)
for p in (GNCDM_DIR, HERE, os.path.join(HERE, "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import gncdm_clora_baseline as CL

repo_root = os.path.dirname(GNCDM_DIR)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
USR = os.path.join(DATA_DIR, "new_user_split")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.6  # 对齐 run_incremental_a0910_user_split.py


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，建议 GPU 服务器。")

    cfg = {
        "name": "a0910",
        "n_user": N_USER,
        "n_item": N_ITEM,
        "n_know": N_KNOW,
        "alpha": ALPHA,
        "new_concepts": "auto",
        "train": os.path.join(USR, "train.csv"),
        "valid": os.path.join(USR, "valid.csv"),
        "test": os.path.join(USR, "test.csv"),
        "Q": os.path.join(DATA_DIR, "Q_matrix.npy"),
    }
    CL.run_user_split(cfg, device, split_tag="user_split")
    print("\n完成：incremental_result/clora_gncdm_lambda_sweep_a0910_user_split.csv")


if __name__ == "__main__":
    main()
