# -*- coding: utf-8 -*-
"""增量学习主实验 —— Junyi 数据集（topic 级共享概念，~5000 users × ~700 items × ~40 concepts）。

复用 run_incremental_math1.py 的管线与 run_incremental_a0910.py 的 auto_new_concepts。
概念来自 junyi 原始 exercise `topic`（多对一共享），由 experiments/_prep_junyi.py 生成四件套：
  data/junyi/Q_matrix.npy + new_random_split/ + new_user_split/。
维度从文件读取（不硬编码）。random_split 走 forward_using_buf 预测（RQ2），
user_split 走 forward 重构（RQ1）。
"""
import os
import sys

gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import numpy as np
import pandas as pd

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "junyi")

# alpha：先用初值 0.9 跑通（与 a0910 random 一致）；per-split 最优留作后续 sweep。
ALPHA = 0.9


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    n_item, n_know = Q.shape

    # n_user：取所有划分文件里的最大 user_id + 1
    n_user = 0
    for split in ("new_random_split", "new_user_split"):
        for f in ("train.csv", "valid.csv", "test.csv"):
            p = os.path.join(DATA_DIR, split, f)
            n_user = max(n_user, int(pd.read_csv(p)["user_id"].max()) + 1)

    new_concepts = auto_new_concepts(Q, new_item_frac=0.34)
    touched = (Q[:, new_concepts] > 0).sum(axis=1) > 0
    print(f"dims: n_user={n_user} n_item={n_item} n_know={n_know}")
    print(f"自动 ΔK：新概念={len(new_concepts)}/{n_know}，"
          f"新题={int(touched.sum())} 旧题={int((~touched).sum())}（旧题不依赖新概念）")

    splits = [
        ("junyi_random_split", "buf",
         os.path.join(DATA_DIR, "new_random_split", "train.csv"),
         os.path.join(DATA_DIR, "new_random_split", "valid.csv"),
         os.path.join(DATA_DIR, "new_random_split", "test.csv")),
        ("junyi_user_split", "recon",
         os.path.join(DATA_DIR, "new_user_split", "train.csv"),
         os.path.join(DATA_DIR, "new_user_split", "valid.csv"),
         os.path.join(DATA_DIR, "new_user_split", "test.csv")),
    ]
    for split_name, mode, tr, va, te in splits:
        set_seed(42)
        run_experiment(split_name, mode, tr, va, te, Q_path, device,
                       n_user=n_user, n_item_total=n_item, n_know_total=n_know,
                       new_concepts=new_concepts, alpha=ALPHA)


if __name__ == "__main__":
    main()
