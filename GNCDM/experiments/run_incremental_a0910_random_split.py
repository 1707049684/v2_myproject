# -*- coding: utf-8 -*-
"""a0910 · random_split 九方法表（6 Ours + EWC/DER++/C-LoRA）。alpha=0.9。

test 用户与训练共享 → 预测口径。run_experiment(buf) 出 6 Ours →
cl_baselines_random_split.run_one() 跑 3 基线直接预测并合并。ΔK 用 auto_new_concepts(0.34)。
产物：incremental_result/all_methods_a0910_random_split.{csv,md}
a0910 题量大(17746)，务必 GPU 服务器（需 avalanche）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts
import cl_baselines_random_split as clbase

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.9


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，建议 GPU 服务器。")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, 0.34)
    rnd = os.path.join(DATA_DIR, "new_random_split")
    tr, va, te = (os.path.join(rnd, f) for f in ("train.csv", "valid.csv", "test.csv"))

    set_seed(42)
    run_experiment("a0910_random_split", "buf", tr, va, te, Q_path, device,
                   n_user=N_USER, n_item_total=N_ITEM, n_know_total=N_KNOW,
                   new_concepts=new_concepts, alpha=ALPHA)
    clbase.run_one({
        "name": "a0910",
        "train": tr, "valid": va, "test": te, "Q": Q_path,
        "n_item": N_ITEM, "n_know": N_KNOW, "new_concepts": "auto",
        "ours_csv": "incremental_results_a0910_random_split.csv",
    }, device)
    print("\n完成：incremental_result/all_methods_a0910_random_split.csv")


if __name__ == "__main__":
    main()
