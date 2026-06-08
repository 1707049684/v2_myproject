# -*- coding: utf-8 -*-
"""Junyi 增量主实验【单文件总调度】——一次跑出两个划分各自的「九方法」对比表。

数据：topic 级共享概念（5000 users × 707 items × 39 concepts），由 _prep_junyi.py 生成
（data/junyi/Q_matrix.npy + new_random_split/ + new_user_split/）。

两个划分口径不同，分别走对应可比协议（均复用既有脚本的函数，不重复实现）：
- junyi_random_split（alpha=0.9）：test 用户与训练共享 → 预测口径。
    先 run_experiment(buf) 出 6 个 Ours 策略 → incremental_results_junyi_random_split.csv；
    再 cl_baselines_random_split.run_one() 跑 EWC/DER++/C-LoRA 直接预测并合并
    → all_methods_junyi_random_split.{csv,md}。
- junyi_user_split（alpha=0.6）：用户互斥 → 统一 support/query 冷启动口径。
    eval_all_methods_user_split.run_one() 在同一份 support/query 上一次跑完 6 Ours + 3 基线
    → all_methods_junyi_user_split.{csv,md}。

ΔK 用同款 auto_new_concepts(0.34)，保证新旧题划分与 Ours 主表一致。
基线骨干为 CognitiveBackbone（非 G-NCDM、无 alpha）；TMD 量级不可与 Ours 直接比，仅看是否>0。

运行（需 avalanche 给 EWC/DER；建议 GPU 服务器）：
    pip install avalanche-lib
    cd GNCDM/experiments
    python run_incremental_junyi.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../GNCDM/experiments
gncdm_dir = os.path.dirname(HERE)                          # .../GNCDM
for p in (HERE, gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts
import cl_baselines_random_split as clbase                # random-split 三基线 + 合并
from eval_all_methods_user_split import run_one as user_split_all_methods  # user-split 九方法

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "junyi")

# 每个划分各自的 alpha（仅影响 Ours / G-NCDM 行；基线无 alpha）。
ALPHA = {"junyi_random_split": 0.9, "junyi_user_split": 0.6}


def _split_paths(split_dir):
    d = os.path.join(DATA_DIR, split_dir)
    return (os.path.join(d, "train.csv"),
            os.path.join(d, "valid.csv"),
            os.path.join(d, "test.csv"))


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    n_item, n_know = int(Q.shape[0]), int(Q.shape[1])

    # n_user：取所有划分文件里的最大 user_id + 1
    n_user = 0
    for split in ("new_random_split", "new_user_split"):
        for f in ("train.csv", "valid.csv", "test.csv"):
            n_user = max(n_user, int(pd.read_csv(
                os.path.join(DATA_DIR, split, f))["user_id"].max()) + 1)

    new_concepts = auto_new_concepts(Q, 0.34)
    touched = (Q[:, new_concepts] > 0).sum(axis=1) > 0
    print(f"dims: n_user={n_user} n_item={n_item} n_know={n_know}")
    print(f"自动 ΔK：新概念={len(new_concepts)}/{n_know}，"
          f"新题={int(touched.sum())} 旧题={int((~touched).sum())}（旧题不依赖新概念）")

    # ============================================================
    # 1) random_split：Ours 6 策略（buf 预测）→ CSV，再 3 基线直接预测并合并
    # ============================================================
    rnd_tr, rnd_va, rnd_te = _split_paths("new_random_split")
    set_seed(42)
    run_experiment("junyi_random_split", "buf", rnd_tr, rnd_va, rnd_te, Q_path, device,
                   n_user=n_user, n_item_total=n_item, n_know_total=n_know,
                   new_concepts=new_concepts, alpha=ALPHA["junyi_random_split"])
    rnd_cfg = {
        "name": "junyi",
        "train": rnd_tr, "valid": rnd_va, "test": rnd_te, "Q": Q_path,
        "n_item": n_item, "n_know": n_know, "new_concepts": "auto",
        "ours_csv": "incremental_results_junyi_random_split.csv",
    }
    clbase.run_one(rnd_cfg, device)   # → all_methods_junyi_random_split.{csv,md}

    # ============================================================
    # 2) user_split：同一份 support/query 上一次跑完 6 Ours + 3 基线
    # ============================================================
    usr_tr, usr_va, usr_te = _split_paths("new_user_split")
    usr_cfg = {
        "train": usr_tr, "valid": usr_va, "test": usr_te, "Q": Q_path,
        "n_user": n_user, "n_item": n_item, "n_know": n_know,
        "new_concepts": new_concepts, "alpha": ALPHA["junyi_user_split"],
    }
    user_split_all_methods("junyi_user_split", usr_cfg, device)  # → all_methods_junyi_user_split.{csv,md}

    print("\n全部完成：")
    print("  incremental_result/all_methods_junyi_random_split.csv")
    print("  incremental_result/all_methods_junyi_user_split.csv")


if __name__ == "__main__":
    main()
