# -*- coding: utf-8 -*-
"""增量学习主实验 —— ASSIST a0910 数据集（4163 users × 17746 items × 123 concepts）。

复用 run_incremental_math1.py 的管线（真实作答日志 + 双口径评测），仅替换数据集
维度/路径/ΔK。random_split 走 forward_using_buf 预测（RQ2），user_split 走 forward
重构（RQ1）。结果写 incremental_result/incremental_results_a0910_{random,user}_split.csv。

注意：a0910 有 17746 题，build_log_mat / evaluate_recon 会构建 (4163, 17746) 的稠密
作答矩阵（约 0.3 GB/个），建议在带 GPU 的服务器上跑（本机 CPU 会很慢）。

ΔK 选择：123 个概念无法手挑，用 auto_new_concepts 自动选「最冷门」概念作为新知识，
使新题占比≈1/3、旧题≈2/3，且严格拓扑保证旧题不依赖任一新概念。
"""
import os
import sys

gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import numpy as np

from run_incremental_math1 import set_seed, run_experiment

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")

# a0910 规格（对齐原仓库 scripts/gncdm_a0910_user_split.sh）
N_USER, N_ITEM, N_KNOW, ALPHA = 4163, 17746, 123, 0.9


def auto_new_concepts(Q, new_item_frac=0.34):
    """自动挑选新概念 ΔK：按「触及题目数」升序（最冷门优先）累加，
    直到触及这些概念的题目占比达到 new_item_frac。返回概念列索引列表。
    这样新题≈frac、旧题=其余，且旧题（不触及任一新概念）天然不依赖新概念。
    """
    n_item = Q.shape[0]
    freq = (Q > 0).sum(axis=0)            # 每个概念被多少题用到
    order = np.argsort(freq)              # 冷门概念在前
    touched = np.zeros(n_item, dtype=bool)
    new_set = []
    for k in order:
        new_set.append(int(k))
        touched |= (Q[:, k] > 0)
        if touched.sum() >= new_item_frac * n_item:
            break
    return sorted(new_set)


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ 检测到 CPU：a0910 题量大(17746)，建议在 GPU 服务器上运行。")

    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, new_item_frac=0.34)
    touched = (Q[:, new_concepts] > 0).sum(axis=1) > 0
    print(f"自动 ΔK：新概念={len(new_concepts)}/{N_KNOW}，"
          f"新题={int(touched.sum())} 旧题={int((~touched).sum())}（旧题不依赖新概念）")

    splits = [
        ("a0910_random_split", "buf",
         os.path.join(DATA_DIR, "new_random_split", "train.csv"),
         os.path.join(DATA_DIR, "new_random_split", "valid.csv"),
         os.path.join(DATA_DIR, "new_random_split", "test.csv")),
        ("a0910_user_split", "recon",
         os.path.join(DATA_DIR, "new_user_split", "train.csv"),
         os.path.join(DATA_DIR, "new_user_split", "valid.csv"),
         os.path.join(DATA_DIR, "new_user_split", "test.csv")),
    ]
    for split_name, mode, tr, va, te in splits:
        set_seed(42)
        run_experiment(split_name, mode, tr, va, te, Q_path, device,
                       n_user=N_USER, n_item_total=N_ITEM, n_know_total=N_KNOW,
                       new_concepts=new_concepts, alpha=ALPHA)


if __name__ == "__main__":
    main()
