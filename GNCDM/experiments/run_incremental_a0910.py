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
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
# per-split 最优 alpha（仅影响 Ours / G-NCDM 行；基线无 alpha）
ALPHA = {"a0910_random_split": 0.9, "a0910_user_split": 0.6}


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
    """单文件总调度：一次产出 a0910 两个划分各自的「九方法」对比表（6 Ours + 3 基线）。

    - a0910_random_split（alpha=0.9，预测口径）：run_experiment(buf) 出 6 Ours →
      cl_baselines_random_split.run_one() 跑 EWC/DER++/C-LoRA 直接预测并合并。
    - a0910_user_split（alpha=0.6，support/query 冷启动口径）：
      eval_all_methods_user_split.run_one() 一次跑完 6 Ours + 3 基线。
    需 avalanche；a0910 题量大(17746)，务必在 GPU 服务器上跑。
    """
    import torch
    import cl_baselines_random_split as clbase
    from eval_all_methods_user_split import run_one as user_split_all_methods

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

    rnd = os.path.join(DATA_DIR, "new_random_split")
    usr = os.path.join(DATA_DIR, "new_user_split")

    # 1) random_split：Ours 6 策略（buf 预测）→ CSV，再 3 基线直接预测并合并
    set_seed(42)
    run_experiment("a0910_random_split", "buf",
                   os.path.join(rnd, "train.csv"), os.path.join(rnd, "valid.csv"),
                   os.path.join(rnd, "test.csv"), Q_path, device,
                   n_user=N_USER, n_item_total=N_ITEM, n_know_total=N_KNOW,
                   new_concepts=new_concepts, alpha=ALPHA["a0910_random_split"])
    clbase.run_one({
        "name": "a0910",
        "train": os.path.join(rnd, "train.csv"), "valid": os.path.join(rnd, "valid.csv"),
        "test": os.path.join(rnd, "test.csv"), "Q": Q_path,
        "n_item": N_ITEM, "n_know": N_KNOW, "new_concepts": "auto",
        "ours_csv": "incremental_results_a0910_random_split.csv",
    }, device)

    # 2) user_split：同一份 support/query 上一次跑完 6 Ours + 3 基线
    user_split_all_methods("a0910_user_split", {
        "train": os.path.join(usr, "train.csv"), "valid": os.path.join(usr, "valid.csv"),
        "test": os.path.join(usr, "test.csv"), "Q": Q_path,
        "n_user": N_USER, "n_item": N_ITEM, "n_know": N_KNOW,
        "new_concepts": new_concepts, "alpha": ALPHA["a0910_user_split"],
    }, device)

    print("\n全部完成：incremental_result/all_methods_a0910_{random,user}_split.csv")


if __name__ == "__main__":
    main()
