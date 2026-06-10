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

gncdm_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # _core/→experiments/→GNCDM/
sys.path.insert(0, gncdm_dir)

import numpy as np

from run_incremental_math1 import set_seed, run_experiment

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")

# a0910 规格（对齐原仓库 scripts/gncdm_a0910_user_split.sh）
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
# per-split 最优 alpha 参考（仅影响 Ours / G-NCDM 行；基线无 alpha）。
# 注：权威值在各 per-split 入口脚本的 ALPHA 常量里硬编码，此 dict 仅备查、未被 import。
ALPHA = {"a0910_random_split": 0.1, "a0910_user_split": 0.6}


def auto_new_concepts(Q, new_item_frac=0.34):
    """自动挑选新概念 ΔK：按「触及题目数」升序（最冷门优先）累加，
    直到触及这些概念的题目占比达到 new_item_frac。返回概念列索引列表。
    这样新题≈frac、旧题=其余，且旧题（不触及任一新概念）天然不依赖新概念。
    """
    n_item = Q.shape[0]
    freq = (Q > 0).sum(axis=0)  # 每个概念被多少题用到
    order = np.argsort(freq)  # 冷门概念在前
    touched = np.zeros(n_item, dtype=bool)
    new_set = []
    for k in order:
        new_set.append(int(k))
        touched |= Q[:, k] > 0
        if touched.sum() >= new_item_frac * n_item:
            break
    return sorted(new_set)


# 本模块现作为库使用（导出 auto_new_concepts / N_USER 等，供拆分脚本与 eval 复用）。
# 跑实验请用拆分脚本（每个只跑一个划分，互不影响）：
#   python run_incremental_a0910_random_split.py   # alpha=0.9 → all_methods_a0910_random_split
#   python run_incremental_a0910_user_split.py     # alpha=0.6 → all_methods_a0910_user_split
if __name__ == "__main__":
    print(
        "本文件已拆分。请运行："
        "\n  python run_incremental_a0910_random_split.py"
        "\n  python run_incremental_a0910_user_split.py"
    )
