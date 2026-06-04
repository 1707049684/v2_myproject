# -*- coding: utf-8 -*-
"""DER++ 持续学习基线（Buzzega et al., NeurIPS 2020）—— ASSIST a0910，random_split。

用于和 a0910 六大策略（incremental_results_a0910_random_split.csv）对比。
划分/口径对齐 run_incremental_a0910.py，骨干刻意沿用简单 CognitiveBackbone（非 G-NCDM）。

运行（需先装 avalanche；务必在 GNCDM/ 根目录执行，建议 GPU 服务器——a0910 有 17746 题）：
    pip install avalanche-lib
    cd GNCDM
    python a0910_der_baseline.py

划分与口径（已对齐 a0910 主实验，random_split 预测口径）：
- ΔK 用 auto_new_concepts(Q, 0.34) 自动选最冷门概念（与主实验同函数、同 frac）→ 严格拓扑二分，
  旧题绝不依赖任一新概念；Task0=旧题基线、Task1=新题增量。
- 数据用 data/a0910/new_random_split/{train,test}.csv → 旧/新任务定义与测试样本与
  incremental_results_a0910_random_split.csv 逐行一致。

可比性边界（同 math1_der_baseline.py）：
- AUC/ACC/F1/RMSE：apples-to-apples，可直接与主表对比（同划分、同测试行、同预测口径）。
- TMD：骨干为 CognitiveBackbone（Embedding+MLP，无概念空间 θ），TMD 在学生 embedding 空间度量，
  **绝对量级不可与 G-NCDM 的概念 θ TMD 直接比**，仅可用于定性/相对趋势（DER 有漂移、非零）。
- 骨干刻意未对齐 G-NCDM（仅作"被认可的顶会 CL 基线"对照），论文不强调 DER 跑在 G-NCDM 架构上。
"""
import os
import sys
import math

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

# Avalanche 组件
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import DER

# 复用 math1 版的骨干网络与评测函数（单一真源，避免重复实现）
from math1_der_baseline import CognitiveBackbone, evaluate_cd_metrics

# 复用主实验的「严格拓扑二分」与 a0910 的「自动 ΔK / 维度常量」作为唯一划分真源
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
import run_incremental_math1 as R          # strict_bipartition / remap_items
import run_incremental_a0910 as A          # auto_new_concepts / N_USER,N_ITEM,N_KNOW,ALPHA / DATA_DIR

N_ITEM_TOTAL = A.N_ITEM   # 17746
N_KNOW_TOTAL = A.N_KNOW   # 123
NEW_ITEM_FRAC = 0.34      # 与 run_incremental_a0910.main() 一致


# ==========================================
# 数据流装载：a0910 random_split + 自动 ΔK 的严格拓扑二分
# ==========================================
def load_a0910_strict_partition():
    Q_path = os.path.join(A.DATA_DIR, "Q_matrix.npy")
    base = os.path.join(A.DATA_DIR, "new_random_split")
    train_path = os.path.join(base, "train.csv")
    test_path = os.path.join(base, "test.csv")
    print(f">>> 加载 {train_path} / {test_path}（a0910 random_split）")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    # 自动选新概念 ΔK（最冷门概念，使新题≈frac），再用主实验同款严格拓扑二分重排。
    new_concepts = A.auto_new_concepts(Q_mat, new_item_frac=NEW_ITEM_FRAC)
    Q_mat, item_id_map, n_item_old, n_know_old = R.strict_bipartition(Q_mat, new_concepts)
    df_train = R.remap_items(df_train, item_id_map)
    df_test = R.remap_items(df_test, item_id_map)
    n_item_new = N_ITEM_TOTAL - n_item_old
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(f">>> 自动 ΔK：新概念={len(new_concepts)}/{N_KNOW_TOTAL}，"
          f"旧题(Task 0)={n_item_old} 新题(Task 1)={n_item_new}, 旧概念={n_know_old}")

    # 双阶段任务流：Task0=旧题(item_id<n_item_old)，Task1=新题；沿用既有 train/test 行。
    train_datasets, test_datasets = [], []
    for lo, hi in [(0, n_item_old), (n_item_old, N_ITEM_TOTAL)]:
        tr = df_train[(df_train["item_id"] >= lo) & (df_train["item_id"] < hi)]
        te = df_test[(df_test["item_id"] >= lo) & (df_test["item_id"] < hi)]
        x_tr = torch.tensor(tr[["user_id", "item_id"]].values, dtype=torch.long)
        y_tr = torch.tensor(tr["score"].values, dtype=torch.long)
        x_te = torch.tensor(te[["user_id", "item_id"]].values, dtype=torch.long)
        y_te = torch.tensor(te["score"].values, dtype=torch.long)
        train_datasets.append(TensorDataset(x_tr, y_tr))
        test_datasets.append(TensorDataset(x_te, y_te))

    benchmark = dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)
    num_students = int(max(df_train["user_id"].max(), df_test["user_id"].max())) + 1
    old_user_ids = torch.tensor(
        df_train[df_train["item_id"] < n_item_old]["user_id"].unique(), dtype=torch.long)
    meta = {
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "num_students": num_students,
        "num_items": N_ITEM_TOTAL,
        "old_user_ids": old_user_ids,
    }
    return benchmark, meta


# ==========================================
# 核心训练流与决战数据提取
# ==========================================
def run_der_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ 检测到 CPU：a0910 题量大(17746)，DER 训练会很慢，建议 GPU 服务器。")

    benchmark, meta = load_a0910_strict_partition()

    # embedding 尺寸按真实数据取（a0910：~4163 用户 / 17746 题）
    model = CognitiveBackbone(num_students=meta["num_students"],
                              num_items=meta["num_items"]).to(device)
    embed_dim = model.student_emb.weight.shape[1]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # mem_size：replay buffer 容量（a0910 数据量大，可按需上调；500 为与 math1 一致的保守默认）
    cl_strategy = DER(
        model, optimizer, criterion,
        mem_size=500, alpha=0.1, beta=0.5,
        train_mb_size=128, train_epochs=10,
        eval_mb_size=256, device=device
    )

    print("\n>>> 启动 DER++ 增量训练（a0910 random_split）...")

    baseline_student_emb = None
    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        task_name = "【旧知识基线】" if task_id == 0 else "【新知识增量】"
        print(f"\n--- 开始训练 Task {task_id} {task_name} ---")
        cl_strategy.train(experience)

        curr_test_ds = benchmark.test_stream[task_id].dataset
        auc, _, _, _ = evaluate_cd_metrics(model, curr_test_ds, device)
        print(f"Task {task_id} 收敛后巅峰 AUC: {auc:.4f}")

        if task_id == 0:
            baseline_student_emb = model.student_emb.weight.data.clone().cpu()

    print("\n" + "=" * 60)
    print(" 终极评估：生成论文决战表格数据（a0910 random_split）")
    print("=" * 60)

    new_dataset = benchmark.test_stream[1].dataset
    auc_new, rmse_new, acc_new, f1_new = evaluate_cd_metrics(model, new_dataset, device)
    old_ds = benchmark.test_stream[0].dataset
    auc_old, rmse_old, acc_old, f1_old = evaluate_cd_metrics(model, old_ds, device)

    # TMD：旧表征学完新任务后的漂移（学生 embedding 空间，整维度，按 √dim 归一化，仅旧任务学生）。
    # ⚠️ 与 G-NCDM 概念 θ 的 TMD 不在同一空间，绝对量级不可直接比，仅看相对趋势。
    final_student_emb = model.student_emb.weight.data.cpu()
    old_ids = meta["old_user_ids"]
    drift = torch.norm(baseline_student_emb[old_ids] - final_student_emb[old_ids], p=2, dim=1)
    tmd = (drift / math.sqrt(embed_dim)).mean().item()

    print("\n请直接将以下数据复制填入你的论文表格中：\n")
    print("| Model | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD (embedding-space, 量级不可与 G-NCDM 直接比) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(f"| DER++ | {auc_old:.4f} | {auc_new:.4f} | {rmse_old:.4f} | {rmse_new:.4f} | {acc_old:.4f} | {acc_new:.4f} | {f1_old:.4f} | {f1_new:.4f} | {tmd:.4f} |")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    run_der_baseline()
