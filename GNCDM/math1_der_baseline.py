# -*- coding: utf-8 -*-
"""DER++ 持续学习基线（Buzzega et al., NeurIPS 2020），用于和六大策略对比。

运行（需先装 avalanche；务必在 GNCDM/ 根目录执行，脚本会把 experiments/ 加入 sys.path 复用主实验划分）：
    pip install avalanche-lib
    cd GNCDM
    python math1_der_baseline.py

划分与口径（已对齐主实验，random_split 预测口径）：
- 复用 run_incremental_math1.strict_bipartition（ΔK={0,1,3,6} → 旧题 13 / 新题 7），
  并使用既有 math1_train/test_0.8_0.2.csv → Task0/Task1 的旧/新任务定义与测试样本
  与六策略 incremental_results_math1_random_split.csv 逐行一致。
- 2 阶段 task stream：Task0=旧题基线、Task1=新题增量；学完 Task1 后回测旧题得 *_old 指标。

可比性边界（重要，避免误读）：
- AUC/ACC/F1/RMSE：apples-to-apples，可直接与主表对比（同划分、同测试行、同预测口径）。
- TMD：本基线骨干为 CognitiveBackbone（Embedding+MLP，非 G-NCDM），无概念空间 θ，
  TMD 在 64 维学生 embedding 空间度量，**绝对量级不可与 G-NCDM 的 7 维概念 θ TMD 直接比**，
  仅可用于定性/相对趋势（DER 有漂移、非零，对照 DNA 的 TMD=0）。
- 骨干刻意未对齐 G-NCDM（仅作"被认可的顶会 CL 基线"对照），论文不强调 DER 跑在 G-NCDM 架构上。
"""
import os
import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Avalanche 组件
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import DER

# 引入认知诊断核心评测算子
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# 复用主实验的「严格拓扑二分」作为唯一划分真源，保证 DER 的旧/新任务定义、测试样本
# 与 run_incremental_math1.py 的六策略逐行一致（否则数字不可比）。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
import run_incremental_math1 as R

# 与主实验一致的数据集常量（math1：20 题 / 11 概念，ΔK={0,1,3,6} → 旧题 13 / 新题 7）
NEW_CONCEPTS = [0, 1, 3, 6]
N_ITEM_TOTAL = 20
N_KNOW_TOTAL = 11

# ==========================================
# 1. 认知诊断专用的基础骨干网络 (Backbone)
# ==========================================
class CognitiveBackbone(nn.Module):
    def __init__(self, num_students, num_items, embed_dim=64):
        super(CognitiveBackbone, self).__init__()
        self.student_emb = nn.Embedding(num_students, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        stu_idx = x[:, 0]
        item_idx = x[:, 1]
        u = self.student_emb(stu_idx)
        v = self.item_emb(item_idx)
        cat_features = torch.cat([u, v], dim=-1)
        logits = self.mlp(cat_features)
        return logits

# ==========================================
# 2. 测量学专用评测函数 (拦截计算 AUC/RMSE/ACC/F1)
# ==========================================
def evaluate_cd_metrics(model, test_dataset, device):
    """提取单个任务的 CD 指标"""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    y_trues, y_probs, y_preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1] # 取预测为 1 的概率
            preds = torch.argmax(logits, dim=1)
            
            y_trues.extend(y.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())
            
    # 防止单个 batch 中只有一类标签导致 AUC 报错
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except ValueError:
        auc = 0.5 
        
    rmse = mean_squared_error(y_trues, y_probs)**0.5
    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, zero_division=0)
    
    return auc, rmse, acc, f1

# ==========================================
# 3. 数据流装载：复用主实验「严格拓扑二分」(ΔK={0,1,3,6}) 并使用既有 train/test 文件
#    —— 旧/新任务定义与测试样本与 run_incremental_math1.py 六策略逐行一致（random_split 口径）。
# ==========================================
def load_math1_strict_partition():
    Q_path = os.path.join(R.DATA_DIR, "math1_Q_matrix.npy")
    train_path = os.path.join(R.DATA_DIR, "math1_train_0.8_0.2.csv")
    test_path = os.path.join(R.DATA_DIR, "math1_test_0.8_0.2.csv")
    print(f">>> 加载 {train_path} / {test_path}（random_split）")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    # 严格拓扑二分（与主实验同一函数、同一 ΔK）：旧题在前、新题在后；旧题绝不依赖新概念。
    Q_mat, item_id_map, n_item_old, n_know_old = R.strict_bipartition(Q_mat, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_id_map)
    df_test = R.remap_items(df_test, item_id_map)
    n_item_new = N_ITEM_TOTAL - n_item_old
    print(f">>> 严格拓扑二分(同主实验, ΔK={NEW_CONCEPTS}): "
          f"旧题(Task 0)={n_item_old} 新题(Task 1)={n_item_new}, 旧概念={n_know_old}")

    # 双阶段任务流：Task0=旧题(item_id<n_item_old)，Task1=新题(item_id>=n_item_old)。
    # 训练/测试沿用既有 train/test 文件的行（不再脚本内重切），保证与主表同口径。
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
    # embedding 尺寸按真实 id 取（覆盖 train+test 出现的最大 id）
    num_students = int(max(df_train["user_id"].max(), df_test["user_id"].max())) + 1
    # 旧任务参与的学生集合（仅这些学生在 Task0 学过旧知识 → 用于衡量旧表征漂移 TMD）
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
# 4. 核心训练流与决战数据提取
# ==========================================
def run_der_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    benchmark, meta = load_math1_strict_partition()

    # embedding 尺寸按真实数据取（math1：4209 用户 / 20 题），不再硬编码 5000
    model = CognitiveBackbone(num_students=meta["num_students"],
                              num_items=meta["num_items"]).to(device)
    embed_dim = model.student_emb.weight.shape[1]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    cl_strategy = DER(
        model, optimizer, criterion,
        mem_size=500, alpha=0.1, beta=0.5, 
        train_mb_size=128, train_epochs=10, 
        eval_mb_size=256, device=device
    )

    print("\n>>> 启动 DER++ 增量训练...")
    
    # 追踪矩阵：用于计算你定义的流形欧式空间 TMD
    baseline_student_emb = None 
    peak_auc_history = {}

    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        task_name = "【旧知识基线】" if task_id == 0 else "【新知识增量】"
        print(f"\n--- 开始训练 Task {task_id} {task_name} ---")
        cl_strategy.train(experience)
        
        # 记录当前任务刚学完时的巅峰 AUC
        curr_test_ds = benchmark.test_stream[task_id].dataset
        auc, _, _, _ = evaluate_cd_metrics(model, curr_test_ds, device)
        peak_auc_history[task_id] = auc
        print(f"Task {task_id} 收敛后巅峰 AUC: {auc:.4f}")
        
        # 获取 Baseline 锚点 (T0 状态的特质向量)
        if task_id == 0:
            # 必须用 clone().detach() 将其从计算图中硬隔离出来并缓存
            baseline_student_emb = model.student_emb.weight.data.clone().cpu()

    print("\n" + "="*60)
    print(" 终极评估：生成论文决战表格数据")
    print("="*60)
    
    # 1. 计算 _new 指标 (即在刚学完的 Task 1 上的性能)
    new_dataset = benchmark.test_stream[1].dataset
    auc_new, rmse_new, acc_new, f1_new = evaluate_cd_metrics(model, new_dataset, device)
    
    # 2. 计算 _old 指标 (即学习完 Task 1 后，回过头去考 Task 0 旧题的性能)
    old_ds = benchmark.test_stream[0].dataset
    auc_old, rmse_old, acc_old, f1_old = evaluate_cd_metrics(model, old_ds, device)
    
    # =========================================================================
    # 3. TMD（Trait Manifold Drift）：旧表征在学完新任务后的漂移
    # -------------------------------------------------------------------------
    # 主实验（G-NCDM）的 TMD 比较的是**概念空间的诊断态 θ**（Theta_buf 的前 K_old 个概念维）。
    # CognitiveBackbone 没有概念对齐的 θ，只有 64 维学生 embedding，其前 7 维并非 7 个知识概念，
    # 故**不能**切 [:, :7] 充当 θ_old（原实现的做法是错误的）。
    # 这里改为度量「学生 trait embedding」在 **Task0 学过旧知识的学生** 上、**整维度**、按 √dim 归一化
    # 的漂移 —— 与 G-NCDM TMD「旧表征学完新任务后移动多少」概念一致。
    # ⚠️ 注意：两者不在同一表征空间（概念 θ∈[0,1]^7 vs 隐 embedding∈R^64），TMD 的**绝对量级不可直接对比**，
    #    可比的是**相对趋势**（DER 是否比 Naive-FT 漂移小、是否仍 >0 而非 DNA 的 0）。
    # =========================================================================
    final_student_emb = model.student_emb.weight.data.cpu()
    old_ids = meta["old_user_ids"]
    drift = torch.norm(baseline_student_emb[old_ids] - final_student_emb[old_ids], p=2, dim=1)
    tmd = (drift / math.sqrt(embed_dim)).mean().item()

    # 打印最终可直接填入论文的 Markdown 表格
    print("\n请直接将以下数据复制填入你的论文表格中：\n")
    print("| Model | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD (embedding-space, 量级不可与 G-NCDM 直接比) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(f"| DER++ | {auc_old:.4f} | {auc_new:.4f} | {rmse_old:.4f} | {rmse_new:.4f} | {acc_old:.4f} | {acc_new:.4f} | {f1_old:.4f} | {f1_new:.4f} | {tmd:.4f} |")
    print("\n" + "="*60)

if __name__ == '__main__':
    run_der_baseline()