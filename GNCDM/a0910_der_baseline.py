# -*- coding: utf-8 -*-
"""DER++ 持续学习基线（Buzzega et al., NeurIPS 2020）—— ASSIST a0910，random_split。【单文件自包含】

用于和 a0910 六大策略（incremental_results_a0910_random_split.csv）对比。
本脚本不依赖 run_incremental_* / math1_der_baseline，所有划分/工具函数均已内联，便于在服务器直接跑。

运行（需先装 avalanche；建议 GPU 服务器——a0910 有 17746 题）：
    pip install avalanche-lib
    python a0910_der_baseline.py

划分与口径（已对齐 a0910 主实验 run_incremental_a0910.py，random_split 预测口径）：
- ΔK 用 auto_new_concepts(Q, 0.34) 自动选最冷门概念 → 严格拓扑二分，旧题绝不依赖任一新概念；
  Task0=旧题基线、Task1=新题增量。
- 数据用 data/a0910/new_random_split/{train,valid,test}.csv → 旧/新任务定义与测试样本与
  incremental_results_a0910_random_split.csv 逐行一致；valid 用于早停。

可比性边界（重要，避免误读）：
- AUC/ACC/F1/RMSE：apples-to-apples，可直接与主表对比（同划分、同测试行、同预测口径）。
- TMD：骨干为 CognitiveBackbone（Embedding+MLP，无概念空间 θ），TMD 在学生 embedding 空间度量，
  **绝对量级不可与 G-NCDM 的概念 θ TMD 直接比**，仅可用于定性/相对趋势（DER 有漂移、非零）。
- 骨干刻意未对齐 G-NCDM（仅作"被认可的顶会 CL 基线"对照），论文不强调 DER 跑在 G-NCDM 架构上。
"""
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

# Avalanche 组件
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import DER

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# ==========================================
# 路径与数据集常量（a0910，对齐 run_incremental_a0910.py）
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # GNCDM/
REPO_ROOT = os.path.dirname(THIS_DIR)                       # Generative-CD-main/
DATA_DIR = os.path.join(REPO_ROOT, "data", "a0910")

N_ITEM_TOTAL = 17746
N_KNOW_TOTAL = 123
NEW_ITEM_FRAC = 0.34       # 与 run_incremental_a0910.main() 一致

# ---- DER++ / 训练超参（便于在服务器上单独 sweep；避免弱化基线）----
# a0910 训练交互约 19 万条、旧任务约 13 万条 → replay buffer 不宜过小，否则旧任务复习不足。
EMBED_DIM = 64            # 骨干隐维（不改 CognitiveBackbone 架构，仅容量；可试 128 进一步增强基线）
LR = 1e-3
TRAIN_EPOCHS = 25         # 早停的最大轮数上限（与主实验 n_epoch=25 对齐）
EARLY_STOP_PATIENCE = 5   # 验证集 ACC 连续多少轮无提升则早停（防过拟合）
TRAIN_MB_SIZE = 128
MEM_SIZE = 5000           # replay buffer 容量（更充分复习旧任务）
DER_ALPHA = 0.5           # DER++ logit 蒸馏权重（加强旧表征保持）
DER_BETA = 0.5            # DER++ replay 标签 CE 权重


def set_seed(seed=42):
    """固定随机种子以保证可复现（对齐 EWC/主实验；DER 此前缺失导致 run 间结果跳动）。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# 内联工具函数（原 run_incremental_math1 / a0910 的同名实现，逐字搬入以自包含）
# ==========================================
def strict_bipartition(Q, new_concepts):
    """严格拓扑二分 + 重排：new_concepts 为新知识 ΔK；旧题=Q 在 ΔK 上全 0 的题。
    重排后旧概念列在前、旧题行在前。返回 (Q_re, item_id_map, n_item_old, n_know_old)。"""
    K = Q.shape[1]
    new_concepts = list(new_concepts)
    old_concepts = [k for k in range(K) if k not in new_concepts]
    concept_perm = old_concepts + new_concepts
    touches_new = Q[:, new_concepts].sum(axis=1) > 0
    old_items = np.where(~touches_new)[0].tolist()
    new_items = np.where(touches_new)[0].tolist()
    item_perm = old_items + new_items
    Q_re = Q[np.ix_(item_perm, concept_perm)].astype(np.float32)
    item_id_map = {old: new for new, old in enumerate(item_perm)}
    return Q_re, item_id_map, len(old_items), len(old_concepts)


def remap_items(df, item_id_map):
    df = df.copy()
    df["item_id"] = df["item_id"].map(item_id_map).astype(int)
    return df


def auto_new_concepts(Q, new_item_frac=0.34):
    """按「触及题目数」升序（最冷门优先）累加概念，直到触及题占比达 new_item_frac。返回概念列索引。"""
    n_item = Q.shape[0]
    freq = (Q > 0).sum(axis=0)
    order = np.argsort(freq)
    touched = np.zeros(n_item, dtype=bool)
    new_set = []
    for k in order:
        new_set.append(int(k))
        touched |= (Q[:, k] > 0)
        if touched.sum() >= new_item_frac * n_item:
            break
    return sorted(new_set)


# ==========================================
# 认知诊断专用骨干网络（简单 Embedding+MLP，非 G-NCDM）
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
            nn.Linear(64, 2),
        )

    def forward(self, x):
        u = self.student_emb(x[:, 0])
        v = self.item_emb(x[:, 1])
        return self.mlp(torch.cat([u, v], dim=-1))


def evaluate_cd_metrics(model, eval_dataset, device):
    """在单个 TensorDataset 上算 CD 指标（AUC/RMSE/ACC/F1）。"""
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    y_trues, y_probs, y_preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_trues.extend(y.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except ValueError:
        auc = 0.5
    rmse = mean_squared_error(y_trues, y_probs) ** 0.5
    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, zero_division=0)
    return auc, rmse, acc, f1


# ==========================================
# 数据流装载：a0910 random_split + 自动 ΔK 的严格拓扑二分
# ==========================================
def load_a0910_strict_partition():
    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    base = os.path.join(DATA_DIR, "new_random_split")
    train_path = os.path.join(base, "train.csv")
    valid_path = os.path.join(base, "valid.csv")
    test_path = os.path.join(base, "test.csv")
    print(f">>> 加载 {train_path} / {valid_path} / {test_path}（a0910 random_split）")

    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    # 自动选新概念 ΔK（最冷门），再严格拓扑二分重排。
    new_concepts = auto_new_concepts(Q_mat, new_item_frac=NEW_ITEM_FRAC)
    Q_mat, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, new_concepts)
    df_train = remap_items(df_train, item_id_map)
    df_valid = remap_items(df_valid, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    n_item_new = N_ITEM_TOTAL - n_item_old
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(f">>> 自动 ΔK：新概念={len(new_concepts)}/{N_KNOW_TOTAL}，"
          f"旧题(Task 0)={n_item_old} 新题(Task 1)={n_item_new}, 旧概念={n_know_old}")

    # 双阶段任务流：Task0=旧题(item_id<n_item_old)，Task1=新题；沿用既有 train/test 行。
    def _ds(df, lo, hi):
        sub = df[(df["item_id"] >= lo) & (df["item_id"] < hi)]
        x = torch.tensor(sub[["user_id", "item_id"]].values, dtype=torch.long)
        y = torch.tensor(sub["score"].values, dtype=torch.long)
        return TensorDataset(x, y)

    ranges = [(0, n_item_old), (n_item_old, N_ITEM_TOTAL)]   # [旧题, 新题]
    train_datasets = [_ds(df_train, lo, hi) for lo, hi in ranges]
    test_datasets = [_ds(df_test, lo, hi) for lo, hi in ranges]
    valid_old_ds = _ds(df_valid, 0, n_item_old)              # 早停监控用
    valid_new_ds = _ds(df_valid, n_item_old, N_ITEM_TOTAL)

    benchmark = dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)
    num_students = int(max(df_train["user_id"].max(), df_valid["user_id"].max(),
                           df_test["user_id"].max())) + 1
    old_user_ids = torch.tensor(
        df_train[df_train["item_id"] < n_item_old]["user_id"].unique(), dtype=torch.long)
    meta = {
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "num_students": num_students,
        "num_items": N_ITEM_TOTAL,
        "old_user_ids": old_user_ids,
        "valid_old_ds": valid_old_ds,
        "valid_new_ds": valid_new_ds,
    }
    return benchmark, meta


# ==========================================
# 核心训练流（按验证集 ACC 早停）与决战数据提取
# ==========================================
def run_der_baseline():
    set_seed(42)   # 可复现：固定随机种子（DER reservoir 采样 / 初始化 / shuffle）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    if device.type == "cpu":
        print("[WARN] 检测到 CPU：a0910 题量大(17746)，DER 训练会很慢，建议 GPU 服务器。")

    benchmark, meta = load_a0910_strict_partition()

    model = CognitiveBackbone(num_students=meta["num_students"],
                              num_items=meta["num_items"], embed_dim=EMBED_DIM).to(device)
    embed_dim = model.student_emb.weight.shape[1]
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # train_epochs=1：DER 内部只跑 1 epoch，外层手动按 valid ACC 早停（对齐主实验"保留最优快照"）
    cl_strategy = DER(
        model, optimizer, criterion,
        mem_size=MEM_SIZE, alpha=DER_ALPHA, beta=DER_BETA,
        train_mb_size=TRAIN_MB_SIZE, train_epochs=1,
        eval_mb_size=256, device=device
    )
    print(f">>> DER++ 超参: mem_size={MEM_SIZE} alpha={DER_ALPHA} beta={DER_BETA} "
          f"max_epochs={TRAIN_EPOCHS} patience={EARLY_STOP_PATIENCE} embed_dim={EMBED_DIM} lr={LR}")

    print("\n>>> 启动 DER++ 增量训练（a0910 random_split，按验证集 ACC 早停）...")

    baseline_student_emb = None
    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        task_name = "【旧知识基线】" if task_id == 0 else "【新知识增量】"
        # 早停验证口径：Task0 监控旧题 valid；Task1 学新+回放旧 → 监控旧+新合并 valid（兼顾保旧与学新）
        if task_id == 0:
            val_ds = meta["valid_old_ds"]
        else:
            val_ds = ConcatDataset([meta["valid_old_ds"], meta["valid_new_ds"]])
        print(f"\n--- 开始训练 Task {task_id} {task_name}"
              f"（最多 {TRAIN_EPOCHS} ep，patience={EARLY_STOP_PATIENCE}）---")

        best_acc, best_state, wait = -1.0, None, 0
        for epoch in range(TRAIN_EPOCHS):
            cl_strategy.train(experience)          # 每次跑 1 epoch
            _, _, val_acc, _ = evaluate_cd_metrics(model, val_ds, device)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            print(f"    [Task {task_id}] epoch {epoch + 1}/{TRAIN_EPOCHS} "
                  f"valid_acc={val_acc:.4f} best={best_acc:.4f} wait={wait}")
            if wait >= EARLY_STOP_PATIENCE:
                print(f"    早停于 epoch {epoch + 1}（valid_acc 连续 {EARLY_STOP_PATIENCE} 轮无提升）")
                break

        # 回到该任务验证集最优快照（再进入下一任务 / 最终评测）
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        curr_test_ds = benchmark.test_stream[task_id].dataset
        auc, _, _, _ = evaluate_cd_metrics(model, curr_test_ds, device)
        print(f"Task {task_id} 选优后 test AUC: {auc:.4f}")

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
