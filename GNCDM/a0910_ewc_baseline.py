# -*- coding: utf-8 -*-
"""EWC 持续学习基线（Elastic Weight Consolidation, Kirkpatrick et al., PNAS 2017）—— a0910 random_split。【单文件自包含】

用于和 a0910 六大策略（incremental_results_a0910_random_split.csv）对比，并通过**扫描惩罚系数
λ_ewc** 展示 EWC 的「稳定性-可塑性窘境」：没有任何单一 λ 能同时保旧+学新，从而反衬 Ours
(DNA/LoRA) 的高旧+高新+TMD=0。

三步协议（avalanche EWC 原生支持）：
1. 锚定记忆：Base 在旧数据 D_old 训到收敛后，avalanche EWC 在 after_training_exp（Task0 结束）
   自动前向算 Fisher 信息矩阵（FIM）+ 保存最优权重锚点 θ*。
2. 增量阶段：新数据 D_new 来时，**不长新分支、不屏蔽旧数据、全参训练**，loss 自动加
   λ·Σ F_i(θ_i−θ*_i)²。
3. 调参刺探：扫描 λ_ewc，记录 old/new 此消彼长的窘境。

运行（需先装 avalanche；建议 GPU 服务器——a0910 有 17746 题，且要跑 |LAMBDA_SWEEP| 遍）：
    pip install avalanche-lib
    python a0910_ewc_baseline.py

实现要点（与 DER 脚本的关键差异）：
- **固定 epoch，不用 DER 那套 per-epoch 早停 hack**：DER 版"train_epochs=1 + 外层多次 train()"
  会让 EWC 在 'separate' 模式把 Task1 自身也加进 Fisher 重要性列表、污染惩罚项。故 EWC 每个任务
  只调用一次 strategy.train(experience)，train_epochs=EWC_EPOCHS 固定。
- 每个 λ 用 set_seed(42) 重置 → Task0 Base 完全一致，仅 λ 作用于 Task1 惩罚（干净对照）。

可比性边界（重要，避免误读）：
- AUC/ACC/F1/RMSE：apples-to-apples，可直接与主表对比（同划分、同测试行、同 random_split 预测口径）。
- TMD：骨干为 CognitiveBackbone（Embedding+MLP，无概念空间 θ），TMD 在学生 embedding 空间度量，
  **绝对量级不可与 G-NCDM 的概念 θ TMD 直接比**，仅可用于定性/相对趋势。
- 骨干刻意未对齐 G-NCDM（仅作"被认可的顶会 CL 基线"对照），论文不强调 EWC 跑在 G-NCDM 架构上。
- 仅 random_split（test 用户与训练共享）；user_split 用户互斥，transductive 骨干无法预测，故不在范围。
"""
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Avalanche 组件
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import EWC

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# ==========================================
# 路径与数据集常量（a0910，对齐 run_incremental_a0910.py）
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # GNCDM/
REPO_ROOT = os.path.dirname(THIS_DIR)                       # Generative-CD-main/
DATA_DIR = os.path.join(REPO_ROOT, "data", "a0910")
SAVE_DIR = os.path.join(THIS_DIR, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)

N_ITEM_TOTAL = 17746
N_KNOW_TOTAL = 123
NEW_ITEM_FRAC = 0.34       # 与 run_incremental_a0910.main() 一致

# ---- EWC / 训练超参（便于在服务器上单独 sweep）----
EMBED_DIM = 64            # 骨干隐维（不改 CognitiveBackbone 架构，仅容量）
LR = 1e-3
EWC_EPOCHS = 15          # 每个任务固定训练轮数（依据 DER 实验：25ep 易过拟合，最优更早；可调）
TRAIN_MB_SIZE = 128
EWC_MODE = "separate"    # 'separate'：每旧任务独立 FIM（2 任务标准 EWC）；'online'：累积+衰减
# λ 扫描：λ=0 即无惩罚（退化为 Naive-FT）；逐级加大 → 稳定性↑可塑性↓
LAMBDA_SWEEP = [0, 1, 10, 100, 1000, 10000]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# 内联工具函数（与 a0910_der_baseline.py 同款，逐字搬入以自包含）
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
    test_path = os.path.join(base, "test.csv")
    print(f">>> 加载 {train_path} / {test_path}（a0910 random_split）")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    # 自动选新概念 ΔK（最冷门），再严格拓扑二分重排。
    new_concepts = auto_new_concepts(Q_mat, new_item_frac=NEW_ITEM_FRAC)
    Q_mat, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, new_concepts)
    df_train = remap_items(df_train, item_id_map)
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
# 单个 λ 的 EWC 两阶段训练 + 评测
# ==========================================
def run_one_lambda(benchmark, meta, ewc_lambda, device):
    """固定 λ 跑一遍 EWC（Task0→FIM 锚定→Task1 加惩罚），返回 old/new 指标 + TMD。"""
    set_seed(42)  # 各 λ 共用同一 Base 起点，仅 λ 作用于 Task1 惩罚
    model = CognitiveBackbone(num_students=meta["num_students"],
                              num_items=meta["num_items"], embed_dim=EMBED_DIM).to(device)
    embed_dim = model.student_emb.weight.shape[1]
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    strategy = EWC(
        model, optimizer, criterion,
        ewc_lambda=ewc_lambda, mode=EWC_MODE,
        train_mb_size=TRAIN_MB_SIZE, train_epochs=EWC_EPOCHS,
        eval_mb_size=256, device=device,
    )

    baseline_student_emb = None
    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        task_name = "【旧知识基线/锚定 FIM】" if task_id == 0 else "【新知识增量/加 EWC 惩罚】"
        print(f"  -- λ={ewc_lambda} Task {task_id} {task_name}（{EWC_EPOCHS} ep 固定）--")
        strategy.train(experience)   # Task0 结束后 avalanche 自动算 FIM + 锚 θ*
        if task_id == 0:
            baseline_student_emb = model.student_emb.weight.data.clone().cpu()

    # 评测：学完 Task1 后回测 test_old / test_new
    auc_new, rmse_new, acc_new, f1_new = evaluate_cd_metrics(model, benchmark.test_stream[1].dataset, device)
    auc_old, rmse_old, acc_old, f1_old = evaluate_cd_metrics(model, benchmark.test_stream[0].dataset, device)

    # TMD：旧任务学生 embedding 整维度、按 √dim 归一化的漂移（embedding 空间，量级不可与概念 θ 比）
    final_student_emb = model.student_emb.weight.data.cpu()
    old_ids = meta["old_user_ids"]
    drift = torch.norm(baseline_student_emb[old_ids] - final_student_emb[old_ids], p=2, dim=1)
    tmd = (drift / math.sqrt(embed_dim)).mean().item()

    return {
        "ewc_lambda": ewc_lambda,
        "AUC_old": auc_old, "AUC_new": auc_new,
        "RMSE_old": rmse_old, "RMSE_new": rmse_new,
        "ACC_old": acc_old, "ACC_new": acc_new,
        "F1_old": f1_old, "F1_new": f1_new,
        "TMD": tmd,
    }


# ==========================================
# λ 扫描主流程
# ==========================================
def run_ewc_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    if device.type == "cpu":
        print("[WARN] 检测到 CPU：a0910 题量大(17746) 且要跑 "
              f"{len(LAMBDA_SWEEP)} 个 λ，会很慢，建议 GPU 服务器。")
    print(f">>> EWC 超参: mode={EWC_MODE} epochs={EWC_EPOCHS} embed_dim={EMBED_DIM} "
          f"lr={LR} lambda_sweep={LAMBDA_SWEEP}")

    benchmark, meta = load_a0910_strict_partition()

    print("\n>>> 启动 EWC λ 扫描（a0910 random_split）...")
    rows = []
    for lam in LAMBDA_SWEEP:
        print(f"\n========== ewc_lambda = {lam} ==========")
        r = run_one_lambda(benchmark, meta, lam, device)
        rows.append(r)
        print(f"  [λ={lam}] 旧: AUC={r['AUC_old']:.4f} ACC={r['ACC_old']:.4f} | "
              f"新: AUC={r['AUC_new']:.4f} ACC={r['ACC_new']:.4f} | TMD={r['TMD']:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, "ewc_lambda_sweep_a0910_random_split.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(" EWC λ 扫描结果（a0910 random_split）—— 稳定性-可塑性窘境")
    print("=" * 60)
    print("\n| ewc_lambda | AUC_old | AUC_new | ACC_old | ACC_new | F1_old | F1_new | TMD (embedding-space, 量级不可与 G-NCDM 直接比) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['ewc_lambda']} | {r['AUC_old']:.4f} | {r['AUC_new']:.4f} | "
              f"{r['ACC_old']:.4f} | {r['ACC_new']:.4f} | {r['F1_old']:.4f} | "
              f"{r['F1_new']:.4f} | {r['TMD']:.4f} |")
    print(f"\n结果已写入 {out}")
    print("\n[窘境提示] 期望趋势：小 λ → 偏学新、旧任务遗忘（TMD 大）；大 λ → 偏保旧、新任务学不动；")
    print("           无单一 λ 能同时达到 Ours(DNA/LoRA) 的「高旧 + 高新 + TMD=0」。")
    print("=" * 60)


if __name__ == '__main__':
    run_ewc_sweep()
