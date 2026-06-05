# -*- coding: utf-8 -*-
"""C-LoRA 持续学习基线（Continual LoRA with Orthogonal Penalty）—— a0910 random_split。【单文件自包含】

模拟 2025 年 C-LoRA (Continual Low-Rank Adaptation) 的核心思想：在增量阶段冻结基座、挂载
LoRA，并以**权重级软正交惩罚 (Soft Orthogonal Penalty)** 近似 C-LoRA 原文中昂贵的 SVD 激活值
零空间投影。增量阶段（Phase 2，仅用新题数据 D_new）loss 重构为：

    L_total = L_CE + λ_ortho · L_ortho
    L_ortho = || W_base.detach() @ ΔW^T ||_F^2 ,   ΔW = scaling · (B @ A)

直觉：强迫新学的 LoRA 权重增量 ΔW 尽量与被冻结基座权重 W_base 的行空间正交（点积 Frobenius
范数趋零），从而把新知识写到旧权重的零空间里、减少对旧任务决策面的干扰。

用于和 a0910 六大策略（incremental_results_a0910_random_split.csv）以及 EWC/DER 基线对比，并
通过**扫描 λ_ortho** 展示 C-LoRA 的「稳定性-可塑性」权衡前沿。

运行（建议 GPU 服务器——a0910 有 17746 题，且要跑 |LAMBDA_SWEEP| 遍；本脚本无需 avalanche/peft）：
    python a0910_clora_baseline.py

实现要点（与 EWC/DER 脚本的关键差异）：
- **C-LoRA 不在 avalanche 内**：冻结+LoRA+正交惩罚逻辑全部手写，比 EWC/DER 的 strategy 更简单。
- **Task0 只训一次 + 快照复用**：每个 λ 从同一 Base 快照恢复后只训 Task1（比 EWC「每 λ 重训
  Task0」更省、且 Task0 逐位一致 → 更干净的对照）。
- **冻结范围**：仅冻结 MLP 基座权重（挂 LoRA），两个 embedding 保持可训（新题 item_emb 必须可
  学；旧用户 student_emb 可训才能让 TMD 度量旧用户漂移，且对齐 EWC/DER 的全参训练口径）。

可比性边界（重要，避免误读）：
- AUC/ACC/F1/RMSE：apples-to-apples，可直接与主表 / EWC / DER 对比（同划分、同测试行、同
  random_split 预测口径）。
- TMD：骨干为 CognitiveBackbone（Embedding+MLP，无概念空间 θ），TMD 在学生 embedding 空间度量，
  **绝对量级不可与 G-NCDM 的概念 θ TMD 直接比**，仅可用于定性/相对趋势。
- λ_ortho **仅约束 MLP 的 ΔW，不约束 embedding**；故 TMD 对 λ 的响应可能弱于 AUC_old，这是
  C-LoRA「权重级约束」的固有特性，论文需说明。
- 骨干刻意未对齐 G-NCDM（仅作"被认可的 CL 基线"对照），不声称 C-LoRA 跑在 G-NCDM 架构上。
- 仅 random_split（test 用户与训练共享）；user_split 用户互斥，transductive 骨干无法预测，故不在范围。
"""
import os
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# ==========================================
# 路径与数据集常量（a0910，对齐 run_incremental_a0910.py / a0910_ewc_baseline.py）
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # GNCDM/
REPO_ROOT = os.path.dirname(THIS_DIR)                       # Generative-CD-main/
DATA_DIR = os.path.join(REPO_ROOT, "data", "a0910")
SAVE_DIR = os.path.join(THIS_DIR, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)

N_ITEM_TOTAL = 17746
N_KNOW_TOTAL = 123
NEW_ITEM_FRAC = 0.34       # 与 run_incremental_a0910.main() 一致

# ---- 训练 / C-LoRA 超参（便于在服务器上单独 sweep）----
EMBED_DIM = 64             # 骨干隐维（不改 CognitiveBackbone 架构，仅容量）
LR = 1e-3
BASE_EPOCHS = 15           # Task0（旧题基座）训练轮数
CLORA_EPOCHS = 15          # Task1（新题增量 + 正交惩罚）训练轮数
TRAIN_MB_SIZE = 128
LORA_RANK = 8              # LoRA 秩（用户 Step 1：rank=8）
LORA_ALPHA = 16            # LoRA 缩放分子（scaling = alpha / rank = 2.0）
# λ_ortho 扫描：λ=0 即无正交约束（退化为无约束 LoRA-FT，下锚）；逐级加大 → 稳定性↑可塑性↓
LAMBDA_ORTHO_SWEEP = [0, 1, 10, 100, 1000, 10000]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# 内联工具函数（与 a0910_ewc_baseline.py 同款，逐字搬入以自包含）
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
# 认知诊断专用骨干网络（简单 Embedding+MLP，非 G-NCDM；与 EWC/DER 同款）
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
# C-LoRA 核心组件：轻量 LoRA wrapper + 权重级软正交惩罚
# ==========================================
class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` base layer with a trainable low-rank update.

    Forward:  y = W_base x + b  +  scaling · (B (A x))
    where A ∈ R^{rank×in}, B ∈ R^{out×rank}, scaling = alpha / rank.

    ``lora_B`` is zero-initialized so that ΔW = B@A = 0 at injection time, i.e. mounting
    the adapter perturbs the (already-trained) base layer by exactly zero — mirroring the
    micro-variance initialization used elsewhere in this repo.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        # Freeze the base weight/bias: only the LoRA factors stay trainable.
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1e-2)
        nn.init.zeros_(self.lora_B.weight)   # ΔW = 0 at start → zero perturbation to base
        self.rank = rank
        self.scaling = alpha / rank

    def delta_w(self) -> torch.Tensor:
        """Effective low-rank weight update ΔW with shape (out_features, in_features)."""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(model: CognitiveBackbone, rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
    """Replace the three ``nn.Linear`` layers inside ``model.mlp`` (Sequential indices 0/2/4)
    with ``LoRALinear`` wrappers in-place. The base MLP weights become frozen; the two
    embedding tables are left untouched (still trainable)."""
    for idx in (0, 2, 4):
        layer = model.mlp[idx]
        assert isinstance(layer, nn.Linear), f"mlp[{idx}] is not nn.Linear: {type(layer)}"
        model.mlp[idx] = LoRALinear(layer, rank=rank, alpha=alpha)
    return model


def orthogonal_penalty(model: nn.Module) -> torch.Tensor:
    """C-LoRA soft orthogonal penalty summed over all mounted LoRA layers:

        L_ortho = Σ_layers || W_base.detach() @ ΔW^T ||_F^2

    Penalizing the Frobenius norm of W_base·ΔW^T drives ΔW towards the (left) null space
    of the frozen base weight, approximating the SVD-based null-space projection of the
    original C-LoRA without any explicit decomposition.
    """
    loss_ortho = model.student_emb.weight.new_zeros(())   # device/dtype-safe scalar 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            W_base = m.base_layer.weight.detach()         # (out, in)
            delta_W = m.delta_w()                         # (out, in)
            # W_base @ ΔW^T : (out,in) @ (in,out) = (out,out); sum of squares = ||·||_F^2
            loss_ortho = loss_ortho + torch.sum((W_base @ delta_W.t()) ** 2)
    return loss_ortho


# ==========================================
# 数据流装载：a0910 random_split + 自动 ΔK 的严格拓扑二分（不经 avalanche）
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

    datasets = {
        "train_old": _ds(df_train, 0, n_item_old),
        "train_new": _ds(df_train, n_item_old, N_ITEM_TOTAL),
        "test_old": _ds(df_test, 0, n_item_old),
        "test_new": _ds(df_test, n_item_old, N_ITEM_TOTAL),
    }
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
    return datasets, meta


# ==========================================
# 训练：单阶段循环（lambda_ortho=None 时为普通 CE；否则加正交软约束）
# ==========================================
def train_phase(model, dataset, epochs, device, lambda_ortho=None, desc="train"):
    """Mini-batch training over a single TensorDataset.

    - lambda_ortho is None  → Phase 0 (Base): plain cross-entropy over the full model.
    - lambda_ortho is float → Phase 2 (C-LoRA): loss = CE + lambda_ortho · L_ortho,
      optimizing only the params left with requires_grad=True (LoRA factors + embeddings).
    """
    model.train()
    loader = DataLoader(dataset, batch_size=TRAIN_MB_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR)

    for epoch in range(epochs):
        total_ce, total_ortho, n_batch = 0.0, 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_ce = criterion(logits, y)
            if lambda_ortho is not None and lambda_ortho > 0:
                loss_ortho = orthogonal_penalty(model)
                loss = loss_ce + lambda_ortho * loss_ortho
                total_ortho += loss_ortho.item()
            else:
                loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_ce += loss_ce.item()
            n_batch += 1
        if lambda_ortho is not None:
            print(f"    [{desc}] epoch {epoch + 1}/{epochs} CE={total_ce / n_batch:.4f} "
                  f"L_ortho(raw)={total_ortho / max(n_batch, 1):.4e}")
        else:
            print(f"    [{desc}] epoch {epoch + 1}/{epochs} CE={total_ce / n_batch:.4f}")


# ==========================================
# 单个 λ 的 C-LoRA 增量训练 + 评测（从 Task0 快照恢复，仅训 Task1）
# ==========================================
def run_one_lambda(base_state, datasets, meta, lambda_ortho, device):
    """从同一 Base 快照恢复 → 挂 LoRA + 冻结 MLP 基座 → 仅训 Task1（带正交惩罚）→ 评测。"""
    set_seed(42)  # 各 λ 的 LoRA 初始化一致，仅 λ 作用于 Task1 惩罚（干净对照）
    model = CognitiveBackbone(num_students=meta["num_students"],
                              num_items=meta["num_items"], embed_dim=EMBED_DIM).to(device)
    model.load_state_dict(base_state)            # 恢复逐位一致的 Task0 基座
    embed_dim = model.student_emb.weight.shape[1]

    # 记录增量前的旧用户 student embedding，用于 TMD（漂移）
    baseline_student_emb = model.student_emb.weight.data.clone().cpu()

    # 冻结 MLP 基座并挂载 LoRA（embedding 保持可训）
    inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    model.to(device)

    print(f"  -- λ_ortho={lambda_ortho} 增量阶段：冻结 MLP 基座 + 挂 LoRA(rank={LORA_RANK}, "
          f"alpha={LORA_ALPHA})，仅训新题 D_new（{CLORA_EPOCHS} ep）--")
    train_phase(model, datasets["train_new"], CLORA_EPOCHS, device,
                lambda_ortho=float(lambda_ortho), desc=f"λ={lambda_ortho}")

    # 评测：学完 Task1 后回测 test_old / test_new
    auc_new, rmse_new, acc_new, f1_new = evaluate_cd_metrics(model, datasets["test_new"], device)
    auc_old, rmse_old, acc_old, f1_old = evaluate_cd_metrics(model, datasets["test_old"], device)

    # TMD：旧任务学生 embedding 整维度、按 √dim 归一化的漂移（embedding 空间，量级不可与概念 θ 比）
    final_student_emb = model.student_emb.weight.data.cpu()
    old_ids = meta["old_user_ids"]
    drift = torch.norm(baseline_student_emb[old_ids] - final_student_emb[old_ids], p=2, dim=1)
    tmd = (drift / math.sqrt(embed_dim)).mean().item()

    return {
        "ortho_lambda": lambda_ortho,
        "AUC_old": auc_old, "AUC_new": auc_new,
        "RMSE_old": rmse_old, "RMSE_new": rmse_new,
        "ACC_old": acc_old, "ACC_new": acc_new,
        "F1_old": f1_old, "F1_new": f1_new,
        "TMD": tmd,
    }


# ==========================================
# λ_ortho 扫描主流程
# ==========================================
def run_clora_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    if device.type == "cpu":
        print("[WARN] 检测到 CPU：a0910 题量大(17746) 且要跑 "
              f"{len(LAMBDA_ORTHO_SWEEP)} 个 λ，会很慢，建议 GPU 服务器。")
    print(f">>> C-LoRA 超参: rank={LORA_RANK} alpha={LORA_ALPHA} embed_dim={EMBED_DIM} lr={LR} "
          f"base_ep={BASE_EPOCHS} clora_ep={CLORA_EPOCHS} lambda_sweep={LAMBDA_ORTHO_SWEEP}")

    datasets, meta = load_a0910_strict_partition()

    # ---- Phase 0: Task0 基座只训一次（旧题 D_old），快照供各 λ 复用 ----
    set_seed(42)
    base_model = CognitiveBackbone(num_students=meta["num_students"],
                                   num_items=meta["num_items"], embed_dim=EMBED_DIM).to(device)
    print(f"\n>>> Phase 0：训练 Base（旧题 D_old，全参，{BASE_EPOCHS} ep）...")
    train_phase(base_model, datasets["train_old"], BASE_EPOCHS, device,
                lambda_ortho=None, desc="Base")
    base_state = copy.deepcopy(base_model.state_dict())   # 逐位一致的 Task0 锚点

    print("\n>>> 启动 C-LoRA λ_ortho 扫描（a0910 random_split）...")
    rows = []
    for lam in LAMBDA_ORTHO_SWEEP:
        print(f"\n========== ortho_lambda = {lam} ==========")
        r = run_one_lambda(base_state, datasets, meta, lam, device)
        rows.append(r)
        print(f"  [λ={lam}] 旧: AUC={r['AUC_old']:.4f} ACC={r['ACC_old']:.4f} | "
              f"新: AUC={r['AUC_new']:.4f} ACC={r['ACC_new']:.4f} | TMD={r['TMD']:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, "clora_lambda_sweep_a0910_random_split.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(" C-LoRA λ_ortho 扫描结果（a0910 random_split）—— 稳定性-可塑性权衡")
    print("=" * 60)
    print("\n| ortho_lambda | AUC_old | AUC_new | ACC_old | ACC_new | F1_old | F1_new | TMD (embedding-space, 量级不可与 G-NCDM 直接比) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['ortho_lambda']} | {r['AUC_old']:.4f} | {r['AUC_new']:.4f} | "
              f"{r['ACC_old']:.4f} | {r['ACC_new']:.4f} | {r['F1_old']:.4f} | "
              f"{r['F1_new']:.4f} | {r['TMD']:.4f} |")
    print(f"\n结果已写入 {out}")
    print("\n[权衡提示] 期望趋势：λ=0 → 无约束 LoRA-FT（偏学新、旧任务受干扰）；λ↑ → 正交约束加强，"
          "旧任务决策面受保护（AUC_old↑），新任务可塑性受限（AUC_new↓）。")
    print("           λ_ortho 仅约束 MLP 的 ΔW、不约束 embedding，故 TMD 对 λ 的响应弱于 AUC_old。")
    print("=" * 60)


if __name__ == '__main__':
    run_clora_sweep()
