# -*- coding: utf-8 -*-
"""G-NCDM + C-LoRA 持续学习基线（Continual LoRA with Orthogonal Penalty）—— a0910 random_split。

【方案二】把 C-LoRA 直接挂载到**原版 G-NCDM 骨干**的 GDF（生成式诊断：f_nn / g_nn）与 IRF
（作答重构：ncd）核心全连接层上，构成 "G-NCDM + C-LoRA" 基线。与 CognitiveBackbone 版
（a0910_clora_baseline.py）相比，本版**与主表六策略同骨干、同口径**，因此：
  - AUC/ACC/F1/RMSE 可直接并入主表对比；
  - **TMD 在真·概念 θ 空间度量（calculate_tmd），量级可与 G-NCDM 概念 θ TMD 直接比**
    （CognitiveBackbone 版只能在 embedding 空间定性，本版无此限制 —— 这是方案二的核心价值）。

核心数学（同 C-LoRA）：增量阶段（Phase 2，仅用新题数据 D_new）loss 重构为
    L_total = L_CE + λ_ortho · L_ortho ,
    L_ortho = Σ_layers || W_base.detach() @ ΔW^T ||_F^2 ,   ΔW = scaling · (B @ A)
惩罚把新学的 LoRA 增量 ΔW 推向冻结基座权重 W_base 的（左）零空间，近似 C-LoRA 原文昂贵的
SVD 激活值零空间投影，减少对旧任务决策面的干扰。

设计与主实验的对接（关键，保证同口径）：
- **不走 expand_topology**：G-NCDM 主实验 Base 是缩减维(n_know_old)、靠 expand 长维；C-LoRA 改为
  **全维建模**（n_item_total / n_know_total / 完整 Q），Phase 1 只喂旧题作答（新题列恒 0、不获梯度
  → Base 不偷看新题），Phase 2 冻结全部基座 + 挂 LoRA 学新题。这正是 C-LoRA "freeze base + adapt"
  的范式，且 is_expanded 始终为 False，走标准（非扩展）前向。
- **挂载层**：自动识别 f_nn / g_nn / ncd 里的所有 nn.Linear / PosLinear（共 8 层）并包 LoRA。
  **刻意排除 theta_agg_mat / psi_agg_mat**：predict_response 对它们走 `F.linear(x, .weight, .bias)`
  直接取张量、不走 module.forward，包 LoRA 会破坏访问；且它们是聚合投影、非 GDF/IRF 诊断核心。
  这两张聚合矩阵在 Phase 2 保持冻结（其新概念列停留在 Phase-1 的 xavier 初始化）。
- 复用 run_incremental_math1 的真实作答管线（train_real / populate_buffers / evaluate_buf /
  strict_bipartition），仅 Phase 2 因需加正交惩罚而单写训练循环。

运行（建议 GPU 服务器：a0910 17746 题、(4163,17746) 稠密作答矩阵 ~0.3GB，且要跑 |sweep| 遍）：
    python a0910_gncdm_clora_baseline.py

可比性边界：
- 仅 a0910 random_split（forward_using_buf 无泄漏预测，test 用户与训练共享）；user_split 用户互斥、
  transductive 不适用，故不在范围。
- λ_ortho 仅约束 f_nn/g_nn/ncd 的 ΔW；聚合矩阵冻结、不含 LoRA。
"""
import os
import sys
import math
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- sys.path：本脚本在 GNCDM/，需同时能 import core.* 与 experiments/run_incremental_* ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # GNCDM/
EXPERIMENTS_DIR = os.path.join(THIS_DIR, "experiments")
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)

from core.model import GNCDM                                     # noqa: E402
from core.train import calculate_tmd                            # noqa: E402
from run_incremental_math1 import (                             # noqa: E402
    set_seed, build_log_mat, strict_bipartition, remap_items,
    train_real, populate_buffers, evaluate_buf, SampleSet,
)
from run_incremental_a0910 import auto_new_concepts             # noqa: E402

# ==========================================
# 路径与数据集常量（a0910，对齐 run_incremental_a0910.py）
# ==========================================
REPO_ROOT = os.path.dirname(THIS_DIR)                            # Generative-CD-main/
DATA_DIR = os.path.join(REPO_ROOT, "data", "a0910")
SAVE_DIR = os.path.join(THIS_DIR, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)

N_USER, N_ITEM, N_KNOW, ALPHA = 4163, 17746, 123, 0.9           # 对齐主实验 a0910
NEW_ITEM_FRAC = 0.34
USER_DIM, ITEM_DIM = 32, 32                                     # 对齐主实验 Base

# ---- 训练 / C-LoRA 超参 ----
LR = 1e-3
BATCH = 256
BASE_EPOCHS = 25            # Phase 1（旧题基座，对齐主实验 Base n_epoch=25）
CLORA_EPOCHS = 25          # Phase 2（新题增量 + 正交惩罚，对齐主实验策略 n_epoch=25）
LORA_RANK = 8              # 用户 Step 1：rank=8
LORA_ALPHA = 16            # 用户 Step 1：alpha=16（scaling = alpha/rank = 2.0）
LAMBDA_ORTHO_SWEEP = [0, 1, 10, 100, 1000, 10000]


# ==========================================
# C-LoRA 核心组件：轻量 LoRA wrapper + 权重级软正交惩罚
# ==========================================
class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` (or ``PosLinear``) base layer with a trainable low-rank update.

    Forward:  y = base_layer(x) + scaling · B(A(x))
    where A ∈ R^{rank×in}, B ∈ R^{out×rank}, scaling = alpha / rank.

    Notes:
    - ``base_layer(x)`` preserves the base layer's own semantics (PosLinear keeps its
      non-negativity transform on the *base* path); the LoRA branch is a standard
      unconstrained low-rank delta, matching the generic C-LoRA treatment.
    - ``lora_B`` is zero-initialized so ΔW = B@A = 0 at injection time → mounting the
      adapter perturbs the (already-trained) base layer by exactly zero.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1e-2)
        nn.init.zeros_(self.lora_B.weight)     # ΔW = 0 at start → zero perturbation
        self.rank = rank
        self.scaling = alpha / rank

    def delta_w(self) -> torch.Tensor:
        """Effective low-rank weight update ΔW, shape (out_features, in_features)."""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora_gncdm(model: GNCDM, rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
    """Wrap every ``nn.Linear``/``PosLinear`` inside GDF (f_nn, g_nn) and IRF (ncd) with
    ``LoRALinear`` in-place. Aggregation matrices (theta_agg_mat / psi_agg_mat) are left
    untouched on purpose — predict_response reads their ``.weight`` directly. Returns the
    number of layers wrapped."""
    n_wrapped = 0
    for container in (model.f_nn, model.g_nn, model.ncd):
        for idx, child in enumerate(container):
            if isinstance(child, nn.Linear):     # PosLinear is a subclass of nn.Linear
                container[idx] = LoRALinear(child, rank=rank, alpha=alpha)
                n_wrapped += 1
    return n_wrapped


def orthogonal_penalty(model: nn.Module) -> torch.Tensor:
    """C-LoRA soft orthogonal penalty over all mounted LoRA layers:
        L_ortho = Σ || W_base.detach() @ ΔW^T ||_F^2
    Drives each ΔW towards the (left) null space of its frozen base weight."""
    loss_ortho = None
    for m in model.modules():
        if isinstance(m, LoRALinear):
            W_base = m.base_layer.weight.detach()             # (out, in)
            delta_W = m.delta_w()                             # (out, in)
            term = torch.sum((W_base @ delta_W.t()) ** 2)     # (out,out) Frobenius^2
            loss_ortho = term if loss_ortho is None else loss_ortho + term
    return loss_ortho


def lora_parameters(model: nn.Module):
    """All trainable LoRA factors (A/B) across the mounted layers."""
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


# ==========================================
# Phase 2 训练循环（标准 G-NCDM 前向 + BCE + λ·正交惩罚）
# ==========================================
def train_clora_phase2(model, samples_df, full_log_mat, params, device,
                       lambda_ortho, n_epoch=CLORA_EPOCHS, batch_size=BATCH, lr=LR, desc="C-LoRA"):
    """与 run_incremental_math1.train_real 的前向一致（喂真实作答日志），但 loss 改为
    L_CE + λ_ortho · L_ortho。仅优化传入的 LoRA 参数。"""
    loader = DataLoader(SampleSet(samples_df), batch_size=batch_size, shuffle=True)
    log_t = torch.tensor(full_log_mat, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(params, lr=lr)
    lam = float(lambda_ortho)

    for epoch in range(n_epoch):
        model.train()
        total_ce, total_ortho = 0.0, 0.0
        for user_ids, item_ids, score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            score = score.to(device).unsqueeze(1)
            user_log = log_t[user_ids]            # (B, n_item) 真实作答
            item_log = log_t[:, item_ids].T       # (B, n_user) 真实作答
            pred = model(user_log, item_log, user_ids, item_ids)
            loss_ce = F.binary_cross_entropy(pred, score)
            if lam > 0:
                loss_ortho = orthogonal_penalty(model)
                loss = loss_ce + lam * loss_ortho
                total_ortho += loss_ortho.item()
            else:
                loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_ce += loss_ce.item()
        print(f"    [{desc}] epoch {epoch + 1}/{n_epoch} CE={total_ce / len(loader):.4f} "
              f"L_ortho(raw)={total_ortho / len(loader):.4e}")
    return model


# ==========================================
# 数据装载：a0910 random_split + 自动 ΔK 严格拓扑二分（全维，不缩减）
# ==========================================
def load_a0910_full_partition():
    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    base = os.path.join(DATA_DIR, "new_random_split")
    print(f">>> 加载 {base}/train.csv, test.csv（a0910 random_split）")
    df_train = pd.read_csv(os.path.join(base, "train.csv"))
    df_test = pd.read_csv(os.path.join(base, "test.csv"))
    Q_mat = np.load(Q_path)

    new_concepts = auto_new_concepts(Q_mat, new_item_frac=NEW_ITEM_FRAC)
    Q_full, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, new_concepts)
    df_train = remap_items(df_train, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    assert Q_full[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(f">>> 自动 ΔK：新概念={len(new_concepts)}/{N_KNOW}，旧题(Task0)={n_item_old} "
          f"新题(Task1)={N_ITEM - n_item_old}，旧概念={n_know_old}")

    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    train_new = df_train[df_train["item_id"] >= n_item_old].copy()
    test_old = df_test[df_test["item_id"] < n_item_old].copy()
    test_new = df_test[df_test["item_id"] >= n_item_old].copy()

    # 全维作答矩阵：log_old_only 新题列恒 0（Phase1 不偷看新题）；log_full 含全部作答（Phase2/评测）
    log_old_only = build_log_mat(train_old, N_USER, N_ITEM)
    log_full = build_log_mat(df_train, N_USER, N_ITEM)

    meta = {
        "Q_full": Q_full,
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "train_old": train_old, "train_new": train_new,
        "test_old": test_old, "test_new": test_new,
        "log_old_only": log_old_only, "log_full": log_full,
    }
    return meta


# ==========================================
# 单个 λ：从 Phase-1 快照恢复 → 冻结基座 + 挂 LoRA → 仅训 Task1（带正交惩罚）→ 评测
# ==========================================
def run_one_lambda(base_state, base_theta_ref, meta, lambda_ortho, device):
    set_seed(42)  # 各 λ 的 LoRA 初始化一致，仅 λ 作用于 Task1 惩罚（干净对照）
    model = GNCDM(n_user=N_USER, n_item=N_ITEM, n_know=N_KNOW, user_dim=USER_DIM, item_dim=ITEM_DIM,
                  alpha=ALPHA, Q_mat=meta["Q_full"], monotonicity_assumption=True, device=device).to(device)
    model.load_state_dict(base_state)        # 恢复逐位一致的 Phase-1 基座

    # 冻结全部基座参数，再在 GDF/IRF 核心层挂 LoRA（仅 LoRA A/B 可训）
    model._freeze_parameters()
    n_wrapped = inject_lora_gncdm(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    model.to(device)
    params = lora_parameters(model)
    print(f"  -- λ_ortho={lambda_ortho}：冻结基座 + 挂 LoRA(rank={LORA_RANK}, alpha={LORA_ALPHA}) "
          f"于 {n_wrapped} 层，可训张量={len(params)}，仅训新题 D_new（{CLORA_EPOCHS} ep）--")

    train_clora_phase2(model, meta["train_new"], meta["log_full"], params, device,
                       lambda_ortho=lambda_ortho, desc=f"λ={lambda_ortho}")

    # 评测（buf 口径，无泄漏）：用全量作答填 buffer，再查 buffer 预测 test_old / test_new
    populate_buffers(model, meta["log_full"], device)
    r_old = evaluate_buf(model, meta["test_old"], device)
    r_new = evaluate_buf(model, meta["test_new"], device)

    # TMD：旧概念 θ 在增量前后的漂移（概念空间，量级可与 G-NCDM 概念 θ TMD 直接比）。
    # calculate_tmd 约定 theta_old 已是 K_old 宽（它内部只把 theta_new 切到 :K_old），而本版 Base
    # 是全维(n_know_total)，故先把参照 θ 切到旧概念列 [:, :n_know_old] 再比对。
    k_old = meta["n_know_old"]
    tmd = calculate_tmd(base_theta_ref[:, :k_old].to(device),
                        model.get_Theta_buf().to(device), k_old)

    return {
        "ortho_lambda": lambda_ortho,
        "AUC_old": r_old["auc"], "AUC_new": r_new["auc"],
        "RMSE_old": r_old["rmse"], "RMSE_new": r_new["rmse"],
        "ACC_old": r_old["acc"], "ACC_new": r_new["acc"],
        "F1_old": r_old["f1"], "F1_new": r_new["f1"],
        "TMD": tmd,
    }


# ==========================================
# λ_ortho 扫描主流程
# ==========================================
def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("[WARN] 检测到 CPU：a0910 题量大(17746) 且要跑 "
              f"{len(LAMBDA_ORTHO_SWEEP)} 个 λ，会很慢，建议 GPU 服务器。")
    print(f">>> G-NCDM+C-LoRA 超参: rank={LORA_RANK} alpha={LORA_ALPHA} alpha_mix={ALPHA} "
          f"base_ep={BASE_EPOCHS} clora_ep={CLORA_EPOCHS} lr={LR} lambda_sweep={LAMBDA_ORTHO_SWEEP}")

    meta = load_a0910_full_partition()

    # ---- Phase 1：Base 全维模型只训一次（旧题 D_old，新题列恒 0），快照供各 λ 复用 ----
    set_seed(42)
    base = GNCDM(n_user=N_USER, n_item=N_ITEM, n_know=N_KNOW, user_dim=USER_DIM, item_dim=ITEM_DIM,
                 alpha=ALPHA, Q_mat=meta["Q_full"], monotonicity_assumption=True, device=device).to(device)
    print(f"\n>>> Phase 1：训练 Base（旧题 D_old，全参，{BASE_EPOCHS} ep）...")
    train_real(base, meta["train_old"], meta["log_old_only"], list(base.parameters()), device,
               n_epoch=BASE_EPOCHS, desc="Base")
    # 增量前的旧概念 θ 参照（用 Base 视角的旧题作答），供 TMD
    populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    base_state = copy.deepcopy(base.state_dict())

    print("\n>>> 启动 G-NCDM+C-LoRA λ_ortho 扫描（a0910 random_split）...")
    rows = []
    for lam in LAMBDA_ORTHO_SWEEP:
        print(f"\n========== ortho_lambda = {lam} ==========")
        r = run_one_lambda(base_state, base_theta_ref, meta, lam, device)
        rows.append(r)
        print(f"  [λ={lam}] 旧: AUC={r['AUC_old']:.4f} ACC={r['ACC_old']:.4f} | "
              f"新: AUC={r['AUC_new']:.4f} ACC={r['ACC_new']:.4f} | TMD={r['TMD']:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, "clora_gncdm_lambda_sweep_a0910_random_split.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(" G-NCDM + C-LoRA  λ_ortho 扫描结果（a0910 random_split）—— 稳定性-可塑性权衡")
    print("=" * 60)
    print("\n| ortho_lambda | AUC_old | AUC_new | ACC_old | ACC_new | F1_old | F1_new | TMD (concept-θ space, 可与 G-NCDM 直接比) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['ortho_lambda']} | {r['AUC_old']:.4f} | {r['AUC_new']:.4f} | "
              f"{r['ACC_old']:.4f} | {r['ACC_new']:.4f} | {r['F1_old']:.4f} | "
              f"{r['F1_new']:.4f} | {r['TMD']:.4f} |")
    print(f"\n结果已写入 {out}")
    print("\n[权衡提示] λ=0 → 无约束 LoRA-FT（偏学新、旧任务受干扰、TMD 大）；λ↑ → 正交约束加强，"
          "旧概念 θ 决策面受保护（AUC_old↑、TMD↓），新任务可塑性受限（AUC_new↓）。")
    print("           与 Ours(DNA/LoRA) 的「高旧 + 高新 + TMD=0」对照：C-LoRA 受单一 λ 权衡所限。")
    print("=" * 60)


if __name__ == "__main__":
    run_sweep()
