"""user_split 九方法统一对比（6 Ours 策略 + 3 CL 基线），**统一 Support/Query 口径**。【单脚本】

动机：之前 a0910 是「Ours 一个脚本 + 三基线一个脚本」分开跑、且 math1 的三 CL 基线从未跑过
user_split。本脚本把 **6 个 Ours 策略（G-NCDM）** 与 **3 个常用 CL 基线（EWC/DER++/C-LoRA，
CognitiveBackbone）** 放进一个脚本，在**同一份 support/query 划分**上评测——所有方法在**完全
相同的 query 行**上算指标，彻底可比，一次产出一张合并对比表。

为什么 support/query：user_split 用户互斥（test 用户训练未见）。transductive 基线必须冷启动
（用 test 用户的 support 拟合 student_emb）；若在「全部作答」上拟合又考同批题会记忆泄漏、AUC 虚
高（曾实测基线反超 Ours 20 点）。故统一用 support/query 留出：每个 test 用户一半作答（support）
做诊断输入 / 拟合，另一半（query，不相交）评测。Ours(G-NCDM) 同口径——`evaluate_recon` 喂
「仅由 support 构建的 log_mat + query 行」，user_log 天然不含被预测项。

可比性边界（务必守住）：
- AUC/ACC/F1/RMSE：九方法同划分、**同一批 query 行**、同 support/query 协议 → 可逐行直接对比。
- RD vs RD\*：Ours 的 RD 在 **G-NCDM 概念 θ 空间**（架构隔离 → 0 或极小）；基线的 RD\* 在
  **CognitiveBackbone 学生 embedding 空间**，**量级不可与 Ours 的 RD 直接比**，只看"是否>0"。
- 骨干不同（基线非 G-NCDM）：勿声称"纯策略"胜出，只说"同划分/同口径下 Ours 全面占优"。

本文件是 **库**（位于 experiments/_core/），提供 user_split 的 `run_one(...)`，被
experiments/ 下的 per-split 入口（run_incremental_{math1,a0910,junyi}_user_split.py）import。
也可直接跑其 __main__ 做自检（建议 GPU 服务器；需 avalanche 给 EWC/DER）：
    pip install avalanche-lib
    python GNCDM/experiments/_core/eval_all_methods_user_split.py
__main__ 默认跑 math1 user_split；把 RUN_A0910=True 也跑 a0910（17746 题，重）。
"""

import copy
import math
import os
import sys

gncdm_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # _core/→experiments/→GNCDM/
sys.path.insert(0, gncdm_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from core.model import GNCDM
from core.train import calculate_rd

# 复用主实验的 G-NCDM 训练/工具函数（import，不改原文件）
from run_incremental_math1 import (
    build_log_mat,
    evaluate_recon,
    fresh_base,
    lora_params,
    new_params,
    populate_buffers,
    remap_items,
    set_seed,
    strict_bipartition,
    train_real,
)
from run_incremental_a0910 import auto_new_concepts
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

SAVE_DIR = os.path.join(gncdm_dir, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- 统一 support/query 协议（Ours 与基线共用同一划分）----
SUPPORT_FRAC = 0.5
SPLIT_SEED = 7

# ---- 是否也跑 a0910（重；默认只跑 math1）----
RUN_A0910 = False
# ---- 是否也跑 junyi（5000×707×39，中等；user_split alpha=0.6）----
RUN_JUNYI = True

# ---- 基线骨干 / 冷启动超参（与 a0910_cl_baselines_user_split.py 对齐）----
EMBED_DIM = 64
LR = 1e-3
TRAIN_MB_SIZE = 128
COLD_START_EPOCHS = 30
COLD_START_LR = 1e-2
COLD_START_BATCH = 512
COLD_START_SEED = 123
# DER++
DER_EPOCHS = 15
MEM_SIZE = 5000
DER_ALPHA = 0.5
DER_BETA = 0.5
# EWC
EWC_EPOCHS = 15
EWC_MODE = "separate"
EWC_LAMBDA_SWEEP = [0, 1, 10, 100, 1000, 10000]
# C-LoRA
BASE_EPOCHS = 15
CLORA_EPOCHS = 15
LORA_RANK = 8
LORA_ALPHA = 16
CLORA_LAMBDA_SWEEP = [0, 1, 10, 100, 1000, 10000]


# ==========================================================================
# 共享：support/query 划分（按用户切一次，Ours 与基线都用这份）
# ==========================================================================
def split_support_query(df, *, split_seed=SPLIT_SEED, support_frac=SUPPORT_FRAC):
    support = df.groupby("user_id", group_keys=False).sample(
        frac=support_frac, random_state=split_seed
    )
    query = df.drop(support.index)
    return support, query


# ==========================================================================
# 基线骨干（CognitiveBackbone + LoRA + 正交惩罚），与 a0910 基线脚本同款（内联自包含）
# ==========================================================================
class CognitiveBackbone(nn.Module):
    def __init__(self, num_students, num_items, embed_dim=64):
        super().__init__()
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


class LoRALinear(nn.Module):
    """冻结 nn.Linear 基座 + 低秩增量；lora_B 零初始化（挂载即 ΔW=0）。"""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1e-2)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def delta_w(self):
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA):
    for idx in (0, 2, 4):
        layer = model.mlp[idx]
        assert isinstance(layer, nn.Linear)
        model.mlp[idx] = LoRALinear(layer, rank=rank, alpha=alpha)
    return model


def orthogonal_penalty(model):
    loss_ortho = model.student_emb.weight.new_zeros(())
    for m in model.modules():
        if isinstance(m, LoRALinear):
            W_base = m.base_layer.weight.detach()
            loss_ortho = loss_ortho + torch.sum((W_base @ m.delta_w().t()) ** 2)
    return loss_ortho


def evaluate_cd_metrics(model, eval_dataset, device):
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    y_trues, y_probs, y_preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)
            y_trues.extend(y.cpu().numpy())
            y_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            y_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except ValueError:
        auc = 0.5
    return (
        auc,
        mean_squared_error(y_trues, y_probs) ** 0.5,
        accuracy_score(y_trues, y_preds),
        f1_score(y_trues, y_preds, zero_division=0),
    )


def coldstart_eval(model, eval_pack, device, *, seed=COLD_START_SEED):
    """Support/Query 冷启动：support 拟合 student_emb → query 的 old/new 上评测（无泄漏）。"""
    em = copy.deepcopy(model).to(device)
    embed_dim = em.item_emb.weight.shape[1]
    torch.manual_seed(seed)
    em.student_emb = nn.Embedding(eval_pack["n_test_users"], embed_dim).to(device)
    for p in em.parameters():
        p.requires_grad = False
    em.student_emb.weight.requires_grad = True
    optimizer = optim.Adam([em.student_emb.weight], lr=COLD_START_LR)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(eval_pack["support_x"], eval_pack["support_y"]),
        batch_size=COLD_START_BATCH,
        shuffle=True,
    )
    em.train()
    for _ in range(COLD_START_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(em(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    old_m = evaluate_cd_metrics(
        em, TensorDataset(eval_pack["query_old_x"], eval_pack["query_old_y"]), device
    )
    new_m = evaluate_cd_metrics(
        em, TensorDataset(eval_pack["query_new_x"], eval_pack["query_new_y"]), device
    )
    return old_m, new_m


def baseline_tmd(baseline_student_emb, model, old_user_ids, embed_dim):
    final = model.student_emb.weight.data.cpu()
    drift = torch.norm(baseline_student_emb[old_user_ids] - final[old_user_ids], p=2, dim=1)
    return (drift / math.sqrt(embed_dim)).mean().item()


# ==========================================================================
# 数据准备：一次构建 Ours / 基线两套表示，共享同一份 support/query 划分
# ==========================================================================
def prepare(cfg, device, *, support_query_seed=SPLIT_SEED, support_frac=SUPPORT_FRAC):
    df_train = pd.read_csv(cfg["train"])
    df_valid = pd.read_csv(cfg["valid"])
    df_test = pd.read_csv(cfg["test"])
    Q_mat = np.load(cfg["Q"])
    n_user, n_item_total, n_know_total = cfg["n_user"], cfg["n_item"], cfg["n_know"]

    Q_mat, item_id_map, n_item_old, n_know_old = strict_bipartition(
        Q_mat, list(cfg["new_concepts"])
    )
    df_train = remap_items(df_train, item_id_map)
    df_valid = remap_items(df_valid, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    n_item_new, n_know_new = n_item_total - n_item_old, n_know_total - n_know_old
    Q_old = Q_mat[:n_item_old, :n_know_old].copy()
    Q_expanded = Q_mat.copy()
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(f"  二分: 旧题={n_item_old} 新题={n_item_new}, 旧概念={n_know_old} 新概念={n_know_new}")

    # 训练集按 old/new 题切分（训练用，两套骨干共用同一行集合）
    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    train_new = df_train[df_train["item_id"] >= n_item_old].copy()

    # === 一次性 support/query 划分（Ours 与基线共用，保证 query 行完全一致）===
    sup_valid, qry_valid = split_support_query(
        df_valid, split_seed=support_query_seed, support_frac=support_frac
    )
    sup_test, qry_test = split_support_query(
        df_test, split_seed=support_query_seed, support_frac=support_frac
    )
    qry_valid_old = qry_valid[qry_valid["item_id"] < n_item_old].copy()
    qry_valid_new = qry_valid[qry_valid["item_id"] >= n_item_old].copy()
    qry_test_old = qry_test[qry_test["item_id"] < n_item_old].copy()
    qry_test_new = qry_test[qry_test["item_id"] >= n_item_old].copy()
    print(
        f"  support/query: test support={len(sup_test)} | query old={len(qry_test_old)} new={len(qry_test_new)}"
    )

    # ---- Ours（G-NCDM）表示：训练 log + 仅 support 构建的评测 log ----
    ours = {
        "log_old": build_log_mat(train_old, n_user, n_item_old),
        "log_full": build_log_mat(df_train, n_user, n_item_total),
        "train_old": train_old,
        "train_new": train_new,
        "sup_valid_old_log": build_log_mat(
            sup_valid[sup_valid["item_id"] < n_item_old], n_user, n_item_old
        ),
        "sup_test_old_log": build_log_mat(
            sup_test[sup_test["item_id"] < n_item_old], n_user, n_item_old
        ),
        "sup_valid_full_log": build_log_mat(sup_valid, n_user, n_item_total),
        "sup_test_full_log": build_log_mat(sup_test, n_user, n_item_total),
        "qry_valid_old": qry_valid_old,
        "qry_valid_new": qry_valid_new,
        "qry_valid": qry_valid,
        "qry_test_old": qry_test_old,
        "qry_test_new": qry_test_new,
        "Q_old": Q_old,
        "Q_expanded": Q_expanded,
        "n_item_old": n_item_old,
        "n_item_new": n_item_new,
        "n_know_old": n_know_old,
        "n_know_new": n_know_new,
    }

    # ---- 基线（CognitiveBackbone）表示：训练 TensorDataset + 冷启动 support/query 包 ----
    def _ds(df, lo, hi):
        sub = df[(df["item_id"] >= lo) & (df["item_id"] < hi)]
        x = torch.tensor(sub[["user_id", "item_id"]].values, dtype=torch.long)
        y = torch.tensor(sub["score"].values, dtype=torch.long)
        return TensorDataset(x, y)

    def _make_eval_pack(support, query_old, query_new):
        users = sorted(
            pd.concat([support["user_id"], query_old["user_id"], query_new["user_id"]]).unique()
        )
        user_remap = {user_id: index for index, user_id in enumerate(users)}

        def _pack(frame):
            uidx = frame["user_id"].map(user_remap).astype(int).values
            x = torch.tensor(np.stack([uidx, frame["item_id"].values], axis=1), dtype=torch.long)
            y = torch.tensor(frame["score"].values, dtype=torch.long)
            return x, y

        support_x, support_y = _pack(support)
        query_old_x, query_old_y = _pack(query_old)
        query_new_x, query_new_y = _pack(query_new)
        return {
            "n_test_users": len(users),
            "support_x": support_x,
            "support_y": support_y,
            "query_old_x": query_old_x,
            "query_old_y": query_old_y,
            "query_new_x": query_new_x,
            "query_new_y": query_new_y,
        }

    base = {
        "train_old_ds": _ds(df_train, 0, n_item_old),
        "train_new_ds": _ds(df_train, n_item_old, n_item_total),
        "num_students": int(df_train["user_id"].max()) + 1,
        "num_items": n_item_total,
        "old_user_ids": torch.tensor(
            df_train[df_train["item_id"] < n_item_old]["user_id"].unique(), dtype=torch.long
        ),
        "valid_eval_pack": _make_eval_pack(sup_valid, qry_valid_old, qry_valid_new),
        "eval_pack": _make_eval_pack(sup_test, qry_test_old, qry_test_new),
    }
    meta = {
        "n_user": n_user,
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "alpha": cfg["alpha"],
        "support_query_seed": support_query_seed,
        "support_frac": support_frac,
        "n_query_old": len(qry_test_old),
        "n_query_new": len(qry_test_new),
    }
    return ours, base, meta


# ==========================================================================
# Ours：六策略（G-NCDM），support/query 评测（复用 evaluate_recon）
# ==========================================================================
def run_ours(ours, meta, device, run_strategies=None):
    """Run Base plus the requested G-NCDM incremental strategies.

    ``Base`` is always required as the common task-1 anchor.  Passing a set in
    ``run_strategies`` avoids training unused ablations while preserving the
    legacy all-strategies behavior for existing callers.
    """
    n_user, alpha = meta["n_user"], meta["alpha"]
    n_item_old, n_know_old = ours["n_item_old"], ours["n_know_old"]
    n_item_new, n_know_new = ours["n_item_new"], ours["n_know_new"]
    log_old, log_full = ours["log_old"], ours["log_full"]
    Q_old, Q_expanded = ours["Q_old"], ours["Q_expanded"]

    def base_eval_fn(m):
        return evaluate_recon(m, ours["qry_valid_old"], ours["sup_valid_old_log"], device)

    def strat_eval_fn(qvalid_df):
        return lambda m: evaluate_recon(m, qvalid_df, ours["sup_valid_full_log"], device)

    def final_old(m):
        return evaluate_recon(m, ours["qry_test_old"], ours["sup_test_full_log"], device)

    def final_new(m):
        return evaluate_recon(m, ours["qry_test_new"], ours["sup_test_full_log"], device)

    rows = []

    def record(name, r_old, r_new, tmd):
        rows.append(
            {
                "Method": name,
                "AUC_old": r_old["auc"],
                "AUC_new": r_new["auc"] if r_new else "-",
                "RMSE_old": r_old["rmse"],
                "RMSE_new": r_new["rmse"] if r_new else "-",
                "ACC_old": r_old["acc"],
                "ACC_new": r_new["acc"] if r_new else "-",
                "F1_old": r_old["f1"],
                "F1_new": r_new["f1"] if r_new else "-",
                "RD": tmd if tmd is not None else "",
            }
        )
        ns = f" | 新 AUC={r_new['auc']:.4f} ACC={r_new['acc']:.4f}" if r_new else ""
        print(f"  [{name}] 旧 AUC={r_old['auc']:.4f} ACC={r_old['acc']:.4f}{ns}")

    print("\n=== Ours-1. Base ===")
    base = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=alpha,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device,
    ).to(device)
    train_real(
        base,
        ours["train_old"],
        log_old,
        list(base.parameters()),
        device,
        n_epoch=25,
        desc="Base",
        eval_fn=base_eval_fn,
    )
    populate_buffers(base, log_old, device)
    base_theta_old = base.get_Theta_buf().clone()
    record(
        "Base",
        evaluate_recon(base, ours["qry_test_old"], ours["sup_test_old_log"], device),
        None,
        None,
    )

    def run_strategy(name, expand_fn, params_fn, train_df, qvalid_df, mask_agg_old=False):
        m = fresh_base(base)
        expand_fn(m)
        populate_buffers(m, log_full, device)
        handles = []
        if mask_agg_old:

            def make_col_mask(k_old):
                def hook(grad):
                    g = grad.clone()
                    g[:, :k_old] = 0.0
                    return g

                return hook

            handles.append(m.theta_agg_mat.weight.register_hook(make_col_mask(n_know_old)))
            handles.append(m.psi_agg_mat.weight.register_hook(make_col_mask(n_know_old)))
        train_real(
            m,
            train_df,
            log_full,
            params_fn(m),
            device,
            n_epoch=25,
            desc=name,
            eval_fn=strat_eval_fn(qvalid_df),
        )
        for h in handles:
            h.remove()
        populate_buffers(m, log_full, device)
        # NOTE: RD 只量 θ 旧维漂移（仅 theta[:, :K_old]），量不到聚合矩阵旧列权重/共享 bias 漂移。
        # θ_old 来自旧编码器 f_nn，而 expand_topology 的 _freeze_parameters() 已冻结 f_nn
        # （requires_grad=False）→ 即便 Ours-Ablated 传 list(m.parameters()) 也训不动它 → RD=0；
        # 但 Ablated 训练了重建后的聚合矩阵旧列权重+bias（新题 Q_old≠0，二者有真实梯度），漂移使旧任务
        # 真退化，看 AUC_old 而非 RD。详细机理见 run_incremental_math1.py 的 run_strategy / DNA 注释。
        tmd = calculate_rd(base_theta_old.to(device), m.get_Theta_buf().to(device), n_know_old)
        record(name, final_old(m), final_new(m), tmd)

    def is_requested(name):
        return run_strategies is None or name in run_strategies

    if is_requested("Ours-Ablated"):
        print("\n=== Ours-2. Ablated ===")
        run_strategy(
            "Ours-Ablated",
            lambda m: m.expand_topology(n_item_new, n_know_new, Q_expanded),
            lambda m: list(m.parameters()),
            ours["train_new"],
            ours["qry_valid_new"],
        )
    if is_requested("Ours (Dynamic DNA)"):
        print("\n=== Ours-3. Dynamic DNA ===")
        run_strategy(
            "Ours (Dynamic DNA)",
            lambda m: m.expand_topology(n_item_new, n_know_new, Q_expanded),
            lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
            ours["train_new"],
            ours["qry_valid_new"],
            mask_agg_old=True,
        )
    if is_requested("Ours (LoRA)"):
        print("\n=== Ours-4. LoRA ===")
        run_strategy(
            "Ours (LoRA)",
            lambda m: m.expand_topology_lora(
                delta_M=n_item_new,
                delta_K=n_know_new,
                Q_expanded=Q_expanded,
                M_old=n_item_old,
                rank=16,
            ),
            lora_params,
            ours["train_new"],
            ours["qry_valid_new"],
        )
    if is_requested("Full Replay Oracle"):
        print("\n=== Ours-5. Full Replay Oracle ===")
        run_strategy(
            "Full Replay Oracle",
            lambda m: m.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded),
            lambda m: list(m.parameters()),
            pd.concat([ours["train_old"], ours["train_new"]], ignore_index=True),
            ours["qry_valid"],
        )
    if is_requested("Naive FT (NFT)"):
        print("\n=== Ours-6. Naive FT ===")
        run_strategy(
            "Naive FT (NFT)",
            lambda m: m.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded),
            lambda m: list(m.parameters()),
            ours["train_new"],
            ours["qry_valid_new"],
        )
    return rows


# ==========================================================================
# 基线：EWC / DER++ / C-LoRA（CognitiveBackbone），共享 support/query 冷启动
# ==========================================================================
def _bench(base):
    from avalanche.benchmarks import dataset_benchmark

    return dataset_benchmark(
        train_datasets=[base["train_old_ds"], base["train_new_ds"]],
        test_datasets=[base["train_old_ds"], base["train_new_ds"]],  # 占位，不用
    )


def _brow(method, old_m, new_m, tmd):
    return {
        "Method": method,
        "AUC_old": old_m[0],
        "AUC_new": new_m[0],
        "RMSE_old": old_m[1],
        "RMSE_new": new_m[1],
        "ACC_old": old_m[2],
        "ACC_new": new_m[2],
        "F1_old": old_m[3],
        "F1_new": new_m[3],
        "RD": tmd,
    }


def run_ewc(base, device, *, seed=42, lambdas=None, eval_pack=None, coldstart_seed=COLD_START_SEED):
    from avalanche.training.supervised import EWC

    print("\n=== Baseline EWC λ 扫描 ===")
    eval_pack = base["eval_pack"] if eval_pack is None else eval_pack
    lambdas = EWC_LAMBDA_SWEEP if lambdas is None else list(lambdas)
    rows = []
    for lam in lambdas:
        set_seed(seed)
        model = CognitiveBackbone(base["num_students"], base["num_items"], EMBED_DIM).to(device)
        strat = EWC(
            model,
            optim.Adam(model.parameters(), lr=LR),
            nn.CrossEntropyLoss(),
            ewc_lambda=lam,
            mode=EWC_MODE,
            train_mb_size=TRAIN_MB_SIZE,
            train_epochs=EWC_EPOCHS,
            eval_mb_size=256,
            device=device,
        )
        b0 = None
        for exp in _bench(base).train_stream:
            strat.train(exp)
            if exp.current_experience == 0:
                b0 = model.student_emb.weight.data.clone().cpu()
        old_m, new_m = coldstart_eval(model, eval_pack, device, seed=coldstart_seed)
        r = _brow(
            f"EWC (lambda={lam})",
            old_m,
            new_m,
            baseline_tmd(b0, model, base["old_user_ids"], EMBED_DIM),
        )
        r["lambda"] = lam
        rows.append(r)
        print(f"  λ={lam}: AUC_old={r['AUC_old']:.4f} AUC_new={r['AUC_new']:.4f}")
    return rows


def run_der(
    base,
    device,
    *,
    seed=42,
    eval_pack=None,
    coldstart_seed=COLD_START_SEED,
    mem_size=MEM_SIZE,
):
    from avalanche.training.supervised import DER

    print("\n=== Baseline DER++ ===")
    eval_pack = base["eval_pack"] if eval_pack is None else eval_pack
    set_seed(seed)
    model = CognitiveBackbone(base["num_students"], base["num_items"], EMBED_DIM).to(device)
    strat = DER(
        model,
        optim.Adam(model.parameters(), lr=LR),
        nn.CrossEntropyLoss(),
        mem_size=mem_size,
        alpha=DER_ALPHA,
        beta=DER_BETA,
        train_mb_size=TRAIN_MB_SIZE,
        train_epochs=DER_EPOCHS,
        eval_mb_size=256,
        device=device,
    )
    b0 = None
    for exp in _bench(base).train_stream:
        strat.train(exp)
        if exp.current_experience == 0:
            b0 = model.student_emb.weight.data.clone().cpu()
    old_m, new_m = coldstart_eval(model, eval_pack, device, seed=coldstart_seed)
    r = _brow(
        f"DER++ (mem={mem_size})",
        old_m,
        new_m,
        baseline_tmd(b0, model, base["old_user_ids"], EMBED_DIM),
    )
    r["mem_size"] = mem_size
    print(f"  DER++: AUC_old={r['AUC_old']:.4f} AUC_new={r['AUC_new']:.4f}")
    return r


def _train_phase(model, dataset, epochs, device, lambda_ortho=None):
    model.train()
    loader = DataLoader(dataset, batch_size=TRAIN_MB_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            if lambda_ortho is not None and lambda_ortho > 0:
                loss = loss + lambda_ortho * orthogonal_penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_clora(
    base, device, *, seed=42, lambdas=None, eval_pack=None, coldstart_seed=COLD_START_SEED
):
    print("\n=== Baseline C-LoRA λ_ortho 扫描 ===")
    eval_pack = base["eval_pack"] if eval_pack is None else eval_pack
    set_seed(seed)
    base_model = CognitiveBackbone(base["num_students"], base["num_items"], EMBED_DIM).to(device)
    _train_phase(base_model, base["train_old_ds"], BASE_EPOCHS, device, lambda_ortho=None)
    base_state = copy.deepcopy(base_model.state_dict())
    lambdas = CLORA_LAMBDA_SWEEP if lambdas is None else list(lambdas)
    rows = []
    for lam in lambdas:
        set_seed(seed)
        model = CognitiveBackbone(base["num_students"], base["num_items"], EMBED_DIM).to(device)
        model.load_state_dict(base_state)
        b0 = model.student_emb.weight.data.clone().cpu()
        inject_lora(model)
        model.to(device)
        _train_phase(model, base["train_new_ds"], CLORA_EPOCHS, device, lambda_ortho=float(lam))
        old_m, new_m = coldstart_eval(model, eval_pack, device, seed=coldstart_seed)
        r = _brow(
            f"C-LoRA (lambda={lam})",
            old_m,
            new_m,
            baseline_tmd(b0, model, base["old_user_ids"], EMBED_DIM),
        )
        r["lambda"] = lam
        rows.append(r)
        print(f"  λ={lam}: AUC_old={r['AUC_old']:.4f} AUC_new={r['AUC_new']:.4f}")
    return rows


def _pick_balanced(rows):
    return max(rows, key=lambda r: (r["AUC_old"] + r["AUC_new"]) / 2.0)


def select_baselines_on_validation(base, device, *, seed=42, coldstart_seed=COLD_START_SEED):
    """Tune EWC/C-LoRA on valid support/query rows, then evaluate once on test.

    The returned rows are intended for the formal per-seed workflow. Keeping
    test selection outside this helper avoids the historic test-set lambda
    selection used by the legacy wide-table writer.
    """

    valid_pack = base["valid_eval_pack"]
    test_pack = base["eval_pack"]
    ewc_valid = run_ewc(
        base,
        device,
        seed=seed,
        eval_pack=valid_pack,
        coldstart_seed=coldstart_seed,
    )
    ewc_lambda = _pick_balanced(ewc_valid)["lambda"]
    ewc_test = run_ewc(
        base,
        device,
        seed=seed,
        lambdas=[ewc_lambda],
        eval_pack=test_pack,
        coldstart_seed=coldstart_seed,
    )[0]
    ewc_test["selected_lambda"] = ewc_lambda
    ewc_test["selection_source"] = "validation_balanced_auc"

    clora_valid = run_clora(
        base,
        device,
        seed=seed,
        eval_pack=valid_pack,
        coldstart_seed=coldstart_seed,
    )
    clora_lambda = _pick_balanced(clora_valid)["lambda"]
    clora_test = run_clora(
        base,
        device,
        seed=seed,
        lambdas=[clora_lambda],
        eval_pack=test_pack,
        coldstart_seed=coldstart_seed,
    )[0]
    clora_test["selected_lambda"] = clora_lambda
    clora_test["selection_source"] = "validation_balanced_auc"

    der_test = run_der(
        base,
        device,
        seed=seed,
        eval_pack=test_pack,
        coldstart_seed=coldstart_seed,
    )
    der_test["selection_source"] = "fixed_hyperparameters"
    return [ewc_test, der_test, clora_test]


def run_fixed_baselines(
    base,
    device,
    *,
    seed=42,
    ewc_lambda,
    der_mem_size=MEM_SIZE,
    clora_lambda=None,
    coldstart_seed=COLD_START_SEED,
):
    """Run only preselected baseline hyperparameters without any lambda sweep."""

    test_pack = base["eval_pack"]
    rows = []
    if ewc_lambda is not None:
        ewc_row = run_ewc(
            base,
            device,
            seed=seed,
            lambdas=[ewc_lambda],
            eval_pack=test_pack,
            coldstart_seed=coldstart_seed,
        )[0]
        ewc_row["selected_lambda"] = ewc_lambda
        ewc_row["selection_source"] = "fixed_from_existing_result"
        rows.append(ewc_row)

    if der_mem_size is not None:
        der_row = run_der(
            base,
            device,
            seed=seed,
            eval_pack=test_pack,
            coldstart_seed=coldstart_seed,
            mem_size=der_mem_size,
        )
        der_row["selection_source"] = "fixed_from_existing_result"
        rows.append(der_row)

    if clora_lambda is not None:
        clora_row = run_clora(
            base,
            device,
            seed=seed,
            lambdas=[clora_lambda],
            eval_pack=test_pack,
            coldstart_seed=coldstart_seed,
        )[0]
        clora_row["selected_lambda"] = clora_lambda
        clora_row["selection_source"] = "fixed_from_existing_result"
        rows.append(clora_row)
    return rows


def run_trial_for_statistics(
    cfg,
    device,
    *,
    train_seed=42,
    support_query_seed=SPLIT_SEED,
    support_frac=SUPPORT_FRAC,
    deterministic=True,
    fixed_baseline_config=None,
    ours_strategies=None,
):
    """Run one leakage-free user-split trial and return raw per-method rows.

    When ``fixed_baseline_config`` is provided, EWC/C-LoRA use its preselected
    values directly and no lambda sweep is run. Otherwise the legacy formal
    path selects lambdas on validation support/query rows.
    """

    set_seed(train_seed, deterministic=deterministic)
    ours, base, meta = prepare(
        cfg,
        device,
        support_query_seed=support_query_seed,
        support_frac=support_frac,
    )
    ours_rows = run_ours(ours, meta, device, run_strategies=ours_strategies)
    coldstart_seed = int(train_seed) + 1_000_003
    if fixed_baseline_config is None:
        baseline_rows = select_baselines_on_validation(
            base, device, seed=train_seed, coldstart_seed=coldstart_seed
        )
    else:
        baseline_rows = run_fixed_baselines(
            base,
            device,
            seed=train_seed,
            ewc_lambda=fixed_baseline_config["ewc_lambda"],
            der_mem_size=fixed_baseline_config["der_mem_size"],
            clora_lambda=fixed_baseline_config.get("clora_lambda"),
            coldstart_seed=coldstart_seed,
        )
    return ours_rows + baseline_rows, meta


# ==========================================================================
# 合并表 + markdown（带 RD 红线脚注）
# ==========================================================================
COLS = [
    "Method",
    "AUC_old",
    "AUC_new",
    "RMSE_old",
    "RMSE_new",
    "ACC_old",
    "ACC_new",
    "F1_old",
    "F1_new",
    "RD",
]


def _fmt(x):
    if isinstance(x, str):
        return x
    return "-" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x:.4f}"


def write_tables(split_name, ours_rows, ewc_rows, der_row, clora_rows):
    ewc_best, clora_best = _pick_balanced(ewc_rows), _pick_balanced(clora_rows)
    merged = list(ours_rows) + [
        {k: ewc_best[k] for k in COLS},
        {k: der_row[k] for k in COLS},
        {k: clora_best[k] for k in COLS},
    ]
    df = pd.DataFrame(merged, columns=COLS)
    csv_path = os.path.join(SAVE_DIR, f"all_methods_{split_name}.csv")
    df.to_csv(csv_path, index=False)

    lines = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in merged:
        lines.append("| " + " | ".join([str(r["Method"])] + [_fmt(r[c]) for c in COLS[1:]]) + " |")
    note = (
        f"\n*口径*：全部 9 方法在 **{split_name}** 同一份 support/query 划分（frac={SUPPORT_FRAC}, "
        f"seed={SPLIT_SEED}）的**同一批 query 行**上评测，AUC/ACC/F1/RMSE 可逐行直接对比。\n"
        f"*RD 红线*：Ours 行的 RD 在 **G-NCDM 概念 θ 空间**（架构隔离→0/极小）；"
        f"EWC/DER/C-LoRA 行的 RD 在 **embedding 空间**，量级**不可**与 Ours 直接比，仅看是否>0。\n"
        f"*均衡点*：EWC λ={ewc_best['lambda']}、C-LoRA λ={clora_best['lambda']}"
        f"（各取 avg(AUC_old,AUC_new) 最大）。骨干不同，勿称纯策略胜出。\n"
    )
    md_path = os.path.join(SAVE_DIR, f"all_methods_{split_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)

    # 同时落盘两条 λ sweep 供画前沿
    pd.DataFrame(ewc_rows).to_csv(
        os.path.join(SAVE_DIR, f"ewc_lambda_sweep_{split_name}.csv"), index=False
    )
    pd.DataFrame(clora_rows).to_csv(
        os.path.join(SAVE_DIR, f"clora_lambda_sweep_{split_name}.csv"), index=False
    )

    print("\n" + "=" * 70)
    print(f" 九方法统一对比（{split_name}，support/query 同口径）")
    print("=" * 70)
    print("\n".join(lines))
    print(f"\n>>> 写入 {csv_path}\n>>> 写入 {md_path}")
    print("=" * 70)


def run_one(split_name, cfg, device):
    print(f"\n{'#' * 72}\n# {split_name}（9 方法统一 support/query 口径）\n{'#' * 72}")
    set_seed(42)
    ours, base, meta = prepare(cfg, device)
    ours_rows = run_ours(ours, meta, device)
    ewc_rows = run_ewc(base, device)
    der_row = run_der(base, device)
    clora_rows = run_clora(base, device)
    write_tables(split_name, ours_rows, ewc_rows, der_row, clora_rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    repo_root = os.path.dirname(gncdm_dir)

    configs = {
        "math1_user_split": {
            "train": os.path.join(repo_root, "data", "math1", "user_split", "train.csv"),
            "valid": os.path.join(repo_root, "data", "math1", "user_split", "valid.csv"),
            "test": os.path.join(repo_root, "data", "math1", "user_split", "test.csv"),
            "Q": os.path.join(gncdm_dir, "data", "math1_Q_matrix.npy"),
            "n_user": 4209,
            "n_item": 20,
            "n_know": 11,
            "new_concepts": [0, 1, 3, 6],
            "alpha": 0.70,
        },
    }
    if RUN_A0910:
        a0910 = os.path.join(repo_root, "data", "a0910")
        Q = np.load(os.path.join(a0910, "Q_matrix.npy"))
        freq = (Q > 0).sum(axis=0)
        touched = np.zeros(Q.shape[0], dtype=bool)
        nc = []
        for k in np.argsort(freq):
            nc.append(int(k))
            touched |= Q[:, k] > 0
            if touched.sum() >= 0.34 * Q.shape[0]:
                break
        configs["a0910_user_split"] = {
            "train": os.path.join(a0910, "new_user_split", "train.csv"),
            "valid": os.path.join(a0910, "new_user_split", "valid.csv"),
            "test": os.path.join(a0910, "new_user_split", "test.csv"),
            "Q": os.path.join(a0910, "Q_matrix.npy"),
            "n_user": 4163,
            "n_item": 17746,
            "n_know": 123,
            # alpha=0.6：a0910 user_split 经 validation 选定（sweep_alpha_a0910_user 全扫 0.1~0.95，
            # valid_ACC 在 0.6 见顶 0.6989；test 端 ACC/F1 最优、test_AUC 仅差 0.3 峰值约 0.0015）。
            # 远优于原作者默认 0.9（test_AUC/ACC 各高约 0.019/0.011）。
            "new_concepts": sorted(nc),
            "alpha": 0.6,
        }

    if RUN_JUNYI:
        junyi = os.path.join(repo_root, "data", "junyi")
        Q = np.load(os.path.join(junyi, "Q_matrix.npy"))
        n_user = 0
        for f in ("train.csv", "valid.csv", "test.csv"):
            n_user = max(
                n_user,
                int(pd.read_csv(os.path.join(junyi, "new_user_split", f))["user_id"].max()) + 1,
            )
        configs["junyi_user_split"] = {
            "train": os.path.join(junyi, "new_user_split", "train.csv"),
            "valid": os.path.join(junyi, "new_user_split", "valid.csv"),
            "test": os.path.join(junyi, "new_user_split", "test.csv"),
            "Q": os.path.join(junyi, "Q_matrix.npy"),
            "n_user": n_user,
            "n_item": Q.shape[0],
            "n_know": Q.shape[1],
            # alpha=0.6：与 a0910 user_split 同值（per-split 最优待 sweep 复核）。
            "new_concepts": auto_new_concepts(Q, 0.34),
            "alpha": 0.6,
        }

    for split_name, cfg in configs.items():
        run_one(split_name, cfg, device)


if __name__ == "__main__":
    main()
