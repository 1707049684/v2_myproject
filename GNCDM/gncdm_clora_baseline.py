"""G-NCDM + C-LoRA 持续学习基线（Continual LoRA + 正交惩罚）—— random_split【方案二·最佳努力版】。

把 C-LoRA 直接挂到**原版 G-NCDM 骨干**的 GDF(f_nn/g_nn)与 IRF(ncd) 全连接层上，与主表六策略
**同骨干、同口径**（AUC/ACC/F1/RMSE 可并入主表；RD 在真·概念 θ 空间，量级可与 Ours 直接比）。

═══ 相对早期退化版（findings 第二十三轮 AUC_new≈0.5）的三处「最佳努力」修复 ═══
A. **归一化正交惩罚**：早期 `L_ortho = Σ‖W_base·ΔWᵀ‖²_F` 被巨阵 `g_nn[0]=Linear(n_user,K)` 的尺度
   主导，λ=1 即碾压 CE、把 ΔW 钉死为 0（退回冻结基座）。现改为**每层除以 ‖W_base‖²_F**，使各层
   惩罚尺度可比、λ 跨层有意义，过渡区间回到可扫范围。
B. **细扫 λ∈(0,1)**：早期 sweep [0,1,10,…] 全落在悬崖右侧；现扫 [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 10]。
C. **解冻新概念聚合列**：早期 `theta_agg_mat/psi_agg_mat` 的新概念列(xavier 初值)被冻结、从未训练
   → 新维度学不动、连 λ=0 都只有 0.65。现把这两张聚合矩阵的**新列 [:, n_know_old:] 加入训练**
   （backward hook 把旧列梯度清零，照搬主实验 DNA 的列分离做法），让新概念真正可学。
   注：新列是**全新参数**（非旧基座漂移），不受正交惩罚约束、直接 CE 学习，符合"学新维度"的需要。

【若仍弱于 Ours】可作论文论点：C-LoRA「冻结基座 + 小幅正交增量」为**已有维度的分布漂移**而设；
G-NCDM 增量要**从零学新概念维度**，正交约束反而阻碍 → 唯有 Ours 的专用新分支胜任。本版已尽最佳
努力调校，结论才站得住。

运行（math1/junyi 小、可本地验证；a0910 17746 题建议 GPU 服务器）：
    cd GNCDM
    python gncdm_clora_baseline.py            # 命令行参数切 "math1" / "a0910" / "junyi"
    python gncdm_clora_baseline.py junyi
    python gncdm_clora_baseline.py a0910
"""

import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # GNCDM/
EXPERIMENTS_DIR = os.path.join(THIS_DIR, "experiments")
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.join(EXPERIMENTS_DIR, "_core"))  # 核心库已移入 experiments/_core/

from core.model import GNCDM  # noqa: E402
from core.train import calculate_rd  # noqa: E402
from run_incremental_math1 import (  # noqa: E402
    SampleSet,
    build_log_mat,
    evaluate_buf,
    evaluate_recon,
    populate_buffers,
    remap_items,
    set_seed,
    strict_bipartition,
    train_real,
)

REPO_ROOT = os.path.dirname(THIS_DIR)
SAVE_DIR = os.path.join(THIS_DIR, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 选择数据集（math1/junyi 可本地快速验证；a0910 建议 GPU 服务器）=====
# 导入时只读取环境变量，避免将其他脚本的命令行参数（如 ``--dataset``）误当成数据集名。
# 直接运行本文件时的命令行数据集选择在底部 ``__main__`` 块处理。
DATASET = os.environ.get("GNCDM_CLORA_DATASET", "math1")
assert DATASET in ("math1", "a0910", "junyi"), (
    f"未知 DATASET={DATASET}（应为 math1 / a0910 / junyi）"
)

CONFIGS = {
    "math1": {
        "n_user": 4209,
        "n_item": 20,
        "n_know": 11,
        "alpha": 0.20,
        "new_concepts": [0, 1, 3, 6],
        "train": os.path.join(THIS_DIR, "data", "math1_train_0.8_0.2.csv"),
        "test": os.path.join(THIS_DIR, "data", "math1_test_0.8_0.2.csv"),
        "Q": os.path.join(THIS_DIR, "data", "math1_Q_matrix.npy"),
    },
    # alpha 与主表 run_incremental_a0910_random_split.py 的 ALPHA 对齐（0.1 全扫见顶，
    # 早前 0.9 未真扫、已被超越，此处同步更新以保证与 Base/Ours 同口径可比）。
    "a0910": {
        "n_user": 4163,
        "n_item": 17746,
        "n_know": 123,
        "alpha": 0.1,
        "new_concepts": "auto",
        "train": os.path.join(REPO_ROOT, "data", "a0910", "new_random_split", "train.csv"),
        "test": os.path.join(REPO_ROOT, "data", "a0910", "new_random_split", "test.csv"),
        "Q": os.path.join(REPO_ROOT, "data", "a0910", "Q_matrix.npy"),
    },
}


def _load_junyi_config():
    """junyi 维度不像 math1/a0910 固定，从文件读（对齐 run_incremental_junyi_random_split.py）。
    alpha=0.1 对齐主表 ALPHA（sweep_junyi_random_alpha.py 全扫选定）。"""
    junyi_dir = os.path.join(REPO_ROOT, "data", "junyi")
    q_path = os.path.join(junyi_dir, "Q_matrix.npy")
    if not os.path.exists(q_path):
        return None
    Q = np.load(q_path)
    n_item, n_know = int(Q.shape[0]), int(Q.shape[1])
    rnd = os.path.join(junyi_dir, "new_random_split")
    train, test = os.path.join(rnd, "train.csv"), os.path.join(rnd, "test.csv")
    n_user = max(
        int(pd.read_csv(os.path.join(rnd, f))["user_id"].max()) + 1
        for f in ("train.csv", "valid.csv", "test.csv")
    )
    return {
        "n_user": n_user,
        "n_item": n_item,
        "n_know": n_know,
        "alpha": 0.1,
        "new_concepts": "auto",
        "train": train,
        "test": test,
        "Q": q_path,
    }


def _config_for_dataset(dataset):
    if dataset not in ("math1", "a0910", "junyi"):
        raise ValueError(f"未知 DATASET={dataset}（应为 math1 / a0910 / junyi）")
    if dataset == "junyi" and dataset not in CONFIGS:
        junyi_cfg = _load_junyi_config()
        assert junyi_cfg is not None, (
            f"未找到 junyi 数据（期望 {os.path.join(REPO_ROOT, 'data', 'junyi')}）"
        )
        CONFIGS[dataset] = junyi_cfg
    return CONFIGS[dataset]


USER_DIM, ITEM_DIM = 32, 32
LR = 1e-3
BATCH = 256
BASE_EPOCHS = 25
CLORA_EPOCHS = 25
LORA_RANK = 8
LORA_ALPHA = 16
NEW_ITEM_FRAC = 0.34
# Fix B：细扫 λ∈(0,1) + 少量 ≥1（归一化后尺度已可比）
LAMBDA_ORTHO_SWEEP = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 10.0]


def auto_new_concepts(Q, new_item_frac=0.34):
    n_item = Q.shape[0]
    freq = (Q > 0).sum(axis=0)
    touched = np.zeros(n_item, dtype=bool)
    new_set = []
    for k in np.argsort(freq):
        new_set.append(int(k))
        touched |= Q[:, k] > 0
        if touched.sum() >= new_item_frac * n_item:
            break
    return sorted(new_set)


# ==========================================
# C-LoRA 组件：LoRA wrapper + 归一化软正交惩罚
# ==========================================
class LoRALinear(nn.Module):
    """冻结 nn.Linear/PosLinear 基座 + 低秩增量；lora_B 零初始化（挂载即 ΔW=0）。"""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1e-2)
        nn.init.zeros_(self.lora_B.weight)
        self.rank = rank
        self.scaling = alpha / rank

    def delta_w(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora_gncdm(model: GNCDM, rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
    """把 GDF(f_nn,g_nn) 与 IRF(ncd) 内所有 nn.Linear/PosLinear 原位包成 LoRALinear。
    聚合矩阵 theta_agg_mat/psi_agg_mat 不包（predict_response 直接读 .weight）。返回包裹层数。"""
    n_wrapped = 0
    for container in (model.f_nn, model.g_nn, model.ncd):
        for idx, child in enumerate(container):
            if isinstance(child, nn.Linear):
                container[idx] = LoRALinear(child, rank=rank, alpha=alpha)
                n_wrapped += 1
    return n_wrapped


def orthogonal_penalty(model: nn.Module) -> torch.Tensor:
    """Fix A：归一化软正交惩罚 L_ortho = Σ_layers ‖W_base·ΔWᵀ‖²_F / ‖W_base‖²_F。
    每层除以基座 Frobenius 范数²，消除巨阵（如 g_nn[0]=Linear(n_user,K)）的尺度主导，
    使 λ 跨层尺度可比、过渡区间回到可扫范围。"""
    loss_ortho = None
    for m in model.modules():
        if isinstance(m, LoRALinear):
            W_base = m.base_layer.weight.detach()  # (out, in)
            delta_W = m.delta_w()  # (out, in)
            num = torch.sum((W_base @ delta_W.t()) ** 2)
            den = torch.sum(W_base**2) + 1e-8
            term = num / den
            loss_ortho = term if loss_ortho is None else loss_ortho + term
    return loss_ortho


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def make_col_mask(k_old):
    """backward hook：把聚合矩阵权重的旧概念列 [:, :k_old] 梯度清零（只训新概念列）。"""

    def hook(grad):
        g = grad.clone()
        g[:, :k_old] = 0.0
        return g

    return hook


# ==========================================
# Phase 2 训练：标准 G-NCDM 前向 + BCE + λ·归一化正交惩罚
# ==========================================
def train_clora_phase2(
    model,
    samples_df,
    full_log_mat,
    params,
    device,
    lambda_ortho,
    n_epoch=CLORA_EPOCHS,
    batch_size=BATCH,
    lr=LR,
    desc="C-LoRA",
    history=None,
    history_eval_fn=None,
):
    """history/history_eval_fn：可选，画收敛曲线用。原函数不做 checkpoint 选优（无 best_state，
    直接用最后一轮），故每 epoch 额外评测不影响任何既有行为。"""
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
            user_log = log_t[user_ids]
            item_log = log_t[:, item_ids].T
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
        if history is not None and history_eval_fn is not None:
            history.append({"epoch": epoch + 1, **history_eval_fn(model)})
        if (epoch + 1) % 5 == 0 or epoch == n_epoch - 1:
            print(
                f"    [{desc}] epoch {epoch + 1}/{n_epoch} CE={total_ce / len(loader):.4f} "
                f"L_ortho(norm)={total_ortho / len(loader):.4e}"
            )
    return model


# ==========================================
# 数据装载：random_split + 严格拓扑二分（全维建模）
# ==========================================
def load_partition(cfg):
    df_train = pd.read_csv(cfg["train"])
    df_test = pd.read_csv(cfg["test"])
    Q_mat = np.load(cfg["Q"])
    n_user, n_item, n_know = cfg["n_user"], cfg["n_item"], cfg["n_know"]

    new_concepts = (
        auto_new_concepts(Q_mat, NEW_ITEM_FRAC)
        if cfg["new_concepts"] == "auto"
        else list(cfg["new_concepts"])
    )
    Q_full, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, new_concepts)
    df_train = remap_items(df_train, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    assert Q_full[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(
        f">>> [{cfg.get('name', DATASET)}] 新概念={len(new_concepts)}/{n_know}, "
        f"旧题(Task0)={n_item_old} "
        f"新题(Task1)={n_item - n_item_old}, 旧概念={n_know_old}"
    )

    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    train_new = df_train[df_train["item_id"] >= n_item_old].copy()
    test_old = df_test[df_test["item_id"] < n_item_old].copy()
    test_new = df_test[df_test["item_id"] >= n_item_old].copy()
    return {
        "Q_full": Q_full,
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "train_old": train_old,
        "train_new": train_new,
        "test_old": test_old,
        "test_new": test_new,
        "log_old_only": build_log_mat(train_old, n_user, n_item),
        "log_full": build_log_mat(df_train, n_user, n_item),
    }


def _split_support_query(df, frac=0.5, seed=7):
    sup = df.groupby("user_id", group_keys=False).sample(frac=frac, random_state=seed)
    return sup, df.drop(sup.index)


def load_partition_user_split(cfg, frac=0.5, seed=7):
    """user_split：需 valid；评测走 support/query（与 eval_all_methods_user_split 同口径）。"""
    df_train = pd.read_csv(cfg["train"])
    df_valid = pd.read_csv(cfg["valid"])
    df_test = pd.read_csv(cfg["test"])
    Q_mat = np.load(cfg["Q"])
    n_user, n_item, n_know = cfg["n_user"], cfg["n_item"], cfg["n_know"]

    new_concepts = (
        auto_new_concepts(Q_mat, NEW_ITEM_FRAC)
        if cfg["new_concepts"] == "auto"
        else list(cfg["new_concepts"])
    )
    Q_full, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, new_concepts)
    df_train = remap_items(df_train, item_id_map)
    df_valid = remap_items(df_valid, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    assert Q_full[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(
        f">>> [user_split] 新概念={len(new_concepts)}/{n_know}, 旧题={n_item_old} "
        f"新题={n_item - n_item_old}, 旧概念={n_know_old}"
    )

    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    train_new = df_train[df_train["item_id"] >= n_item_old].copy()
    sup_test, qry_test = _split_support_query(df_test, frac=frac, seed=seed)
    qry_test_old = qry_test[qry_test["item_id"] < n_item_old].copy()
    qry_test_new = qry_test[qry_test["item_id"] >= n_item_old].copy()
    print(
        f"  support/query test: support={len(sup_test)} "
        f"query old={len(qry_test_old)} new={len(qry_test_new)}"
    )
    return {
        "Q_full": Q_full,
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "train_old": train_old,
        "train_new": train_new,
        "log_old_only": build_log_mat(train_old, n_user, n_item),
        "log_full": build_log_mat(df_train, n_user, n_item),
        "sup_test_full_log": build_log_mat(sup_test, n_user, n_item),
        "qry_test_old": qry_test_old,
        "qry_test_new": qry_test_new,
    }


def run_user_split(
    cfg,
    device,
    split_tag="user_split",
    lambda_sweep=None,
    *,
    train_seed=42,
    support_query_seed=7,
    support_frac=0.5,
    write_output=True,
):
    """G-NCDM+C-LoRA · user_split · λ 扫描 · support/query 评测。"""
    sweep = LAMBDA_ORTHO_SWEEP if lambda_sweep is None else lambda_sweep
    meta = load_partition_user_split(cfg, frac=support_frac, seed=support_query_seed)

    set_seed(train_seed)
    base = _new_model(cfg, meta, device)
    print(f"\n>>> Phase 1：Base（旧题，{BASE_EPOCHS} ep）...")
    train_real(
        base,
        meta["train_old"],
        meta["log_old_only"],
        list(base.parameters()),
        device,
        n_epoch=BASE_EPOCHS,
        desc="Base(US)",
    )
    populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    base_state = copy.deepcopy(base.state_dict())

    rows = []
    for lam in sweep:
        print(f"\n========== ortho_lambda = {lam} ==========")
        set_seed(train_seed)
        model = _new_model(cfg, meta, device)
        model.load_state_dict(base_state)
        n_know_old = meta["n_know_old"]
        model._freeze_parameters()
        inject_lora_gncdm(model, rank=LORA_RANK, alpha=LORA_ALPHA)
        model.theta_agg_mat.weight.requires_grad = True
        model.psi_agg_mat.weight.requires_grad = True
        handles = [
            model.theta_agg_mat.weight.register_hook(make_col_mask(n_know_old)),
            model.psi_agg_mat.weight.register_hook(make_col_mask(n_know_old)),
        ]
        model.to(device)
        params = lora_parameters(model) + [model.theta_agg_mat.weight, model.psi_agg_mat.weight]
        train_clora_phase2(
            model,
            meta["train_new"],
            meta["log_full"],
            params,
            device,
            lambda_ortho=lam,
            desc=f"λ={lam}",
        )
        for h in handles:
            h.remove()
        populate_buffers(model, meta["log_full"], device)
        r_old = evaluate_recon(model, meta["qry_test_old"], meta["sup_test_full_log"], device)
        r_new = evaluate_recon(model, meta["qry_test_new"], meta["sup_test_full_log"], device)
        tmd = calculate_rd(
            base_theta_ref[:, :n_know_old].to(device), model.get_Theta_buf().to(device), n_know_old
        )
        r = {
            "ortho_lambda": lam,
            "AUC_old": r_old["auc"],
            "AUC_new": r_new["auc"],
            "RMSE_old": r_old["rmse"],
            "RMSE_new": r_new["rmse"],
            "ACC_old": r_old["acc"],
            "ACC_new": r_new["acc"],
            "F1_old": r_old["f1"],
            "F1_new": r_new["f1"],
            "RD": tmd,
        }
        rows.append(r)
        print(
            f"  [λ={lam}] 旧: AUC={r['AUC_old']:.4f} ACC={r['ACC_old']:.4f} | "
            f"新: AUC={r['AUC_new']:.4f} ACC={r['ACC_new']:.4f} | RD={r['RD']:.4f}"
        )

    if write_output:
        out = os.path.join(
            SAVE_DIR, f"clora_gncdm_lambda_sweep_{cfg.get('name', 'ds')}_{split_tag}.csv"
        )
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n写入 {out}")
    return rows


def run_fixed_user_split(
    cfg,
    device,
    *,
    lambda_ortho,
    seed,
    support_query_seed=7,
    support_frac=0.5,
):
    """Run exactly one preselected G-NCDM C-LoRA configuration for one seed."""

    row = run_user_split(
        cfg,
        device,
        lambda_sweep=[lambda_ortho],
        train_seed=seed,
        support_query_seed=support_query_seed,
        support_frac=support_frac,
        write_output=False,
    )[0]
    return {
        "Method": f"C-LoRA-GNCDM (lambda={lambda_ortho:g})",
        **row,
        "selected_lambda": lambda_ortho,
        "selection_source": "fixed_from_existing_result",
    }


def _new_model(cfg, meta, device):
    return GNCDM(
        n_user=cfg["n_user"],
        n_item=cfg["n_item"],
        n_know=cfg["n_know"],
        user_dim=USER_DIM,
        item_dim=ITEM_DIM,
        alpha=cfg["alpha"],
        Q_mat=meta["Q_full"],
        monotonicity_assumption=True,
        device=device,
    ).to(device)


# ==========================================
# 单个 λ：恢复 Phase-1 基座 → 冻结 + 挂 LoRA + 解冻新概念聚合列 → 训 Task1 → 评测
# ==========================================
def run_one_lambda(
    cfg,
    base_state,
    base_theta_ref,
    meta,
    lambda_ortho,
    device,
    history=None,
    history_eval_fn=None,
    n_epoch=CLORA_EPOCHS,
    seed=42,
):
    set_seed(seed)
    model = _new_model(cfg, meta, device)
    model.load_state_dict(base_state)
    n_know_old = meta["n_know_old"]

    model._freeze_parameters()
    n_wrapped = inject_lora_gncdm(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    # Fix C：解冻两张聚合矩阵的新概念列（旧列梯度由 hook 清零）
    model.theta_agg_mat.weight.requires_grad = True
    model.psi_agg_mat.weight.requires_grad = True
    handles = [
        model.theta_agg_mat.weight.register_hook(make_col_mask(n_know_old)),
        model.psi_agg_mat.weight.register_hook(make_col_mask(n_know_old)),
    ]
    model.to(device)
    params = lora_parameters(model) + [model.theta_agg_mat.weight, model.psi_agg_mat.weight]
    print(
        f"  -- λ_ortho={lambda_ortho}：冻结基座 + LoRA({n_wrapped} 层) + 新概念聚合列；"
        f"可训张量={len(params)}（{n_epoch} ep）--"
    )

    train_clora_phase2(
        model,
        meta["train_new"],
        meta["log_full"],
        params,
        device,
        lambda_ortho=lambda_ortho,
        n_epoch=n_epoch,
        desc=f"λ={lambda_ortho}",
        history=history,
        history_eval_fn=history_eval_fn,
    )
    for h in handles:
        h.remove()

    populate_buffers(model, meta["log_full"], device)
    r_old = evaluate_buf(model, meta["test_old"], device)
    r_new = evaluate_buf(model, meta["test_new"], device)
    tmd = calculate_rd(
        base_theta_ref[:, :n_know_old].to(device), model.get_Theta_buf().to(device), n_know_old
    )
    return {
        "ortho_lambda": lambda_ortho,
        "AUC_old": r_old["auc"],
        "AUC_new": r_new["auc"],
        "RMSE_old": r_old["rmse"],
        "RMSE_new": r_new["rmse"],
        "ACC_old": r_old["acc"],
        "ACC_new": r_new["acc"],
        "F1_old": r_old["f1"],
        "F1_new": r_new["f1"],
        "RD": tmd,
    }


def run_fixed_random_split(cfg, device, *, lambda_ortho, seed):
    """Run exactly one preselected G-NCDM C-LoRA configuration for one seed."""

    meta = load_partition(cfg)
    set_seed(seed)
    base = _new_model(cfg, meta, device)
    print(f"\n>>> Phase 1：Base（旧题，{BASE_EPOCHS} ep；seed={seed}）...")
    train_real(
        base,
        meta["train_old"],
        meta["log_old_only"],
        list(base.parameters()),
        device,
        n_epoch=BASE_EPOCHS,
        desc="Base",
    )
    populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    base_state = copy.deepcopy(base.state_dict())
    row = run_one_lambda(
        cfg,
        base_state,
        base_theta_ref,
        meta,
        lambda_ortho,
        device,
        seed=seed,
    )
    return {
        "Method": f"C-LoRA-GNCDM (lambda={lambda_ortho:g})",
        **row,
        "selected_lambda": lambda_ortho,
        "selection_source": "fixed_from_existing_result",
    }


def run_sweep(dataset=DATASET):
    cfg = _config_for_dataset(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | dataset = {dataset}")
    if device.type == "cpu" and dataset == "a0910":
        print("[WARN] CPU 上 a0910(17746 题) 跑 G-NCDM+C-LoRA sweep 很慢，建议 GPU 服务器。")
    if dataset == "junyi":
        print(f"[junyi] dims: n_user={cfg['n_user']} n_item={cfg['n_item']} n_know={cfg['n_know']}")
    print(
        f">>> 超参: rank={LORA_RANK} alpha={LORA_ALPHA} alpha_mix={cfg['alpha']} "
        f"base_ep={BASE_EPOCHS} clora_ep={CLORA_EPOCHS} lr={LR} λ_sweep={LAMBDA_ORTHO_SWEEP}"
    )

    meta = load_partition(cfg)

    # Phase 1：Base 全维（旧题；新题列恒 0、不获梯度），快照供各 λ 复用
    set_seed(42)
    base = _new_model(cfg, meta, device)
    print(f"\n>>> Phase 1：训练 Base（旧题 D_old，全参，{BASE_EPOCHS} ep）...")
    train_real(
        base,
        meta["train_old"],
        meta["log_old_only"],
        list(base.parameters()),
        device,
        n_epoch=BASE_EPOCHS,
        desc="Base",
    )
    populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    base_state = copy.deepcopy(base.state_dict())

    print("\n>>> 启动 G-NCDM+C-LoRA λ_ortho 扫描（最佳努力版）...")
    rows = []
    for lam in LAMBDA_ORTHO_SWEEP:
        print(f"\n========== ortho_lambda = {lam} ==========")
        r = run_one_lambda(cfg, base_state, base_theta_ref, meta, lam, device)
        rows.append(r)
        print(
            f"  [λ={lam}] 旧: AUC={r['AUC_old']:.4f} ACC={r['ACC_old']:.4f} | "
            f"新: AUC={r['AUC_new']:.4f} ACC={r['ACC_new']:.4f} | RD={r['RD']:.4f}"
        )

    out = os.path.join(SAVE_DIR, f"clora_gncdm_lambda_sweep_{dataset}_random_split.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n" + "=" * 64)
    print(f" G-NCDM + C-LoRA λ_ortho 扫描（{dataset} random_split，最佳努力版）")
    print("=" * 64)
    print("\n| λ_ortho | AUC_old | AUC_new | ACC_old | ACC_new | F1_old | F1_new | RD(concept-θ) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['ortho_lambda']} | {r['AUC_old']:.4f} | {r['AUC_new']:.4f} | "
            f"{r['ACC_old']:.4f} | {r['ACC_new']:.4f} | {r['F1_old']:.4f} | "
            f"{r['F1_new']:.4f} | {r['RD']:.4f} |"
        )
    print(f"\n结果已写入 {out}")
    print("=" * 64)


if __name__ == "__main__":
    cli_dataset = sys.argv[1] if len(sys.argv) > 1 else DATASET
    run_sweep(cli_dataset)
