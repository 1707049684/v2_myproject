# -*- coding: utf-8 -*-
"""DER++ (Dark Experience Replay++, NeurIPS 2020) 高保真移植到 G-NCDM。

相比 X-DER（TPAMI 2022），DER++ 去掉了：
  - L_Future（ΔK 潜通道反激活）
  - 动态记忆修正（memory revision, z_past clamp 更新）

训练目标（Task2）：
    L_total = L_BCE(B_new) + α·L_KD(B_buf) + β·L_BCE_buf(B_buf)
  - L_BCE     : 新题流 BCE on σ(z_current)。
  - L_KD      : logit 空间蒸馏 || z_current − z_past ||_2^2（忠实 DER++ 原意）。
  - L_BCE_buf : buffer 旧题真实分数 y 的 BCE（DER++ β 项，防历史物理表征崩溃）。

骨干/扩容：fresh_base(base) + full_replay_oracle_expand_topology，与 X-DER 一致（单一共享网络、
全参可训）。记忆缓冲区存 {u, i, s, z_past}，z_past 为 Task1 末的 raw logit。

口径：random_split / mode='buf' 无泄漏预测（populate_buffers + evaluate_buf），与 Ours 主表
及 X-DER 行直接可比。产物：incremental_result/derpp_{ds}_random_split.{csv,md}（单行）。
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(os.path.dirname(HERE))  # _core/ -> experiments/ -> GNCDM/
for _p in (HERE, gncdm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_incremental_math1 import (  # noqa: E402
    SAVE_DIR,
    SampleSet,
    build_log_mat,
    evaluate_buf,
    fresh_base,
    populate_buffers,
    remap_items,
    set_seed,
    strict_bipartition,
    train_real,
)
from core.model import GNCDM  # noqa: E402
from core.train import calculate_tmd  # noqa: E402

# 复用 X-DER 的 logit 抽取与 buffer 构建（接口完全一致）
from run_xder import _forward_logit, build_buffer  # noqa: E402

COLS = [
    "Method", "AUC_old", "AUC_new", "RMSE_old", "RMSE_new",
    "ACC_old", "ACC_new", "F1_old", "F1_new", "TMD",
]


def train_derpp(
    model,
    train_new,
    log_full,
    buffer,
    device,
    valid_eval_fn,
    alpha_kd=0.5,
    beta_buf=0.5,
    n_epoch=25,
    lr=1e-3,
    batch_size=256,
    buf_batch=256,
):
    """Task2：L_BCE(new) + α·L_KD(buf) + β·L_BCE_buf(buf)，全参可训。
    无 L_Future，无 memory revision（DER++ 原版）。按 valid acc 选最优快照。"""
    loader = DataLoader(SampleSet(train_new), batch_size=batch_size, shuffle=True)
    log_t = torch.tensor(log_full, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_buf = buffer["u"].shape[0]
    best_metric, best_state = -1.0, None

    for epoch in range(n_epoch):
        model.train()
        tot, t_bce, t_kd, t_bbuf = 0.0, 0.0, 0.0, 0.0
        for user_ids, item_ids, score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            score = score.to(device).unsqueeze(1)

            # L_BCE: 新题流
            z_new, _ = _forward_logit(model, log_t[user_ids], log_t[:, item_ids].T, user_ids, item_ids)
            loss_bce = F.binary_cross_entropy_with_logits(z_new, score)

            # 采 buffer 旧题
            sel = torch.randint(0, n_buf, (min(buf_batch, n_buf),), device=device)
            bu, bi = buffer["u"][sel], buffer["i"][sel]
            by = buffer["s"][sel].unsqueeze(1)
            bz = buffer["z_past"][sel].unsqueeze(1)
            z_buf, _ = _forward_logit(model, log_t[bu], log_t[:, bi].T, bu, bi)

            # L_KD: logit 空间蒸馏
            loss_kd = F.mse_loss(z_buf, bz)
            # L_BCE_buf: buffer 真实分数硬标签
            loss_bbuf = F.binary_cross_entropy_with_logits(z_buf, by)

            loss = loss_bce + alpha_kd * loss_kd + beta_buf * loss_bbuf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
            t_bce += loss_bce.item()
            t_kd += loss_kd.item()
            t_bbuf += loss_bbuf.item()

        vr = valid_eval_fn(model)
        vm = vr["acc"]
        if vm > best_metric:
            best_metric = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        nb = len(loader)
        print(
            f"    [DER++] epoch {epoch + 1}/{n_epoch} loss={tot / nb:.4f} "
            f"(bce={t_bce / nb:.4f} kd={t_kd / nb:.4f} bce_buf={t_bbuf / nb:.4f}) "
            f"valid_acc={vm:.4f} best={best_metric:.4f}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def run_derpp(
    split_name,
    ds_name,
    train_path,
    valid_path,
    test_path,
    Q_path,
    device,
    n_user,
    n_item_total,
    n_know_total,
    new_concepts,
    alpha=0.2,
    alpha_kd=0.5,
    beta_buf=0.5,
    buffer_size=5000,
    n_epoch=25,
    lr=1e-3,
    batch_size=256,
    buf_batch=256,
):
    """在一个数据集的 random_split 上跑 DER++，产出单行结果（列与 all_methods 一致）。"""
    print(f"\n{'#' * 70}\n# DER++  {split_name}  (G-NCDM 骨干, mode=buf)\n{'#' * 70}")
    set_seed(42)

    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    NEW_CONCEPTS = list(new_concepts)
    Q_mat, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, NEW_CONCEPTS)
    df_train = remap_items(df_train, item_id_map)
    df_valid = remap_items(df_valid, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    n_item_new, n_know_new = n_item_total - n_item_old, n_know_total - n_know_old
    Q_old = Q_mat[:n_item_old, :n_know_old].copy()
    Q_expanded = Q_mat.copy()
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(
        f"严格拓扑二分 dK={NEW_CONCEPTS}: 旧题={n_item_old} 新题={n_item_new}, "
        f"旧概念={n_know_old} 新概念={n_know_new}"
    )

    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    train_new = df_train[df_train["item_id"] >= n_item_old].copy()
    test_old = df_test[df_test["item_id"] < n_item_old].copy()
    test_new = df_test[df_test["item_id"] >= n_item_old].copy()
    valid_old = df_valid[df_valid["item_id"] < n_item_old].copy()
    valid_new = df_valid[df_valid["item_id"] >= n_item_old].copy()
    valid_comb = pd.concat([valid_old, valid_new], ignore_index=True)

    log_old = build_log_mat(train_old, n_user, n_item_old)
    log_full = build_log_mat(df_train, n_user, n_item_total)

    # Task1: 训练 base（旧题，全参），与 Ours 的 Base 同口径
    print("\n=== Task1: Base (DER++ 起点) ===")
    base = GNCDM(
        n_user=n_user, n_item=n_item_old, n_know=n_know_old,
        user_dim=32, item_dim=32, alpha=alpha, Q_mat=Q_old,
        monotonicity_assumption=True, device=device,
    ).to(device)

    def base_eval_fn(m):
        populate_buffers(m, log_old, device)
        return evaluate_buf(m, valid_old, device)

    train_real(
        base, train_old, log_old, list(base.parameters()), device,
        n_epoch=25, desc="Base(DER++)", eval_fn=base_eval_fn,
    )
    populate_buffers(base, log_old, device)
    base_theta_old = base.get_Theta_buf().clone()

    # 记忆库（raw logit z_past，与 X-DER 同口径）
    buffer = build_buffer(base, train_old, log_old, device, buffer_size)

    # Task2: 扩容（共享网络全参可训）+ DER++ 训练
    print("\n=== Task2: DER++ (L_BCE + α·L_KD + β·L_BCE_buf) ===")
    model = fresh_base(base)
    model.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded)

    def valid_eval_fn(m):
        populate_buffers(m, log_full, device)
        return evaluate_buf(m, valid_comb, device)

    train_derpp(
        model, train_new, log_full, buffer, device, valid_eval_fn,
        alpha_kd=alpha_kd, beta_buf=beta_buf,
        n_epoch=n_epoch, lr=lr, batch_size=batch_size, buf_batch=buf_batch,
    )

    # 评测（buf 无泄漏）+ TMD
    populate_buffers(model, log_full, device)
    r_old = evaluate_buf(model, test_old, device)
    r_new = evaluate_buf(model, test_new, device)
    tmd = calculate_tmd(base_theta_old.to(device), model.get_Theta_buf().to(device), n_know_old)

    method = f"DER++ (mem={buffer_size})"
    row = {
        "Method": method,
        "AUC_old": r_old["auc"], "AUC_new": r_new["auc"],
        "RMSE_old": r_old["rmse"], "RMSE_new": r_new["rmse"],
        "ACC_old": r_old["acc"], "ACC_new": r_new["acc"],
        "F1_old": r_old["f1"], "F1_new": r_new["f1"],
        "TMD": tmd,
    }
    print(
        f"\n  [{method}]\n"
        f"    旧: AUC={r_old['auc']:.4f} ACC={r_old['acc']:.4f} F1={r_old['f1']:.4f}\n"
        f"    新: AUC={r_new['auc']:.4f} ACC={r_new['acc']:.4f} F1={r_new['f1']:.4f}\n"
        f"    TMD={tmd:.4f}（与 Ours/X-DER 同 θ 空间，可直接对比）"
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, f"derpp_{ds_name}_random_split.csv")
    pd.DataFrame([row], columns=COLS).to_csv(csv_path, index=False)

    def _fmt(x):
        return x if isinstance(x, str) else f"{x:.4f}"

    lines = [
        "| " + " | ".join(COLS) + " |",
        "|" + "|".join(["---"] * len(COLS)) + "|",
        "| " + " | ".join([row["Method"]] + [_fmt(row[c]) for c in COLS[1:]]) + " |",
    ]
    note = (
        f"\n*口径*：{ds_name} random_split，G-NCDM 骨干，buf 无泄漏预测，与 Ours/X-DER 主表逐行可比。\n"
        f"*DER++ 损失*：L_BCE(new) + α·L_KD(logit) + β·L_BCE_buf；无 L_Future，无 memory revision。\n"
        f"*超参*：mem={buffer_size}, α={alpha_kd}, β={beta_buf}, alpha={alpha}, epochs={n_epoch}。\n"
        "*TMD*：与 Ours/X-DER 同在 G-NCDM 概念 θ 空间（calculate_tmd 取前 K_old 列），可直接比。\n"
        "*与 X-DER 区别*：去掉了 L_Future（ΔK 反激活）和动态记忆修正（z_past clamp 更新）。\n"
    )
    with open(os.path.join(SAVE_DIR, f"derpp_{ds_name}_random_split.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)

    print(f"\n>>> 写入 {csv_path}")
    return row
