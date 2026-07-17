# -*- coding: utf-8 -*-
"""X-DER (eXtended Dark Experience Replay, TPAMI 2022 / arXiv:2201.00766) 高保真移植到 G-NCDM。

为什么用 G-NCDM(而非 cl_baselines 的 CognitiveBackbone):X-DER 的 future/anti-activation 需要
"新概念 ΔK 在模型里是可寻址的参数通道"。CognitiveBackbone 只吃 (user_id,item_id)、不读 Q、
无概念轴,ΔK 无落点;G-NCDM 的 θ/ψ 是 per-concept、读 Q 掩码,ΔK 落成实在的列。额外红利:
X-DER 与 Ours 同骨干 → RD 在同一个概念 θ 空间,可与 Ours 行直接对比。

CDM 没有"分类输出头",故按 route A 把 X-DER 核心算子高保真映射到 BCE + 潜在特质 θ 底座:

记忆缓冲区 M 每条目存 {y^(s)_i(作答向量,用 log_full 索引,不逐条存), e_j, y_ij, z_past},
其中 z_past 是**未过 sigmoid 的 raw logit**(确保 logit-space KD)。

X-DER 复合目标(Task2):
    L_total = L_BCE(B_new) + α·L_KD(B_buf) + β·L_BCE_buf(B_buf) + λ·L_Future(B_buf)
  - L_BCE     : 新题流 BCE on σ(z_current)。
  - L_KD      : logit 空间蒸馏 || z_current − z_past ||_2^2(忠实 X-DER 原意,对齐 raw logit)。
  - L_BCE_buf : buffer 旧题真实分数 y 的 BCE,防历史物理表征崩溃(DER++ β 项)。
  - L_Future  : 潜在特质反激活 mean( max(0, θ_current[:, K_old:])^2 )。STB 下旧题不含新概念,
                读 buffer 旧题时新概念掌握度应被强力抑制(CDM 无类槽,以 ΔK 潜通道再诠释 X-DER
                future-prep,论文需注明这是 CDM 适配而非原类头机制)。
动态记忆修正:每 epoch 末 z_past ← clamp(z_current, z_past−γ, z_past+γ),受约束更新 soft target。

骨干/扩容:fresh_base(base) + full_replay_oracle_expand_topology —— 单一共享网络、全参可训
(回放类 CL 的底座,非架构隔离)。

口径:random_split / mode='buf' 无泄漏预测(populate_buffers + forward_using_buf),逐行对齐
Ours 主表。产物:incremental_result/xder_{ds}_random_split.{csv,md}(单行,列与 all_methods 一致)。
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

from run_incremental_math1 import (  # noqa: E402  (依赖 sys.path 注入)
    SAVE_DIR,
    SampleSet,
    build_log_mat,
    evaluate_buf,
    evaluate_recon,
    fresh_base,
    populate_buffers,
    remap_items,
    set_seed,
    strict_bipartition,
    train_real,
)
from core.model import GNCDM  # noqa: E402
from core.train import calculate_rd  # noqa: E402

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


def _forward_logit(model, user_log, item_log, user_id, item_id):
    """返回 (raw logit 未过 sigmoid, theta)。X-DER 需 logit-space KD 与 θ 的 ΔK 列。

    G-NCDM 末端 ncd 以 Sigmoid 收尾;ncd[:-1] 去掉末层 Sigmoid 即得 raw logit。
    base(未扩容)与 full_replay_oracle 扩容后的共享网络 predict_response 同构(均为
    F.linear 聚合 + ncd),故同一 helper 通吃。
    """
    theta, psi = model.diagnose_theta_psi(user_log, item_log)
    Q_batch = model.Q_mat[item_id].squeeze(dim=1)
    theta_agg = F.linear(theta * Q_batch, model.theta_agg_mat.weight, model.theta_agg_mat.bias)
    psi_agg = F.linear(psi * Q_batch, model.psi_agg_mat.weight, model.psi_agg_mat.bias)
    logit = model.ncd[:-1](theta_agg - psi_agg)
    return logit, theta


def build_buffer(base, train_old, log_old, device, buffer_size, seed=42):
    """记忆库:从 train_old 采样 (user, old_item, score),并存 base(Task1 末)对该作答的
    **raw logit** z_past 作为 soft target。作答向量 y^(s) 用 log_full 索引,不逐条存。"""
    rng = np.random.RandomState(seed)
    n = len(train_old)
    idx = rng.choice(n, size=min(buffer_size, n), replace=False)
    sub = train_old.iloc[idx]
    u = torch.tensor(sub["user_id"].values, dtype=torch.long, device=device)
    i = torch.tensor(sub["item_id"].values, dtype=torch.long, device=device)
    s = torch.tensor(sub["score"].values, dtype=torch.float32, device=device)
    log_t = torch.tensor(log_old, dtype=torch.float32, device=device)
    base.eval()
    with torch.no_grad():
        z_past, _ = _forward_logit(base, log_t[u], log_t[:, i].T, u, i)
        z_past = z_past.reshape(-1).clone()
    print(f"  [X-DER] buffer 大小={u.shape[0]}（从 {n} 条旧题作答采样,z_past=raw logit）")
    return {"u": u, "i": i, "s": s, "z_past": z_past}


def _revise_memory(model, buffer, log_t, gamma, chunk=1024):
    """动态记忆修正:z_past ← clamp(z_current, z_past−γ, z_past+γ),受约束更新软标签。"""
    model.eval()
    u, i, z_old = buffer["u"], buffer["i"], buffer["z_past"]
    n = u.shape[0]
    with torch.no_grad():
        new_z = torch.empty_like(z_old)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            zc, _ = _forward_logit(model, log_t[u[a:b]], log_t[:, i[a:b]].T, u[a:b], i[a:b])
            new_z[a:b] = zc.reshape(-1)
        buffer["z_past"] = torch.clamp(new_z, min=z_old - gamma, max=z_old + gamma)


def train_xder(
    model,
    train_new,
    log_full,
    buffer,
    device,
    valid_eval_fn,
    n_know_old,
    alpha_kd=0.5,
    beta_buf=0.5,
    lam_future=0.5,
    gamma=0.75,
    n_epoch=15,
    lr=1e-3,
    batch_size=256,
    buf_batch=256,
    history=None,
    history_eval_fn=None,
):
    """Task2:L_BCE(new)+α·L_KD+β·L_BCE_buf+λ·L_Future,全参可训;每 epoch 末 memory revision。
    按 combined valid acc 选最优快照。

    history/history_eval_fn：可选，画收敛曲线用。给定则每 epoch 额外调
    history_eval_fn(model)（不给则退化用 vr，即 combined valid）记录一条，不影响 best_state 选优。
    """
    loader = DataLoader(SampleSet(train_new), batch_size=batch_size, shuffle=True)
    log_t = torch.tensor(log_full, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_buf = buffer["u"].shape[0]
    best_metric, best_state = -1.0, None
    for epoch in range(n_epoch):
        model.train()
        tot, t_bce, t_kd, t_bbuf, t_fut = 0.0, 0.0, 0.0, 0.0, 0.0
        for user_ids, item_ids, score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            score = score.to(device).unsqueeze(1)

            # --- L_BCE: 新题流 (on σ(z_current)) ---
            z_new, _ = _forward_logit(
                model, log_t[user_ids], log_t[:, item_ids].T, user_ids, item_ids
            )
            loss_bce = F.binary_cross_entropy_with_logits(z_new, score)

            # --- 采 buffer 旧题 ---
            sel = torch.randint(0, n_buf, (min(buf_batch, n_buf),), device=device)
            bu, bi = buffer["u"][sel], buffer["i"][sel]
            by = buffer["s"][sel].unsqueeze(1)
            bz = buffer["z_past"][sel].unsqueeze(1)
            z_buf, theta_buf = _forward_logit(model, log_t[bu], log_t[:, bi].T, bu, bi)

            # --- L_KD: logit 空间蒸馏 ---
            loss_kd = F.mse_loss(z_buf, bz)
            # --- L_BCE_buf: buffer 真实分数硬标签 ---
            loss_bbuf = F.binary_cross_entropy_with_logits(z_buf, by)
            # --- L_Future: 潜在特质反激活 mean(relu(θ_ΔK)^2) ---
            theta_new = theta_buf[:, n_know_old:]
            loss_fut = F.relu(theta_new).pow(2).mean()

            loss = loss_bce + alpha_kd * loss_kd + beta_buf * loss_bbuf + lam_future * loss_fut
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
            t_bce += loss_bce.item()
            t_kd += loss_kd.item()
            t_bbuf += loss_bbuf.item()
            t_fut += loss_fut.item()

        # --- 动态记忆修正(epoch 末) ---
        _revise_memory(model, buffer, log_t, gamma)

        vr = valid_eval_fn(model)
        vm = vr["acc"]
        if vm > best_metric:
            best_metric = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if history is not None:
            hr = history_eval_fn(model) if history_eval_fn is not None else vr
            history.append({"epoch": epoch + 1, **hr})
        nb = len(loader)
        print(
            f"    [X-DER] epoch {epoch + 1}/{n_epoch} loss={tot / nb:.4f} "
            f"(bce={t_bce / nb:.4f} kd={t_kd / nb:.4f} bce_buf={t_bbuf / nb:.4f} "
            f"fut={t_fut / nb:.4f}) valid_acc={vm:.4f} best={best_metric:.4f}"
        )
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def run_xder(
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
    lam_future=0.5,
    gamma=0.75,
    buffer_size=5000,
    n_epoch=15,
    lr=1e-3,
    batch_size=256,
    buf_batch=256,
    history=None,
    history_eval_fn=None,
    seed=42,
    write_artifacts=True,
):
    """在一个数据集的 random_split 上跑 X-DER,产出单行结果(列与 all_methods 一致)。

    history/history_eval_fn：透传给 train_xder，画收敛曲线用，不影响原有训练/选优行为。
    """
    print(f"\n{'#' * 70}\n# X-DER  {split_name}  (G-NCDM 骨干, mode=buf)\n{'#' * 70}")
    set_seed(seed)

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
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念,二分失败!"
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

    # ---- Task1: 训练 base(旧题, 全参), 与 Ours 的 Base 同口径 ----
    print("\n=== Task1: Base (X-DER 起点) ===")
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

    def base_eval_fn(m):
        populate_buffers(m, log_old, device)
        return evaluate_buf(m, valid_old, device)

    train_real(
        base,
        train_old,
        log_old,
        list(base.parameters()),
        device,
        n_epoch=15,
        desc="Base(X-DER)",
        eval_fn=base_eval_fn,
    )
    populate_buffers(base, log_old, device)
    base_theta_old = base.get_Theta_buf().clone()  # RD 参照

    # ---- 记忆库(存 raw logit) ----
    buffer = build_buffer(base, train_old, log_old, device, buffer_size, seed=seed)

    # ---- Task2: 扩容(共享网络全参可训) + X-DER 训练 ----
    print("\n=== Task2: X-DER (logit-KD + BCE_buf + anti-activation + memory revision) ===")
    model = fresh_base(base)
    model.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded)

    def valid_eval_fn(m):
        populate_buffers(m, log_full, device)
        return evaluate_buf(m, valid_comb, device)

    train_xder(
        model,
        train_new,
        log_full,
        buffer,
        device,
        valid_eval_fn,
        n_know_old,
        alpha_kd=alpha_kd,
        beta_buf=beta_buf,
        lam_future=lam_future,
        gamma=gamma,
        n_epoch=n_epoch,
        lr=lr,
        batch_size=batch_size,
        buf_batch=buf_batch,
        history=history,
        history_eval_fn=history_eval_fn,
    )

    # ---- 评测(buf 无泄漏, 逐行对齐 Ours) + RD ----
    populate_buffers(model, log_full, device)
    r_old = evaluate_buf(model, test_old, device)
    r_new = evaluate_buf(model, test_new, device)
    tmd = calculate_rd(base_theta_old.to(device), model.get_Theta_buf().to(device), n_know_old)

    method = f"X-DER (mem={buffer_size})"
    row = {
        "Method": method,
        "AUC_old": r_old["auc"],
        "AUC_new": r_new["auc"],
        "RMSE_old": r_old["rmse"],
        "RMSE_new": r_new["rmse"],
        "ACC_old": r_old["acc"],
        "ACC_new": r_new["acc"],
        "F1_old": r_old["f1"],
        "F1_new": r_new["f1"],
        "RD": tmd,
        "mem_size": buffer_size,
        "selection_source": "fixed_from_existing_result",
    }
    print(
        f"\n  [{method}]\n"
        f"    旧: AUC={r_old['auc']:.4f} ACC={r_old['acc']:.4f} F1={r_old['f1']:.4f}\n"
        f"    新: AUC={r_new['auc']:.4f} ACC={r_new['acc']:.4f} F1={r_new['f1']:.4f}\n"
        f"    RD={tmd:.4f}（与 Ours 同 θ 空间,可直接对比）"
    )

    if write_artifacts:
        os.makedirs(SAVE_DIR, exist_ok=True)
        csv_path = os.path.join(SAVE_DIR, f"xder_{ds_name}_random_split.csv")
        pd.DataFrame([row], columns=COLS).to_csv(csv_path, index=False)

        def _fmt(x):
            return x if isinstance(x, str) else f"{x:.4f}"

        lines = [
            "| " + " | ".join(COLS) + " |",
            "|" + "|".join(["---"] * len(COLS)) + "|",
            "| " + " | ".join([row["Method"]] + [_fmt(row[c]) for c in COLS[1:]]) + " |",
        ]
        note = (
            f"\n*口径*:{ds_name} random_split,G-NCDM 骨干,buf 无泄漏预测,与 Ours 主表逐行可比。\n"
            f"*X-DER 损失*:L_BCE(new)+α·L_KD(logit)+β·L_BCE_buf+λ·L_Future;memory revision(γ clamp)。\n"
            f"*超参*:mem={buffer_size}, α={alpha_kd}, β={beta_buf}, λ={lam_future}, γ={gamma}, "
            f"alpha={alpha}, epochs={n_epoch}。\n"
            "*RD*:与 Ours 同在 G-NCDM 概念 θ 空间(calculate_rd 取前 K_old 列),可与 Ours 行直接比。\n"
            "*L_Future*:CDM 无类槽,以 ΔK 潜通道反激活 mean(relu(θ_ΔK)^2) 再诠释 X-DER future-prep,"
            "论文须注明为 CDM 适配而非原类头机制。\n"
        )
        with open(
            os.path.join(SAVE_DIR, f"xder_{ds_name}_random_split.md"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(lines) + "\n" + note)

        print(f"\n>>> 写入 {csv_path}")
        print("    合并进 all_methods 主表请把该行交给我（或人工 append）。")
    return row


# ── support/query 协议常量（与 eval_all_methods_user_split 保持一致）──
_SUPPORT_FRAC = 0.5
_SPLIT_SEED = 7


def _split_sq(df, *, frac=_SUPPORT_FRAC, seed=_SPLIT_SEED):
    sup = df.groupby("user_id", group_keys=False).sample(frac=frac, random_state=seed)
    return sup, df.drop(sup.index)


def run_xder_user_split(
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
    lam_future=0.5,
    gamma=0.75,
    buffer_size=5000,
    n_epoch=15,
    lr=1e-3,
    batch_size=256,
    buf_batch=256,
    seed=42,
    support_query_seed=_SPLIT_SEED,
    support_frac=_SUPPORT_FRAC,
    artifact_dir=None,
):
    """user_split 口径的 X-DER（G-NCDM 骨干）。

    Task1/Task2 训练与 random_split 版完全相同；评测切换为 evaluate_recon +
    support/query 留出（SUPPORT_FRAC=0.5, SPLIT_SEED=7，与 eval_all_methods_user_split 一致）。
    产物：incremental_result/xder_{ds_name}_user_split.{csv,md}（单行，列同 all_methods）。
    """
    print(f"\n{'#' * 70}\n# X-DER  {split_name}  (G-NCDM 骨干, user_split 口径)\n{'#' * 70}")
    set_seed(seed)
    artifact_dir = SAVE_DIR if artifact_dir is None else str(artifact_dir)

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

    # support/query 划分（valid + test 各划一次，与 eval_all_methods_user_split 同口径）
    sup_valid, qry_valid = _split_sq(df_valid, frac=support_frac, seed=support_query_seed)
    sup_test, qry_test = _split_sq(df_test, frac=support_frac, seed=support_query_seed)
    qry_valid_old = qry_valid[qry_valid["item_id"] < n_item_old].copy()
    qry_valid_comb = qry_valid.copy()
    qry_test_old = qry_test[qry_test["item_id"] < n_item_old].copy()
    qry_test_new = qry_test[qry_test["item_id"] >= n_item_old].copy()
    print(
        f"  support/query: test support={len(sup_test)} | "
        f"query old={len(qry_test_old)} new={len(qry_test_new)}"
    )

    # log 矩阵（训练用全量；评测用 support-only）
    log_old = build_log_mat(train_old, n_user, n_item_old)
    log_full = build_log_mat(df_train, n_user, n_item_total)
    sup_valid_full_log = build_log_mat(sup_valid, n_user, n_item_total)
    sup_test_full_log = build_log_mat(sup_test, n_user, n_item_total)

    # Task1：训练 Base（与 random_split 版同口径）
    print("\n=== Task1: Base (X-DER user_split 起点) ===")
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

    def base_eval_fn(m):
        sup_old_log = build_log_mat(
            sup_valid[sup_valid["item_id"] < n_item_old], n_user, n_item_old
        )
        return evaluate_recon(m, qry_valid_old, sup_old_log, device)

    train_real(
        base,
        train_old,
        log_old,
        list(base.parameters()),
        device,
        n_epoch=15,
        desc="Base(X-DER-US)",
        eval_fn=base_eval_fn,
    )
    populate_buffers(base, log_old, device)
    base_theta_old = base.get_Theta_buf().clone()

    # 记忆库（raw logit，与 random_split 版相同）
    buffer = build_buffer(base, train_old, log_old, device, buffer_size, seed=seed)

    # Task2：扩容 + X-DER 训练，valid 用 evaluate_recon（support/query）
    print("\n=== Task2: X-DER (user_split eval，logit-KD + BCE_buf + anti-act + mem-revision) ===")
    model = fresh_base(base)
    model.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded)

    def valid_eval_fn(m):
        return evaluate_recon(m, qry_valid_comb, sup_valid_full_log, device)

    train_xder(
        model,
        train_new,
        log_full,
        buffer,
        device,
        valid_eval_fn,
        n_know_old,
        alpha_kd=alpha_kd,
        beta_buf=beta_buf,
        lam_future=lam_future,
        gamma=gamma,
        n_epoch=n_epoch,
        lr=lr,
        batch_size=batch_size,
        buf_batch=buf_batch,
    )

    # 评测（support/query，test 集）+ RD
    populate_buffers(model, log_full, device)
    r_old = evaluate_recon(model, qry_test_old, sup_test_full_log, device)
    r_new = evaluate_recon(model, qry_test_new, sup_test_full_log, device)
    tmd = calculate_rd(base_theta_old.to(device), model.get_Theta_buf().to(device), n_know_old)

    method = f"X-DER (mem={buffer_size})"
    row = {
        "Method": method,
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
    print(
        f"\n  [{method}]\n"
        f"    旧: AUC={r_old['auc']:.4f} ACC={r_old['acc']:.4f} F1={r_old['f1']:.4f}\n"
        f"    新: AUC={r_new['auc']:.4f} ACC={r_new['acc']:.4f} F1={r_new['f1']:.4f}\n"
        f"    RD={tmd:.4f}（与 Ours 同 θ 空间，可直接对比）"
    )

    os.makedirs(artifact_dir, exist_ok=True)
    csv_path = os.path.join(artifact_dir, f"xder_{ds_name}_user_split.csv")
    pd.DataFrame([row], columns=COLS).to_csv(csv_path, index=False)

    def _fmt(x):
        return x if isinstance(x, str) else f"{x:.4f}"

    lines = [
        "| " + " | ".join(COLS) + " |",
        "|" + "|".join(["---"] * len(COLS)) + "|",
        "| " + " | ".join([row["Method"]] + [_fmt(row[c]) for c in COLS[1:]]) + " |",
    ]
    note = (
        f"\n*口径*：{ds_name} user_split，G-NCDM 骨干，support/query 留出"
        f"（frac={support_frac}, seed={support_query_seed}），与 eval_all_methods_user_split 同口径。\n"
        f"*X-DER 损失*：L_BCE(new)+α·L_KD(logit)+β·L_BCE_buf+λ·L_Future；memory revision(γ clamp)。\n"
        f"*超参*：mem={buffer_size}, α={alpha_kd}, β={beta_buf}, λ={lam_future}, γ={gamma}, "
        f"alpha={alpha}, epochs={n_epoch}。\n"
        "*RD*：与 Ours 同在 G-NCDM 概念 θ 空间（populate_buffers 取训练用户），可与 Ours 行直接比。\n"
    )
    with open(
        os.path.join(artifact_dir, f"xder_{ds_name}_user_split.md"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(lines) + "\n" + note)

    print(f"\n>>> 写入 {csv_path}")
    return row
