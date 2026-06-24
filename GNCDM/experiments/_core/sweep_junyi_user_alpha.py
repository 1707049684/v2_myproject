# -*- coding: utf-8 -*-
"""Alpha sweep on junyi user_split — 按 valid AUC 选 Ours 最优 alpha。

junyi user_split 当前 alpha=0.6 是沿用 a0910 user_split 的初值、**从未真扫过**
（CLAUDE.md / memory 均标注「暂未扫，沿用初值」）。本脚本按 user_split 正确口径
（**support/query 冷启动重构 evaluate_recon**，与 eval_all_methods_user_split.run_ours
完全一致；直接复用其 split_support_query，frac/seed 同源）扫 alpha：

  每个 alpha 训 Base + Ours(Dynamic DNA) + Ours(LoRA)，
  **在 valid 上按 DNA mean(valid AUC_old, AUC_new) 选 alpha、test 上报数**。

只用 G-NCDM，不需要 avalanche。junyi 稠密版(1000×712×39)，CPU 可跑但慢、GPU 更快。
产物：incremental_result/alpha_sweep_junyi_user_split.csv
"""

import os

import numpy as np
import pandas as pd
import torch

import run_incremental_math1 as R
from core.model import GNCDM
from eval_all_methods_user_split import split_support_query  # 同源 support/query 划分
from run_incremental_a0910 import auto_new_concepts

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # _core→experiments→GNCDM→repo
DATA = os.path.join(REPO, "data", "junyi", "new_user_split")
Q_PATH = os.path.join(REPO, "data", "junyi", "Q_matrix.npy")
ALPHAS = [round(0.1 * k, 2) for k in range(1, 10)] + [0.95]  # 0.1..0.9, 0.95
N_EPOCH = 25
RANK = 16


def load():
    Q = np.load(Q_PATH)
    n_item_total, n_know_total = Q.shape
    nc = auto_new_concepts(Q, 0.34)
    df_tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    df_va = pd.read_csv(os.path.join(DATA, "valid.csv"))
    df_te = pd.read_csv(os.path.join(DATA, "test.csv"))
    n_user = int(max(df_tr.user_id.max(), df_va.user_id.max(), df_te.user_id.max())) + 1

    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, nc)
    df_tr, df_va, df_te = (R.remap_items(d, item_map) for d in (df_tr, df_va, df_te))
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"

    train_old = df_tr[df_tr.item_id < n_item_old].copy()
    train_new = df_tr[df_tr.item_id >= n_item_old].copy()

    # support/query 划分（与最终九方法表同源同口径）
    sup_va, qry_va = split_support_query(df_va)
    sup_te, qry_te = split_support_query(df_te)
    qry_va_old = qry_va[qry_va.item_id < n_item_old].copy()
    qry_va_new = qry_va[qry_va.item_id >= n_item_old].copy()
    qry_te_old = qry_te[qry_te.item_id < n_item_old].copy()
    qry_te_new = qry_te[qry_te.item_id >= n_item_old].copy()

    c = dict(
        n_user=n_user,
        n_item_total=n_item_total,
        n_know_total=n_know_total,
        n_item_old=n_item_old,
        n_know_old=n_know_old,
        n_item_new=n_item_total - n_item_old,
        n_know_new=n_know_total - n_know_old,
        Q_old=Q_mat[:n_item_old, :n_know_old].copy(),
        Q_exp=Q_mat.copy(),
        train_old=train_old,
        train_new=train_new,
        # 训练 log
        log_old=R.build_log_mat(train_old, n_user, n_item_old),
        log_full=R.build_log_mat(df_tr, n_user, n_item_total),
        # 仅 support 构建的评测 log（无泄漏）
        sup_va_old_log=R.build_log_mat(sup_va[sup_va.item_id < n_item_old], n_user, n_item_old),
        sup_te_old_log=R.build_log_mat(sup_te[sup_te.item_id < n_item_old], n_user, n_item_old),
        sup_va_full_log=R.build_log_mat(sup_va, n_user, n_item_total),
        sup_te_full_log=R.build_log_mat(sup_te, n_user, n_item_total),
        qry_va_old=qry_va_old,
        qry_va_new=qry_va_new,
        qry_te_old=qry_te_old,
        qry_te_new=qry_te_new,
    )
    return c


def train_dna(base, c, device):
    m = R.fresh_base(base)
    m.expand_topology(c["n_item_new"], c["n_know_new"], c["Q_exp"])
    R.populate_buffers(m, c["log_full"], device)
    k_old = c["n_know_old"]

    def mask(grad):
        g = grad.clone()
        g[:, :k_old] = 0.0
        return g

    h1 = m.theta_agg_mat.weight.register_hook(mask)
    h2 = m.psi_agg_mat.weight.register_hook(mask)
    R.train_real(
        m,
        c["train_new"],
        c["log_full"],
        R.new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        device,
        n_epoch=N_EPOCH,
        desc="DNA",
        eval_fn=lambda mm: R.evaluate_recon(mm, c["qry_va_new"], c["sup_va_full_log"], device),
    )
    h1.remove()
    h2.remove()
    R.populate_buffers(m, c["log_full"], device)
    return m


def train_lora(base, c, device):
    m = R.fresh_base(base)
    m.expand_topology_lora(
        delta_M=c["n_item_new"],
        delta_K=c["n_know_new"],
        Q_expanded=c["Q_exp"],
        M_old=c["n_item_old"],
        rank=RANK,
    )
    R.populate_buffers(m, c["log_full"], device)
    R.train_real(
        m,
        c["train_new"],
        c["log_full"],
        R.lora_params(m),
        device,
        n_epoch=N_EPOCH,
        desc="LoRA",
        eval_fn=lambda mm: R.evaluate_recon(mm, c["qry_va_new"], c["sup_va_full_log"], device),
    )
    R.populate_buffers(m, c["log_full"], device)
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    c = load()
    print(
        f"dims: n_user={c['n_user']} n_item={c['n_item_total']} n_know={c['n_know_total']} "
        f"| 旧题={c['n_item_old']} 新题={c['n_item_new']} 旧概念={c['n_know_old']} 新概念={c['n_know_new']}"
    )

    rows = []
    for a in ALPHAS:
        R.set_seed(42)
        base = GNCDM(
            n_user=c["n_user"],
            n_item=c["n_item_old"],
            n_know=c["n_know_old"],
            user_dim=32,
            item_dim=32,
            alpha=a,
            Q_mat=c["Q_old"],
            monotonicity_assumption=True,
            device=device,
        ).to(device)
        R.train_real(
            base,
            c["train_old"],
            c["log_old"],
            list(base.parameters()),
            device,
            n_epoch=N_EPOCH,
            desc=f"Base(a={a})",
            eval_fn=lambda m: R.evaluate_recon(m, c["qry_va_old"], c["sup_va_old_log"], device),
        )
        R.populate_buffers(base, c["log_old"], device)
        b_va = R.evaluate_recon(base, c["qry_va_old"], c["sup_va_old_log"], device)
        b_te = R.evaluate_recon(base, c["qry_te_old"], c["sup_te_old_log"], device)

        dna = train_dna(base, c, device)
        dna_v_new = R.evaluate_recon(dna, c["qry_va_new"], c["sup_va_full_log"], device)
        dna_t_old = R.evaluate_recon(dna, c["qry_te_old"], c["sup_te_full_log"], device)
        dna_t_new = R.evaluate_recon(dna, c["qry_te_new"], c["sup_te_full_log"], device)

        lora = train_lora(base, c, device)
        lora_v_new = R.evaluate_recon(lora, c["qry_va_new"], c["sup_va_full_log"], device)
        lora_t_old = R.evaluate_recon(lora, c["qry_te_old"], c["sup_te_full_log"], device)
        lora_t_new = R.evaluate_recon(lora, c["qry_te_new"], c["sup_te_full_log"], device)

        sel = 0.5 * (b_va["auc"] + dna_v_new["auc"])  # 选择标准：DNA mean(valid AUC_old, AUC_new)
        rows.append(
            {
                "alpha": a,
                "sel_DNA_validAUC": round(sel, 4),
                "Base_va_AUCold": b_va["auc"],
                "DNA_va_AUCnew": dna_v_new["auc"],
                "LoRA_va_AUCnew": lora_v_new["auc"],
                "Base_te_AUCold": b_te["auc"],
                "Base_te_ACCold": b_te["acc"],
                "DNA_te_AUCold": dna_t_old["auc"],
                "DNA_te_AUCnew": dna_t_new["auc"],
                "DNA_te_ACCnew": dna_t_new["acc"],
                "DNA_te_F1new": dna_t_new["f1"],
                "LoRA_te_AUCold": lora_t_old["auc"],
                "LoRA_te_AUCnew": lora_t_new["auc"],
                "LoRA_te_ACCnew": lora_t_new["acc"],
                "LoRA_te_F1new": lora_t_new["f1"],
            }
        )
        print(
            f"alpha={a:.2f} | selAUC={sel:.4f} | Base va AUCold={b_va['auc']:.4f} te AUCold={b_te['auc']:.4f} "
            f"| DNA te AUCnew={dna_t_new['auc']:.4f} ACCnew={dna_t_new['acc']:.4f} "
            f"| LoRA te AUCnew={lora_t_new['auc']:.4f} ACCnew={lora_t_new['acc']:.4f}"
        )

    df = pd.DataFrame(rows)
    out = os.path.join(R.SAVE_DIR, "alpha_sweep_junyi_user_split.csv")
    df.to_csv(out, index=False)
    best = df.sort_values("sel_DNA_validAUC", ascending=False).iloc[0]
    print("\n=== 全表（按 alpha）===")
    print(df.to_string(index=False))
    print(
        f"\n>>> 选择标准 DNA mean(valid AUC_old,AUC_new) 最优 alpha = {best['alpha']:.2f} "
        f"(selAUC={best['sel_DNA_validAUC']:.4f})"
    )
    print(
        f">>> 该 alpha 下 test：Base AUCold={best['Base_te_AUCold']:.4f} ACCold={best['Base_te_ACCold']:.4f} | "
        f"DNA AUCnew={best['DNA_te_AUCnew']:.4f} ACCnew={best['DNA_te_ACCnew']:.4f} | "
        f"LoRA AUCnew={best['LoRA_te_AUCnew']:.4f} ACCnew={best['LoRA_te_ACCnew']:.4f}"
    )
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
