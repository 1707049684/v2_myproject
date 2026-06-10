# -*- coding: utf-8 -*-
"""Alpha sweep on junyi random_split — 选 Ours 性能最优的 alpha（一次性，跑完即删）。

buf 预测口径（mode='buf'）。每个 alpha 训 Base + Ours(Dynamic DNA) + Ours(LoRA)，
在 valid 上选 alpha、test 上报数（不只挑 test）。只用 G-NCDM，不需要 avalanche。
选择标准：Ours(DNA) 的 mean(valid AUC_old, valid AUC_new) 最大（旗舰、架构隔离零遗忘）。
"""

import os

import numpy as np
import pandas as pd
import torch

import run_incremental_math1 as R
from core.model import GNCDM
from run_incremental_a0910 import auto_new_concepts

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # _core→experiments→GNCDM→repo
DATA = os.path.join(REPO, "data", "junyi")
ALPHAS = [round(0.1 * k, 2) for k in range(1, 10)] + [0.95]  # 0.1..0.9, 0.95
N_EPOCH = 25
RANK = 16


def load():
    Q = np.load(os.path.join(DATA, "Q_matrix.npy"))
    n_item_total, n_know_total = Q.shape
    nc = auto_new_concepts(Q, 0.34)
    df_tr = pd.read_csv(os.path.join(DATA, "new_random_split", "train.csv"))
    df_va = pd.read_csv(os.path.join(DATA, "new_random_split", "valid.csv"))
    df_te = pd.read_csv(os.path.join(DATA, "new_random_split", "test.csv"))
    n_user = int(max(df_tr.user_id.max(), df_va.user_id.max(), df_te.user_id.max())) + 1
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, nc)
    df_tr, df_va, df_te = (R.remap_items(d, item_map) for d in (df_tr, df_va, df_te))
    ctx = dict(
        n_user=n_user,
        n_item_total=n_item_total,
        n_know_total=n_know_total,
        n_item_old=n_item_old,
        n_know_old=n_know_old,
        n_item_new=n_item_total - n_item_old,
        n_know_new=n_know_total - n_know_old,
        Q_old=Q_mat[:n_item_old, :n_know_old].copy(),
        Q_exp=Q_mat.copy(),
        train_old=df_tr[df_tr.item_id < n_item_old].copy(),
        train_new=df_tr[df_tr.item_id >= n_item_old].copy(),
        valid_old=df_va[df_va.item_id < n_item_old].copy(),
        valid_new=df_va[df_va.item_id >= n_item_old].copy(),
        test_old=df_te[df_te.item_id < n_item_old].copy(),
        test_new=df_te[df_te.item_id >= n_item_old].copy(),
    )
    ctx["log_old"] = R.build_log_mat(ctx["train_old"], n_user, n_item_old)
    ctx["log_full"] = R.build_log_mat(df_tr, n_user, n_item_total)
    return ctx


def strat_eval(c, device, valid_df):
    def fn(m):
        R.populate_buffers(m, c["log_full"], device)
        return R.evaluate_buf(m, valid_df, device)

    return fn


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
        eval_fn=strat_eval(c, device, c["valid_new"]),
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
        eval_fn=strat_eval(c, device, c["valid_new"]),
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
            eval_fn=lambda m: (
                R.populate_buffers(m, c["log_old"], device),
                R.evaluate_buf(m, c["valid_old"], device),
            )[1],
        )
        R.populate_buffers(base, c["log_old"], device)
        b_va, b_te = (
            R.evaluate_buf(base, c["valid_old"], device),
            R.evaluate_buf(base, c["test_old"], device),
        )

        dna = train_dna(base, c, device)
        dna_v_new = R.evaluate_buf(dna, c["valid_new"], device)
        dna_t_old, dna_t_new = (
            R.evaluate_buf(dna, c["test_old"], device),
            R.evaluate_buf(dna, c["test_new"], device),
        )

        lora = train_lora(base, c, device)
        lora_v_new = R.evaluate_buf(lora, c["valid_new"], device)
        lora_t_old, lora_t_new = (
            R.evaluate_buf(lora, c["test_old"], device),
            R.evaluate_buf(lora, c["test_new"], device),
        )

        sel = 0.5 * (b_va["auc"] + dna_v_new["auc"])  # 选择标准：DNA mean(valid AUC_old, AUC_new)
        rows.append(
            {
                "alpha": a,
                "sel_DNA_validAUC": round(sel, 4),
                "Base_te_AUCold": b_te["auc"],
                "Base_te_ACCold": b_te["acc"],
                "DNA_te_AUCnew": dna_t_new["auc"],
                "DNA_te_ACCnew": dna_t_new["acc"],
                "DNA_te_F1new": dna_t_new["f1"],
                "LoRA_te_AUCnew": lora_t_new["auc"],
                "LoRA_te_ACCnew": lora_t_new["acc"],
                "LoRA_te_F1new": lora_t_new["f1"],
                "DNA_va_AUCnew": dna_v_new["auc"],
                "LoRA_va_AUCnew": lora_v_new["auc"],
            }
        )
        print(
            f"alpha={a:.2f} | selAUC={sel:.4f} | Base te AUCold={b_te['auc']:.4f} ACCold={b_te['acc']:.4f} "
            f"| DNA te AUCnew={dna_t_new['auc']:.4f} ACCnew={dna_t_new['acc']:.4f} "
            f"| LoRA te AUCnew={lora_t_new['auc']:.4f} ACCnew={lora_t_new['acc']:.4f}"
        )

    df = pd.DataFrame(rows)
    out = os.path.join(R.SAVE_DIR, "alpha_sweep_junyi_random_split.csv")
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
