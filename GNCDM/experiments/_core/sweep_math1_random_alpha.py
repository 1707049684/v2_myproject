# -*- coding: utf-8 -*-
"""Alpha sweep on math1 random_split — 按 DNA mean(valid ACC) 选最优 alpha。

与 a0910/junyi 同口径：buf 预测；每个 alpha 训 Base + Ours(Dynamic DNA) + Ours(LoRA)；
选择标准 = 0.5 * (Base valid ACC_old + DNA valid ACC_new)；test 只上报。
math1 主实验 n_epoch=15、ΔK=[0,1,3,6]、LoRA rank=4。

运行：cd GNCDM/experiments/_core && python sweep_math1_random_alpha.py
产物：incremental_result/alpha_sweep_math1_random_split.csv
"""

import os
import sys

# _core → experiments → GNCDM
_CORE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_CORE)
_GNCDM = os.path.dirname(_EXP)
for p in (_CORE, _EXP, _GNCDM):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import torch

import run_incremental_math1 as R
from core.model import GNCDM

DATA = os.path.join(_GNCDM, "data")
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHAS = [round(0.05 * k, 2) for k in range(1, 20)]  # 0.05..0.95
N_EPOCH = 15
RANK = 4  # math1 n_know_new=4；与 run_incremental_math1 的 min(16, n_know_new) 一致
N_USER, N_ITEM, N_KNOW = 4209, 20, 11


def load():
    Q = __import__("numpy").load(os.path.join(DATA, "math1_Q_matrix.npy"))
    df_tr = pd.read_csv(os.path.join(DATA, "math1_train_0.8_0.2.csv"))
    df_va = pd.read_csv(os.path.join(DATA, "math1_valid_0.8_0.2.csv"))
    df_te = pd.read_csv(os.path.join(DATA, "math1_test_0.8_0.2.csv"))
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_tr, df_va, df_te = (R.remap_items(d, item_map) for d in (df_tr, df_va, df_te))
    n_item_new, n_know_new = N_ITEM - n_item_old, N_KNOW - n_know_old
    ctx = dict(
        n_user=N_USER,
        n_item_total=N_ITEM,
        n_know_total=N_KNOW,
        n_item_old=n_item_old,
        n_know_old=n_know_old,
        n_item_new=n_item_new,
        n_know_new=n_know_new,
        Q_old=Q_mat[:n_item_old, :n_know_old].copy(),
        Q_exp=Q_mat.copy(),
        train_old=df_tr[df_tr.item_id < n_item_old].copy(),
        train_new=df_tr[df_tr.item_id >= n_item_old].copy(),
        valid_old=df_va[df_va.item_id < n_item_old].copy(),
        valid_new=df_va[df_va.item_id >= n_item_old].copy(),
        test_old=df_te[df_te.item_id < n_item_old].copy(),
        test_new=df_te[df_te.item_id >= n_item_old].copy(),
    )
    ctx["log_old"] = R.build_log_mat(ctx["train_old"], N_USER, n_item_old)
    ctx["log_full"] = R.build_log_mat(df_tr, N_USER, N_ITEM)
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
        select_metric="acc",
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
        select_metric="acc",
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
    print(
        f"选择标准: DNA mean(valid ACC_old, ACC_new); n_epoch={N_EPOCH}; alphas={ALPHAS[0]}..{ALPHAS[-1]}"
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
            select_metric="acc",
        )
        R.populate_buffers(base, c["log_old"], device)
        b_va = R.evaluate_buf(base, c["valid_old"], device)
        b_te = R.evaluate_buf(base, c["test_old"], device)

        dna = train_dna(base, c, device)
        dna_v_new = R.evaluate_buf(dna, c["valid_new"], device)
        dna_t_old = R.evaluate_buf(dna, c["test_old"], device)
        dna_t_new = R.evaluate_buf(dna, c["test_new"], device)

        lora = train_lora(base, c, device)
        lora_v_new = R.evaluate_buf(lora, c["valid_new"], device)
        lora_t_new = R.evaluate_buf(lora, c["test_new"], device)

        # DNA 架构隔离 → valid ACC_old ≡ Base valid ACC_old
        sel = 0.5 * (b_va["acc"] + dna_v_new["acc"])
        rows.append(
            {
                "alpha": a,
                "sel_DNA_validACC": round(sel, 4),
                "Base_va_ACCold": b_va["acc"],
                "DNA_va_ACCnew": dna_v_new["acc"],
                "Base_te_AUCold": b_te["auc"],
                "Base_te_ACCold": b_te["acc"],
                "DNA_te_AUCold": dna_t_old["auc"],
                "DNA_te_ACCold": dna_t_old["acc"],
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
            f"alpha={a:.2f} | selACC={sel:.4f} "
            f"(va ACCold={b_va['acc']:.4f} ACCnew={dna_v_new['acc']:.4f}) | "
            f"Base te ACCold={b_te['acc']:.4f} | "
            f"DNA te ACCnew={dna_t_new['acc']:.4f} | LoRA te ACCnew={lora_t_new['acc']:.4f}"
        )

    df = pd.DataFrame(rows)
    out = os.path.join(R.SAVE_DIR, "alpha_sweep_math1_random_split.csv")
    # 备份旧 Base-only 表
    if os.path.isfile(out):
        bak = out.replace(".csv", "_base_only_legacy.csv")
        if not os.path.isfile(bak):
            os.replace(out, bak)
            print(f"旧表备份 → {bak}")
    df.to_csv(out, index=False)
    best = df.sort_values("sel_DNA_validACC", ascending=False).iloc[0]
    print("\n=== 全表（按 sel_DNA_validACC 降序）===")
    print(df.sort_values("sel_DNA_validACC", ascending=False).to_string(index=False))
    print(
        f"\n>>> 选择标准 DNA mean(valid ACC_old,ACC_new) 最优 alpha = {best['alpha']:.2f} "
        f"(selACC={best['sel_DNA_validACC']:.4f})"
    )
    print(
        f">>> 该 alpha 下 test：Base ACCold={best['Base_te_ACCold']:.4f} | "
        f"DNA ACCnew={best['DNA_te_ACCnew']:.4f} | LoRA ACCnew={best['LoRA_te_ACCnew']:.4f}"
    )
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
