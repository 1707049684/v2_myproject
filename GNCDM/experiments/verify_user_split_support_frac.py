"""验证 support_frac × multi-seed 下 LoRA vs DNA 的 AUC_new 差距（a0910 user_split）。

假说：support 越少 → LoRA 低秩隐式正则优势越大（gap = AUC_new(LoRA)-AUC_new(DNA) 随 frac↓ 而↑）。
结果：若 gap 无单调趋势则假说为噪声。

用法（从 GNCDM/experiments/ 目录运行）：
    cd GNCDM/experiments
    python verify_user_split_support_frac.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_core"))

import numpy as np
import pandas as pd
import torch

# 触发 eval_all_methods_user_split 的 sys.path 扩展（把 gncdm_dir 加进来）
import eval_all_methods_user_split as evals
from eval_all_methods_user_split import evaluate_recon, prepare
from run_incremental_math1 import (
    evaluate_recon,  # noqa: F811 (same function, imported via evals chain)
    fresh_base,
    lora_params,
    new_params,
    populate_buffers,
    set_seed,
    train_real,
)
from core.model import GNCDM

# ---- a0910 config（与 eval_all_methods_user_split.main() 完全一致）----
gncdm_dir = evals.gncdm_dir
repo_root = os.path.dirname(gncdm_dir)
a0910_dir = os.path.join(repo_root, "data", "a0910")
Q_npy = os.path.join(a0910_dir, "Q_matrix.npy")
_Q = np.load(Q_npy)
_freq = (_Q > 0).sum(axis=0)
_touched = np.zeros(_Q.shape[0], dtype=bool)
_nc: list[int] = []
for _k in np.argsort(_freq):
    _nc.append(int(_k))
    _touched |= _Q[:, _k] > 0
    if _touched.sum() >= 0.34 * _Q.shape[0]:
        break

CFG = {
    "train": os.path.join(a0910_dir, "new_user_split", "train.csv"),
    "valid": os.path.join(a0910_dir, "new_user_split", "valid.csv"),
    "test": os.path.join(a0910_dir, "new_user_split", "test.csv"),
    "Q": Q_npy,
    "n_user": 4163,
    "n_item": 17746,
    "n_know": 123,
    "new_concepts": sorted(_nc),
    "alpha": 0.6,
}

FRACS = [0.2, 0.35, 0.5, 0.7]   # support 比例：越小 → 冷启动越难
SEEDS = [7, 42, 1]               # support/query 划分种子


# ==========================================================================
# 仅跑 Base + DNA + LoRA（跳过 Ablated / Full Replay / Naive FT）
# ==========================================================================
def run_dna_lora(ours, meta, device):
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

    # Base 训练固定 seed=42，隔离模型初始化对结果的影响
    set_seed(42)
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
        base, ours["train_old"], log_old, list(base.parameters()),
        device, n_epoch=25, desc="Base", eval_fn=base_eval_fn,
    )
    populate_buffers(base, log_old, device)

    def run_strat(name, expand_fn, params_fn, mask_agg_old=False):
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
            m, ours["train_new"], log_full, params_fn(m), device,
            n_epoch=25, desc=name, eval_fn=strat_eval_fn(ours["qry_valid_new"]),
        )
        for h in handles:
            h.remove()
        populate_buffers(m, log_full, device)
        ro, rn = final_old(m), final_new(m)
        return {
            "AUC_old": ro["auc"], "AUC_new": rn["auc"],
            "ACC_new": rn["acc"], "F1_new": rn["f1"], "RMSE_new": rn["rmse"],
        }

    dna = run_strat(
        "DNA",
        lambda m: m.expand_topology(n_item_new, n_know_new, Q_expanded),
        lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        mask_agg_old=True,
    )
    lora = run_strat(
        "LoRA",
        lambda m: m.expand_topology_lora(
            delta_M=n_item_new, delta_K=n_know_new,
            Q_expanded=Q_expanded, M_old=n_item_old, rank=16,
        ),
        lora_params,
    )
    gap = lora["AUC_new"] - dna["AUC_new"]
    print(
        f"  DNA  AUC_new={dna['AUC_new']:.4f}  "
        f"LoRA AUC_new={lora['AUC_new']:.4f}  "
        f"gap={gap:+.4f}"
    )
    return dna, lora


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  fracs={FRACS}  seeds={SEEDS}\n")

    rows = []
    for frac in FRACS:
        gaps = []
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"frac={frac}  seed={seed}")
            print("=" * 60)
            # 猴子补丁：让 split_support_query 读到本轮的 frac/seed
            evals.SUPPORT_FRAC = frac
            evals.SPLIT_SEED = seed
            ours_data, _base_data, meta = prepare(CFG, device)
            dna, lora = run_dna_lora(ours_data, meta, device)
            gap = lora["AUC_new"] - dna["AUC_new"]
            gaps.append(gap)
            rows.append({
                "frac": frac,
                "seed": seed,
                "AUC_new_DNA": dna["AUC_new"],
                "AUC_new_LoRA": lora["AUC_new"],
                "gap_LoRA_minus_DNA": gap,
                "AUC_old": dna["AUC_old"],
            })
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
        print(
            f"\n  [frac={frac}] gap mean={mean_gap:+.4f} std={std_gap:.4f}"
            f"  ({'LoRA>DNA' if mean_gap > 0 else 'DNA>=LoRA'})"
        )

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "..", "incremental_result", "verify_usersplit_frac_sweep.csv")
    out = os.path.normpath(out)
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print("汇总（假说：gap 随 frac↓ 而↑ → LoRA 在冷启动场景越优）")
    print("=" * 60)
    summary = (
        df.groupby("frac")["gap_LoRA_minus_DNA"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("frac")
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # 单调性判断：gap 是否随 frac↓ 单调↑
    fracs_sorted = summary["frac"].tolist()
    means_sorted = summary["mean"].tolist()
    monotone = all(
        means_sorted[i] >= means_sorted[i + 1] for i in range(len(means_sorted) - 1)
    )
    print(f"\n单调性（gap 随 frac↓ 而↑）: {'YES - 假说成立' if monotone else 'NO - 假说不成立/噪声'}")
    print(f"\n>>> 写入 {out}")


if __name__ == "__main__":
    main()
