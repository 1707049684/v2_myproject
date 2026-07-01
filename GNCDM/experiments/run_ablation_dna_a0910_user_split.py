# -*- coding: utf-8 -*-
"""DNA 组件细粒度消融 · a0910 · user_split

对比 Ours(Dynamic DNA) 的两个核心组件分别消融的效果：

| 名称                     | OrthoMask | FrozenBias |
|--------------------------|-----------|------------|
| Ours (Dynamic DNA)       | ✓         | ✓          |
| Ablated (w/o OrthoMask)  | ✗         | ✓          |  <- 只消融正交掩码
| Ablated (w/o FrozenBias) | ✓         | ✗          |  <- 只消融冻结共享偏置
| Ours-Ablated             | ✗         | ✗          |  <- 同时消融两者（原始参照）

OrthoMask：theta_agg_mat/psi_agg_mat 旧列梯度归零（register_hook）。
FrozenBias：DNA 只训 new_params + agg_mat.weight（bias 冻结）；消融则加入 agg_mat.bias。

口径：a0910 user_split，support/query 冷启动同 eval_all_methods_user_split（SUPPORT_FRAC=0.5）。
产物：incremental_result/ablation_dna_a0910_user_split.{csv,md}
运行：cd GNCDM/experiments && python run_ablation_dna_a0910_user_split.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from eval_all_methods_user_split import SAVE_DIR, prepare
from run_incremental_math1 import (
    evaluate_recon,
    fresh_base,
    new_params,
    populate_buffers,
    set_seed,
    train_real,
)
from run_incremental_a0910 import auto_new_concepts
from core.model import GNCDM
from core.train import calculate_rd

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.6

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


def run_ablations(ours, meta, device):
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

    # Base
    print("\n=== Base ===")
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

    def run_strategy(name, params_fn, mask_agg_old=False):
        m = fresh_base(base)
        m.expand_topology(n_item_new, n_know_new, Q_expanded)
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
            ours["train_new"],
            log_full,
            params_fn(m),
            device,
            n_epoch=25,
            desc=name,
            eval_fn=strat_eval_fn(ours["qry_valid_new"]),
        )
        for h in handles:
            h.remove()
        populate_buffers(m, log_full, device)
        tmd = calculate_rd(base_theta_old.to(device), m.get_Theta_buf().to(device), n_know_old)
        record(name, final_old(m), final_new(m), tmd)

    # 参照：Ours-Ablated（同时消融两者）
    print("\n=== Ours-Ablated（同时消融 OrthoMask + FrozenBias，参照）===")
    run_strategy(
        "Ours-Ablated",
        lambda m: [p for p in m.parameters() if p.requires_grad],
        mask_agg_old=False,
    )

    # 参照：Ours(Dynamic DNA)（两者均保留）
    print("\n=== Ours(Dynamic DNA)（OrthoMask + FrozenBias 均保留，参照）===")
    run_strategy(
        "Ours (Dynamic DNA)",
        lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        mask_agg_old=True,
    )

    # 消融1：只消融 OrthoMask（保留 FrozenBias）
    print("\n=== Ablated (w/o OrthoMask)（OrthoMask 消融，FrozenBias 保留）===")
    run_strategy(
        "Ablated (w/o OrthoMask)",
        lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        mask_agg_old=False,
    )

    # 消融2：只消融 FrozenBias（保留 OrthoMask，加入 bias 参数）
    print("\n=== Ablated (w/o FrozenBias)（FrozenBias 消融，OrthoMask 保留）===")
    run_strategy(
        "Ablated (w/o FrozenBias)",
        lambda m: new_params(m)
        + [
            m.theta_agg_mat.weight,
            m.theta_agg_mat.bias,
            m.psi_agg_mat.weight,
            m.psi_agg_mat.bias,
        ],
        mask_agg_old=True,
    )

    return rows


def write_table(rows):
    def _fmt(x):
        if isinstance(x, str):
            return x
        return "-" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x:.4f}"

    df = pd.DataFrame(rows, columns=COLS)
    csv_path = os.path.join(SAVE_DIR, "ablation_dna_a0910_user_split.csv")
    df.to_csv(csv_path, index=False)

    lines = [
        "| " + " | ".join(COLS) + " |",
        "|" + "|".join(["---"] * len(COLS)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([str(r["Method"])] + [_fmt(r[c]) for c in COLS[1:]]) + " |")

    note = (
        "\n*消融对象*：OrthoMask = theta_agg_mat/psi_agg_mat 旧列梯度归零（hook）；"
        "FrozenBias = DNA 只训 new_params+agg_mat.weight，不训 bias。\n"
        "*口径*：a0910 user_split，support/query（frac=0.5, seed=7）同 eval_all_methods 一致。\n"
        "*RD*：G-NCDM 概念 θ 空间（架构隔离→0/极小），仅 w/o-FrozenBias 训了 bias 可能略有漂移。\n"
    )
    md_path = os.path.join(SAVE_DIR, "ablation_dna_a0910_user_split.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)

    print("\n" + "=" * 70)
    print(" DNA 组件消融（a0910 user_split）")
    print("=" * 70)
    print("\n".join(lines))
    print(f"\n>>> 写入 {csv_path}\n>>> 写入 {md_path}")
    print("=" * 70)


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，建议 GPU 服务器。")

    set_seed(42)
    Q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    Q = np.load(Q_path)
    usr = os.path.join(DATA_DIR, "new_user_split")

    cfg = {
        "train": os.path.join(usr, "train.csv"),
        "valid": os.path.join(usr, "valid.csv"),
        "test": os.path.join(usr, "test.csv"),
        "Q": Q_path,
        "n_user": N_USER,
        "n_item": N_ITEM,
        "n_know": N_KNOW,
        "new_concepts": auto_new_concepts(Q, 0.34),
        "alpha": ALPHA,
    }

    ours, _, meta = prepare(cfg, device)
    rows = run_ablations(ours, meta, device)
    write_table(rows)
    print("\n完成：incremental_result/ablation_dna_a0910_user_split.csv")


if __name__ == "__main__":
    main()
