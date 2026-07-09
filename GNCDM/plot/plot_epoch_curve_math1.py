# -*- coding: utf-8 -*-
"""图 A（4 策略版）：math1 random_split ACC_new-epoch 曲线。训练走 run_strategy + buf_strategy_specs。

运行：cd GNCDM/plot && python plot_epoch_curve_math1.py
"""

import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(PLOT_DIR)
EXPERIMENTS_DIR = os.path.join(GNCDM_DIR, "experiments")
for p in (GNCDM_DIR, EXPERIMENTS_DIR, os.path.join(EXPERIMENTS_DIR, "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_incremental_math1 as R
from core.model import GNCDM

DATA_DIR = R.DATA_DIR
SAVE_DIR = R.SAVE_DIR
ALPHA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_EPOCH = 15
N_USER, N_ITEM_TOTAL, N_KNOW_TOTAL = 4209, 20, 11
CURVE_STRATEGIES = [
    "Ours (Dynamic DNA)",
    "Ours (LoRA)",
    "Full Replay Oracle",
    "Naive FT (NFT)",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R.set_seed(42)
    Q = np.load(os.path.join(DATA_DIR, "math1_Q_matrix.npy"))
    df_train = pd.read_csv(os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"))
    df_valid = pd.read_csv(os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"))
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = N_ITEM_TOTAL - n_item_old, N_KNOW_TOTAL - n_know_old
    train_old = df_train[df_train.item_id < n_item_old].copy()
    train_new = df_train[df_train.item_id >= n_item_old].copy()
    valid_old = df_valid[df_valid.item_id < n_item_old].copy()
    valid_new = df_valid[df_valid.item_id >= n_item_old].copy()
    log_old = R.build_log_mat(train_old, N_USER, n_item_old)
    log_full = R.build_log_mat(df_train, N_USER, N_ITEM_TOTAL)

    base = GNCDM(
        n_user=N_USER, n_item=n_item_old, n_know=n_know_old,
        user_dim=32, item_dim=32, alpha=ALPHA,
        Q_mat=Q_mat[:n_item_old, :n_know_old].copy(),
        monotonicity_assumption=True, device=device,
    ).to(device)
    R.train_real(
        base, train_old, log_old, list(base.parameters()), device,
        n_epoch=N_EPOCH, desc="Base",
        eval_fn=lambda m: (R.populate_buffers(m, log_old, device), R.evaluate_buf(m, valid_old, device))[1],
    )

    def strat_eval_fn(valid_df):
        return lambda m: (R.populate_buffers(m, log_full, device), R.evaluate_buf(m, valid_df, device))[1]

    specs = R.buf_strategy_specs(
        n_item_new, n_know_new, n_item_old, Q_mat.copy(), train_old, train_new, valid_old, valid_new
    )
    rs_kw = dict(
        log_full=log_full, n_know_old=n_know_old, device=device,
        strat_eval_fn=strat_eval_fn,
        final_old=lambda m: R.evaluate_buf(m, valid_old, device),
        final_new=lambda m: R.evaluate_buf(m, valid_new, device),
        n_epoch=N_EPOCH,
        curve_eval_fn=lambda m: strat_eval_fn(valid_new)(m),
    )

    rows = []
    for name in CURVE_STRATEGIES:
        history = []
        R.run_strategy(base, name, record_fn=None, history=history, **specs[name], **rs_kw)
        rows.extend({"Model": name, "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]} for h in history)
        print(f"[{name}] {len(history)} ep")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "epoch_curve_math1_random_split.csv"), index=False)
    plt.figure(figsize=(5.5, 4))
    markers = {"Ours (Dynamic DNA)": "o", "Ours (LoRA)": "s", "Full Replay Oracle": "^", "Naive FT (NFT)": "d"}
    for name in CURVE_STRATEGIES:
        sub = df[df.Model == name]
        plt.plot(sub.epoch, sub.ACC_new, marker=markers[name], label=name, linewidth=1.8, markersize=5)
    plt.xlabel("Training epoch")
    plt.ylabel("ACC_new (validation)")
    plt.title("Math1 random_split: convergence of ACC_new")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "epoch_curve_math1_random_split.png"), dpi=200)


if __name__ == "__main__":
    main()
