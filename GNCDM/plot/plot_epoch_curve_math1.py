# -*- coding: utf-8 -*-
"""图 A：效率-效果曲线 —— ACC_new 随训练 epoch 的收敛速度对比。

math1 random_split，alpha=0.20（与 all_methods_math1_random_split.csv 同口径）。
对比 4 个从同一个 Base 扩展出来的策略：Ours(DNA) / Ours(LoRA) / Full-Replay-Oracle /
Naive-FT，每个 epoch 都在 valid_new 上评一次，画出 ACC_new-epoch 曲线：越靠左上（少
epoch 达到高 ACC）说明该策略"效率"越高。

产物：
  incremental_result/epoch_curve_math1_random_split.csv
  incremental_result/epoch_curve_math1_random_split.png
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


def load():
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

    return dict(
        n_item_old=n_item_old,
        n_know_old=n_know_old,
        n_item_new=n_item_new,
        n_know_new=n_know_new,
        Q_old=Q_mat[:n_item_old, :n_know_old].copy(),
        Q_exp=Q_mat.copy(),
        train_old=train_old,
        train_new=train_new,
        valid_old=valid_old,
        valid_new=valid_new,
        log_old=R.build_log_mat(train_old, N_USER, n_item_old),
        log_full=R.build_log_mat(df_train, N_USER, N_ITEM_TOTAL),
    )


def train_base(c, device):
    R.set_seed(42)
    base = GNCDM(
        n_user=N_USER,
        n_item=c["n_item_old"],
        n_know=c["n_know_old"],
        user_dim=32,
        item_dim=32,
        alpha=ALPHA,
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
        desc="Base",
        eval_fn=lambda m: (
            R.populate_buffers(m, c["log_old"], device),
            R.evaluate_buf(m, c["valid_old"], device),
        )[1],
    )
    return base


def new_task_eval_fn(c, device):
    """所有策略统一用 valid_new 上的 buffer 无泄漏 ACC 作曲线纵轴，口径一致才可比。"""

    def fn(m):
        R.populate_buffers(m, c["log_full"], device)
        return R.evaluate_buf(m, c["valid_new"], device)

    return fn


STRATEGIES = {
    "Ours (Dynamic DNA)": dict(
        expand=lambda m, c: m.expand_topology(c["n_item_new"], c["n_know_new"], c["Q_exp"]),
        params=lambda m: R.new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        train_df=lambda c: c["train_new"],
    ),
    "Ours (LoRA)": dict(
        expand=lambda m, c: m.expand_topology_lora(
            delta_M=c["n_item_new"],
            delta_K=c["n_know_new"],
            Q_expanded=c["Q_exp"],
            M_old=c["n_item_old"],
            rank=min(16, c["n_know_new"]),
        ),
        params=R.lora_params,
        train_df=lambda c: c["train_new"],
    ),
    "Full Replay Oracle": dict(
        expand=lambda m, c: m.full_replay_oracle_expand_topology(
            c["n_item_new"], c["n_know_new"], c["Q_exp"]
        ),
        params=lambda m: list(m.parameters()),
        train_df=lambda c: pd.concat([c["train_old"], c["train_new"]], ignore_index=True),
    ),
    "Naive FT (NFT)": dict(
        expand=lambda m, c: m.full_replay_oracle_expand_topology(
            c["n_item_new"], c["n_know_new"], c["Q_exp"]
        ),
        params=lambda m: list(m.parameters()),
        train_df=lambda c: c["train_new"],
    ),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    c = load()
    base = train_base(c, device)

    rows = []
    for name, spec in STRATEGIES.items():
        m = R.fresh_base(base)
        spec["expand"](m, c)
        R.populate_buffers(m, c["log_full"], device)
        history = []
        R.train_real(
            m,
            spec["train_df"](c),
            c["log_full"],
            spec["params"](m),
            device,
            n_epoch=N_EPOCH,
            desc=name,
            eval_fn=new_task_eval_fn(c, device),
            history=history,
        )
        for h in history:
            rows.append({"Model": name, "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]})
        print(f"[{name}] 完成，{len(history)} 个 epoch 记录")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "epoch_curve_math1_random_split.csv")
    df.to_csv(out_csv, index=False)
    print(f"写入 {out_csv}")

    plt.figure(figsize=(5.5, 4))
    markers = {"Ours (Dynamic DNA)": "o", "Ours (LoRA)": "s", "Full Replay Oracle": "^", "Naive FT (NFT)": "d"}
    for name in STRATEGIES:
        sub = df[df.Model == name]
        plt.plot(sub.epoch, sub.ACC_new, marker=markers[name], label=name, linewidth=1.8, markersize=5)
    plt.xlabel("Training epoch")
    plt.ylabel("ACC_new (validation)")
    plt.title("Math1 random_split: convergence of ACC_new")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(SAVE_DIR, "epoch_curve_math1_random_split.png")
    plt.savefig(out_png, dpi=200)
    print(f"写入 {out_png}")


if __name__ == "__main__":
    main()
