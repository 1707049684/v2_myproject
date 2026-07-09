# -*- coding: utf-8 -*-
"""图 A（GNCDM 骨干）：math1 random_split 的 ACC_new/ACC_old-epoch 曲线。

CLEAN-Full/Full-Replay 直接调 run_incremental_math1.run_strategy + buf_strategy_specs
（与主实验同一套训练配方）；X-DER / C-LoRA-GNCDM 仍走各自基线脚本。不含 CLEAN-LoRA。

运行：cd GNCDM/plot && python plot_epoch_curve_gncdm_math1.py [--epochs 25]
产物：incremental_result/epoch_curve_gncdm_math1_random_split_ep{N}.csv
"""

import argparse
import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(PLOT_DIR)
EXPERIMENTS_DIR = os.path.join(GNCDM_DIR, "experiments")
for p in (GNCDM_DIR, EXPERIMENTS_DIR, os.path.join(EXPERIMENTS_DIR, "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

# gncdm_clora_baseline 把 sys.argv[1] 当数据集名；先摘掉本脚本的 --epochs
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--epochs", type=int, default=25)
_cli_epochs, _unknown = _pre.parse_known_args()
if "--epochs" in sys.argv:
    i = sys.argv.index("--epochs")
    del sys.argv[i : i + 2]
sys.argv.extend(_unknown)

import pandas as pd
import torch

import run_incremental_math1 as R
from core.model import GNCDM
from run_xder import run_xder
import gncdm_clora_baseline as CL

DATA_DIR = R.DATA_DIR
SAVE_DIR = R.SAVE_DIR
ALPHA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_USER, N_ITEM_TOTAL, N_KNOW_TOTAL = 4209, 20, 11
CLORA_LAMBDA = 0.1

# 官方策略名 -> 图例名
PLOT_NAMES = {
    "Ours (Dynamic DNA)": "CLEAN-Full",
    "Full Replay Oracle": "Full-Replay",
}


def load_ctx(device, n_epoch):
    R.set_seed(42)
    Q = __import__("numpy").load(os.path.join(DATA_DIR, "math1_Q_matrix.npy"))
    df_train = pd.read_csv(os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"))
    df_valid = pd.read_csv(os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"))
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = N_ITEM_TOTAL - n_item_old, N_KNOW_TOTAL - n_know_old
    Q_expanded = Q_mat.copy()
    train_old = df_train[df_train.item_id < n_item_old].copy()
    train_new = df_train[df_train.item_id >= n_item_old].copy()
    valid_old = df_valid[df_valid.item_id < n_item_old].copy()
    valid_new = df_valid[df_valid.item_id >= n_item_old].copy()
    log_old = R.build_log_mat(train_old, N_USER, n_item_old)
    log_full = R.build_log_mat(df_train, N_USER, N_ITEM_TOTAL)

    base = GNCDM(
        n_user=N_USER,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=ALPHA,
        Q_mat=Q_mat[:n_item_old, :n_know_old].copy(),
        monotonicity_assumption=True,
        device=device,
    ).to(device)
    R.train_real(
        base,
        train_old,
        log_old,
        list(base.parameters()),
        device,
        n_epoch=n_epoch,
        desc="Base",
        eval_fn=lambda m: (
            R.populate_buffers(m, log_old, device),
            R.evaluate_buf(m, valid_old, device),
        )[1],
    )

    def strat_eval_fn(valid_df):
        return lambda m: (
            R.populate_buffers(m, log_full, device),
            R.evaluate_buf(m, valid_df, device),
        )[1]

    def curve_eval_fn(m):
        R.populate_buffers(m, log_full, device)
        old_m = R.evaluate_buf(m, valid_old, device)
        new_m = R.evaluate_buf(m, valid_new, device)
        return {
            "acc": new_m["acc"],
            "auc": new_m["auc"],
            "acc_old": old_m["acc"],
            "auc_old": old_m["auc"],
        }

    specs = R.buf_strategy_specs(
        n_item_new, n_know_new, n_item_old, Q_expanded, train_old, train_new, valid_old, valid_new
    )
    rs_kw = dict(
        log_full=log_full,
        n_know_old=n_know_old,
        device=device,
        strat_eval_fn=strat_eval_fn,
        final_old=lambda m: R.evaluate_buf(m, valid_old, device),
        final_new=lambda m: R.evaluate_buf(m, valid_new, device),
        n_epoch=n_epoch,
        curve_eval_fn=curve_eval_fn,
    )
    return specs, rs_kw, base, curve_eval_fn, n_epoch


def history_row(label, h):
    row = {"Model": label, "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]}
    if "acc_old" in h:
        row["ACC_old"] = h["acc_old"]
        row["AUC_old"] = h["auc_old"]
    return row


def run_gncdm_family(specs, rs_kw, base):
    rows = []
    for official, label in PLOT_NAMES.items():
        history = []
        R.run_strategy(base, official, record_fn=None, history=history, **specs[official], **rs_kw)
        rows.extend(history_row(label, h) for h in history)
        print(f"[{label}] {len(history)} ep,末 ACC_new={history[-1]['acc']:.4f} ACC_old={history[-1]['acc_old']:.4f}")
    return rows


def run_xder_curve(curve_eval_fn, device, n_epoch):
    history = []
    row = run_xder(
        split_name="math1_random_split(curve)",
        ds_name="math1_curve",
        train_path=os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"),
        valid_path=os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"),
        test_path=os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv"),
        Q_path=os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
        device=device,
        n_user=N_USER,
        n_item_total=N_ITEM_TOTAL,
        n_know_total=N_KNOW_TOTAL,
        new_concepts=NEW_CONCEPTS,
        alpha=ALPHA,
        n_epoch=n_epoch,
        history=history,
        history_eval_fn=curve_eval_fn,
    )
    print(f"[X-DER] 末 ACC_new={history[-1]['acc']:.4f} | test={row['ACC_new']:.4f}")
    return [history_row("X-DER", h) for h in history]


def run_clora_gncdm_curve(device, n_epoch):
    cfg = CL.CONFIGS["math1"]
    meta = CL.load_partition(cfg)
    CL.set_seed(42)
    base = CL._new_model(cfg, meta, device)
    CL.train_real(
        base, meta["train_old"], meta["log_old_only"], list(base.parameters()), device,
        n_epoch=CL.BASE_EPOCHS, desc="Base(CLoRA)",
    )
    CL.populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    import copy as _copy
    base_state = _copy.deepcopy(base.state_dict())

    def eval_fn(m):
        CL.populate_buffers(m, meta["log_full"], device)
        old_m = CL.evaluate_buf(m, meta["test_old"], device)
        new_m = CL.evaluate_buf(m, meta["test_new"], device)
        return {"acc": new_m["acc"], "auc": new_m["auc"], "acc_old": old_m["acc"], "auc_old": old_m["auc"]}

    history = []
    r = CL.run_one_lambda(
        cfg,
        base_state,
        base_theta_ref,
        meta,
        CLORA_LAMBDA,
        device,
        history=history,
        history_eval_fn=eval_fn,
        n_epoch=n_epoch,
    )
    print(f"[C-LoRA-GNCDM] 末 ACC_new={history[-1]['acc']:.4f} | test={r['ACC_new']:.4f}")
    return [history_row("C-LoRA-GNCDM", h) for h in history]


def main():
    n_epoch = _cli_epochs.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | epochs = {n_epoch}")
    specs, rs_kw, base, curve_eval_fn, _ = load_ctx(device, n_epoch)
    rows = run_gncdm_family(specs, rs_kw, base)
    rows += run_xder_curve(curve_eval_fn, device, n_epoch)
    rows += run_clora_gncdm_curve(device, n_epoch)
    out = os.path.join(SAVE_DIR, f"epoch_curve_gncdm_math1_random_split_ep{n_epoch}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
