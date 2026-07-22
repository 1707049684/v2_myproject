# -*- coding: utf-8 -*-
"""Junyi random_split ACC_new/ACC_old epoch curves (GNCDM backbone). Mirrors math1.

Run on GPU server:
  cd GNCDM/plot && python plot_epoch_curve_gncdm_junyi.py --epochs 15
"""

import argparse
import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(PLOT_DIR)
REPO_ROOT = os.path.dirname(GNCDM_DIR)
EXPERIMENTS_DIR = os.path.join(GNCDM_DIR, "experiments")
for p in (GNCDM_DIR, EXPERIMENTS_DIR, os.path.join(EXPERIMENTS_DIR, "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--epochs", type=int, default=15)
_cli_epochs, _unknown = _pre.parse_known_args()
if "--epochs" in sys.argv:
    i = sys.argv.index("--epochs")
    del sys.argv[i : i + 2]
sys.argv.extend(_unknown)

import numpy as np
import pandas as pd
import torch

import gncdm_clora_baseline as CL
import run_incremental_math1 as R
from core.model import GNCDM
from run_incremental_a0910 import auto_new_concepts
from run_xder import run_xder

DATA_DIR = os.path.join(REPO_ROOT, "data", "junyi")
SAVE_DIR = R.SAVE_DIR
ALPHA = 0.1
CLORA_LAMBDA = 0.1  # all_methods_junyi_random_split.csv

PLOT_NAMES = {
    "Ours (Dynamic DNA)": "CLEAN-Full",
    "Full Replay Oracle": "Full-Replay",
}


def load_ctx(device, n_epoch):
    R.set_seed(42)
    Q = np.load(os.path.join(DATA_DIR, "Q_matrix.npy"))
    rnd = os.path.join(DATA_DIR, "new_random_split")
    df_train = pd.read_csv(os.path.join(rnd, "train.csv"))
    df_valid = pd.read_csv(os.path.join(rnd, "valid.csv"))
    n_user = max(
        int(pd.read_csv(os.path.join(rnd, f))["user_id"].max()) + 1
        for f in ("train.csv", "valid.csv", "test.csv")
    )
    n_item_total, n_know_total = int(Q.shape[0]), int(Q.shape[1])
    new_concepts = auto_new_concepts(Q, 0.34)
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, new_concepts)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = n_item_total - n_item_old, n_know_total - n_know_old
    train_old = df_train[df_train.item_id < n_item_old].copy()
    train_new = df_train[df_train.item_id >= n_item_old].copy()
    valid_old = df_valid[df_valid.item_id < n_item_old].copy()
    valid_new = df_valid[df_valid.item_id >= n_item_old].copy()
    log_old = R.build_log_mat(train_old, n_user, n_item_old)
    log_full = R.build_log_mat(df_train, n_user, n_item_total)

    print(
        f"junyi dims n_user={n_user} items={n_item_old}+{n_item_new} "
        f"know={n_know_old}+{n_know_new} alpha={ALPHA}"
    )

    base = GNCDM(
        n_user=n_user,
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
        n_item_new, n_know_new, n_item_old, Q_mat.copy(), train_old, train_new, valid_old, valid_new
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
    meta = {
        "n_user": n_user,
        "n_item_total": n_item_total,
        "n_know_total": n_know_total,
        "new_concepts": new_concepts,
        "train": os.path.join(rnd, "train.csv"),
        "valid": os.path.join(rnd, "valid.csv"),
        "test": os.path.join(rnd, "test.csv"),
        "Q": os.path.join(DATA_DIR, "Q_matrix.npy"),
    }
    return specs, rs_kw, base, curve_eval_fn, meta


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
        print(
            f"[{label}] {len(history)} ep, end ACC_new={history[-1]['acc']:.4f} "
            f"ACC_old={history[-1]['acc_old']:.4f}"
        )
    return rows


def run_xder_curve(curve_eval_fn, device, n_epoch, meta):
    history = []
    row = run_xder(
        split_name="junyi_random_split(curve)",
        ds_name="junyi_curve",
        train_path=meta["train"],
        valid_path=meta["valid"],
        test_path=meta["test"],
        Q_path=meta["Q"],
        device=device,
        n_user=meta["n_user"],
        n_item_total=meta["n_item_total"],
        n_know_total=meta["n_know_total"],
        new_concepts=meta["new_concepts"],
        alpha=ALPHA,
        n_epoch=n_epoch,
        history=history,
        history_eval_fn=curve_eval_fn,
        write_artifacts=False,
    )
    print(f"[X-DER] end ACC_new={history[-1]['acc']:.4f} | test={row['ACC_new']:.4f}")
    return [history_row("X-DER", h) for h in history]


def run_clora_gncdm_curve(device, n_epoch):
    cfg = CL._config_for_dataset("junyi")
    cfg = dict(cfg)
    cfg["name"] = "junyi"
    meta = CL.load_partition(cfg)
    CL.set_seed(42)
    base = CL._new_model(cfg, meta, device)
    CL.train_real(
        base,
        meta["train_old"],
        meta["log_old_only"],
        list(base.parameters()),
        device,
        n_epoch=n_epoch,
        desc="Base(CLoRA)",
    )
    CL.populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    import copy as _copy

    base_state = _copy.deepcopy(base.state_dict())

    def eval_fn(m):
        CL.populate_buffers(m, meta["log_full"], device)
        old_m = CL.evaluate_buf(m, meta["test_old"], device)
        new_m = CL.evaluate_buf(m, meta["test_new"], device)
        return {
            "acc": new_m["acc"],
            "auc": new_m["auc"],
            "acc_old": old_m["acc"],
            "auc_old": old_m["auc"],
        }

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
    print(f"[C-LoRA-GNCDM] end ACC_new={history[-1]['acc']:.4f} | test={r['ACC_new']:.4f}")
    return [history_row("C-LoRA-GNCDM", h) for h in history]


def main():
    n_epoch = _cli_epochs.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | epochs = {n_epoch}")
    specs, rs_kw, base, curve_eval_fn, meta = load_ctx(device, n_epoch)
    rows = run_gncdm_family(specs, rs_kw, base)
    rows += run_xder_curve(curve_eval_fn, device, n_epoch, meta)
    rows += run_clora_gncdm_curve(device, n_epoch)
    out = os.path.join(SAVE_DIR, f"epoch_curve_gncdm_junyi_random_split_ep{n_epoch}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
