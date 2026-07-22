# -*- coding: utf-8 -*-
"""Junyi random_split EWC/DER++ epoch curves. Needs avalanche.

  cd GNCDM/plot && python plot_epoch_curve_avalanche_junyi.py --epochs 15
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

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import cl_baselines_random_split as CB

EWC_LAMBDA = 1000  # all_methods_junyi_random_split.csv
SAVE_DIR = CB.SAVE_DIR


def ewc_curve(meta, device, curve_max_epoch):
    from avalanche.core import SupervisedPlugin
    from avalanche.training.supervised import EWC

    history = []

    class NewTaskAccPlugin(SupervisedPlugin):
        def __init__(self):
            super().__init__()
            self.epoch_ctr = 0

        def after_training_epoch(self, strategy, **kwargs):
            if strategy.experience.current_experience != 1:
                return
            self.epoch_ctr += 1
            _, _, acc_old, _ = CB.evaluate_cd_metrics(strategy.model, meta["valid_old_ds"], device)
            _, _, acc_new, _ = CB.evaluate_cd_metrics(strategy.model, meta["valid_new_ds"], device)
            history.append({"epoch": self.epoch_ctr, "acc": acc_new, "acc_old": acc_old})

    CB.set_seed(42)
    model = CB.CognitiveBackbone(meta["num_students"], meta["num_items"], CB.EMBED_DIM).to(device)
    strat = EWC(
        model,
        optim.Adam(model.parameters(), lr=CB.LR),
        nn.CrossEntropyLoss(),
        ewc_lambda=EWC_LAMBDA,
        mode=CB.EWC_MODE,
        train_mb_size=CB.TRAIN_MB_SIZE,
        train_epochs=curve_max_epoch,
        eval_mb_size=256,
        device=device,
        plugins=[NewTaskAccPlugin()],
    )
    for exp in CB._bench(meta).train_stream:
        strat.train(exp)
    return [
        {"Model": "EWC", "epoch": h["epoch"], "ACC_new": h["acc"], "ACC_old": h["acc_old"]}
        for h in history
    ]


def der_curve(meta, device, curve_max_epoch):
    from avalanche.training.supervised import DER

    history = []
    CB.set_seed(42)
    model = CB.CognitiveBackbone(meta["num_students"], meta["num_items"], CB.EMBED_DIM).to(device)
    strat = DER(
        model,
        optim.Adam(model.parameters(), lr=CB.LR),
        nn.CrossEntropyLoss(),
        mem_size=CB.MEM_SIZE,
        alpha=CB.DER_ALPHA,
        beta=CB.DER_BETA,
        train_mb_size=CB.TRAIN_MB_SIZE,
        train_epochs=CB.DER_EPOCHS_INNER,
        eval_mb_size=256,
        device=device,
    )
    for exp in CB._bench(meta).train_stream:
        tid = exp.current_experience
        val_ds = (
            meta["valid_old_ds"]
            if tid == 0
            else CB.ConcatDataset([meta["valid_old_ds"], meta["valid_new_ds"]])
        )
        best_acc, best_state = -1.0, None
        ep = 0
        max_ep = curve_max_epoch if tid == 1 else CB.TRAIN_EPOCHS
        for _ in range(max_ep):
            strat.train(exp)
            _, _, val_acc, _ = CB.evaluate_cd_metrics(model, val_ds, device)
            if tid == 1:
                ep += 1
                _, _, acc_old, _ = CB.evaluate_cd_metrics(model, meta["valid_old_ds"], device)
                _, _, acc_new, _ = CB.evaluate_cd_metrics(model, meta["valid_new_ds"], device)
                history.append({"epoch": ep, "acc": acc_new, "acc_old": acc_old})
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return [
        {"Model": "DER++", "epoch": h["epoch"], "ACC_new": h["acc"], "ACC_old": h["acc_old"]}
        for h in history
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | epochs = {args.epochs}")
    junyi = os.path.join(REPO_ROOT, "data", "junyi")
    cfg = {
        "name": "junyi",
        "train": os.path.join(junyi, "new_random_split", "train.csv"),
        "valid": os.path.join(junyi, "new_random_split", "valid.csv"),
        "test": os.path.join(junyi, "new_random_split", "test.csv"),
        "Q": os.path.join(junyi, "Q_matrix.npy"),
        "n_item": int(__import__("numpy").load(os.path.join(junyi, "Q_matrix.npy")).shape[0]),
        "n_know": int(__import__("numpy").load(os.path.join(junyi, "Q_matrix.npy")).shape[1]),
        "new_concepts": "auto",
    }
    meta = CB.load_random(cfg)
    rows = ewc_curve(meta, device, args.epochs)
    rows += der_curve(meta, device, args.epochs)
    out = os.path.join(SAVE_DIR, f"epoch_curve_avalanche_junyi_random_split_ep{args.epochs}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
