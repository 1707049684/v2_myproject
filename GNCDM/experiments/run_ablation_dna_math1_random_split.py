"""Run CLEAN component ablations on the Math1 random split."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(HERE)
for path in (GNCDM_DIR, HERE, os.path.join(HERE, "_core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.model import GNCDM  # noqa: E402
from incremental.metrics import calculate_rd  # noqa: E402
from run_incremental_math1 import (  # noqa: E402
    DATA_DIR,
    SAVE_DIR,
    build_log_mat,
    evaluate_buf,
    fresh_base,
    new_params,
    populate_buffers,
    remap_items,
    set_seed,
    strict_bipartition,
    train_real,
)

N_USER, N_ITEM, N_KNOW = 4209, 20, 11
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHA = 0.20
N_EPOCH = 15


def make_col_mask(n_know_old):
    def hook(grad):
        masked = grad.clone()
        masked[:, :n_know_old] = 0.0
        return masked

    return hook


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    q_mat = np.load(os.path.join(DATA_DIR, "math1_Q_matrix.npy"))
    train = pd.read_csv(os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"))
    valid = pd.read_csv(os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv"))
    q_mat, item_map, n_item_old, n_know_old = strict_bipartition(q_mat, NEW_CONCEPTS)
    train, valid, test = (remap_items(frame, item_map) for frame in (train, valid, test))
    n_item_new, n_know_new = N_ITEM - n_item_old, N_KNOW - n_know_old

    train_old = train[train.item_id < n_item_old].copy()
    train_new = train[train.item_id >= n_item_old].copy()
    valid_old = valid[valid.item_id < n_item_old].copy()
    valid_new = valid[valid.item_id >= n_item_old].copy()
    test_old = test[test.item_id < n_item_old].copy()
    test_new = test[test.item_id >= n_item_old].copy()
    log_old = build_log_mat(train_old, N_USER, n_item_old)
    log_full = build_log_mat(train, N_USER, N_ITEM)

    base = GNCDM(
        n_user=N_USER,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=ALPHA,
        Q_mat=q_mat[:n_item_old, :n_know_old].copy(),
        monotonicity_assumption=True,
        device=device,
    ).to(device)

    def base_eval(model):
        populate_buffers(model, log_old, device)
        return evaluate_buf(model, valid_old, device)

    train_real(
        base,
        train_old,
        log_old,
        list(base.parameters()),
        device,
        n_epoch=N_EPOCH,
        desc="Base",
        eval_fn=base_eval,
    )
    populate_buffers(base, log_old, device)
    base_theta = base.get_Theta_buf().clone()

    methods = [
        (
            "CLEAN(w/o OCM)",
            lambda m: [p for p in m.parameters() if p.requires_grad],
            False,
        ),
        (
            "CLEAN(w/o OrthoMask)",
            lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
            False,
        ),
        (
            "CLEAN(w/o FrozenBias)",
            lambda m: new_params(m)
            + [
                m.theta_agg_mat.weight,
                m.theta_agg_mat.bias,
                m.psi_agg_mat.weight,
                m.psi_agg_mat.bias,
            ],
            True,
        ),
        (
            "CLEAN-Full",
            lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
            True,
        ),
    ]

    rows = []
    for name, params_fn, mask_agg_old in methods:
        print(f"\n=== {name} ===")
        model = fresh_base(base)
        model.expand_topology(n_item_new, n_know_new, q_mat.copy())
        populate_buffers(model, log_full, device)
        handles = []
        if mask_agg_old:
            handles.append(model.theta_agg_mat.weight.register_hook(make_col_mask(n_know_old)))
            handles.append(model.psi_agg_mat.weight.register_hook(make_col_mask(n_know_old)))

        def incremental_eval(candidate):
            populate_buffers(candidate, log_full, device)
            return evaluate_buf(candidate, valid_new, device)

        train_real(
            model,
            train_new,
            log_full,
            params_fn(model),
            device,
            n_epoch=N_EPOCH,
            desc=name,
            eval_fn=incremental_eval,
        )
        for handle in handles:
            handle.remove()
        populate_buffers(model, log_full, device)
        old, new = evaluate_buf(model, test_old, device), evaluate_buf(model, test_new, device)
        rows.append(
            {
                "Model": name,
                "ACC_old": old["acc"],
                "ACC_new": new["acc"],
                "F1_old": old["f1"],
                "F1_new": new["f1"],
                "RMSE_old": old["rmse"],
                "RMSE_new": new["rmse"],
                "RD": calculate_rd(base_theta.to(device), model.get_Theta_buf().to(device), n_know_old),
                "AUC_old": old["auc"],
                "AUC_new": new["auc"],
            }
        )

    output = os.path.join(SAVE_DIR, "ablation_dna_math1_random_split.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nresults written to {output}")


if __name__ == "__main__":
    main()
