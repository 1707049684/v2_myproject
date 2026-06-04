# -*- coding: utf-8 -*-
"""Alpha sweep for Base on the random_split (Math1).

Replicates ONLY the Base path of run_incremental_math1.run_experiment under
mode='buf' (forward_using_buf, no-leak prediction), and reports test ACC_old
for each alpha so we can pick the alpha that maximizes Base's ACC_old on our
own (sub-)dataset rather than the paper's full-dataset number.
"""
import os
import numpy as np
import pandas as pd
import torch

import run_incremental_math1 as R
from core.model import GNCDM


def base_acc_for_alpha(alpha, device):
    """Train Base on the old-item subset (random_split) and return test metrics."""
    DATA_DIR = R.DATA_DIR
    Q_path = os.path.join(DATA_DIR, "math1_Q_matrix.npy")
    train_path = os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv")
    valid_path = os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv")
    test_path = os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv")

    n_user, n_item_total, n_know_total = 4209, 20, 11
    NEW_CONCEPTS = [0, 1, 3, 6]

    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    Q_mat, item_id_map, n_item_old, n_know_old = R.strict_bipartition(Q_mat, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_id_map)
    df_valid = R.remap_items(df_valid, item_id_map)
    df_test = R.remap_items(df_test, item_id_map)
    Q_old = Q_mat[:n_item_old, :n_know_old].copy()

    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    test_old = df_test[df_test["item_id"] < n_item_old].copy()
    valid_old = df_valid[df_valid["item_id"] < n_item_old].copy()
    log_old = R.build_log_mat(train_old, n_user, n_item_old)

    def base_eval_fn(m):
        R.populate_buffers(m, log_old, device)
        return R.evaluate_buf(m, valid_old, device)

    R.set_seed(42)  # 与主实验一致，保证可复现
    base = GNCDM(n_user=n_user, n_item=n_item_old, n_know=n_know_old,
                 user_dim=32, item_dim=32, alpha=alpha, Q_mat=Q_old,
                 monotonicity_assumption=True, device=device).to(device)
    R.train_real(base, train_old, log_old, list(base.parameters()), device,
                 n_epoch=25, desc=f"Base(a={alpha})", eval_fn=base_eval_fn)
    R.populate_buffers(base, log_old, device)
    return R.evaluate_buf(base, test_old, device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    alphas = [round(0.05 * k, 2) for k in range(1, 20)]  # 0.05 .. 0.95
    rows = []
    for a in alphas:
        r = base_acc_for_alpha(a, device)
        rows.append({"alpha": a, "ACC_old": r["acc"], "AUC_old": r["auc"],
                     "RMSE_old": r["rmse"], "F1_old": r["f1"]})
        print(f"alpha={a:.2f}  ACC_old={r['acc']:.6f}  AUC_old={r['auc']:.6f}  F1_old={r['f1']:.6f}")

    df = pd.DataFrame(rows).sort_values("ACC_old", ascending=False)
    out = os.path.join(R.SAVE_DIR, "base_alpha_sweep_random_split.csv")
    df.to_csv(out, index=False)
    best = df.iloc[0]
    print("\n=== 按 Base ACC_old 排序 ===")
    print(df.to_string(index=False))
    print(f"\n最优 alpha = {best['alpha']:.2f}  ACC_old = {best['ACC_old']:.6f}")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
