"""Math1 random_split main entry: original methods plus ICDM-WWW24.

Run from ``GNCDM/experiments``.  ``--icdm-only`` runs only the migrated graph
baseline and upserts its row into an existing all-methods table.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for path in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import cl_baselines_random_split as clbase  # noqa: E402
from run_icdm_ww24 import DEFAULT_EPOCHS, run_math1_random_split  # noqa: E402
from run_incremental_math1 import run_experiment, set_seed  # noqa: E402

DATA_DIR = os.path.join(gncdm_dir, "data")
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHA = 0.20
STRATEGY_SELECT_METRIC = "acc"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Math1 random-split baselines")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--icdm-only", action="store_true", help="only run ICDM-WWW24")
    mode.add_argument("--skip-icdm", action="store_true", help="run the original table only")
    parser.add_argument("--icdm-epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args(argv)


def main(argv=None):
    import torch

    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    q_path = os.path.join(DATA_DIR, "math1_Q_matrix.npy")
    train_path = os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv")
    valid_path = os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv")
    test_path = os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv")
    cfg = {
        "name": "math1",
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
        "Q": q_path,
        "n_user": 4209,
        "n_item": 20,
        "n_know": 11,
        "new_concepts": NEW_CONCEPTS,
        "ours_csv": "incremental_results_math1_random_split.csv",
    }

    if not args.icdm_only:
        set_seed(42)
        run_experiment(
            "math1_random_split",
            "buf",
            train_path,
            valid_path,
            test_path,
            q_path,
            device,
            n_user=4209,
            n_item_total=20,
            n_know_total=11,
            new_concepts=NEW_CONCEPTS,
            alpha=ALPHA,
            strategy_select_metric=STRATEGY_SELECT_METRIC,
        )
        clbase.run_one(cfg, device)
    if not args.skip_icdm:
        run_math1_random_split(cfg, device, epochs=args.icdm_epochs, append=True)
    print("\n完成：incremental_result/all_methods_math1_random_split.csv")


if __name__ == "__main__":
    main()
