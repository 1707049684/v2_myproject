"""Math1 user_split main entry: original methods plus ICDM-WWW24.

Run from ``GNCDM/experiments``.  ``--icdm-only`` runs only the migrated
inductive baseline and upserts its row into an existing all-methods table.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for path in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_all_methods_user_split import run_one as user_split_all_methods  # noqa: E402
from run_icdm_ww24 import DEFAULT_EPOCHS, run_math1_user_split  # noqa: E402

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(gncdm_dir, "data")
NEW_CONCEPTS = [0, 1, 3, 6]
ALPHA = 0.70


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Math1 user-split baselines")
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
    cfg = {
        "train": os.path.join(repo_root, "data", "math1", "user_split", "train.csv"),
        "valid": os.path.join(repo_root, "data", "math1", "user_split", "valid.csv"),
        "test": os.path.join(repo_root, "data", "math1", "user_split", "test.csv"),
        "Q": os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
        "n_user": 4209,
        "n_item": 20,
        "n_know": 11,
        "new_concepts": NEW_CONCEPTS,
        "alpha": ALPHA,
    }
    if not args.icdm_only:
        user_split_all_methods("math1_user_split", cfg, device)
    if not args.skip_icdm:
        run_math1_user_split(cfg, device, epochs=args.icdm_epochs, append=True)
    print("\n完成：incremental_result/all_methods_math1_user_split.csv")


if __name__ == "__main__":
    main()
