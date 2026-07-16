"""a0910 random_split main entry: original methods plus ICDM-WWW24."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for path in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import cl_baselines_random_split as clbase  # noqa: E402
from run_icdm_ww24 import DEFAULT_EPOCHS, run_random_split  # noqa: E402
from run_incremental_a0910 import auto_new_concepts  # noqa: E402
from run_incremental_math1 import run_experiment, set_seed  # noqa: E402

repo_root = os.path.dirname(gncdm_dir)
DATA_DIR = os.path.join(repo_root, "data", "a0910")
N_USER, N_ITEM, N_KNOW = 4163, 17746, 123
ALPHA = 0.1


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a0910 random-split baselines")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--icdm-only", action="store_true", help="only run ICDM-WWW24")
    mode.add_argument("--skip-icdm", action="store_true", help="run the original table only")
    parser.add_argument("--icdm-epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--icdm-dim", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    import torch

    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cpu":
        print("注意：a0910 有 17746 道题，CPU 运行会慢于 GPU。")

    q_path = os.path.join(DATA_DIR, "Q_matrix.npy")
    q = np.load(q_path)
    new_concepts = auto_new_concepts(q, 0.34)
    random_dir = os.path.join(DATA_DIR, "new_random_split")
    train_path, valid_path, test_path = (
        os.path.join(random_dir, filename) for filename in ("train.csv", "valid.csv", "test.csv")
    )
    cfg = {
        "name": "a0910",
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
        "Q": q_path,
        "n_user": N_USER,
        "n_item": N_ITEM,
        "n_know": N_KNOW,
        "new_concepts": new_concepts,
        "ours_csv": "incremental_results_a0910_random_split.csv",
    }

    if not args.icdm_only:
        set_seed(42)
        run_experiment(
            "a0910_random_split",
            "buf",
            train_path,
            valid_path,
            test_path,
            q_path,
            device,
            n_user=N_USER,
            n_item_total=N_ITEM,
            n_know_total=N_KNOW,
            new_concepts=new_concepts,
            alpha=ALPHA,
        )
        clbase.run_one(cfg, device)
    if not args.skip_icdm:
        run_random_split(
            cfg,
            device,
            epochs=args.icdm_epochs,
            dim=args.icdm_dim,
            append=True,
        )
    print("\n完成：incremental_result/all_methods_a0910_random_split.csv")


if __name__ == "__main__":
    main()
