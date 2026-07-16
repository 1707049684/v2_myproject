# -*- coding: utf-8 -*-
"""Old-item prediction distributions for CLEAN aggregation ablations.

Variants (match run_ablation_dna_a0910_user_split.py):
  Base
  CLEAN-Full              = OrthoMask ✓  FrozenBias ✓   (Ours Dynamic DNA)
  CLEAN (w/o OrthoMask)   = OrthoMask ✗  FrozenBias ✓
  CLEAN (w/o FrozenBias)  = OrthoMask ✓  FrozenBias ✗
  CLEAN (w/o OCM)         = OrthoMask ✗  FrozenBias ✗   (Ours-Ablated)

user_split support/query reconstruction on qry_test_old.
AlignedUMAP on θ_old cannot separate these (RD=0); ŷ on old items can.

Run:
  cd GNCDM/plot
  python plot_old_pred_dist_ablation.py --dataset math1
  python plot_old_pred_dist_ablation.py --dataset a0910   # needs GPU

Out:
  incremental_result/old_pred_dist_ablation_{dataset}_user_split.{png,csv}
  incremental_result/additonal/old_pred_dist_ablation_{dataset}_user_split.{pdf,svg}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PLOT_DIR = Path(__file__).resolve().parent
GNCDM_DIR = PLOT_DIR.parent
EXPERIMENTS_DIR = GNCDM_DIR / "experiments"
REPO_ROOT = GNCDM_DIR.parent
SAVE_DIR = GNCDM_DIR / "incremental_result"
EXTRA_DIR = SAVE_DIR / "additonal"
CACHE_DIR = SAVE_DIR / "old_pred_dist_cache"

for p in (str(GNCDM_DIR), str(EXPERIMENTS_DIR), str(EXPERIMENTS_DIR / "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_all_methods_user_split import prepare  # noqa: E402
from run_incremental_a0910 import auto_new_concepts  # noqa: E402
from run_incremental_math1 import (  # noqa: E402
    SampleSet,
    fresh_base,
    new_params,
    populate_buffers,
    set_seed,
    train_real,
)
from core.model import GNCDM  # noqa: E402

# Display name, cache tag, color
METHODS = [
    ("Base", "base", "#222222"),
    ("CLEAN-Full", "clean_full", "#0F4D92"),
    ("CLEAN (w/o OrthoMask)", "wo_orthomask", "#C47A2C"),
    ("CLEAN (w/o FrozenBias)", "wo_frozenbias", "#7A4E9A"),
    ("CLEAN (w/o OCM)", "wo_ocm", "#B64342"),
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def dataset_cfg(name: str) -> dict:
    if name == "math1":
        return {
            "train": str(REPO_ROOT / "data" / "math1" / "user_split" / "train.csv"),
            "valid": str(REPO_ROOT / "data" / "math1" / "user_split" / "valid.csv"),
            "test": str(REPO_ROOT / "data" / "math1" / "user_split" / "test.csv"),
            "Q": str(GNCDM_DIR / "data" / "math1_Q_matrix.npy"),
            "n_user": 4209,
            "n_item": 20,
            "n_know": 11,
            "new_concepts": [0, 1, 3, 6],
            "alpha": 0.70,
            "n_epoch_base": 15,
            "n_epoch_inc": 15,
        }
    if name == "a0910":
        a0910 = REPO_ROOT / "data" / "a0910"
        Q = np.load(a0910 / "Q_matrix.npy")
        return {
            "train": str(a0910 / "new_user_split" / "train.csv"),
            "valid": str(a0910 / "new_user_split" / "valid.csv"),
            "test": str(a0910 / "new_user_split" / "test.csv"),
            "Q": str(a0910 / "Q_matrix.npy"),
            "n_user": 4163,
            "n_item": 17746,
            "n_know": 123,
            "new_concepts": auto_new_concepts(Q, 0.34),
            "alpha": 0.6,
            "n_epoch_base": 25,
            "n_epoch_inc": 25,
        }
    raise ValueError(f"unknown dataset {name}")


def collect_preds(model, eval_df, eval_log_mat, device, batch_size=256) -> np.ndarray:
    """Return ŷ on eval_df under reconstruction logs (same as evaluate_recon)."""
    model.eval()
    log_t = torch.tensor(eval_log_mat, dtype=torch.float32, device=device)
    loader = DataLoader(SampleSet(eval_df), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for user_ids, item_ids, _score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_log = log_t[user_ids]
            item_log = log_t[:, item_ids].T
            pred = model(user_log, item_log, user_ids, item_ids)
            preds.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


def make_col_mask(k_old: int):
    def hook(grad):
        g = grad.clone()
        g[:, :k_old] = 0.0
        return g

    return hook


def train_variants(ours, meta, device, n_epoch_base: int, n_epoch_inc: int) -> dict[str, np.ndarray]:
    n_user, alpha = meta["n_user"], meta["alpha"]
    n_item_old, n_know_old = ours["n_item_old"], ours["n_know_old"]
    n_item_new, n_know_new = ours["n_item_new"], ours["n_know_new"]
    log_old, log_full = ours["log_old"], ours["log_full"]
    Q_old, Q_expanded = ours["Q_old"], ours["Q_expanded"]

    def base_eval_fn(m):
        from run_incremental_math1 import evaluate_recon

        return evaluate_recon(m, ours["qry_valid_old"], ours["sup_valid_old_log"], device)

    def strat_eval_fn(qvalid_df):
        from run_incremental_math1 import evaluate_recon

        return lambda m: evaluate_recon(m, qvalid_df, ours["sup_valid_full_log"], device)

    print("=== Base ===")
    base = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=alpha,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device,
    ).to(device)
    train_real(
        base,
        ours["train_old"],
        log_old,
        list(base.parameters()),
        device,
        n_epoch=n_epoch_base,
        desc="Base",
        eval_fn=base_eval_fn,
    )
    populate_buffers(base, log_old, device)
    out = {
        "Base": collect_preds(base, ours["qry_test_old"], ours["sup_test_old_log"], device),
    }

    specs = {
        "CLEAN-Full": dict(
            params_fn=lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
            mask_agg_old=True,
        ),
        "CLEAN (w/o OrthoMask)": dict(
            params_fn=lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
            mask_agg_old=False,
        ),
        "CLEAN (w/o FrozenBias)": dict(
            params_fn=lambda m: new_params(m)
            + [
                m.theta_agg_mat.weight,
                m.theta_agg_mat.bias,
                m.psi_agg_mat.weight,
                m.psi_agg_mat.bias,
            ],
            mask_agg_old=True,
        ),
        "CLEAN (w/o OCM)": dict(
            params_fn=lambda m: [p for p in m.parameters() if p.requires_grad],
            mask_agg_old=False,
        ),
    }

    for name, spec in specs.items():
        print(f"=== {name} ===")
        m = fresh_base(base)
        m.expand_topology(n_item_new, n_know_new, Q_expanded)
        populate_buffers(m, log_full, device)
        handles = []
        if spec["mask_agg_old"]:
            handles.append(m.theta_agg_mat.weight.register_hook(make_col_mask(n_know_old)))
            handles.append(m.psi_agg_mat.weight.register_hook(make_col_mask(n_know_old)))
        train_real(
            m,
            ours["train_new"],
            log_full,
            spec["params_fn"](m),
            device,
            n_epoch=n_epoch_inc,
            desc=name,
            eval_fn=strat_eval_fn(ours["qry_valid_new"]),
        )
        for h in handles:
            h.remove()
        populate_buffers(m, log_full, device)
        out[name] = collect_preds(m, ours["qry_test_old"], ours["sup_test_full_log"], device)

    return out


def cache_paths(dataset: str) -> dict[str, Path]:
    d = CACHE_DIR / f"{dataset}_user_split"
    d.mkdir(parents=True, exist_ok=True)
    return {label: d / f"{tag}.npy" for label, tag, _ in METHODS}


def load_or_train(dataset: str, reuse: bool, device) -> dict[str, np.ndarray]:
    paths = cache_paths(dataset)
    if reuse and all(p.exists() for p in paths.values()):
        print(f"reuse cached preds under {paths['Base'].parent}")
        return {label: np.load(paths[label]) for label, _, _ in METHODS}

    cfg = dataset_cfg(dataset)
    set_seed(42)
    ours, _, meta = prepare(cfg, device)
    preds = train_variants(
        ours,
        meta,
        device,
        n_epoch_base=cfg["n_epoch_base"],
        n_epoch_inc=cfg["n_epoch_inc"],
    )
    for label, _, _ in METHODS:
        np.save(paths[label], preds[label])
    return preds


def summary_table(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    base = preds["Base"]
    rows = []
    for label, _, _ in METHODS:
        y = preds[label]
        rows.append(
            {
                "Method": label,
                "mean_pred": float(y.mean()),
                "std_pred": float(y.std()),
                "mean_abs_diff_vs_Base": float(np.abs(y - base).mean()),
                "max_abs_diff_vs_Base": float(np.abs(y - base).max()),
                "frac_diff_gt_1e-6": float((np.abs(y - base) > 1e-6).mean()),
            }
        )
    return pd.DataFrame(rows)


def _density(y: np.ndarray, xs: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 2 or np.allclose(y, y[0]):
        dens = np.zeros_like(xs)
        dens[np.argmin(np.abs(xs - float(y.mean())))] = 1.0
        return dens
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(y)
        return kde(xs)
    except Exception:
        hist, edges = np.histogram(y, bins=40, range=(0.0, 1.0), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(xs, centers, hist, left=0.0, right=0.0)


def plot_dists(preds: dict[str, np.ndarray], dataset: str) -> Path:
    """Two panels: (a) |ŷ−ŷ_Base| — identity claim; (b) raw ŷ density."""
    setup_style()
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    base = preds["Base"]
    title_ds = "Math1" if dataset == "math1" else "ASSIST a0910"

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), dpi=200)

    # (a) absolute deviation from Base — CLEAN-Full collapses to 0
    ax = axes[0]
    xs_d = np.linspace(0.0, 0.85, 256)
    for label, _, color in METHODS:
        if label == "Base":
            continue
        diff = np.abs(preds[label] - base)
        dens = _density(diff, xs_d)
        lw = 1.6 if label == "CLEAN-Full" else 1.2
        ax.plot(xs_d, dens, color=color, linewidth=lw, label=label)
    ax.set_xlabel(r"$|\hat{y}-\hat{y}_{\mathrm{Base}}|$ on old items")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(bottom=0.0)
    ax.set_title("(a) Deviation from Base")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.4)

    # (b) raw ŷ — Base dashed; CLEAN-Full should overlap Base
    ax = axes[1]
    xs = np.linspace(0.0, 1.0, 256)
    for label, _, color in METHODS:
        dens = _density(preds[label], xs)
        ls = "--" if label == "Base" else "-"
        lw = 1.6 if label in ("Base", "CLEAN-Full") else 1.1
        ax.plot(xs, dens, color=color, linestyle=ls, linewidth=lw, label=label)
    ax.set_xlabel(r"Predicted score $\hat{y}$ on old items")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    # Cap y so FrozenBias spike does not hide other curves; spike still visible at left
    other_max = max(
        _density(preds[lab], xs).max()
        for lab, _, _ in METHODS
        if lab != "CLEAN (w/o FrozenBias)"
    )
    ax.set_ylim(0.0, max(3.0, 1.35 * other_max))
    ax.set_title("(b) Prediction density")
    ax.legend(loc="upper right", fontsize=5.5, ncol=1, handlelength=1.4)

    fig.suptitle(
        f"Old-item predictions after incremental update ({title_ds}, user split)",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()

    stem = f"old_pred_dist_ablation_{dataset}_user_split"
    png = SAVE_DIR / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(EXTRA_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(EXTRA_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["math1", "a0910"], default="math1")
    parser.add_argument("--reuse", action="store_true", help="reuse cached ŷ .npy")
    parser.add_argument("--force", action="store_true", help="retrain even if cache exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} dataset={args.dataset}")
    if args.dataset == "a0910" and device.type == "cpu":
        print("WARNING: a0910 on CPU is very slow; prefer a GPU host.")

    reuse = args.reuse and not args.force
    preds = load_or_train(args.dataset, reuse=reuse, device=device)
    df = summary_table(preds)
    csv_path = SAVE_DIR / f"old_pred_dist_ablation_{args.dataset}_user_split.csv"
    df.to_csv(csv_path, index=False)
    png = plot_dists(preds, args.dataset)
    print(df.to_string(index=False))
    print(f"wrote {csv_path}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
