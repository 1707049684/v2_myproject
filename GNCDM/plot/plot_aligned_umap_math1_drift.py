# -*- coding: utf-8 -*-
"""AlignedUMAP of old-concept traits across incremental stages (Math1).

Design (scheme A — two panels):
  (a) t=0  Base training end:     θ_old^(0) ∈ R^{N×K}
  (b) t=1  After incremental:     θ_old^(1) ∈ R^{N×K}
Only the first K old-concept dims are kept (supports zero-drift claim).
Shared learners U^(0) ∩ U^(1) are aligned via AlignedUMAP relations.

Run:
  cd GNCDM/plot
  python plot_aligned_umap_math1_drift.py --method full --reuse
  python plot_aligned_umap_math1_drift.py --method lora --reuse

Out:
  incremental_result/aligned_umap_math1_{clean_full|clean_lora}_old_drift.{png,pdf,svg}
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
from umap import AlignedUMAP

PLOT_DIR = Path(__file__).resolve().parent
GNCDM_DIR = PLOT_DIR.parent
EXPERIMENTS_DIR = GNCDM_DIR / "experiments"
for p in (str(GNCDM_DIR), str(EXPERIMENTS_DIR), str(EXPERIMENTS_DIR / "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_incremental_math1 as R
from core.model import GNCDM

SAVE_DIR = Path(R.SAVE_DIR)
CACHE_DIR = SAVE_DIR / "aligned_umap_cache_math1"
DATA_DIR = Path(R.DATA_DIR)

N_USER, N_ITEM, N_KNOW = 4209, 20, 11
ALPHA_DNA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_EPOCH = 15
SEED = 42

METHODS = {
    "full": ("Ours (Dynamic DNA)", "CLEAN-Full", True),
    "lora": ("Ours (LoRA)", "CLEAN-LoRA", False),
}


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
        }
    )


def pick_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_theta(model: GNCDM, log_mat: np.ndarray, device: torch.device, batch=256) -> np.ndarray:
    model.eval()
    log_t = torch.tensor(log_mat, dtype=torch.float32, device=device)
    chunks = []
    for i in range(0, log_t.shape[0], batch):
        chunks.append(model.diagnose_theta(log_t[i : i + batch]).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def score_from_log(log_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_obs = (log_mat != 0).sum(axis=1)
    n_correct = (log_mat > 0).sum(axis=1)
    keep = n_obs > 0
    score = np.full(log_mat.shape[0], np.nan, dtype=np.float64)
    score[keep] = n_correct[keep] / n_obs[keep]
    return score, keep


def train_base_and_incremental(method_key: str, device: torch.device, n_epoch: int):
    """Return (theta_old_t0, theta_old_t1, score, user_ids, k_old)."""
    strat_name, _, mask_agg = METHODS[method_key]
    R.set_seed(SEED)
    Q = np.load(DATA_DIR / "math1_Q_matrix.npy")
    df_train = pd.read_csv(DATA_DIR / "math1_train_0.8_0.2.csv")
    df_valid = pd.read_csv(DATA_DIR / "math1_valid_0.8_0.2.csv")

    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = N_ITEM - n_item_old, N_KNOW - n_know_old
    Q_expanded = Q_mat.copy()

    train_old = df_train[df_train.item_id < n_item_old].copy()
    train_new = df_train[df_train.item_id >= n_item_old].copy()
    valid_old = df_valid[df_valid.item_id < n_item_old].copy()
    valid_new = df_valid[df_valid.item_id >= n_item_old].copy()
    log_old = R.build_log_mat(train_old, N_USER, n_item_old)
    log_full = R.build_log_mat(df_train, N_USER, N_ITEM)

    base = GNCDM(
        n_user=N_USER,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=ALPHA_DNA,
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
        desc="Base(t=0)",
        eval_fn=lambda m: (
            R.populate_buffers(m, log_old, device),
            R.evaluate_buf(m, valid_old, device),
        )[1],
    )

    # t=0: diagnose on old-item log → θ ∈ R^{N×K}
    theta0_all = extract_theta(base, log_old, device)

    specs = R.buf_strategy_specs(
        n_item_new, n_know_new, n_item_old, Q_expanded, train_old, train_new, valid_old, valid_new
    )
    spec = specs[strat_name]

    def strat_eval_fn(valid_df):
        return lambda m: (
            R.populate_buffers(m, log_full, device),
            R.evaluate_buf(m, valid_df, device),
        )[1]

    model = R.run_strategy(
        base,
        strat_name,
        spec["expand_fn"],
        spec["params_fn"],
        spec["train_df"],
        spec["valid_df"],
        log_full=log_full,
        n_know_old=n_know_old,
        device=device,
        strat_eval_fn=strat_eval_fn,
        final_old=lambda m: R.evaluate_buf(m, valid_old, device),
        final_new=lambda m: R.evaluate_buf(m, valid_new, device),
        n_epoch=n_epoch,
        mask_agg_old=mask_agg,
    )

    # t=1: full log → take first K old dims only
    theta1_all = extract_theta(model, log_full, device)[:, :n_know_old]

    # Shared learners: observed on old items at t=0 (and thus present at t=1).
    score_all, keep = score_from_log(log_old)
    user_ids = np.where(keep)[0]
    theta0 = theta0_all[user_ids]
    theta1 = theta1_all[user_ids]
    score = score_all[user_ids]

    drift = np.linalg.norm(theta1 - theta0, axis=1)
    print(
        f"[{METHODS[method_key][1]}] K_old={n_know_old} N_shared={len(user_ids)} "
        f"|θ1-θ0|_2: mean={drift.mean():.6f} max={drift.max():.6f}"
    )
    return theta0, theta1, score, user_ids, n_know_old


def load_or_train(method_key: str, reuse: bool, n_epoch: int, device: torch.device):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = METHODS[method_key][1].lower().replace("-", "_")
    paths = {
        "t0": CACHE_DIR / f"theta_old_t0_{tag}.npy",
        "t1": CACHE_DIR / f"theta_old_t1_{tag}.npy",
        "score": CACHE_DIR / f"score_old_{tag}.npy",
        "uids": CACHE_DIR / f"user_ids_{tag}.npy",
        "meta": CACHE_DIR / f"meta_{tag}.npz",
    }
    if reuse and all(paths[k].exists() for k in ("t0", "t1", "score", "uids")):
        print(f"reuse cached θ_old from {CACHE_DIR} ({tag})")
        return (
            np.load(paths["t0"]),
            np.load(paths["t1"]),
            np.load(paths["score"]),
            np.load(paths["uids"]),
        )

    theta0, theta1, score, uids, k_old = train_base_and_incremental(method_key, device, n_epoch)
    np.save(paths["t0"], theta0)
    np.save(paths["t1"], theta1)
    np.save(paths["score"], score)
    np.save(paths["uids"], uids)
    paths["meta"].write_text(f"k_old={k_old}\nn_shared={len(uids)}\n", encoding="utf-8")
    # drop stale embeddings
    for p in CACHE_DIR.glob(f"aligned_xy_*_{tag}.npy"):
        p.unlink()
    return theta0, theta1, score, uids


def run_aligned_umap(theta0: np.ndarray, theta1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = theta0.shape[0]
    # Identity map: row i in t=0 ↔ row i in t=1 (shared-learner matrices already aligned).
    relations = [{i: i for i in range(n)}]
    reducer = AlignedUMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        n_epochs=200,
    )
    reducer.fit([theta0, theta1], relations=relations)
    z0, z1 = reducer.embeddings_
    return np.asarray(z0), np.asarray(z1)


def draw_panel(ax, xy, score, title):
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=score,
        cmap="RdBu",
        s=5,
        alpha=0.85,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_aspect("equal", adjustable="datalim")
    return sc


def plot_dual(z0, z1, score, method_label: str, out_stem: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    draw_panel(axes[0], z0, score, f"(a) Base training (t=0)\n{method_label}")
    sc = draw_panel(axes[1], z1, score, f"(b) After incremental update (t=1)\n{method_label}")
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Score rate (old items)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        r"AlignedUMAP of old-concept traits $\theta_{\mathrm{old}}$ (Math1)",
        fontsize=9.5,
        y=1.04,
    )
    fig.subplots_adjust(wspace=0.12, right=0.90, left=0.04, top=0.82, bottom=0.06)
    for ext in ("png", "pdf", "svg"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="full")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument("--force-embed", action="store_true", help="recompute AlignedUMAP")
    args = parser.parse_args()

    device = pick_device()
    _, label, _ = METHODS[args.method]
    tag = label.lower().replace("-", "_")

    theta0, theta1, score, _uids = load_or_train(args.method, args.reuse, args.epochs, device)

    z0_path = CACHE_DIR / f"aligned_xy_t0_{tag}.npy"
    z1_path = CACHE_DIR / f"aligned_xy_t1_{tag}.npy"
    if args.reuse and not args.force_embed and z0_path.exists() and z1_path.exists():
        print(f"reuse AlignedUMAP embeddings {z0_path.name}")
        z0, z1 = np.load(z0_path), np.load(z1_path)
    else:
        print(f"AlignedUMAP on θ_old {theta0.shape} → {theta1.shape} ...")
        z0, z1 = run_aligned_umap(theta0, theta1)
        np.save(z0_path, z0)
        np.save(z1_path, z1)

    # Embedding drift (should be tiny for CLEAN-Full if θ_old nearly identical).
    emb_drift = np.linalg.norm(z1 - z0, axis=1)
    print(f"2D embedding drift |z1-z0|_2: mean={emb_drift.mean():.6f} max={emb_drift.max():.6f}")

    out = SAVE_DIR / f"aligned_umap_math1_{tag}_old_drift"
    plot_dual(z0, z1, score, label, out)


if __name__ == "__main__":
    main()
