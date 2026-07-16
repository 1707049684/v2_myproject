# -*- coding: utf-8 -*-
"""AlignedUMAP of old-item prediction vectors on Math1 random_split.

Unlike θ_old AlignedUMAP (RD=0 for all CLEAN ablations), this embeds each
learner's old-item score predictions ŷ ∈ R^{M_old} at t=0 (Base) vs t=1
(after incremental), so aggregation ablations become visible.

Variants (same as DNA ablation):
  CLEAN-Full / CLEAN (w/o OrthoMask) / CLEAN (w/o FrozenBias) / CLEAN (w/o OCM)

Protocol: Math1 random_split (0.8/0.2 CSVs), α=0.20, strict bipartition,
forward_using_buf after populate_buffers (RQ2).

Run:
  cd GNCDM/plot
  python plot_aligned_umap_math1_pred_ablation.py
  python plot_aligned_umap_math1_pred_ablation.py --reuse

Out:
  incremental_result/aligned_umap_math1_pred_ablation_grid.png
  incremental_result/aligned_umap_math1_{tag}_pred_drift.png
  pdf/svg → incremental_result/additonal/
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
EXTRA_DIR = SAVE_DIR / "additonal"
CACHE_DIR = SAVE_DIR / "aligned_umap_pred_cache_math1"
DATA_DIR = Path(R.DATA_DIR)

N_USER, N_ITEM, N_KNOW = 4209, 20, 11
ALPHA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_EPOCH = 15
SEED = 42

# (display label, cache tag, params_fn factory, mask_agg_old)
METHODS = [
    (
        "CLEAN-Full",
        "clean_full",
        lambda m: R.new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        True,
    ),
    (
        "CLEAN (w/o OrthoMask)",
        "wo_orthomask",
        lambda m: R.new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        False,
    ),
    (
        "CLEAN (w/o FrozenBias)",
        "wo_frozenbias",
        lambda m: R.new_params(m)
        + [
            m.theta_agg_mat.weight,
            m.theta_agg_mat.bias,
            m.psi_agg_mat.weight,
            m.psi_agg_mat.bias,
        ],
        True,
    ),
    (
        "CLEAN (w/o OCM)",
        "wo_ocm",
        lambda m: [p for p in m.parameters() if p.requires_grad],
        False,
    ),
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
        }
    )


def pick_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def score_from_log(log_mat: np.ndarray):
    obs = log_mat >= 0
    n_obs = obs.sum(axis=1)
    n_correct = np.where(obs, log_mat, 0).sum(axis=1)
    keep = n_obs > 0
    score = np.full(log_mat.shape[0], np.nan, dtype=np.float64)
    score[keep] = n_correct[keep] / n_obs[keep]
    return score, keep


@torch.no_grad()
def pred_matrix_old(model: GNCDM, user_ids: np.ndarray, n_item_old: int, device, batch=256) -> np.ndarray:
    """ŷ ∈ R^{N×M_old} via forward_using_buf (random_split / RQ2)."""
    model.eval()
    n = len(user_ids)
    out = np.zeros((n, n_item_old), dtype=np.float64)
    item_all = torch.arange(n_item_old, device=device, dtype=torch.long)
    for i in range(0, n, batch):
        u = torch.as_tensor(user_ids[i : i + batch], device=device, dtype=torch.long)
        b = u.shape[0]
        # repeat users × items
        uu = u.repeat_interleave(n_item_old)
        ii = item_all.repeat(b)
        pred = model.forward_using_buf(uu, ii).detach().cpu().numpy().reshape(b, n_item_old)
        out[i : i + b] = pred
    return out


def make_col_mask(k_old: int):
    def hook(grad):
        g = grad.clone()
        g[:, :k_old] = 0.0
        return g

    return hook


def prepare_data():
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
    return {
        "Q_old": Q_mat[:n_item_old, :n_know_old].copy(),
        "Q_expanded": Q_expanded,
        "n_item_old": n_item_old,
        "n_know_old": n_know_old,
        "n_item_new": n_item_new,
        "n_know_new": n_know_new,
        "train_old": train_old,
        "train_new": train_new,
        "valid_old": valid_old,
        "valid_new": valid_new,
        "log_old": log_old,
        "log_full": log_full,
    }


def train_base(d, device, n_epoch: int) -> GNCDM:
    base = GNCDM(
        n_user=N_USER,
        n_item=d["n_item_old"],
        n_know=d["n_know_old"],
        user_dim=32,
        item_dim=32,
        alpha=ALPHA,
        Q_mat=d["Q_old"],
        monotonicity_assumption=True,
        device=device,
    ).to(device)

    def eval_fn(m):
        R.populate_buffers(m, d["log_old"], device)
        return R.evaluate_buf(m, d["valid_old"], device)

    R.train_real(
        base,
        d["train_old"],
        d["log_old"],
        list(base.parameters()),
        device,
        n_epoch=n_epoch,
        desc="Base(t=0)",
        eval_fn=eval_fn,
    )
    R.populate_buffers(base, d["log_old"], device)
    return base


def run_one_method(base, d, label, params_fn, mask_agg_old, device, n_epoch: int) -> GNCDM:
    m = R.fresh_base(base)
    m.expand_topology(d["n_item_new"], d["n_know_new"], d["Q_expanded"])
    R.populate_buffers(m, d["log_full"], device)
    handles = []
    if mask_agg_old:
        handles.append(m.theta_agg_mat.weight.register_hook(make_col_mask(d["n_know_old"])))
        handles.append(m.psi_agg_mat.weight.register_hook(make_col_mask(d["n_know_old"])))

    def eval_fn(model):
        R.populate_buffers(model, d["log_full"], device)
        return R.evaluate_buf(model, d["valid_new"], device)

    R.train_real(
        m,
        d["train_new"],
        d["log_full"],
        params_fn(m),
        device,
        n_epoch=n_epoch,
        desc=label,
        eval_fn=eval_fn,
    )
    for h in handles:
        h.remove()
    R.populate_buffers(m, d["log_full"], device)
    return m


def run_aligned_umap(y0: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align ŷ^(0) and ŷ^(1). If predictions are identical, reuse one embedding.

    AlignedUMAP can still jitter two views even when inputs are bitwise-equal; that
    spurious 2D drift must not be magnified as if it were real prediction change.
    """
    if np.allclose(y0, y1, rtol=0.0, atol=1e-12):
        from umap import UMAP

        z = UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            random_state=SEED,
            n_epochs=200,
        ).fit_transform(y0)
        z = np.asarray(z)
        return z, z.copy()

    n = y0.shape[0]
    relations = [{i: i for i in range(n)}]
    reducer = AlignedUMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        n_epochs=200,
    )
    reducer.fit([y0, y1], relations=relations)
    z0, z1 = reducer.embeddings_
    return np.asarray(z0), np.asarray(z1)


def draw_panel(ax, xy, score, title):
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=score,
        cmap="RdBu",
        s=4,
        alpha=0.85,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_aspect("equal", adjustable="datalim")
    return sc


def save_fig(fig, stem: Path) -> None:
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    png = SAVE_DIR / f"{stem.name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(EXTRA_DIR / f"{stem.name}.pdf", bbox_inches="tight")
    fig.savefig(EXTRA_DIR / f"{stem.name}.svg", bbox_inches="tight")
    print(f"wrote {png}")
    plt.close(fig)


def plot_dual(z0, z1, score, label: str, stem: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    draw_panel(axes[0], z0, score, f"(a) Base training (t=0)\n{label}")
    sc = draw_panel(axes[1], z1, score, f"(b) After incremental update (t=1)\n{label}")
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Score rate (old items)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        r"AlignedUMAP of old-item predictions $\hat{y}_{\mathrm{old}}$ (Math1, random split)",
        fontsize=9.5,
        y=1.04,
    )
    fig.subplots_adjust(wspace=0.12, right=0.90, left=0.04, top=0.82, bottom=0.06)
    save_fig(fig, stem)


def plot_grid(results: list[dict], score: np.ndarray, stem: Path) -> None:
    setup_style()
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(6.8, 2.2 * n))
    sc = None
    for r, row in enumerate(results):
        draw_panel(axes[r, 0], results[r]["z0"], score, f"{results[r]['label']}  |  t=0")
        sc = draw_panel(axes[r, 1], results[r]["z1"], score, f"{results[r]['label']}  |  t=1")
        drift = results[r]["mean_l2"]
        axes[r, 1].text(
            0.98,
            0.02,
            rf"mean $\|\hat y^{{(1)}}-\hat y^{{(0)}}\|_2$={drift:.3f}",
            transform=axes[r, 1].transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="#333333",
        )
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Score rate (old items)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        r"AlignedUMAP of old-item predictions $\hat{y}_{\mathrm{old}}\in\mathbb{R}^{M_{\mathrm{old}}}$"
        "\nMath1 random split — CLEAN aggregation ablations",
        fontsize=9.5,
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.08, right=0.90, left=0.04, top=0.93, bottom=0.03)
    save_fig(fig, stem)


def _subsample_idx(n: int, k: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def plot_magnified_aligned_grid(
    results: list[dict],
    score: np.ndarray,
    *,
    magnify: float = 5.0,
    stem: Path,
) -> None:
    # 2x2 methods; each cell = (t=0 | t=1×mag)
    order = [
        ("CLEAN-Full", "CLEAN"),
        ("CLEAN (w/o OCM)", "w/o OCM"),
        ("CLEAN (w/o OrthoMask)", "w/o OrthoMask"),
        ("CLEAN (w/o FrozenBias)", "w/o FrozenBias"),
    ]
    by_label = {r["label"]: r for r in results}
    setup_style()
    fig = plt.figure(figsize=(7.2, 6.4))
    outer = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22, left=0.04, right=0.88, top=0.86, bottom=0.04)
    sc = None
    for i, (lab, short) in enumerate(order):
        r, c = divmod(i, 2)
        inner = outer[r, c].subgridspec(1, 2, wspace=0.08)
        ax0 = fig.add_subplot(inner[0, 0])
        ax1 = fig.add_subplot(inner[0, 1])
        rec = by_label[lab]
        z0, z1 = rec["z0"], rec["z1"]
        if rec["mean_l2"] <= 1e-12 or np.allclose(rec["y0"], rec["y1"], atol=1e-12):
            z1 = z0
        z1m = z0 + magnify * (z1 - z0)
        pts = np.vstack([z0, z1m])
        pad = 0.05 * max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
        xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ylim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        sc = draw_panel(ax0, z0, score, f"{short}\nt=0")
        draw_panel(ax1, z1m, score, f"{short}\nt=1 (×{magnify:g})")
        for ax in (ax0, ax1):
            ax.set_aspect("auto")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax1.text(
            0.98,
            0.02,
            rf"mean $\|\Delta\hat y\|_2$={rec['mean_l2']:.3f}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="#333333",
        )
    cax = fig.add_axes([0.90, 0.14, 0.018, 0.68])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Score rate (old items)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        r"AlignedUMAP of old-item predictions $\hat{y}_{\mathrm{old}}$ (Math1, random split)"
        "\n"
        + rf"$z'=z^{{(0)}}+{magnify:g}\,(z^{{(1)}}-z^{{(0)}})$",
        fontsize=9.5,
        y=0.97,
    )
    save_fig(fig, stem)


def plot_trajectory_grid(
    results: list[dict],
    *,
    arrow_scale: float = 3.0,
    n_arrows: int = 500,
    stem: Path,
) -> None:
    """Standard CL visualization: aligned t=0→t=1 trajectories, displacements magnified.

    Points colored by ||Δŷ||_2. Arrows (subsampled) show embedding motion ×arrow_scale.
    """
    setup_style()
    n = len(results)
    # Shared color scale across methods for fair comparison
    vmax = max(float(np.linalg.norm(r["y1"] - r["y0"], axis=1).max()) for r in results)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.2))
    axes = axes.ravel()
    sc = None
    for i, r in enumerate(results):
        ax = axes[i]
        z0, z1 = r["z0"], r["z1"]
        d_y = np.linalg.norm(r["y1"] - r["y0"], axis=1)
        # Midpoints + magnified displacement for arrows
        dz = z1 - z0
        z_end = z0 + arrow_scale * dz

        sc = ax.scatter(
            z0[:, 0],
            z0[:, 1],
            c=d_y,
            cmap="magma",
            s=6,
            alpha=0.75,
            linewidths=0,
            vmin=0.0,
            vmax=vmax,
            zorder=1,
        )
        idx = _subsample_idx(len(z0), n_arrows)
        ax.quiver(
            z0[idx, 0],
            z0[idx, 1],
            (z_end - z0)[idx, 0],
            (z_end - z0)[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0022,
            headwidth=3.5,
            headlength=4.0,
            color="#1a1a1a",
            alpha=0.35,
            zorder=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(
            f"{r['label']}\n"
            + rf"mean $\|\Delta\hat y\|_2$={r['mean_l2']:.3f}",
            fontsize=8,
            pad=4,
        )

    cax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$\|\hat y^{(1)}-\hat y^{(0)}\|_2$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        r"AlignedUMAP trajectories of $\hat y_{\mathrm{old}}$ (Math1, random split)"
        f"\nArrows: $t{{=}}0\\to t{{=}}1$ displacements magnified $\\times${arrow_scale:g} for visibility",
        fontsize=9.5,
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.32, wspace=0.12, right=0.90, left=0.04, top=0.90, bottom=0.04)
    save_fig(fig, stem)


def plot_delta_umap_grid(results: list[dict], stem: Path) -> None:
    """Embed residual Δŷ = ŷ¹−ŷ⁰ (standard residual / change map).

    CLEAN-Full collapses near the origin; ablations spread — maximizes visual contrast.
    """
    from umap import UMAP

    setup_style()
    n = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.2))
    axes = axes.ravel()

    # Fit a joint UMAP on stacked residuals so panels share geometry
    deltas = [r["y1"] - r["y0"] for r in results]
    lengths = [d.shape[0] for d in deltas]
    stacked = np.vstack(deltas)
    # CLEAN-Full is exactly 0; add tiny noise only for UMAP numerical stability, then plot at true zeros
    stacked_fit = stacked.copy()
    zero_rows = np.linalg.norm(stacked_fit, axis=1) < 1e-12
    if zero_rows.any():
        rng = np.random.default_rng(SEED)
        stacked_fit[zero_rows] = rng.normal(0.0, 1e-6, size=stacked_fit[zero_rows].shape)

    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        n_epochs=200,
    )
    xy_all = reducer.fit_transform(stacked_fit)
    # Map exact-zero residuals to the centroid of their tiny-noise cloud → near-origin blob
    if zero_rows.any():
        xy_all[zero_rows] = xy_all[zero_rows].mean(axis=0)

    vmax = max(float(np.linalg.norm(d, axis=1).max()) for d in deltas)
    vmax = max(vmax, 1e-6)
    offset = 0
    sc = None
    for i, r in enumerate(results):
        ax = axes[i]
        sl = slice(offset, offset + lengths[i])
        xy = xy_all[sl]
        d_y = np.linalg.norm(deltas[i], axis=1)
        offset += lengths[i]
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=d_y,
            cmap="magma",
            s=6,
            alpha=0.8,
            linewidths=0,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(
            f"{r['label']}\n"
            + rf"UMAP$(\Delta\hat y)$, mean $\|\Delta\hat y\|_2$={r['mean_l2']:.3f}",
            fontsize=8,
            pad=4,
        )

    cax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$\|\Delta\hat y\|_2$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(
        r"Residual UMAP of prediction change $\Delta\hat y=\hat y^{(1)}-\hat y^{(0)}$"
        "\nMath1 random split — change-only embedding (amplifies ablation contrast)",
        fontsize=9.5,
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.32, wspace=0.12, right=0.90, left=0.04, top=0.90, bottom=0.04)
    save_fig(fig, stem)


def load_or_compute(reuse: bool, n_epoch: int, device):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    score_path = CACHE_DIR / "score_old.npy"
    uids_path = CACHE_DIR / "user_ids.npy"

    results = []
    # Resume: load finished methods; retrain only missing ones (need Base).
    if reuse and score_path.exists() and uids_path.exists():
        score = np.load(score_path)
        uids = np.load(uids_path)
        missing = [
            (lab, tag, pfn, mask)
            for lab, tag, pfn, mask in METHODS
            if not (
                (CACHE_DIR / f"y0_{tag}.npy").exists() and (CACHE_DIR / f"y1_{tag}.npy").exists()
            )
        ]
        if not missing:
            print(f"reuse cached y-hat under {CACHE_DIR}")
            for label, tag, _, _ in METHODS:
                y0 = np.load(CACHE_DIR / f"y0_{tag}.npy")
                y1 = np.load(CACHE_DIR / f"y1_{tag}.npy")
                results.append({"label": label, "tag": tag, "y0": y0, "y1": y1})
            return results, score, uids
        print(f"resume: {len(METHODS) - len(missing)} cached, {len(missing)} to train")
        d = prepare_data()
        base = train_base(d, device, n_epoch)
        y0_shared = None
        for label, tag, _, _ in METHODS:
            p0, p1 = CACHE_DIR / f"y0_{tag}.npy", CACHE_DIR / f"y1_{tag}.npy"
            if p0.exists() and p1.exists():
                y0 = np.load(p0)
                y1 = np.load(p1)
                y0_shared = y0
                results.append({"label": label, "tag": tag, "y0": y0, "y1": y1})
        if y0_shared is None:
            R.populate_buffers(base, d["log_old"], device)
            y0_shared = pred_matrix_old(base, uids, d["n_item_old"], device)
        for label, tag, params_fn, mask in missing:
            print(f"=== {label} ===")
            m = run_one_method(base, d, label, params_fn, mask, device, n_epoch)
            y1 = pred_matrix_old(m, uids, d["n_item_old"], device)
            np.save(CACHE_DIR / f"y0_{tag}.npy", y0_shared)
            np.save(CACHE_DIR / f"y1_{tag}.npy", y1)
            mean_l2 = float(np.linalg.norm(y1 - y0_shared, axis=1).mean())
            print(f"  mean |y1-y0|_2 = {mean_l2:.6f}")
            results.append({"label": label, "tag": tag, "y0": y0_shared, "y1": y1})
        # keep METHODS order
        order = {tag: i for i, (_, tag, _, _) in enumerate(METHODS)}
        results.sort(key=lambda r: order[r["tag"]])
        return results, score, uids

    d = prepare_data()
    base = train_base(d, device, n_epoch)
    score_all, keep = score_from_log(d["log_old"])
    uids = np.where(keep)[0]
    score = score_all[uids]
    y0 = pred_matrix_old(base, uids, d["n_item_old"], device)
    np.save(score_path, score)
    np.save(uids_path, uids)

    for label, tag, params_fn, mask in METHODS:
        print(f"=== {label} ===")
        m = run_one_method(base, d, label, params_fn, mask, device, n_epoch)
        y1 = pred_matrix_old(m, uids, d["n_item_old"], device)
        np.save(CACHE_DIR / f"y0_{tag}.npy", y0)
        np.save(CACHE_DIR / f"y1_{tag}.npy", y1)
        mean_l2 = float(np.linalg.norm(y1 - y0, axis=1).mean())
        print(f"  mean |y1-y0|_2 = {mean_l2:.6f}")
        results.append({"label": label, "tag": tag, "y0": y0, "y1": y1})
    return results, score, uids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument(
        "--magnify",
        type=float,
        default=5.0,
        help="displacement magnification for AlignedUMAP t=1 / arrows (default 5)",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"device={device}")
    results, score, _uids = load_or_compute(args.reuse, args.epochs, device)

    embedded = []
    for r in results:
        z0_p = CACHE_DIR / f"aligned_xy_t0_{r['tag']}.npy"
        z1_p = CACHE_DIR / f"aligned_xy_t1_{r['tag']}.npy"
        if args.reuse and not args.force_embed and z0_p.exists() and z1_p.exists():
            z0, z1 = np.load(z0_p), np.load(z1_p)
            print(f"reuse AlignedUMAP {r['tag']}")
        else:
            print(f"AlignedUMAP {r['label']} {r['y0'].shape} ...")
            z0, z1 = run_aligned_umap(r["y0"], r["y1"])
            np.save(z0_p, z0)
            np.save(z1_p, z1)
        mean_l2 = float(np.linalg.norm(r["y1"] - r["y0"], axis=1).mean())
        emb = float(np.linalg.norm(z1 - z0, axis=1).mean())
        print(f"  y L2 mean={mean_l2:.4f} | 2D emb drift mean={emb:.4f}")
        plot_dual(z0, z1, score, r["label"], SAVE_DIR / f"aligned_umap_math1_{r['tag']}_pred_drift")
        embedded.append({**r, "z0": z0, "z1": z1, "mean_l2": mean_l2})

    plot_grid(embedded, score, SAVE_DIR / "aligned_umap_math1_pred_ablation_grid")
    mag = args.magnify
    plot_magnified_aligned_grid(
        embedded,
        score,
        magnify=mag,
        stem=SAVE_DIR / f"aligned_umap_math1_pred_ablation_mag{mag:g}",
    )
    plot_trajectory_grid(
        embedded,
        arrow_scale=mag,
        n_arrows=500,
        stem=SAVE_DIR / f"aligned_umap_math1_pred_ablation_traj_x{mag:g}",
    )
    plot_delta_umap_grid(embedded, SAVE_DIR / "aligned_umap_math1_pred_ablation_delta")

    rows = [
        {
            "Method": e["label"],
            "mean_L2_y": e["mean_l2"],
            "mean_L2_emb": float(np.linalg.norm(e["z1"] - e["z0"], axis=1).mean()),
        }
        for e in embedded
    ]
    csv_path = SAVE_DIR / "aligned_umap_math1_pred_ablation_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
