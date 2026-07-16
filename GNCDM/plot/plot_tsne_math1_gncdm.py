# -*- coding: utf-8 -*-
"""t-SNE of Math1 learner traits θ: G-NCDM | CLEAN-Full | CLEAN-LoRA.

Reuses θ / score_rate cached by plot_umap_math1_gncdm.py under
incremental_result/umap_cache_math1/. Retrains if --reuse is off and
cache is missing (imports trainers from the UMAP script).

Color: RdBu — blue = higher score rate, red = lower (paper Fig.10).

Run:
  cd GNCDM/plot
  python plot_tsne_math1_gncdm.py --reuse

Out:
  incremental_result/tsne_math1_gncdm_full_lora.{png,pdf,svg}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

PLOT_DIR = Path(__file__).resolve().parent
GNCDM_DIR = PLOT_DIR.parent
EXPERIMENTS_DIR = GNCDM_DIR / "experiments"
for p in (str(GNCDM_DIR), str(EXPERIMENTS_DIR), str(EXPERIMENTS_DIR / "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse UMAP script trainers / cache helpers.
import plot_umap_math1_gncdm as U

SAVE_DIR = U.SAVE_DIR
CACHE_DIR = U.CACHE_DIR
TSNE_SEED = 42

PANELS = [
    ("gncdm", "G-NCDM", "theta_gncdm.npy", "tsne_xy_gncdm.npy", "(a) G-NCDM @ Math1"),
    ("full", "CLEAN-Full", "theta_clean_full.npy", "tsne_xy_clean_full.npy", "(b) CLEAN-Full @ Math1"),
    ("lora", "CLEAN-LoRA", "theta_clean_lora.npy", "tsne_xy_clean_lora.npy", "(c) CLEAN-LoRA @ Math1"),
]

TRAINERS = {
    "gncdm": U.train_full_gncdm,
    "full": U.train_clean_full,
    "lora": U.train_clean_lora,
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


def run_tsne(theta: np.ndarray) -> np.ndarray:
    # perplexity must be < n_samples; 30 is standard for ~4k points.
    reducer = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=TSNE_SEED,
        max_iter=1000,
    )
    return reducer.fit_transform(theta)


def tsne_xy(theta: np.ndarray, xy_path: Path, reuse: bool) -> np.ndarray:
    if reuse and xy_path.exists():
        print(f"reuse t-SNE {xy_path}")
        return np.load(xy_path)
    print(f"t-SNE on θ {theta.shape} ...")
    xy = run_tsne(theta)
    np.save(xy_path, xy)
    return xy


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


def plot_panels(panel_data: list, out_stem: Path) -> None:
    setup_style()
    n = len(panel_data)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.1))
    if n == 1:
        axes = [axes]
    sc = None
    for ax, (xy, score, title) in zip(axes, panel_data):
        sc = draw_panel(ax, xy, score, title)
    cax = fig.add_axes([0.93, 0.18, 0.012, 0.64])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Score rate", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(wspace=0.08, right=0.91, left=0.03, top=0.90, bottom=0.06)
    for ext in ("png", "pdf", "svg"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="reuse cached θ / t-SNE xy")
    parser.add_argument("--epochs", type=int, default=U.N_EPOCH)
    parser.add_argument(
        "--force-tsne",
        action="store_true",
        help="recompute t-SNE even if xy cache exists",
    )
    args = parser.parse_args()
    device = U.pick_device()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    panels = []
    for key, name, theta_f, xy_f, title in PANELS:
        train_fn = TRAINERS[key]
        theta, score = U.load_or_train_panel(
            name, theta_f, xy_f.replace("tsne_", "umap_"), train_fn, args.reuse, args.epochs, device
        )
        # load_or_train_panel may unlink umap xy; t-SNE has its own cache file.
        xy_path = CACHE_DIR / xy_f
        reuse_xy = args.reuse and not args.force_tsne
        xy = tsne_xy(theta, xy_path, reuse_xy)
        panels.append((xy, score, title))

    n = min(len(p[1]) for p in panels)
    panels = [(xy[:n], score[:n], title) for xy, score, title in panels]
    plot_panels(panels, SAVE_DIR / "tsne_math1_gncdm_full_lora")


if __name__ == "__main__":
    main()
