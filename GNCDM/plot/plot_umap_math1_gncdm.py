# -*- coding: utf-8 -*-
"""UMAP of Math1 learner traits θ (score-rate colored, Fig.10 style).

Compares:
  --compare all3        G-NCDM | CLEAN-Full | CLEAN-LoRA  (default)
  --compare full_lora   CLEAN-Full vs CLEAN-LoRA
  --compare gncdm_full  G-NCDM vs CLEAN-Full

Run:
  cd GNCDM/plot
  python plot_umap_math1_gncdm.py --reuse

Out:
  incremental_result/umap_math1_gncdm_full_lora.{png,pdf,svg}
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
import umap

PLOT_DIR = Path(__file__).resolve().parent
GNCDM_DIR = PLOT_DIR.parent
EXPERIMENTS_DIR = GNCDM_DIR / "experiments"
for p in (str(GNCDM_DIR), str(EXPERIMENTS_DIR), str(EXPERIMENTS_DIR / "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_incremental_math1 as R
from core.model import GNCDM

SAVE_DIR = Path(R.SAVE_DIR)
CACHE_DIR = SAVE_DIR / "umap_cache_math1"
DATA_DIR = Path(R.DATA_DIR)

N_USER, N_ITEM, N_KNOW = 4209, 20, 11
ALPHA_FULL = 0.8
ALPHA_DNA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_EPOCH = 15
UMAP_SEED = 42


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
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def score_from_log(log_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_obs = (log_mat != 0).sum(axis=1)
    n_correct = (log_mat > 0).sum(axis=1)
    keep = n_obs > 0
    score = np.full(log_mat.shape[0], np.nan, dtype=np.float64)
    score[keep] = n_correct[keep] / n_obs[keep]
    return score, keep


@torch.no_grad()
def extract_theta(model: GNCDM, log_mat: np.ndarray, device: torch.device, batch=256) -> np.ndarray:
    model.eval()
    log_t = torch.tensor(log_mat, dtype=torch.float32, device=device)
    chunks = []
    for i in range(0, log_t.shape[0], batch):
        chunks.append(model.diagnose_theta(log_t[i : i + batch]).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_umap(theta: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=UMAP_SEED,
    )
    return reducer.fit_transform(theta)


def train_full_gncdm(device: torch.device, n_epoch: int) -> tuple[GNCDM, np.ndarray]:
    R.set_seed(42)
    Q = np.load(DATA_DIR / "math1_Q_matrix.npy")
    df_train = pd.read_csv(DATA_DIR / "math1_train_0.8_0.2.csv")
    df_valid = pd.read_csv(DATA_DIR / "math1_valid_0.8_0.2.csv")
    log_train = R.build_log_mat(df_train, N_USER, N_ITEM)

    model = GNCDM(
        n_user=N_USER,
        n_item=N_ITEM,
        n_know=N_KNOW,
        user_dim=32,
        item_dim=32,
        alpha=ALPHA_FULL,
        Q_mat=Q.copy(),
        monotonicity_assumption=True,
        device=device,
    ).to(device)

    R.train_real(
        model,
        df_train,
        log_train,
        list(model.parameters()),
        device,
        n_epoch=n_epoch,
        desc="G-NCDM-full",
        eval_fn=lambda m: (
            R.populate_buffers(m, log_train, device),
            R.evaluate_buf(m, df_valid, device),
        )[1],
    )
    return model, log_train


def _prepare_incremental(device: torch.device, n_epoch: int):
    """Train Base(old) once; return base, specs, logs, and eval helpers."""
    R.set_seed(42)
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
        desc="Base",
        eval_fn=lambda m: (
            R.populate_buffers(m, log_old, device),
            R.evaluate_buf(m, valid_old, device),
        )[1],
    )

    specs = R.buf_strategy_specs(
        n_item_new, n_know_new, n_item_old, Q_expanded, train_old, train_new, valid_old, valid_new
    )

    def strat_eval_fn(valid_df):
        return lambda m: (
            R.populate_buffers(m, log_full, device),
            R.evaluate_buf(m, valid_df, device),
        )[1]

    rs_kw = dict(
        log_full=log_full,
        n_know_old=n_know_old,
        device=device,
        strat_eval_fn=strat_eval_fn,
        final_old=lambda m: R.evaluate_buf(m, valid_old, device),
        final_new=lambda m: R.evaluate_buf(m, valid_new, device),
        n_epoch=n_epoch,
    )
    return base, specs, log_full, rs_kw


def _run_named_strategy(strat_name: str, device: torch.device, n_epoch: int) -> tuple[GNCDM, np.ndarray]:
    base, specs, log_full, rs_kw = _prepare_incremental(device, n_epoch)
    spec = specs[strat_name]
    model = R.run_strategy(
        base,
        strat_name,
        spec["expand_fn"],
        spec["params_fn"],
        spec["train_df"],
        spec["valid_df"],
        mask_agg_old=spec.get("mask_agg_old", False),
        **rs_kw,
    )
    return model, log_full


def train_clean_full(device: torch.device, n_epoch: int) -> tuple[GNCDM, np.ndarray]:
    return _run_named_strategy("Ours (Dynamic DNA)", device, n_epoch)


def train_clean_lora(device: torch.device, n_epoch: int) -> tuple[GNCDM, np.ndarray]:
    return _run_named_strategy("Ours (LoRA)", device, n_epoch)


def load_or_train_panel(
    name: str,
    theta_name: str,
    xy_name: str,
    train_fn,
    reuse: bool,
    n_epoch: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    theta_path = CACHE_DIR / theta_name
    xy_path = CACHE_DIR / xy_name
    score_path = CACHE_DIR / "score_rate.npy"

    if reuse and theta_path.exists() and score_path.exists():
        print(f"[{name}] reuse θ {theta_path}")
        return np.load(theta_path), np.load(score_path)

    print(f"[{name}] train (epoch={n_epoch}, device={device}) ...")
    model, log_mat = train_fn(device, n_epoch)
    score, keep = score_from_log(log_mat)
    theta = extract_theta(model, log_mat, device)[keep]
    score_kept = score[keep]
    np.save(theta_path, theta)
    # Shared score rate from first panel that writes it; overwrite is fine (same train users).
    np.save(score_path, score_kept)
    torch.save(model.state_dict(), CACHE_DIR / f"{name.replace(' ', '_').lower()}.pt")
    if xy_path.exists():
        xy_path.unlink()
    print(f"[{name}] cached θ {theta.shape}")
    return theta, score_kept


def umap_xy(theta: np.ndarray, xy_path: Path, reuse: bool) -> np.ndarray:
    if reuse and xy_path.exists():
        print(f"reuse UMAP {xy_path}")
        return np.load(xy_path)
    print(f"UMAP on θ {theta.shape} ...")
    xy = run_umap(theta)
    np.save(xy_path, xy)
    return xy


def draw_panel(ax, xy, score, title):
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=score,
        # Paper Fig.10: blue = higher score rate, red = lower (matplotlib RdBu).
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


def plot_panels(panel_data: list[tuple], out_stem: Path) -> None:
    """panel_data: list of (xy, score, title)."""
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


PANELS = {
    "gncdm": ("G-NCDM", "theta_gncdm.npy", "umap_xy_gncdm.npy", train_full_gncdm),
    "full": ("CLEAN-Full", "theta_clean_full.npy", "umap_xy_clean_full.npy", train_clean_full),
    "lora": ("CLEAN-LoRA", "theta_clean_lora.npy", "umap_xy_clean_lora.npy", train_clean_lora),
}

COMPARE = {
    "all3": (
        [
            ("gncdm", "(a) G-NCDM @ Math1"),
            ("full", "(b) CLEAN-Full @ Math1"),
            ("lora", "(c) CLEAN-LoRA @ Math1"),
        ],
        "umap_math1_gncdm_full_lora",
    ),
    "full_lora": (
        [
            ("full", "(a) CLEAN-Full @ Math1"),
            ("lora", "(b) CLEAN-LoRA @ Math1"),
        ],
        "umap_math1_clean_full_vs_lora",
    ),
    "gncdm_full": (
        [
            ("gncdm", "(a) G-NCDM @ Math1"),
            ("full", "(b) CLEAN-Full @ Math1"),
        ],
        "umap_math1_gncdm_vs_clean_full",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="reuse cached theta/UMAP")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument(
        "--compare",
        choices=list(COMPARE),
        default="all3",
        help="panels to plot (default: G-NCDM + CLEAN-Full + CLEAN-LoRA)",
    )
    args = parser.parse_args()
    device = pick_device()

    specs, out_name = COMPARE[args.compare]
    panels = []
    for key, title in specs:
        name, theta_f, xy_f, train_fn = PANELS[key]
        theta, score = load_or_train_panel(
            name, theta_f, xy_f, train_fn, args.reuse, args.epochs, device
        )
        xy = umap_xy(theta, CACHE_DIR / xy_f, args.reuse)
        panels.append((xy, score, title))

    n = min(len(p[1]) for p in panels)
    panels = [(xy[:n], score[:n], title) for xy, score, title in panels]
    plot_panels(panels, SAVE_DIR / out_name)


if __name__ == "__main__":
    main()
