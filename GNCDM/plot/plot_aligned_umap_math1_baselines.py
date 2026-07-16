# -*- coding: utf-8 -*-
"""AlignedUMAP of θ_old (t=0 vs t=1) for Math1 G-NCDM-backbone methods.

Runnable (true concept-θ space):
  --method freplay | xder | clora | full | lora | all_gncdm

NOT runnable as θ_old (different representation space):
  EWC / DER++ / table-C-LoRA  → CognitiveBackbone student_emb only
  ICD                         → EduCDM NCD trait (3rd space + separate venv)

Scheme A: two panels, shared score-rate colorbar, AlignedUMAP on first K old dims.

Run:
  cd GNCDM/plot
  python plot_aligned_umap_math1_baselines.py --method all_gncdm
  python plot_aligned_umap_math1_baselines.py --method freplay --reuse
"""

from __future__ import annotations

import argparse
import copy
import os
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

# gncdm_clora_baseline reads sys.argv[1] as dataset name — isolate before import.
_argv_bak = sys.argv[:]
sys.argv = [sys.argv[0], "math1"]
import gncdm_clora_baseline as CL  # noqa: E402

sys.argv = _argv_bak

import run_incremental_math1 as R  # noqa: E402
from core.model import GNCDM  # noqa: E402
from run_xder import build_buffer, train_xder  # noqa: E402

SAVE_DIR = Path(R.SAVE_DIR)
CACHE_DIR = SAVE_DIR / "aligned_umap_cache_math1"
DATA_DIR = Path(R.DATA_DIR)

N_USER, N_ITEM, N_KNOW = 4209, 20, 11
ALPHA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_EPOCH = 15
SEED = 42
CLORA_LAMBDA = 0.1  # matches plot_epoch_curve_gncdm_math1.py

# method_key -> (cache_tag, figure_label)
METHOD_META = {
    "full": ("clean_full", "CLEAN-Full"),
    "lora": ("clean_lora", "CLEAN-LoRA"),
    "freplay": ("full_replay", "Full Replay Oracle"),
    "xder": ("xder", "X-DER"),
    "clora": ("clora_gncdm", "C-LoRA-GNCDM"),
}

CANNOT_DO = {
    "EWC": "CognitiveBackbone student_emb only — no G-NCDM concept θ_old",
    "DER++": "CognitiveBackbone student_emb only — no G-NCDM concept θ_old",
    "C-LoRA (table / CognitiveBackbone)": "embedding-space C-LoRA; use C-LoRA-GNCDM instead",
    "ICD": "EduCDM NCD trait space + isolated venv — not diagnose_theta",
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
def extract_theta(model, log_mat, device, batch=256) -> np.ndarray:
    model.eval()
    log_t = torch.tensor(log_mat, dtype=torch.float32, device=device)
    chunks = []
    for i in range(0, log_t.shape[0], batch):
        chunks.append(model.diagnose_theta(log_t[i : i + batch]).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def score_keep_from_log(log_mat: np.ndarray):
    n_obs = (log_mat != 0).sum(axis=1)
    n_correct = (log_mat > 0).sum(axis=1)
    keep = n_obs > 0
    score = np.full(log_mat.shape[0], np.nan, dtype=np.float64)
    score[keep] = n_correct[keep] / n_obs[keep]
    return score, keep


def _shared_pack(theta0_all, theta1_all, k_old, log_for_score):
    score_all, keep = score_keep_from_log(log_for_score)
    uids = np.where(keep)[0]
    theta0 = theta0_all[uids][:, :k_old]
    theta1 = theta1_all[uids][:, :k_old]
    score = score_all[uids]
    drift = np.linalg.norm(theta1 - theta0, axis=1)
    print(f"  N_shared={len(uids)} K_old={k_old} |θ1-θ0|_2 mean={drift.mean():.6f} max={drift.max():.6f}")
    return theta0, theta1, score, uids


def train_dna_family(method_key: str, device, n_epoch: int):
    """full / lora / freplay via buf_strategy_specs."""
    name_map = {
        "full": "Ours (Dynamic DNA)",
        "lora": "Ours (LoRA)",
        "freplay": "Full Replay Oracle",
    }
    strat_name = name_map[method_key]
    R.set_seed(SEED)
    Q = np.load(DATA_DIR / "math1_Q_matrix.npy")
    df_train = pd.read_csv(DATA_DIR / "math1_train_0.8_0.2.csv")
    df_valid = pd.read_csv(DATA_DIR / "math1_valid_0.8_0.2.csv")
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = N_ITEM - n_item_old, N_KNOW - n_know_old
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
        alpha=ALPHA,
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
        desc=f"Base({method_key})",
        eval_fn=lambda m: (
            R.populate_buffers(m, log_old, device),
            R.evaluate_buf(m, valid_old, device),
        )[1],
    )
    theta0_all = extract_theta(base, log_old, device)

    specs = R.buf_strategy_specs(
        n_item_new, n_know_new, n_item_old, Q_mat.copy(), train_old, train_new, valid_old, valid_new
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
        mask_agg_old=spec.get("mask_agg_old", False),
    )
    theta1_all = extract_theta(model, log_full, device)
    return _shared_pack(theta0_all, theta1_all, n_know_old, log_old)


def train_xder_pair(device, n_epoch: int):
    R.set_seed(SEED)
    Q = np.load(DATA_DIR / "math1_Q_matrix.npy")
    df_train = pd.read_csv(DATA_DIR / "math1_train_0.8_0.2.csv")
    df_valid = pd.read_csv(DATA_DIR / "math1_valid_0.8_0.2.csv")
    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = N_ITEM - n_item_old, N_KNOW - n_know_old
    train_old = df_train[df_train.item_id < n_item_old].copy()
    train_new = df_train[df_train.item_id >= n_item_old].copy()
    valid_old = df_valid[df_valid.item_id < n_item_old].copy()
    valid_new = df_valid[df_valid.item_id >= n_item_old].copy()
    valid_comb = pd.concat([valid_old, valid_new], ignore_index=True)
    log_old = R.build_log_mat(train_old, N_USER, n_item_old)
    log_full = R.build_log_mat(df_train, N_USER, N_ITEM)
    Q_expanded = Q_mat.copy()

    base = GNCDM(
        n_user=N_USER,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=ALPHA,
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
        desc="Base(X-DER)",
        eval_fn=lambda m: (
            R.populate_buffers(m, log_old, device),
            R.evaluate_buf(m, valid_old, device),
        )[1],
    )
    theta0_all = extract_theta(base, log_old, device)
    buffer = build_buffer(base, train_old, log_old, device, buffer_size=5000)

    model = R.fresh_base(base)
    model.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded)

    def valid_eval_fn(m):
        R.populate_buffers(m, log_full, device)
        return R.evaluate_buf(m, valid_comb, device)

    train_xder(
        model,
        train_new,
        log_full,
        buffer,
        device,
        valid_eval_fn,
        n_know_old,
        n_epoch=n_epoch,
    )
    theta1_all = extract_theta(model, log_full, device)
    return _shared_pack(theta0_all, theta1_all, n_know_old, log_old)


def train_clora_gncdm(device, n_epoch: int):
    cfg = CL.CONFIGS["math1"]
    meta = CL.load_partition(cfg)
    k_old = meta["n_know_old"]
    CL.set_seed(SEED)
    base = CL._new_model(cfg, meta, device)
    CL.train_real(
        base,
        meta["train_old"],
        meta["log_old_only"],
        list(base.parameters()),
        device,
        n_epoch=n_epoch,
        desc="Base(CLoRA)",
    )
    theta0_all = extract_theta(base, meta["log_old_only"], device)
    CL.populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    base_state = copy.deepcopy(base.state_dict())

    # run_one_lambda trains but discards model; re-run inject+train to keep model
    CL.set_seed(SEED)
    model = CL._new_model(cfg, meta, device)
    model.load_state_dict(base_state)
    model._freeze_parameters()
    CL.inject_lora_gncdm(model, rank=CL.LORA_RANK, alpha=CL.LORA_ALPHA)
    model.theta_agg_mat.weight.requires_grad = True
    model.psi_agg_mat.weight.requires_grad = True
    handles = [
        model.theta_agg_mat.weight.register_hook(CL.make_col_mask(k_old)),
        model.psi_agg_mat.weight.register_hook(CL.make_col_mask(k_old)),
    ]
    model.to(device)
    params = CL.lora_parameters(model) + [model.theta_agg_mat.weight, model.psi_agg_mat.weight]
    CL.train_clora_phase2(
        model,
        meta["train_new"],
        meta["log_full"],
        params,
        device,
        lambda_ortho=CLORA_LAMBDA,
        n_epoch=n_epoch,
        desc=f"λ={CLORA_LAMBDA}",
    )
    for h in handles:
        h.remove()
    theta1_all = extract_theta(model, meta["log_full"], device)
    # score from old-item observations in padded log
    return _shared_pack(theta0_all, theta1_all, k_old, meta["log_old_only"])


TRAINERS = {
    "full": train_dna_family,
    "lora": train_dna_family,
    "freplay": train_dna_family,
    "xder": lambda key, device, n_epoch: train_xder_pair(device, n_epoch),
    "clora": lambda key, device, n_epoch: train_clora_gncdm(device, n_epoch),
}


def load_or_train(method_key: str, reuse: bool, n_epoch: int, device):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = METHOD_META[method_key][0]
    paths = {
        "t0": CACHE_DIR / f"theta_old_t0_{tag}.npy",
        "t1": CACHE_DIR / f"theta_old_t1_{tag}.npy",
        "score": CACHE_DIR / f"score_old_{tag}.npy",
        "uids": CACHE_DIR / f"user_ids_{tag}.npy",
    }
    if reuse and all(p.exists() for p in paths.values()):
        print(f"[{METHOD_META[method_key][1]}] reuse cache {tag}")
        return np.load(paths["t0"]), np.load(paths["t1"]), np.load(paths["score"])

    print(f"\n===== train {METHOD_META[method_key][1]} =====")
    trainer = TRAINERS[method_key]
    if method_key in ("full", "lora", "freplay"):
        theta0, theta1, score, _uids = trainer(method_key, device, n_epoch)
    else:
        theta0, theta1, score, _uids = trainer(method_key, device, n_epoch)
    np.save(paths["t0"], theta0)
    np.save(paths["t1"], theta1)
    np.save(paths["score"], score)
    np.save(paths["uids"], _uids)
    for p in CACHE_DIR.glob(f"aligned_xy_*_{tag}.npy"):
        p.unlink()
    return theta0, theta1, score


def run_aligned_umap(theta0, theta1):
    n = theta0.shape[0]
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


def plot_dual(z0, z1, score, label: str, out_stem: Path):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    draw_panel(axes[0], z0, score, f"(a) Base training (t=0)\n{label}")
    sc = draw_panel(axes[1], z1, score, f"(b) After incremental update (t=1)\n{label}")
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


def run_one(method_key: str, reuse: bool, force_embed: bool, n_epoch: int, device):
    tag, label = METHOD_META[method_key]
    theta0, theta1, score = load_or_train(method_key, reuse, n_epoch, device)
    z0_path = CACHE_DIR / f"aligned_xy_t0_{tag}.npy"
    z1_path = CACHE_DIR / f"aligned_xy_t1_{tag}.npy"
    if reuse and not force_embed and z0_path.exists() and z1_path.exists():
        z0, z1 = np.load(z0_path), np.load(z1_path)
        print(f"reuse AlignedUMAP {tag}")
    else:
        print(f"AlignedUMAP {label} {theta0.shape} ...")
        z0, z1 = run_aligned_umap(theta0, theta1)
        np.save(z0_path, z0)
        np.save(z1_path, z1)
    emb = np.linalg.norm(z1 - z0, axis=1)
    print(f"  2D |z1-z0|_2 mean={emb.mean():.6f} max={emb.max():.6f}")
    plot_dual(z0, z1, score, label, SAVE_DIR / f"aligned_umap_math1_{tag}_old_drift")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=list(METHOD_META) + ["all_gncdm"],
        default="all_gncdm",
        help="all_gncdm = freplay+xder+clora (plus optional full/lora)",
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument(
        "--also-ours",
        action="store_true",
        help="when all_gncdm, also run CLEAN-Full / CLEAN-LoRA",
    )
    args = parser.parse_args()
    device = pick_device()
    print(f"device={device} torch={torch.__version__}")
    print("Cannot do as θ_old AlignedUMAP:")
    for k, v in CANNOT_DO.items():
        print(f"  - {k}: {v}")

    if args.method == "all_gncdm":
        keys = ["freplay", "xder", "clora"]
        if args.also_ours:
            keys = ["full", "lora"] + keys
    else:
        keys = [args.method]

    for key in keys:
        run_one(key, args.reuse, args.force_embed, args.epochs, device)


if __name__ == "__main__":
    main()
