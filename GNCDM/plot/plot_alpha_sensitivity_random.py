# -*- coding: utf-8 -*-
"""Sensitivity analysis of the mixing coefficient α on random splits.

Reads Math1 / junyi / a0910 sweep CSVs under incremental_result/, plots
sel_DNA_validACC = DNA mean(valid ACC_old, ACC_new) vs alpha, marks the
chosen operating point, and overlays DNA test ACC_new.

Location: GNCDM/plot/plot_alpha_sensitivity_random.py
Run:      cd GNCDM/plot && python plot_alpha_sensitivity_random.py
Out:      incremental_result/alpha_sensitivity_random_split.{png,pdf,svg}
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
GNCDM_DIR = PLOT_DIR.parent
SAVE_DIR = GNCDM_DIR / "incremental_result"

# (panel title, candidate CSV names, chosen alpha used in main experiments)
DATASETS = [
    ("Math1", ["alpha_sweep_math1_random_split.csv"], 0.20),
    ("junyi", ["alpha_sweep_junyi_random_split.csv"], 0.10),
    (
        "ASSIST a0910",
        [
            "alpha_sweep_a0910_random_split.csv",
            "alpha_sweep_a0910_random_split .csv",  # legacy typo (trailing space)
        ],
        0.10,
    ),
]

COLOR_SEL = "#0F4D92"
COLOR_TEST = "#42949E"
COLOR_MARK = "#B64342"


def load_sweep(candidates: list[str]) -> pd.DataFrame:
    for name in candidates:
        path = SAVE_DIR / name
        if path.exists():
            df = pd.read_csv(path).sort_values("alpha").reset_index(drop=True)
            if "sel_DNA_validACC" not in df.columns:
                raise KeyError(f"{path.name} missing sel_DNA_validACC")
            return df
    raise FileNotFoundError(f"none of {candidates} found under {SAVE_DIR}")


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


def plot_panel(ax, title: str, df: pd.DataFrame, chosen_alpha: float, show_legend: bool) -> None:
    x = df["alpha"].to_numpy()
    y_sel = df["sel_DNA_validACC"].to_numpy()
    ax.plot(
        x,
        y_sel,
        color=COLOR_SEL,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label="sel (valid mean ACC)",
        zorder=3,
    )

    if "DNA_te_ACCnew" in df.columns:
        ax.plot(
            x,
            df["DNA_te_ACCnew"].to_numpy(),
            color=COLOR_TEST,
            marker="s",
            markersize=3.5,
            linewidth=1.2,
            linestyle="--",
            label="CLEAN-Full test ACC_new",
            zorder=2,
        )

    # Mark chosen alpha used by the main experiment entrypoint.
    row = df.loc[(df["alpha"] - chosen_alpha).abs().idxmin()]
    ax.axvline(chosen_alpha, color=COLOR_MARK, linestyle=":", linewidth=1.0, alpha=0.85, zorder=1)
    ax.scatter(
        [row["alpha"]],
        [row["sel_DNA_validACC"]],
        s=48,
        color=COLOR_MARK,
        marker="*",
        zorder=4,
        label=f"chosen α={chosen_alpha:.2f}",
    )

    best_row = df.loc[df["sel_DNA_validACC"].idxmax()]
    ax.set_title(f"{title}  (peak α={best_row['alpha']:.2f})", fontsize=9, pad=4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_xlim(min(x) - 0.02, max(x) + 0.02)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    if show_legend:
        ax.legend(loc="lower left", fontsize=6.5, handlelength=2.2)


def main() -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=False)

    for i, (title, candidates, chosen) in enumerate(DATASETS):
        df = load_sweep(candidates)
        plot_panel(axes[i], title, df, chosen, show_legend=(i == 0))
        if i == 0:
            axes[i].set_ylabel("Accuracy")

    fig.suptitle(
        r"Sensitivity analysis of the mixing coefficient $\alpha$ on random splits",
        fontsize=9.5,
        y=1.02,
    )
    fig.tight_layout(w_pad=1.2)

    stem = SAVE_DIR / "alpha_sensitivity_random_split"
    for ext in ("png", "pdf", "svg"):
        out = f"{stem}.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
