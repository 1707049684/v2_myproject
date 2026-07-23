# -*- coding: utf-8 -*-
"""Merge junyi epoch CSVs; plot ACC_new / ACC_old as one 1x2 figure.

  cd GNCDM/plot && python plot_epoch_curve_final_junyi.py --epochs 25
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_epoch_curve_final_math1 import ICD_NOTE, ORDER, STYLE, epochs_for_plot

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "incremental_result")


def paths(n_epoch):
    tag = f"ep{n_epoch}"
    return {
        "gncdm": os.path.join(SAVE_DIR, f"epoch_curve_gncdm_junyi_random_split_{tag}.csv"),
        "aval": os.path.join(SAVE_DIR, f"epoch_curve_avalanche_junyi_random_split_{tag}.csv"),
        "out_csv": os.path.join(SAVE_DIR, f"epoch_curve_junyi_random_split_final_{tag}.csv"),
        "out_png": os.path.join(SAVE_DIR, f"epoch_curve_junyi_random_split_final_{tag}.png"),
    }


def draw_panel(ax, df, n_epoch, ycol, ylabel, title, legend_loc):
    for name in ORDER:
        sub = epochs_for_plot(df[df.Model == name].sort_values("epoch"), n_epoch)
        if sub.empty or ycol not in sub.columns:
            print(f"[WARN] skip {name} ({ycol})")
            continue
        ax.plot(sub.epoch, sub[ycol], label=name, linewidth=1.8, markersize=5, **STYLE[name])
    ax.set_xlabel("Training epoch (Task2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, n_epoch)
    if n_epoch >= 25:
        ax.set_xticks(range(1, n_epoch + 1, 2))
    ax.legend(fontsize=7, ncol=2, loc=legend_loc)
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    n = args.epochs
    p = paths(n)
    df = pd.concat([pd.read_csv(p["gncdm"]), pd.read_csv(p["aval"])], ignore_index=True)
    df = df[df.epoch <= n].copy()
    df.to_csv(p["out_csv"], index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    draw_panel(
        axes[0],
        df,
        n,
        "ACC_new",
        "ACC_new",
        f"(a) junyi ACC_new (ep1–{n})",
        "lower right",
    )
    draw_panel(
        axes[1],
        df,
        n,
        "ACC_old",
        "ACC_old",
        f"(b) junyi ACC_old (ep1–{n})",
        "lower left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(
        0.02,
        0.01,
        ICD_NOTE.replace("math1", "junyi"),
        fontsize=6.5,
        color="#555555",
        va="bottom",
    )
    fig.savefig(p["out_png"], dpi=200)
    print(f"wrote {p['out_csv']}")
    print(f"wrote {p['out_png']}")


if __name__ == "__main__":
    main()
