# -*- coding: utf-8 -*-
"""Merge junyi epoch CSVs and plot ACC_new / ACC_old (same style as math1 final).

  cd GNCDM/plot && python plot_epoch_curve_final_junyi.py --epochs 15
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_epoch_curve_final_math1 import (
    ICD_NOTE,
    ORDER,
    STYLE,
    apply_x_axis,
    epochs_for_plot,
)

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(os.path.dirname(PLOT_DIR), "incremental_result")


def paths(n_epoch):
    tag = f"ep{n_epoch}"
    return {
        "gncdm": os.path.join(SAVE_DIR, f"epoch_curve_gncdm_junyi_random_split_{tag}.csv"),
        "aval": os.path.join(SAVE_DIR, f"epoch_curve_avalanche_junyi_random_split_{tag}.csv"),
        "out_csv": os.path.join(SAVE_DIR, f"epoch_curve_junyi_random_split_final_{tag}.csv"),
        "out_png_new": os.path.join(SAVE_DIR, f"epoch_curve_junyi_random_split_final_{tag}.png"),
        "out_png_old": os.path.join(
            SAVE_DIR, f"epoch_curve_junyi_random_split_final_old_{tag}.png"
        ),
    }


def load_merged(n_epoch):
    p = paths(n_epoch)
    df = pd.concat([pd.read_csv(p["gncdm"]), pd.read_csv(p["aval"])], ignore_index=True)
    df = df[df.epoch <= n_epoch].copy()
    df.to_csv(p["out_csv"], index=False)
    print(f"wrote {p['out_csv']}")
    return df, p


def plot_one(df, n_epoch, ycol, ylabel, title, out_png, legend_loc):
    plt.figure(figsize=(7, 5.6))
    for name in ORDER:
        sub = epochs_for_plot(df[df.Model == name].sort_values("epoch"), n_epoch)
        if sub.empty or ycol not in sub.columns:
            print(f"[WARN] skip {name} ({ycol})")
            continue
        plt.plot(sub.epoch, sub[ycol], label=name, linewidth=1.8, markersize=6, **STYLE[name])
    plt.xlabel("Training epoch (Task2 / new-item incremental stage)")
    plt.ylabel(ylabel)
    plt.title(title)
    apply_x_axis(n_epoch)
    plt.legend(fontsize=8, ncol=2, loc=legend_loc)
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0.09, 1, 1))
    plt.gcf().text(0.02, 0.01, ICD_NOTE.replace("math1", "junyi"), fontsize=6.5, color="#555555", va="bottom")
    plt.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    df, p = load_merged(args.epochs)
    plot_one(
        df,
        args.epochs,
        "ACC_new",
        "ACC_new (new items)",
        f"junyi random_split: efficiency vs. effectiveness (ep1–{args.epochs})",
        p["out_png_new"],
        "lower right",
    )
    plot_one(
        df,
        args.epochs,
        "ACC_old",
        "ACC_old (old items)",
        f"junyi random_split: old-task retention vs. training cost (ep1–{args.epochs})",
        p["out_png_old"],
        "lower left",
    )


if __name__ == "__main__":
    main()
