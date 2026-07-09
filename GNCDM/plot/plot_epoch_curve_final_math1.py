# -*- coding: utf-8 -*-
"""图 A 终版：math1 random_split，6 个模型的 ACC_new-epoch 收敛曲线（效率 vs 效果）。

汇总 2 个产物脚本各自跑出的 CSV（按 epoch 后缀区分，如 _ep15 / _ep25）：
  plot_epoch_curve_gncdm_math1.py       -> epoch_curve_gncdm_math1_random_split_ep{N}.csv
  plot_epoch_curve_avalanche_math1.py   -> epoch_curve_avalanche_math1_random_split_ep{N}.csv

运行：cd GNCDM/plot && python plot_epoch_curve_final_math1.py [--epochs 25]
产物：incremental_result/epoch_curve_math1_random_split_final_ep{N}.{csv,png}
"""

import argparse
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(PLOT_DIR)
SAVE_DIR = os.path.join(GNCDM_DIR, "incremental_result")

STYLE = {
    "CLEAN-Full": dict(marker="o", linestyle="-", color="#1f77b4"),
    "Full-Replay": dict(marker="^", linestyle="-", color="#2ca02c"),
    "X-DER": dict(marker="v", linestyle="-", color="#9467bd"),
    "C-LoRA-GNCDM": dict(marker="D", linestyle="-", color="#8c564b"),
    "EWC": dict(marker="P", linestyle="-", color="#e377c2"),
    "DER++": dict(marker="X", linestyle="-", color="#7f7f7f"),
}
ORDER = ["CLEAN-Full", "Full-Replay", "EWC", "DER++", "C-LoRA-GNCDM", "X-DER"]

ICD_NOTE = (
    "ICD not shown: single-pass streaming method, no epoch-wise retraining\n"
    "(turning-point gate never fired on new items) -> not comparable on this x-axis.\n"
    "See all_methods_math1_random_split.csv for its final-metric comparison."
)


def curve_paths(n_epoch):
    tag = f"ep{n_epoch}"
    return {
        "gncdm": os.path.join(SAVE_DIR, f"epoch_curve_gncdm_math1_random_split_{tag}.csv"),
        "aval": os.path.join(SAVE_DIR, f"epoch_curve_avalanche_math1_random_split_{tag}.csv"),
        "out_csv": os.path.join(SAVE_DIR, f"epoch_curve_math1_random_split_final_{tag}.csv"),
        "out_png": os.path.join(SAVE_DIR, f"epoch_curve_math1_random_split_final_{tag}.png"),
    }


def epochs_for_plot(sub, n_epoch):
    """25 ep: plot odd epochs only (1,3,5,...) — skip every other node."""
    if n_epoch >= 25:
        return sub[sub.epoch % 2 == 1]
    return sub


def apply_x_axis(n_epoch):
    plt.xlim(1, n_epoch)
    if n_epoch >= 25:
        plt.xticks(range(1, n_epoch + 1, 2))


def plot_acc_new(n_epoch):
    paths = curve_paths(n_epoch)
    df_gncdm = pd.read_csv(paths["gncdm"])
    df_aval = pd.read_csv(paths["aval"])
    df = pd.concat([df_gncdm, df_aval], ignore_index=True)
    df = df[df.epoch <= n_epoch].copy()
    df.to_csv(paths["out_csv"], index=False)
    print(f"合并写入 {paths['out_csv']}")

    plt.figure(figsize=(7, 5.6))
    for name in ORDER:
        sub = epochs_for_plot(df[df.Model == name].sort_values("epoch"), n_epoch)
        if sub.empty:
            print(f"[WARN] 缺少 {name} 的数据，跳过")
            continue
        st = STYLE[name]
        plt.plot(
            sub.epoch,
            sub.ACC_new,
            label=name,
            linewidth=1.8,
            markersize=6,
            **st,
        )
    plt.xlabel("Training epoch (Task2 / new-item incremental stage)")
    plt.ylabel("ACC_new (new items)")
    plt.title(f"math1 random_split: efficiency vs. effectiveness (ep1–{n_epoch})")
    apply_x_axis(n_epoch)
    plt.legend(fontsize=8, ncol=2, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0.09, 1, 1))
    plt.gcf().text(0.02, 0.01, ICD_NOTE, fontsize=6.5, color="#555555", va="bottom")
    plt.savefig(paths["out_png"], dpi=200)
    print(f"写入 {paths['out_png']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="横轴上限 epoch（默认 25）")
    args = parser.parse_args()
    plot_acc_new(args.epochs)


if __name__ == "__main__":
    main()
