# -*- coding: utf-8 -*-
"""图 B：math1 random_split，6 个模型的 ACC_old-epoch 收敛曲线（旧任务保持 vs 训练成本）。

运行：cd GNCDM/plot && python plot_epoch_curve_final_math1_old.py [--epochs 25]
产物：incremental_result/epoch_curve_math1_random_split_final_old_ep{N}.{csv,png}
"""

import argparse
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_epoch_curve_final_math1 import (
    ICD_NOTE,
    ORDER,
    SAVE_DIR,
    STYLE,
    apply_x_axis,
    curve_paths,
    epochs_for_plot,
)


def plot_acc_old(n_epoch):
    paths = curve_paths(n_epoch)
    out_csv = os.path.join(SAVE_DIR, f"epoch_curve_math1_random_split_final_old_ep{n_epoch}.csv")
    out_png = os.path.join(SAVE_DIR, f"epoch_curve_math1_random_split_final_old_ep{n_epoch}.png")

    df_gncdm = pd.read_csv(paths["gncdm"])
    df_aval = pd.read_csv(paths["aval"])
    df = pd.concat([df_gncdm, df_aval], ignore_index=True)
    df = df[df.epoch <= n_epoch].copy()
    df.to_csv(out_csv, index=False)
    print(f"合并写入 {out_csv}")

    plt.figure(figsize=(7, 5.6))
    for name in ORDER:
        sub = epochs_for_plot(df[df.Model == name].sort_values("epoch"), n_epoch)
        if sub.empty:
            print(f"[WARN] 缺少 {name} 的数据，跳过")
            continue
        if "ACC_old" not in sub.columns:
            print(f"[WARN] {name} 缺少 ACC_old 列，请先重跑数据脚本")
            continue
        st = STYLE[name]
        plt.plot(
            sub.epoch,
            sub.ACC_old,
            label=name,
            linewidth=1.8,
            markersize=6,
            **st,
        )
    plt.xlabel("Training epoch (Task2 / new-item incremental stage)")
    plt.ylabel("ACC_old (old items)")
    plt.title(f"math1 random_split: old-task retention vs. training cost (ep1–{n_epoch})")
    apply_x_axis(n_epoch)
    plt.legend(fontsize=8, ncol=2, loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0.09, 1, 1))
    plt.gcf().text(0.02, 0.01, ICD_NOTE, fontsize=6.5, color="#555555", va="bottom")
    plt.savefig(out_png, dpi=200)
    print(f"写入 {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="横轴上限 epoch（默认 25）")
    args = parser.parse_args()
    plot_acc_old(args.epochs)


if __name__ == "__main__":
    main()
