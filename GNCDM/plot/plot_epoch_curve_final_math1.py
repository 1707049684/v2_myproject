# -*- coding: utf-8 -*-
"""图 A 终版：math1 random_split，7 个模型的 ACC_new-epoch 收敛曲线（效率 vs 效果）。

汇总 2 个产物脚本各自跑出的 CSV：
  plot_epoch_curve_gncdm_math1.py       -> epoch_curve_gncdm_math1_random_split.csv
      (CLEAN-Full / CLEAN-LoRA / Full-Replay / X-DER / C-LoRA-GNCDM)
  plot_epoch_curve_avalanche_math1.py   -> epoch_curve_avalanche_math1_random_split.csv
      (EWC / DER++，需 avalanche-lib，跑在 _scratch/clbase-venv)

为什么不画 ICD：本图的 x 轴是"同一份新题训练集被重复训练的第几个 epoch"，7 条曲线里每一
格都对应一次真实的梯度更新。ICD 是单遍流式方法（新题训练集只切成 chunk 顺序过一遍，且
是否更新参数由 turning_point() 按分布漂移量门控决定）——实测该方法在新题阶段的 25 个
chunk 里一次更新都没触发（依赖旧题阶段学到的编码器零样本泛化），因此它没有"随 epoch 收
敛"这个过程，跟本图的横轴定义（重复训练的进度）不是同一件事，放进来会造成"两种不同训练
范式共享一个 x 轴"的误导。ICD 仍然出现在 `all_methods_math1_random_split.csv` 的最终指
标对比里（那里比的是各方法用自己协议跑完后的结果，不涉及训练进度，比较成立）。
ICD 曲线数据/脚本见 `experiments/run_icd_math1_curve.py`（不用于本图，留作旁证）。

运行：cd GNCDM/plot && python plot_epoch_curve_final_math1.py
产物：incremental_result/epoch_curve_math1_random_split_final.png
"""

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
    "CLEAN-LoRA": dict(marker="s", linestyle="-", color="#ff7f0e"),
    "Full-Replay": dict(marker="^", linestyle="-", color="#2ca02c"),
    "X-DER": dict(marker="v", linestyle="-", color="#9467bd"),
    "C-LoRA-GNCDM": dict(marker="D", linestyle="-", color="#8c564b"),
    "EWC": dict(marker="P", linestyle="-", color="#e377c2"),
    "DER++": dict(marker="X", linestyle="-", color="#7f7f7f"),
}
ORDER = ["CLEAN-Full", "CLEAN-LoRA", "Full-Replay", "EWC", "DER++", "C-LoRA-GNCDM", "X-DER"]

# 图内脚注：解释 ICD 为何不在本图出现（单遍流式 + turning-point 门控，没有"随 epoch 收敛"
# 这个过程，跟本图 x 轴的"重复训练进度"定义不是同一件事；最终指标仍在总表里跟其它方法比）。
ICD_NOTE = (
    "ICD not shown: single-pass streaming method, no epoch-wise retraining\n"
    "(turning-point gate never fired on new items) -> not comparable on this x-axis.\n"
    "See all_methods_math1_random_split.csv for its final-metric comparison."
)


def main():
    df_gncdm = pd.read_csv(os.path.join(SAVE_DIR, "epoch_curve_gncdm_math1_random_split.csv"))
    df_aval = pd.read_csv(os.path.join(SAVE_DIR, "epoch_curve_avalanche_math1_random_split.csv"))
    df = pd.concat([df_gncdm, df_aval], ignore_index=True)

    out_csv = os.path.join(SAVE_DIR, "epoch_curve_math1_random_split_final.csv")
    df.to_csv(out_csv, index=False)
    print(f"合并写入 {out_csv}")

    plt.figure(figsize=(7, 5.6))
    for name in ORDER:
        sub = df[df.Model == name].sort_values("epoch")
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
    plt.title("math1 random_split: efficiency vs. effectiveness")
    plt.legend(fontsize=8, ncol=2, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0.09, 1, 1))
    plt.gcf().text(0.02, 0.01, ICD_NOTE, fontsize=6.5, color="#555555", va="bottom")
    out_png = os.path.join(SAVE_DIR, "epoch_curve_math1_random_split_final.png")
    plt.savefig(out_png, dpi=200)
    print(f"写入 {out_png}")


if __name__ == "__main__":
    main()
