# -*- coding: utf-8 -*-
"""一次性探针：只跑 xes random_split 的 DNA + LoRA（外加必跑的 Base），alpha=0.1。

目的：看 alpha=0.1 下代表模型 Ours (Dynamic DNA) / Ours (LoRA) 的 AUC_new 相比旧表
（alpha=0.20）有没有变好。选优口径与旧表一致（train_real 默认 select_metric='acc'，
与基线 DER++ 早停同口径），所以唯一变量是 alpha。只看这两条，不碰基线、不碰 X-DER。

⚠️ 用独立 split_name="xes_dnalora_probe_a01"，结果写到
   incremental_result/incremental_results_xes_dnalora_probe_a01.csv，
   **不会覆盖** 主表 incremental_results_xes_random_split.csv（6 行 Ours 中间表）。
口径与主入口一致：buf 无泄漏预测、ΔK=auto_new_concepts(0.34)。仅 alpha=0.1 不同。

这是探索性脚本，看完结论即可删（不入主流程）。
运行：cd GNCDM/experiments && python probe_xes_dnalora_a01.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts

DATA_DIR = os.path.join(gncdm_dir, "data")
PREFIX = "xes"
N_USER, N_ITEM, N_KNOW = 3402, 7056, 837
ALPHA = 0.1


def main():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | alpha={ALPHA} | 检查点按 valid ACC 选优（与旧表同口径）")
    if device.type == "cpu":
        print("⚠️ XES3G5M 子集交互量大(~1M),建议 GPU 服务器。")

    Q_path = os.path.join(DATA_DIR, f"{PREFIX}_Q_matrix.npy")
    Q = np.load(Q_path)
    new_concepts = auto_new_concepts(Q, 0.34)
    tr, va, te = (
        os.path.join(DATA_DIR, f"{PREFIX}_{s}_0.8_0.1_0.1.csv") for s in ("train", "valid", "test")
    )

    set_seed(42)
    results = run_experiment(
        "xes_dnalora_probe_a01",  # 独立 split → 不覆盖主表
        "buf",
        tr,
        va,
        te,
        Q_path,
        device,
        n_user=N_USER,
        n_item_total=N_ITEM,
        n_know_total=N_KNOW,
        new_concepts=new_concepts,
        alpha=ALPHA,
        run_strategies={"Ours (Dynamic DNA)", "Ours (LoRA)"},  # 只跑这两条
    )

    # 对照打印（与 all_methods 旧值并列，方便一眼看 AUC_new 有没有抬）
    old_ref = {
        "Ours (Dynamic DNA)": (0.7868, 0.7224),  # 旧表 alpha=0.20（同为 acc 选优）
        "Ours (LoRA)": (0.7868, 0.7367),
    }
    print("\n" + "=" * 64)
    print(" 探针结果对照（旧 = all_methods alpha=0.20/acc 选优）")
    print("=" * 64)
    print(f"{'Method':<22}{'AUC_old':>10}{'AUC_new':>10}   vs 旧(old/new)")
    for r in results:
        if r["Model"] not in old_ref:
            continue
        ao, an = r["AUC_old"], r["AUC_new"]
        ro, rn = old_ref[r["Model"]]
        dn = an - rn if isinstance(an, (int, float)) else float("nan")
        print(f"{r['Model']:<22}{ao:>10.4f}{an:>10.4f}   旧 {ro:.4f}/{rn:.4f}  ΔAUC_new={dn:+.4f}")
    print(
        "\n（注：选优口径与旧表一致(acc)，唯一变量是 alpha 0.20→0.1，ΔAUC_new 即纯 alpha 效应。）"
    )


if __name__ == "__main__":
    main()
