# -*- coding: utf-8 -*-
"""验证 Ours-LoRA vs Ours-DNA 的相对性能受什么驱动（数据量？概念数 ΔK？），
并用多 seed 配对比较判断差异是否真实（而非噪声）。

两种扫描轴（--sweep）：
  frac   : 固定 split/ΔK，缩放【新任务训练数据量】(默认 new-only：只子采样新题作答行，
           旧题满量 → 旧任务恒定、TMD≈0)。检验“数据量越大 LoRA 越强”。
  deltak : 固定数据，改变【新概念个数 ΔK】(挑最冷门的 N 个概念)。检验“概念越多 LoRA 越强”。
           ΔK 与 rank=16 的关系是关键：ΔK≤16 瓶颈失效 → LoRA≈DNA；ΔK>16 且概念相关 →
           低秩先验可能让 LoRA 反超。**务必配 --fix-new-rows 锁住新题作答量**，否则概念多
           会顺带带来更多新题/数据，混淆“概念数”与“数据量”两个变量。

多 seed（--seeds）：DNA 与 LoRA 在同一 seed 下用【同一份数据 + 同一 run_experiment 调用】，
  天然配对。汇总按 seed 配对算 gap=AUC_new(LoRA)-AUC_new(DNA)，报 mean±std 与显著性
  （|mean_gap| > 2·SE 才算差异可信）。数据子采样用固定 data_seed=42（各 seed 同一份数据），
  只让模型 init/训练随机性随 seed 变 → 直接检验“gap 是不是 init 噪声”。

口径：完全复用 _core/run_incremental_math1.run_experiment（buffer 预测口径，与主实验一致），
仅注入子采样/改 ΔK 后的训练集，再取 "Ours (Dynamic DNA)" / "Ours (LoRA)" 两行。

注意：a0910(17746 题)每点跑满 6 策略，务必 GPU；先用 --dataset math1 跑通流程再上 a0910。
本脚本为辅助验证脚本，验证完可删（结论沉淀进 docs / findings）。
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
gncdm_dir = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "_core"), gncdm_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from run_incremental_math1 import set_seed, run_experiment
from run_incremental_a0910 import auto_new_concepts

repo_root = os.path.dirname(gncdm_dir)
MATH1_DATA_DIR = os.path.join(gncdm_dir, "data")  # math1 在 GNCDM/data/
DS_DATA_DIR = os.path.join(repo_root, "data")  # a0910/junyi 在 repo_root/data/<ds>/
TMP_DIR = os.path.join(HERE, "_scale_tmp")
RESULT_DIR = os.path.join(gncdm_dir, "incremental_result")
DATA_SEED = 42  # 子采样固定 seed：各训练 seed 共用同一份数据，隔离 init/训练随机性

DNA_KEY = "Ours (Dynamic DNA)"
LORA_KEY = "Ours (LoRA)"


def get_config(ds):
    """返回数据集配置：Q 路径、train/valid/test 路径、维度、默认新概念 ΔK（与主实验一致）。"""
    if ds == "math1":
        return {
            "Q": os.path.join(MATH1_DATA_DIR, "math1_Q_matrix.npy"),
            "train": os.path.join(MATH1_DATA_DIR, "math1_train_0.8_0.2.csv"),
            "valid": os.path.join(MATH1_DATA_DIR, "math1_valid_0.8_0.2.csv"),
            "test": os.path.join(MATH1_DATA_DIR, "math1_test_0.8_0.2.csv"),
            "n_user": 4209,
            "n_item": 20,
            "n_know": 11,
            "new_concepts": [0, 1, 3, 6],
        }
    if ds in ("a0910", "junyi"):
        base = os.path.join(DS_DATA_DIR, ds)
        Q_path = os.path.join(base, "Q_matrix.npy")
        Q = np.load(Q_path)
        rnd = os.path.join(base, "new_random_split")
        tr, va, te = (os.path.join(rnd, f) for f in ("train.csv", "valid.csv", "test.csv"))
        if ds == "a0910":
            n_user, n_item, n_know = 4163, 17746, 123
        else:  # junyi：维度从文件读（对齐 run_incremental_junyi_random_split.py）
            n_item, n_know = int(Q.shape[0]), int(Q.shape[1])
            n_user = max(int(pd.read_csv(f)["user_id"].max()) + 1 for f in (tr, va, te))
        return {
            "Q": Q_path,
            "train": tr,
            "valid": va,
            "test": te,
            "n_user": n_user,
            "n_item": n_item,
            "n_know": n_know,
            "new_concepts": auto_new_concepts(Q, 0.34),
        }
    raise ValueError(f"unknown dataset: {ds}")


def select_cold_concepts(Q, n):
    """挑最冷门的 n 个概念（按被多少题使用升序）作为新概念 ΔK。
    冷门优先 → 新题占比小、旧题多，且旧题天然不依赖这些新概念。"""
    freq = (Q > 0).sum(axis=0)
    order = np.argsort(freq)  # 冷门在前
    return sorted(int(k) for k in order[:n])


def make_subsampled_train(cfg, ds, new_concepts, frac, mode, fix_new_rows):
    """生成训练集 CSV，返回 (路径, 实际新题作答行数)。
    在【原始 item_id 空间】操作（run_experiment 内部再 strict_bipartition+remap）。
    新题 = Q 在任一新概念列上非零的题。子采样用固定 DATA_SEED（各训练 seed 共用同一份数据）。
    fix_new_rows 优先于 frac：把新题作答行数压到该预算（用于 deltak 扫描时解耦数据量）。"""
    df = pd.read_csv(cfg["train"])
    Q = np.load(cfg["Q"])
    new_items = set(np.where(Q[:, new_concepts].sum(axis=1) > 0)[0].tolist())
    is_new_row = df["item_id"].isin(new_items)
    full_new = int(is_new_row.sum())

    # 无需子采样：满量 new、无预算、frac>=1
    if mode == "new" and fix_new_rows is None and frac >= 1.0:
        return cfg["train"], full_new

    if mode == "all":
        out_df = df.sample(frac=frac, random_state=DATA_SEED).reset_index(drop=True)
        n_new = int(out_df["item_id"].isin(new_items).sum())
    else:  # mode == "new"
        new_rows = df[is_new_row]
        old_rows = df[~is_new_row]
        if fix_new_rows is not None and len(new_rows) > fix_new_rows:
            new_rows = new_rows.sample(n=fix_new_rows, random_state=DATA_SEED)
        elif frac < 1.0:
            new_rows = new_rows.sample(frac=frac, random_state=DATA_SEED)
        out_df = pd.concat([old_rows, new_rows], ignore_index=True)
        n_new = len(new_rows)

    os.makedirs(TMP_DIR, exist_ok=True)
    tag = f"{ds}_dk{len(new_concepts)}_{mode}_f{frac}_fix{fix_new_rows}"
    out_path = os.path.join(TMP_DIR, f"train_{tag}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path, n_new


def run_point(cfg, ds, new_concepts, frac, alpha, mode, device, seed, fix_new_rows):
    """跑单个点（一个 ΔK/frac/alpha/seed），返回 DNA 与 LoRA 两行记录。"""
    set_seed(seed)  # 仅影响模型 init/训练随机性（数据用固定 DATA_SEED）
    train_path, n_new = make_subsampled_train(cfg, ds, new_concepts, frac, mode, fix_new_rows)
    delta_k = len(new_concepts)
    split_name = f"{ds}_dk{delta_k}_f{frac}_a{alpha}_s{seed}"
    res = run_experiment(
        split_name,
        "buf",
        train_path,
        cfg["valid"],
        cfg["test"],
        cfg["Q"],
        device,
        n_user=cfg["n_user"],
        n_item_total=cfg["n_item"],
        n_know_total=cfg["n_know"],
        new_concepts=new_concepts,
        alpha=alpha,
        run_strategies={DNA_KEY, LORA_KEY},  # 只跑 DNA+LoRA（Base 仍跑，作为扩展基座），省算力
    )
    side = os.path.join(RESULT_DIR, f"incremental_results_{split_name}.csv")
    if os.path.exists(side):
        os.remove(side)  # 删 run_experiment 的逐点副产物，避免污染目录

    by_model = {r["Model"]: r for r in res}
    rows = []
    for key in (DNA_KEY, LORA_KEY):
        r = by_model[key]
        rows.append(
            {
                "dataset": ds,
                "sweep_mode": mode,
                "delta_k": delta_k,
                "frac": frac,
                "alpha": alpha,
                "seed": seed,
                "method": "DNA" if key == DNA_KEY else "LoRA",
                "new_train_rows": n_new,
                "auc_old": float(r["AUC_old"]),
                "auc_new": float(r["AUC_new"]),
                "acc_new": float(r["ACC_new"]),
                "tmd": float(r["TMD"]) if r["TMD"] != "" else float("nan"),
            }
        )
    return rows


def summarize(df, xcol, out_csv):
    """多 seed 配对汇总：每个 (xval, alpha) 下按 seed 算 gap=LoRA-DNA，报 mean±std + 显著性。"""
    print("\n" + "=" * 78)
    print(f"汇总（配对 gap=AUC_new[LoRA]-AUC_new[DNA]，按 {xcol} × alpha；多 seed 取均值）")
    print("=" * 78)
    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha]
        print(f"\n--- alpha = {alpha} ---")
        # 每 seed 配对：index=(xval,seed) → 列 DNA/LoRA
        piv = sub.pivot_table(index=[xcol, "seed"], columns="method", values="auc_new")
        if "DNA" not in piv or "LoRA" not in piv:
            print("  缺 DNA 或 LoRA 数据，跳过。")
            continue
        piv["gap"] = piv["LoRA"] - piv["DNA"]
        header = f"  {xcol:>8} | {'DNA(mean)':>10} {'LoRA(mean)':>10} | {'gap mean±std':>16} {'n':>3} {'sig?':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        gaps_by_x = []
        for xval, g in piv.groupby(level=0):
            n = len(g)
            dna_m = g["DNA"].mean()
            lora_m = g["LoRA"].mean()
            gap_m = g["gap"].mean()
            gap_s = g["gap"].std(ddof=1) if n > 1 else float("nan")
            if n > 1 and gap_s == gap_s and gap_s > 0:
                se = gap_s / np.sqrt(n)
                sig = "✅" if abs(gap_m) > 2 * se else "✗(噪声)"
            else:
                sig = "需多seed"
            std_str = f"{gap_s:.4f}" if gap_s == gap_s else "  -  "
            print(
                f"  {xval:>8} | {dna_m:>10.4f} {lora_m:>10.4f} | "
                f"{gap_m:>+8.4f}±{std_str:>6} {n:>3} {sig:>6}"
            )
            gaps_by_x.append((xval, gap_m))
        if len(gaps_by_x) >= 2:
            x0, g0 = gaps_by_x[0]
            x1, g1 = gaps_by_x[-1]
            trend = "上升↑" if g1 > g0 else "下降↓"
            crossed = (g0 < 0) and (g1 > 0)
            print(
                f"  → gap 随 {xcol} 趋势: {g0:+.4f}({x0}) → {g1:+.4f}({x1}) ({trend}); "
                f"{'✅ 出现交叉(LoRA 反超)' if crossed else '未交叉'}"
            )
    print(f"\n明细已写入: {out_csv}")


def maybe_plot(df, xcol, out_png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] 跳过绘图（matplotlib 不可用：{e}）")
        return
    alphas = sorted(df["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 4.5), squeeze=False)
    for j, alpha in enumerate(alphas):
        ax = axes[0][j]
        sub = df[df["alpha"] == alpha]
        for method, mk in (("DNA", "o-"), ("LoRA", "s-")):
            s = sub[sub["method"] == method]
            agg = s.groupby(xcol)["auc_new"].agg(["mean", "std"]).sort_index()
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt=mk, capsize=3,
                        label=f"Ours-{method}")
        ax.set_title(f"AUC_new vs {xcol} (alpha={alpha})")
        ax.set_xlabel(xcol)
        ax.set_ylabel("AUC_new")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[plot] 已保存: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="LoRA vs DNA: 数据量 / 概念数 缩放验证（多 seed）")
    ap.add_argument("--dataset", default="a0910", choices=["a0910", "junyi", "math1"])
    ap.add_argument("--sweep", default="frac", choices=["frac", "deltak"],
                    help="frac=扫新任务数据量; deltak=扫新概念个数 ΔK")
    ap.add_argument("--mode", default="new", choices=["new", "all"],
                    help="frac 扫描用: new=只子采样新题作答行; all=整份训练集子采样")
    ap.add_argument("--fracs", default="0.1,0.25,0.5,0.75,1.0", help="frac 扫描的比例列表")
    ap.add_argument("--delta-ks", default="8,16,32,64", help="deltak 扫描的新概念个数列表")
    ap.add_argument("--frac", type=float, default=1.0, help="deltak 扫描时固定的 frac")
    ap.add_argument("--fix-new-rows", type=int, default=None,
                    help="deltak 扫描强烈建议设置：把新题作答行数锁到该预算以解耦数据量")
    ap.add_argument("--alphas", default="0.1,0.9", help="alpha 列表（解耦 alpha）")
    ap.add_argument("--seeds", default="42", help="训练 seed 列表，逗号分隔（多 seed 估方差）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if args.dataset == "a0910" and device.type == "cpu":
        print("⚠️ a0910 题量大(17746)，CPU 会很慢，建议 GPU。")

    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    cfg = get_config(args.dataset)

    # 构造扫描点：每个点 = (frac, new_concepts)
    if args.sweep == "frac":
        xcol = "frac"
        points = [(f, cfg["new_concepts"]) for f in (float(x) for x in args.fracs.split(","))]
        mode = args.mode
        if args.fix_new_rows is not None:
            print("⚠️ frac 扫描下忽略 --fix-new-rows（frac 本身就是数据量旋钮）")
        fix_new_rows = None
    else:  # deltak
        xcol = "delta_k"
        Q = np.load(cfg["Q"])
        dks = [int(x) for x in args.delta_ks.split(",")]
        for n in dks:
            if n >= cfg["n_know"]:
                raise SystemExit(f"ΔK={n} ≥ 总概念数 {cfg['n_know']}，请减小")
        points = [(args.frac, select_cold_concepts(Q, n)) for n in dks]
        mode = "new"  # deltak 扫描固定 new 口径
        fix_new_rows = args.fix_new_rows
        if fix_new_rows is None:
            print("⚠️ 未设 --fix-new-rows：概念数与新题数据量未解耦，结果会混淆两个变量！")

    print(f"dataset={args.dataset} dims=({cfg['n_user']},{cfg['n_item']},{cfg['n_know']})")
    print(f"sweep={args.sweep} xcol={xcol} 点数={len(points)} alphas={alphas} seeds={seeds} "
          f"fix_new_rows={fix_new_rows}")
    print(f"将运行 {len(points) * len(alphas) * len(seeds)} 个点 × 6 策略（可中断续跑）")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_csv = args.out or os.path.join(
        RESULT_DIR, f"scaling_lora_vs_dna_{args.dataset}_{args.sweep}.csv"
    )
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["delta_k"], prev["frac"], prev["alpha"], prev["seed"]))
        print(f"续跑：已有 {len(done)} 个 (ΔK,frac,alpha,seed) 点，跳过。")

    for alpha in alphas:
        for frac, ncs in points:
            for seed in seeds:
                key = (len(ncs), frac, alpha, seed)
                if key in done:
                    continue
                print(f"\n>>> ΔK={len(ncs)} frac={frac} alpha={alpha} seed={seed}")
                rows = run_point(cfg, args.dataset, ncs, frac, alpha, mode, device,
                                 seed, fix_new_rows)
                df_now = pd.DataFrame(rows)
                if os.path.exists(out_csv):
                    df_now = pd.concat([pd.read_csv(out_csv), df_now], ignore_index=True)
                df_now.drop_duplicates(
                    subset=["dataset", "delta_k", "frac", "alpha", "seed", "method"],
                    keep="last",
                ).to_csv(out_csv, index=False)

    df = pd.read_csv(out_csv)
    df = df[df["dataset"] == args.dataset]
    summarize(df, xcol, out_csv)
    if not args.no_plot:
        maybe_plot(df, xcol, out_csv.replace(".csv", f"_{args.sweep}.png"))


if __name__ == "__main__":
    main()
