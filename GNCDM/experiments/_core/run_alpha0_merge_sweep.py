# -*- coding: utf-8 -*-
"""补跑 random_split 的 alpha=0，并合并进已有 alpha_sweep CSV。

同口径：Base + Ours(DNA) + Ours(LoRA)，选优指标 sel = mean(valid ACC_old, ACC_new)。
合并目标：
  incremental_result/alpha_sweep_junyi_random_split.csv
  incremental_result/alpha_sweep_a0910_random_split.csv

服务器运行（需 GPU；从 experiments/ 启动）：
  cd GNCDM/experiments
  CUDA_VISIBLE_DEVICES=0 python _core/run_alpha0_merge_sweep.py --dataset both
  CUDA_VISIBLE_DEVICES=0 python _core/run_alpha0_merge_sweep.py --dataset junyi
  CUDA_VISIBLE_DEVICES=0 python _core/run_alpha0_merge_sweep.py --dataset a0910

可选：
  --alpha 0.0          # 默认 0
  --device cuda:0      # 默认自动选 cuda / cpu
  --epochs 25          # 覆盖 sweep 脚本里的 N_EPOCH
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))  # experiments/_core
EXP = os.path.dirname(HERE)  # experiments
GNCDM = os.path.dirname(EXP)  # GNCDM
for p in (HERE, EXP, GNCDM):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.model import GNCDM
import run_incremental_math1 as R

JOBS = {
    "junyi": ("sweep_junyi_random_alpha", "alpha_sweep_junyi_random_split.csv"),
    "a0910": ("sweep_a0910_random_alpha", "alpha_sweep_a0910_random_split.csv"),
}


def pick_device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_alpha(mod, alpha: float, device: torch.device, n_epoch: int) -> dict:
    print(f"\n===== [{mod.__name__}] alpha={alpha} device={device} n_epoch={n_epoch} =====")
    if device.type == "cpu" and "a0910" in mod.__name__:
        print("WARNING: a0910 on CPU is very slow; prefer a GPU node.")

    c = mod.load()
    print(
        f"dims: n_user={c['n_user']} n_item={c['n_item_total']} n_know={c['n_know_total']} "
        f"| old={c['n_item_old']}/{c['n_know_old']} new={c['n_item_new']}/{c['n_know_new']}"
    )

    # Temporarily override module N_EPOCH used inside train_dna / train_lora.
    old_ep = getattr(mod, "N_EPOCH", n_epoch)
    mod.N_EPOCH = n_epoch

    R.set_seed(42)
    base = GNCDM(
        n_user=c["n_user"],
        n_item=c["n_item_old"],
        n_know=c["n_know_old"],
        user_dim=32,
        item_dim=32,
        alpha=alpha,
        Q_mat=c["Q_old"],
        monotonicity_assumption=True,
        device=device,
    ).to(device)
    R.train_real(
        base,
        c["train_old"],
        c["log_old"],
        list(base.parameters()),
        device,
        n_epoch=n_epoch,
        desc=f"Base(a={alpha})",
        eval_fn=lambda m: (
            R.populate_buffers(m, c["log_old"], device),
            R.evaluate_buf(m, c["valid_old"], device),
        )[1],
    )
    R.populate_buffers(base, c["log_old"], device)
    b_va = R.evaluate_buf(base, c["valid_old"], device)
    b_te = R.evaluate_buf(base, c["test_old"], device)

    dna = mod.train_dna(base, c, device)
    dna_v_new = R.evaluate_buf(dna, c["valid_new"], device)
    dna_t_new = R.evaluate_buf(dna, c["test_new"], device)

    lora = mod.train_lora(base, c, device)
    lora_v_new = R.evaluate_buf(lora, c["valid_new"], device)
    lora_t_new = R.evaluate_buf(lora, c["test_new"], device)

    mod.N_EPOCH = old_ep

    sel = 0.5 * (b_va["acc"] + dna_v_new["acc"])
    row = {
        "alpha": alpha,
        "sel_DNA_validACC": round(sel, 4),
        "Base_te_AUCold": b_te["auc"],
        "Base_te_ACCold": b_te["acc"],
        "DNA_te_AUCnew": dna_t_new["auc"],
        "DNA_te_ACCnew": dna_t_new["acc"],
        "DNA_te_F1new": dna_t_new["f1"],
        "LoRA_te_AUCnew": lora_t_new["auc"],
        "LoRA_te_ACCnew": lora_t_new["acc"],
        "LoRA_te_F1new": lora_t_new["f1"],
        "DNA_va_AUCnew": dna_v_new["auc"],
        "LoRA_va_AUCnew": lora_v_new["auc"],
    }
    print(
        f"alpha={alpha:.2f} | sel={sel:.4f} | Base te ACCold={b_te['acc']:.4f} "
        f"| DNA te ACCnew={dna_t_new['acc']:.4f} | LoRA te ACCnew={lora_t_new['acc']:.4f}"
    )
    return row


def merge_row(csv_name: str, row: dict) -> str:
    out = os.path.join(R.SAVE_DIR, csv_name)
    df_new = pd.DataFrame([row])
    if os.path.exists(out):
        df = pd.read_csv(out)
        # Drop any existing row with the same alpha (float-safe).
        df = df[(df["alpha"] - row["alpha"]).abs() > 1e-9]
        df = pd.concat([df_new, df], ignore_index=True)
    else:
        df = df_new
    df = df.sort_values("alpha").reset_index(drop=True)
    df.to_csv(out, index=False)
    best = df.sort_values("sel_DNA_validACC", ascending=False).iloc[0]
    print(f"wrote {out} (n={len(df)})")
    print(df.to_string(index=False))
    print(
        f">>> best alpha={best['alpha']:.2f} sel={best['sel_DNA_validACC']:.4f} "
        f"(alphas={list(df['alpha'].values)})"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="补跑 alpha=0 并合并到 alpha_sweep CSV")
    parser.add_argument(
        "--dataset",
        choices=["junyi", "a0910", "both"],
        default="both",
        help="要跑的数据集（默认 both）",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="要补的 alpha（默认 0.0）")
    parser.add_argument("--device", type=str, default=None, help="如 cuda:0 / cpu；默认自动")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖 sweep 的 N_EPOCH（junyi/a0910 默认 25）",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    names = ["junyi", "a0910"] if args.dataset == "both" else [args.dataset]
    for name in names:
        mod_name, csv_name = JOBS[name]
        mod = importlib.import_module(mod_name)
        n_epoch = args.epochs if args.epochs is not None else mod.N_EPOCH
        row = run_one_alpha(mod, args.alpha, device, n_epoch)
        merge_row(csv_name, row)


if __name__ == "__main__":
    main()
