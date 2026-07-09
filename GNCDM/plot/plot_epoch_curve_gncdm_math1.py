# -*- coding: utf-8 -*-
"""图 A（GNCDM 骨干部分）：ACC_new 随训练 epoch 的收敛曲线，math1 random_split，alpha=0.20。

产出 5 条曲线（不含需要 avalanche 的 EWC/DER++，见 plot_epoch_curve_avalanche_math1.py）：
  CLEAN-Full（= Ours Dynamic DNA）/ CLEAN-LoRA（= Ours LoRA）/ Full-Replay（Full Replay Oracle）
  / X-DER（G-NCDM 骨干移植版）/ C-LoRA-GNCDM（同框架 C-LoRA，lambda=0.1，math1 最优）

统一用 valid_new 上 buffer 无泄漏 ACC 做纵轴（C-LoRA-GNCDM 因该基线脚本本身不读 valid 文件、
且其训练本就不做逐epoch checkpoint 选优，改用 test_new 监控，不影响任何训练/选优行为）。

产物：incremental_result/epoch_curve_gncdm_math1_random_split.csv
运行：cd GNCDM/plot && python plot_epoch_curve_gncdm_math1.py
"""

import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(PLOT_DIR)
EXPERIMENTS_DIR = os.path.join(GNCDM_DIR, "experiments")
for p in (GNCDM_DIR, EXPERIMENTS_DIR, os.path.join(EXPERIMENTS_DIR, "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch

import run_incremental_math1 as R
from core.model import GNCDM
from run_xder import run_xder
import gncdm_clora_baseline as CL

DATA_DIR = R.DATA_DIR
SAVE_DIR = R.SAVE_DIR
ALPHA = 0.20
NEW_CONCEPTS = [0, 1, 3, 6]
N_EPOCH = 15
N_USER, N_ITEM_TOTAL, N_KNOW_TOTAL = 4209, 20, 11
CLORA_LAMBDA = 0.1  # math1 最优（clora_gncdm_lambda_sweep_random_split.csv 见顶）


def load():
    Q = np.load(os.path.join(DATA_DIR, "math1_Q_matrix.npy"))
    df_train = pd.read_csv(os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"))
    df_valid = pd.read_csv(os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"))

    Q_mat, item_map, n_item_old, n_know_old = R.strict_bipartition(Q, NEW_CONCEPTS)
    df_train = R.remap_items(df_train, item_map)
    df_valid = R.remap_items(df_valid, item_map)
    n_item_new, n_know_new = N_ITEM_TOTAL - n_item_old, N_KNOW_TOTAL - n_know_old

    train_old = df_train[df_train.item_id < n_item_old].copy()
    train_new = df_train[df_train.item_id >= n_item_old].copy()
    valid_old = df_valid[df_valid.item_id < n_item_old].copy()
    valid_new = df_valid[df_valid.item_id >= n_item_old].copy()

    return dict(
        n_item_old=n_item_old,
        n_know_old=n_know_old,
        n_item_new=n_item_new,
        n_know_new=n_know_new,
        Q_old=Q_mat[:n_item_old, :n_know_old].copy(),
        Q_exp=Q_mat.copy(),
        train_old=train_old,
        train_new=train_new,
        valid_old=valid_old,
        valid_new=valid_new,
        log_old=R.build_log_mat(train_old, N_USER, n_item_old),
        log_full=R.build_log_mat(df_train, N_USER, N_ITEM_TOTAL),
    )


def train_base(c, device):
    R.set_seed(42)
    base = GNCDM(
        n_user=N_USER,
        n_item=c["n_item_old"],
        n_know=c["n_know_old"],
        user_dim=32,
        item_dim=32,
        alpha=ALPHA,
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
        n_epoch=N_EPOCH,
        desc="Base",
        eval_fn=lambda m: (
            R.populate_buffers(m, c["log_old"], device),
            R.evaluate_buf(m, c["valid_old"], device),
        )[1],
    )
    return base


def new_task_eval_fn(c, device):
    def fn(m):
        R.populate_buffers(m, c["log_full"], device)
        return R.evaluate_buf(m, c["valid_new"], device)

    return fn


STRATEGIES = {
    "CLEAN-Full": dict(
        expand=lambda m, c: m.expand_topology(c["n_item_new"], c["n_know_new"], c["Q_exp"]),
        params=lambda m: R.new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        train_df=lambda c: c["train_new"],
    ),
    "CLEAN-LoRA": dict(
        expand=lambda m, c: m.expand_topology_lora(
            delta_M=c["n_item_new"],
            delta_K=c["n_know_new"],
            Q_expanded=c["Q_exp"],
            M_old=c["n_item_old"],
            rank=min(16, c["n_know_new"]),
        ),
        params=R.lora_params,
        train_df=lambda c: c["train_new"],
    ),
    "Full-Replay": dict(
        expand=lambda m, c: m.full_replay_oracle_expand_topology(
            c["n_item_new"], c["n_know_new"], c["Q_exp"]
        ),
        params=lambda m: list(m.parameters()),
        train_df=lambda c: pd.concat([c["train_old"], c["train_new"]], ignore_index=True),
    ),
}


def run_gncdm_family(c, device):
    base = train_base(c, device)
    rows = []
    for name, spec in STRATEGIES.items():
        m = R.fresh_base(base)
        spec["expand"](m, c)
        R.populate_buffers(m, c["log_full"], device)
        history = []
        R.train_real(
            m,
            spec["train_df"](c),
            c["log_full"],
            spec["params"](m),
            device,
            n_epoch=N_EPOCH,
            desc=name,
            eval_fn=new_task_eval_fn(c, device),
            history=history,
        )
        for h in history:
            rows.append({"Model": name, "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]})
        print(f"[{name}] 完成，{len(history)} 个 epoch 记录，末轮 ACC_new={history[-1]['acc']:.4f}")
    return rows


def run_xder_curve(c, device):
    """history_eval_fn 用同一份 c（load() 产出，与其它 4 条曲线共享同一 bipartition/seed）的
    valid_new，保证纵轴口径统一为"新题验证 ACC"，而非 run_xder 内部选优用的 combined valid。"""
    history = []
    row = run_xder(
        split_name="math1_random_split(curve)",
        ds_name="math1_curve",
        train_path=os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"),
        valid_path=os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"),
        test_path=os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv"),
        Q_path=os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
        device=device,
        n_user=N_USER,
        n_item_total=N_ITEM_TOTAL,
        n_know_total=N_KNOW_TOTAL,
        new_concepts=NEW_CONCEPTS,
        alpha=ALPHA,
        history=history,
        history_eval_fn=new_task_eval_fn(c, device),
    )
    print(f"[X-DER] 末轮(valid_new) ACC={history[-1]['acc']:.4f} | 官方 test ACC_new={row['ACC_new']:.4f}")
    rows = [{"Model": "X-DER", "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]} for h in history]
    return rows


def run_clora_gncdm_curve(device):
    cfg = CL.CONFIGS["math1"]
    meta = CL.load_partition(cfg)
    CL.set_seed(42)
    base = CL._new_model(cfg, meta, device)
    CL.train_real(
        base,
        meta["train_old"],
        meta["log_old_only"],
        list(base.parameters()),
        device,
        n_epoch=CL.BASE_EPOCHS,
        desc="Base(CLoRA)",
    )
    CL.populate_buffers(base, meta["log_old_only"], device)
    base_theta_ref = base.get_Theta_buf().clone()
    import copy as _copy

    base_state = _copy.deepcopy(base.state_dict())

    def eval_fn(m):
        CL.populate_buffers(m, meta["log_full"], device)
        return CL.evaluate_buf(m, meta["test_new"], device)

    history = []
    r = CL.run_one_lambda(
        cfg, base_state, base_theta_ref, meta, CLORA_LAMBDA, device, history=history, history_eval_fn=eval_fn
    )
    print(
        f"[C-LoRA-GNCDM] 末轮(test_new 监控) ACC={history[-1]['acc']:.4f} | 官方 test ACC_new={r['ACC_new']:.4f}"
    )
    rows = [
        {"Model": "C-LoRA-GNCDM", "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]}
        for h in history
    ]
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    c = load()

    rows = run_gncdm_family(c, device)
    rows += run_xder_curve(c, device)
    rows += run_clora_gncdm_curve(device)

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, "epoch_curve_gncdm_math1_random_split.csv")
    df.to_csv(out, index=False)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
