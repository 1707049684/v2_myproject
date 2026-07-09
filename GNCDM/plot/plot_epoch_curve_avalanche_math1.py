# -*- coding: utf-8 -*-
"""图 A（Avalanche 骨干部分）：EWC / DER++ 在 math1 random_split 上的 ACC_new/ACC_old-epoch 收敛曲线。

必须用装了 avalanche-lib 的解释器跑（本机在 d:\\CD_continue\\_scratch\\clbase-venv）：
    d:\\CD_continue\\_scratch\\clbase-venv\\Scripts\\python.exe plot_epoch_curve_avalanche_math1.py

只做"监控式"改动：
- EWC：用一个 avalanche SupervisedPlugin 的 after_training_epoch 钩子，在 strat.train(exp) 内部
  每个 epoch 结束时读一次 valid_new_ds 的 ACC，只在 target 实验（Task1=新题）里记录；不改变
  strat.train(exp) 单次调用的结构，Fisher 计算时机与官方 run_ewc 完全一致，零风险。
- DER++：官方 run_der 本就是"外层 for epoch + strat.train(exp, train_epochs=1)"的手写循环，
  直接在循环里加一次 valid_new_ds 评测记录，不改变 best_acc/wait/checkpoint 选优逻辑。

lambda/mem_size 取官方 all_methods_math1_random_split.csv 里已选定的值（EWC lambda=1000，
DER++ mem=5000），只跑这一组，不做完整 sweep。

产物：incremental_result/epoch_curve_avalanche_math1_random_split_ep{N}.csv
运行：cd GNCDM/plot && python plot_epoch_curve_avalanche_math1.py [--epochs 25]
"""

import argparse
import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(PLOT_DIR)
EXPERIMENTS_DIR = os.path.join(GNCDM_DIR, "experiments")
for p in (GNCDM_DIR, EXPERIMENTS_DIR, os.path.join(EXPERIMENTS_DIR, "_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

import cl_baselines_random_split as CB

EWC_LAMBDA = 1000  # 官方 all_methods_math1_random_split.csv 里 "EWC (lambda=1000)" 那一行
SAVE_DIR = CB.SAVE_DIR


def ewc_curve(meta, device, curve_max_epoch):
    from avalanche.training.supervised import EWC
    from avalanche.core import SupervisedPlugin

    history = []

    class NewTaskAccPlugin(SupervisedPlugin):
        def __init__(self):
            super().__init__()
            self.epoch_ctr = 0

        def after_training_epoch(self, strategy, **kwargs):
            if strategy.experience.current_experience != 1:  # 只记 Task1=新题
                return
            self.epoch_ctr += 1
            _, _, acc_old, _ = CB.evaluate_cd_metrics(strategy.model, meta["valid_old_ds"], device)
            _, _, acc_new, _ = CB.evaluate_cd_metrics(strategy.model, meta["valid_new_ds"], device)
            history.append({"epoch": self.epoch_ctr, "acc": acc_new, "acc_old": acc_old})

    CB.set_seed(42)
    model = CB.CognitiveBackbone(meta["num_students"], meta["num_items"], CB.EMBED_DIM).to(device)
    strat = EWC(
        model,
        optim.Adam(model.parameters(), lr=CB.LR),
        nn.CrossEntropyLoss(),
        ewc_lambda=EWC_LAMBDA,
        mode=CB.EWC_MODE,
        train_mb_size=CB.TRAIN_MB_SIZE,
        train_epochs=curve_max_epoch,
        eval_mb_size=256,
        device=device,
        plugins=[NewTaskAccPlugin()],
    )
    for exp in CB._bench(meta).train_stream:
        strat.train(exp)
    old_m, new_m = CB._eval_both(model, meta, device)
    print(
        f"[EWC] 末轮 valid_new ACC={history[-1]['acc']:.4f} "
        f"valid_old ACC={history[-1]['acc_old']:.4f} | 本次 test ACC_new={new_m[2]:.4f}"
    )
    return [
        {"Model": "EWC", "epoch": h["epoch"], "ACC_new": h["acc"], "ACC_old": h["acc_old"]} for h in history
    ]


def der_curve(meta, device, curve_max_epoch):
    from avalanche.training.supervised import DER

    history = []
    CB.set_seed(42)
    model = CB.CognitiveBackbone(meta["num_students"], meta["num_items"], CB.EMBED_DIM).to(device)
    strat = DER(
        model,
        optim.Adam(model.parameters(), lr=CB.LR),
        nn.CrossEntropyLoss(),
        mem_size=CB.MEM_SIZE,
        alpha=CB.DER_ALPHA,
        beta=CB.DER_BETA,
        train_mb_size=CB.TRAIN_MB_SIZE,
        train_epochs=CB.DER_EPOCHS_INNER,
        eval_mb_size=256,
        device=device,
    )
    for exp in CB._bench(meta).train_stream:
        tid = exp.current_experience
        val_ds = (
            meta["valid_old_ds"]
            if tid == 0
            else CB.ConcatDataset([meta["valid_old_ds"], meta["valid_new_ds"]])
        )
        best_acc, best_state = -1.0, None
        ep = 0
        max_ep = curve_max_epoch if tid == 1 else CB.TRAIN_EPOCHS
        for _ in range(max_ep):
            strat.train(exp)
            _, _, val_acc, _ = CB.evaluate_cd_metrics(model, val_ds, device)
            if tid == 1:
                ep += 1
                _, _, acc_old, _ = CB.evaluate_cd_metrics(model, meta["valid_old_ds"], device)
                _, _, acc_new, _ = CB.evaluate_cd_metrics(model, meta["valid_new_ds"], device)
                history.append({"epoch": ep, "acc": acc_new, "acc_old": acc_old})
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    old_m, new_m = CB._eval_both(model, meta, device)
    print(
        f"[DER++] 末轮 valid_new ACC={history[-1]['acc']:.4f} "
        f"valid_old ACC={history[-1]['acc_old']:.4f} | 本次 test ACC_new={new_m[2]:.4f}"
    )
    return [
        {"Model": "DER++", "epoch": h["epoch"], "ACC_new": h["acc"], "ACC_old": h["acc_old"]}
        for h in history
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Task2 训练轮数（默认 25）")
    args = parser.parse_args()
    curve_max_epoch = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | epochs = {curve_max_epoch}")
    cfg = {
        "name": "math1",
        "train": os.path.join(CB.DATA_DIR, "math1_train_0.8_0.2.csv"),
        "valid": os.path.join(CB.DATA_DIR, "math1_valid_0.8_0.2.csv"),
        "test": os.path.join(CB.DATA_DIR, "math1_test_0.8_0.2.csv"),
        "Q": os.path.join(CB.DATA_DIR, "math1_Q_matrix.npy"),
        "n_item": 20,
        "n_know": 11,
        "new_concepts": [0, 1, 3, 6],
    }
    meta = CB.load_random(cfg)

    rows = ewc_curve(meta, device, curve_max_epoch)
    rows += der_curve(meta, device, curve_max_epoch)

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, f"epoch_curve_avalanche_math1_random_split_ep{curve_max_epoch}.csv")
    df.to_csv(out, index=False)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
