"""ICD 在 math1 random_split 上的 "ACC_new vs 训练进度" 曲线，取代原来的 IRT 参考线。

与 `run_icd_math1_A.py`（官方 ICD 基线行的来源）完全同一套超参/同一次 train() 调用结构，
不新建/不拆分 train() 调用（ICD 的 warmup/turning-point 是按“整条 stream 的长度”一次性算好的，
拆成多次 train() 调用会把这个计算打乱，参见 run_icd_math1_A.py 顶部注释）。

做法（监控式，不改变任何训练动态）：子类化 `ICD` 只重写 `eval()`——先调用父类原有逻辑
（保留 stableness/trait 等日志），再在"新题阶段"的每个 stream chunk 后，额外用固定的
new_test 集合跑一次 `eval_subset`，记录到 history。这跟其它 7 条曲线的"外挂式监控评测、
不碰训练/选优逻辑"是同一套原则；只是 ICD 没有 epoch 概念，x 轴换成"新题阶段的 stream chunk
序号"（STREAM_PER_STAGE=25 个 chunk，与其它曲线最多 25 epoch 的量级相近，可放在同一张图上比）。

运行（须用装了 EduCDM 的解释器）：
    d:\\CD_continue\\_scratch\\icd-venv\\Scripts\\python.exe run_icd_math1_curve.py
产物：GNCDM/incremental_result/epoch_curve_icd_math1_random_split.csv
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

from EduCDM.ICD.ICD import ICD
from EduCDM.ICD.etl import dict_etl, inc_stream, item2knowledge, item2users, transform, user2items

HERE = os.path.dirname(os.path.abspath(__file__))  # GNCDM/experiments
DATA = os.path.join(HERE, "..", "data")  # GNCDM/data
OUT = os.path.join(HERE, "icd_out_curve_math1")
os.makedirs(OUT, exist_ok=True)
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("icd_curve")

NEW_CONCEPTS = [0, 1, 3, 6]
USER_N, ITEM_N, KNOW_N = 4209, 20, 11
ALPHA, TOLERANCE, BETA, WARMUP, EPOCH, STREAM_PER_STAGE = 0.2, 0.2, 0.9, 0.1, 1, 25


def strict_bipartition(Q, new_concepts):
    K = Q.shape[1]
    new_concepts = list(new_concepts)
    old_concepts = [k for k in range(K) if k not in new_concepts]
    concept_perm = old_concepts + new_concepts
    touches_new = Q[:, new_concepts].sum(axis=1) > 0
    item_perm = np.where(~touches_new)[0].tolist() + np.where(touches_new)[0].tolist()
    Q_re = Q[np.ix_(item_perm, concept_perm)].astype(np.float32)
    item_id_map = {old: new for new, old in enumerate(item_perm)}
    return Q_re, item_id_map, len(np.where(~touches_new)[0]), len(old_concepts)


def remap_items(df, item_id_map):
    df = df.copy()
    df["item_id"] = df["item_id"].map(item_id_map)
    return df


Q = np.load(os.path.join(DATA, "math1_Q_matrix.npy"))
Q_re, item_id_map, n_item_old, n_know_old = strict_bipartition(Q, NEW_CONCEPTS)
print(f"strict_bipartition: old items={n_item_old} new={ITEM_N - n_item_old}")

tr = remap_items(pd.read_csv(os.path.join(DATA, "math1_train_0.8_0.2.csv")), item_id_map)
va = remap_items(pd.read_csv(os.path.join(DATA, "math1_valid_0.8_0.2.csv")), item_id_map)

item_csv = os.path.join(OUT, "item.csv")
pd.DataFrame(
    [
        {"item_id": i, "knowledge_code": str([int(k) + 1 for k in np.where(Q_re[i] > 0)[0]])}
        for i in range(ITEM_N)
    ]
).to_csv(item_csv, index=False)
i2k = item2knowledge(item_csv)

old_tr, new_tr = tr[tr.item_id < n_item_old], tr[tr.item_id >= n_item_old]
new_va = va[va.item_id >= n_item_old]  # 曲线纵轴用 valid_new，跟其它 7 条曲线口径一致


def chunks(df, n):
    return list(inc_stream(df, max(1, int(len(df) // n))))


old_chunks = chunks(old_tr, STREAM_PER_STAGE)
new_chunks = chunks(new_tr, STREAM_PER_STAGE)
print(f"stream: stage1 old={len(old_chunks)} chunks, stage2 new={len(new_chunks)} chunks")

# 固定邻接表：跟 run_icd_math1_A.py 的最终评测一致，用"全量训练交互"构建 u2i/i2u，
# 每一步曲线评测只是换 net 的权重，不换邻接表（无泄漏口径与官方脚本相同）。
u2i_full = user2items(tr)
i2u_full = item2users(tr)


def eval_subset(net, df_subset):
    net.eval()
    yt, yp = [], []
    data = transform(
        df_subset, u2i_full, i2u_full, i2k, KNOW_N, batch_size=256, silent=True, allow_missing="skip"
    )
    with torch.no_grad():
        for uid, U, um, iid, I, im, IK, r in data:
            pred, *_ = net(U, um, I, im, IK)
            yp.extend(pred.tolist())
            yt.extend(r.tolist())
    yt, yp = np.array(yt), np.array(yp)
    yl = (yp >= 0.5).astype(int)
    auc = roc_auc_score(yt, yp) if len(set(yt.tolist())) > 1 else float("nan")
    return {
        "auc": auc,
        "rmse": mean_squared_error(yt, yp) ** 0.5,
        "acc": accuracy_score(yt, yl),
        "f1": f1_score(yt, yl),
    }


class ICDWithHistory(ICD):
    """只重写 eval()：先跑父类原逻辑（保留 stableness/trait 日志不变），
    再在新题阶段（i >= new_stage_start）额外记一条 valid_new 曲线点。不碰 train()。"""

    def __init__(self, *args, history=None, new_stage_start=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = history
        self._new_stage_start = new_stage_start

    def eval(self, i, inc_train_df_list, inc_test_data, pre_dict2, inc_u2i, inc_i2u, tps, wfs):
        super().eval(i, inc_train_df_list, inc_test_data, pre_dict2, inc_u2i, inc_i2u, tps, wfs)
        if self._history is not None and i >= self._new_stage_start:
            r = eval_subset(self.net, new_va)
            # tps_so_far：截至当前 chunk 累计触发过的 turning-point 次数，用来核实"新题阶段
            # 是否真的发生过再训练"——本次跑（tolerance=0.2）里 new 阶段全程都不再增长，
            # 说明曲线是平线由此而来，不是评测代码的问题。
            self._history.append(
                {"epoch": i - self._new_stage_start + 1, "tps_so_far": len(tps), **r}
            )


def main():
    os.chdir(OUT)
    history = []
    model = ICDWithHistory(
        "ncd",
        USER_N,
        ITEM_N,
        KNOW_N,
        epoch=EPOCH,
        weight_decay=0,
        inner_metrics=True,  # 每个 stream step 都调用 eval()，才能拿到逐 chunk 曲线
        logger=logger,
        alpha=ALPHA,
        ctx="cpu",
        history=history,
        new_stage_start=len(old_chunks),
    )
    model.train(old_chunks + new_chunks, i2k, beta=BETA, warmup_ratio=WARMUP, tolerance=TOLERANCE)

    print(
        f"[ICD] 新题阶段共记录 {len(history)} 个 chunk 点，末点 ACC_new={history[-1]['acc']:.4f}，"
        f"turning-point 触发次数从 {history[0]['tps_so_far']} 到 {history[-1]['tps_so_far']}"
    )
    rows = [
        {"Model": "ICD", "epoch": h["epoch"], "ACC_new": h["acc"], "AUC_new": h["auc"]} for h in history
    ]
    out_dir = os.path.join(HERE, "..", "incremental_result")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "epoch_curve_icd_math1_random_split.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"写入 {out_csv}")


if __name__ == "__main__":
    main()
