"""ICD baseline on a0910 (ASSISTments) random_split — Approach A (old/new split).

Run on a GPU server (a0910 has 17746 items). Aligned column-wise to
all_methods_a0910_random_split.csv: same strict_bipartition + auto_new_concepts(Q, 0.34),
ICD official hparams (cdm=ncd, alpha=0.2, tolerance=0.2, beta=0.9, epoch=1, warmup=0.1),
neighborhood caps max_u2i=128/max_i2u=64. Protocol: one stream old->new (stage1 old items
then stage2 new items), final one-shot eval on OLD-item test and NEW-item test separately;
RD = old-user NCD-trait L2 drift (old-only reference model = state after stage 1).

ENV (isolated venv; see GNCDM/docs/ICD_baseline_repro.md):
  pip install torch EduCDM  (+ longling baize fire); any pandas — this script monkeypatches
  EduCDM's groupby(['col']) pandas>=2.2 tuple-key bug at runtime, so NO source edit needed.

Usage:
  python run_icd_a0910_A.py [DATA_DIR] [CTX] [STREAM_PER_STAGE] [SPLIT_TAG]
    DATA_DIR         dir containing Q_matrix.npy and new_{SPLIT_TAG}/{train,valid,test}.csv
                     (default: <repo>/data/a0910)
    CTX              'cuda:0' (default if available) or 'cpu'
    STREAM_PER_STAGE chunks per stage (default 25)
    SPLIT_TAG        random_split (default) or user_split
"""

import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

# ---- config ----
DATASET = "a0910"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DATA = os.path.join(_REPO, "data", DATASET)

_extra = list(sys.argv[1:])
if _extra and _extra[-1] in ("random_split", "user_split"):
    _SPLIT_FROM_ARGV = _extra.pop()
else:
    _SPLIT_FROM_ARGV = "random_split"
if _extra and os.path.isfile(os.path.join(_extra[0], "Q_matrix.npy")):
    DATA_DIR = os.path.abspath(_extra.pop(0))
else:
    DATA_DIR = os.path.abspath(_DEFAULT_DATA)
CTX = _extra.pop(0) if _extra else ("cuda:0" if torch.cuda.is_available() else "cpu")
STREAM_PER_STAGE = int(_extra.pop(0)) if _extra else 25
SPLIT_TAG = _SPLIT_FROM_ARGV
assert SPLIT_TAG in ("random_split", "user_split"), f"bad SPLIT_TAG={SPLIT_TAG}"

NEW_ITEM_FRAC = 0.34
SUPPORT_FRAC, SPLIT_SEED = 0.5, 7  # user_split 与 eval_all_methods_user_split 同口径
# alpha/tolerance/beta/epoch: EduCDM examples/ICD/ICD.py main() defaults.
# warmup_ratio=0 (not the generic 0.1): bigdata-ustc/ICD's own a0910 example command is
# `pure_stream_inc_run.py --dataset a0910 --cdm ncd --alpha 0.2 --beta 0.9 --tolerance 0.2
#  --inner_metrics True --warmup_ratio 0`; 0.1 is borrowed from a different ("math") example
# dataset's generic main() defaults and produced miscalibrated old-test predictions on a0910
# (AUC_old > 0.5 but ACC_old < 0.5 -> thresholded-at-0.5 outputs shifted, not "no signal").
# epoch=1 (official default; DO NOT bump). Tried epoch=3: old-test AUC/ACC improved
# (0.60->0.63 / 0.54->0.60) but new-test AUC collapsed near-random (0.6675->0.5549) because
# `epoch` re-trains for N passes on EVERY stream chunk inside ICD.train(), not "overall" -
# it distorts turning_point()/momentum_weight_update() dynamics across the old->new stage
# boundary, overfitting stage-1 chunks at the cost of stage-2 (new-item) generalization.
# Net effect is a regression, not a fix -> reverted to 1.
ALPHA, TOLERANCE, BETA, WARMUP, EPOCH = 0.2, 0.2, 0.9, 0.0, 1
MAX_U2I, MAX_I2U = 128, 64  # official a0910 caps (large item/user count)
EVAL_BS = 256
SUPPORT_FRAC = float(os.environ.get("ICD_SUPPORT_FRAC", str(SUPPORT_FRAC)))
SPLIT_SEED = int(os.environ.get("ICD_SUPPORT_QUERY_SEED", str(SPLIT_SEED)))
SEED = int(os.environ.get("ICD_TRAIN_SEED", "0"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"icd_out_{DATASET}")
os.makedirs(OUT, exist_ok=True)
# Resolve before os.chdir(OUT): a relative ICD_OUTPUT_CSV would otherwise land under OUT/.
if os.environ.get("ICD_OUTPUT_CSV"):
    os.environ["ICD_OUTPUT_CSV"] = os.path.abspath(os.environ["ICD_OUTPUT_CSV"])
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(f"icd_{DATASET}")


# ---- runtime patch: EduCDM's groupby(['col']) yields tuple keys on pandas>=2.2 -> transform
# skips every row (empty data). Replace user2items/item2users with scalar-groupby versions in
# BOTH the etl and ICD modules so ICD.train picks them up. Safe on any pandas version. ----
def _patch_groupby():
    import EduCDM.ICD.ICD as _icd
    import EduCDM.ICD.etl as _etl

    def user2items(df, dict2=None):
        ul = {}
        if dict2:
            dict2.u2i = ul
        for uid, g in df.groupby("user_id"):
            if dict2:
                dict2.add_user_items_responses(uid, g["item_id"], g["score"])
            ul[uid] = [int(x) for x in (g["item_id"] * 2 + g["score"] + 1)]
        return ul

    def item2users(df, dict2=None):
        il = {}
        for iid, g in df.groupby("item_id"):
            if dict2:
                dict2.add_item_users_responses(iid, g["user_id"], g["score"])
            il[iid] = [int(x) for x in (g["user_id"] * 2 + g["score"] + 1)]
        return il

    _etl.user2items = _icd.user2items = user2items
    _etl.item2users = _icd.item2users = item2users
    return user2items, item2users


user2items, item2users = _patch_groupby()
from EduCDM.ICD.ICD import ICD  # noqa: E402
from EduCDM.ICD.etl import dict_etl, inc_stream, item2knowledge, transform  # noqa: E402


# ---- project-aligned bipartition (verbatim from cl_baselines_random_split.py) ----
def auto_new_concepts(Q, frac=0.34):
    n_item = Q.shape[0]
    freq = (Q > 0).sum(axis=0)
    touched = np.zeros(n_item, dtype=bool)
    new_set = []
    for k in np.argsort(freq):
        new_set.append(int(k))
        touched |= Q[:, k] > 0
        if touched.sum() >= frac * n_item:
            break
    return sorted(new_set)


def strict_bipartition(Q, new_concepts):
    K = Q.shape[1]
    new_concepts = list(new_concepts)
    old_concepts = [k for k in range(K) if k not in new_concepts]
    concept_perm = old_concepts + new_concepts
    touches_new = Q[:, new_concepts].sum(axis=1) > 0
    item_perm = np.where(~touches_new)[0].tolist() + np.where(touches_new)[0].tolist()
    Q_re = Q[np.ix_(item_perm, concept_perm)].astype(np.float32)
    item_id_map = {old: new for new, old in enumerate(item_perm)}
    return Q_re, item_id_map, int((~touches_new).sum()), len(old_concepts)


def remap_items(df, item_id_map):
    df = df.copy()
    df["item_id"] = df["item_id"].map(item_id_map).astype(int)
    return df


def split_support_query(df, frac=SUPPORT_FRAC, seed=SPLIT_SEED):
    sup = df.groupby("user_id", group_keys=False).sample(frac=frac, random_state=seed)
    return sup, df.drop(sup.index)


# ---- load + bipartition ----
rnd = os.path.join(DATA_DIR, f"new_{SPLIT_TAG}")
Q = np.load(os.path.join(DATA_DIR, "Q_matrix.npy"))
ITEM_N, KNOW_N = int(Q.shape[0]), int(Q.shape[1])
tr = pd.read_csv(os.path.join(rnd, "train.csv"))
va = pd.read_csv(os.path.join(rnd, "valid.csv"))
te = pd.read_csv(os.path.join(rnd, "test.csv"))
USER_N = int(max(tr.user_id.max(), va.user_id.max(), te.user_id.max())) + 1

new_concepts = auto_new_concepts(Q, NEW_ITEM_FRAC)
Q_re, item_id_map, n_item_old, n_know_old = strict_bipartition(Q, new_concepts)
tr, te = remap_items(tr, item_id_map), remap_items(te, item_id_map)
print(f"[{DATASET}] dims user={USER_N} item={ITEM_N} know={KNOW_N} | ctx={CTX}")
print(
    f"bipartition: new_concepts={len(new_concepts)}/{KNOW_N}, old items={n_item_old} "
    f"new={ITEM_N - n_item_old}, old concepts={n_know_old}"
)

item_csv = os.path.join(OUT, "item.csv")
pd.DataFrame(
    [
        {"item_id": i, "knowledge_code": str([int(k) + 1 for k in np.where(Q_re[i] > 0)[0]])}
        for i in range(ITEM_N)
    ]
).to_csv(item_csv, index=False)
i2k = item2knowledge(item_csv)

old_tr, new_tr = tr[tr.item_id < n_item_old], tr[tr.item_id >= n_item_old]
old_te, new_te = te[te.item_id < n_item_old], te[te.item_id >= n_item_old]
print(f"train old/new rows: {len(old_tr)}/{len(new_tr)}; test old/new: {len(old_te)}/{len(new_te)}")


def chunks(df, n):
    return list(inc_stream(df, max(1, int(len(df) // n))))


def make_icd():
    return ICD(
        "ncd",
        USER_N,
        ITEM_N,
        KNOW_N,
        epoch=EPOCH,
        weight_decay=0,
        inner_metrics=False,
        logger=logger,
        alpha=ALPHA,
        ctx=CTX,
    )


old_users = sorted(old_tr.user_id.unique().tolist())
u2i_old = user2items(old_tr)


def old_user_traits(_net):
    # EduCDM's get_net() auto-wraps in nn.DataParallel when >1 GPU is visible; DataParallel
    # only proxies forward/__call__, so custom methods need the same unwrap ICD.py uses
    # internally (`_net.module if isinstance(_net, torch.nn.DataParallel) else _net`).
    _net = _net.module if isinstance(_net, torch.nn.DataParallel) else _net
    out = _net.get_user_profiles(dict_etl(old_users, u2i_old, batch_size=EVAL_BS))
    return out["u_trait"].detach().cpu().numpy()


os.chdir(OUT)

# main model: one stream old->new (stage1 then stage2)
model = make_icd()
net = model.net
model.train(
    chunks(old_tr, STREAM_PER_STAGE) + chunks(new_tr, STREAM_PER_STAGE),
    i2k,
    beta=BETA,
    warmup_ratio=WARMUP,
    tolerance=TOLERANCE,
    max_u2i=MAX_U2I,
    max_i2u=MAX_I2U,
)
trait_after = old_user_traits(net)

# reference model: old-only (= after stage 1) for RD baseline
ref = make_icd()
ref.train(
    chunks(old_tr, STREAM_PER_STAGE),
    i2k,
    beta=BETA,
    warmup_ratio=WARMUP,
    tolerance=TOLERANCE,
    max_u2i=MAX_U2I,
    max_i2u=MAX_I2U,
)
trait_before = old_user_traits(ref.net)
RD = float(np.linalg.norm(trait_after - trait_before, axis=1).mean())
print(f"RD (old-user trait drift, L2): {RD:.6f}")

net.eval()
dev = next(net.parameters()).device
_thr = 0.5  # random_split / 默认：与主表其它方法一致
sup_te = None
if SPLIT_TAG == "user_split":
    # test 用户训练集不可见；用 train+support 建邻域，在 query 上评（与 G-NCDM user_split 一致）
    sup_te, qry_te = split_support_query(te)
    graph_df = pd.concat([tr, sup_te], ignore_index=True)
    u2i, i2u = user2items(graph_df), item2users(graph_df)
    old_eval = qry_te[qry_te.item_id < n_item_old]
    new_eval = qry_te[qry_te.item_id >= n_item_old]
    print(f"user_split eval: support={len(sup_te)} | query old={len(old_eval)} new={len(new_eval)}")
else:
    u2i, i2u = user2items(tr), item2users(tr)
    old_eval, new_eval = old_te, new_te


def predict_scores(df_subset):
    yt, yp = [], []
    data = transform(
        df_subset,
        u2i,
        i2u,
        i2k,
        KNOW_N,
        EVAL_BS,
        max_u2i=MAX_U2I,
        max_i2u=MAX_I2U,
        silent=True,
        allow_missing="skip",
    )
    with torch.no_grad():
        for uid, U, um, iid, I, im, IK, r in data:
            pred, *_ = net(U.to(dev), um.to(dev), I.to(dev), im.to(dev), IK.to(dev))
            yp.extend(pred.detach().cpu().tolist())
            yt.extend(r.tolist())
    return np.array(yt), np.array(yp)


def threshold_from_support(yt, yp):
    """在 support 上用 Youden J 选阈值（冷启动合法：support 评测时可见，不碰 query）。"""
    from sklearn.metrics import roc_curve

    if len(yt) == 0 or len(set(yt.tolist())) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(yt, yp)
    j = tpr - fpr
    return float(thr[int(j.argmax())])


if SPLIT_TAG == "user_split" and sup_te is not None:
    yt_s, yp_s = predict_scores(sup_te)
    _thr = threshold_from_support(yt_s, yp_s)
    acc_s05 = accuracy_score(yt_s, (yp_s >= 0.5).astype(int)) if len(yt_s) else float("nan")
    acc_st = accuracy_score(yt_s, (yp_s >= _thr).astype(int)) if len(yt_s) else float("nan")
    print(
        f"user_split threshold: Youden on support → t={_thr:.4f} "
        f"(support ACC@0.5={acc_s05:.4f} → ACC@t={acc_st:.4f}); "
        f"ACC/F1 on query use this t; AUC/RMSE unchanged"
    )


def eval_subset(df_subset, thr=_thr):
    yt, yp = predict_scores(df_subset)
    if len(yt) == 0:
        print("    [diag] n=0 (no evaluable rows — check user_split support/query graph)")
        nan = float("nan")
        return nan, nan, nan, nan, 0
    yl = (yp >= thr).astype(int)
    auc = roc_auc_score(yt, yp) if len(set(yt.tolist())) > 1 else float("nan")
    acc = accuracy_score(yt, yl)
    print(
        f"    [diag] n={len(yt)} pos_rate(true)={yt.mean():.4f} "
        f"pred: mean={yp.mean():.4f} std={yp.std():.4f} min={yp.min():.4f} max={yp.max():.4f} "
        f"| thr={thr:.4f} ACC={acc:.4f}"
    )
    return auc, mean_squared_error(yt, yp) ** 0.5, acc, f1_score(yt, yl), len(yt)


auc_o, rmse_o, acc_o, f1_o, n_o = eval_subset(old_eval)
auc_n, rmse_n, acc_n, f1_n, n_n = eval_subset(new_eval)

row = {
    "Method": "ICD",
    "AUC_old": auc_o,
    "AUC_new": auc_n,
    "RMSE_old": rmse_o,
    "RMSE_new": rmse_n,
    "ACC_old": acc_o,
    "ACC_new": acc_n,
    "F1_old": f1_o,
    "F1_new": f1_n,
    "RD": RD,
}
print(f"\n===== ICD (cdm=ncd) on {DATASET}_{SPLIT_TAG} — Approach A =====")
print(f"  old-test n={n_o}: AUC={auc_o:.4f} ACC={acc_o:.4f} F1={f1_o:.4f} RMSE={rmse_o:.4f}")
print(f"  new-test n={n_n}: AUC={auc_n:.4f} ACC={acc_n:.4f} F1={f1_n:.4f} RMSE={rmse_n:.4f}")
if SPLIT_TAG == "user_split":
    print(
        f"\n[NOTE] user_split ACC/F1 使用 support 上 Youden 阈值 t={_thr:.4f}（冷启动合法校准）；\n"
        f"  AUC/RMSE 与阈值无关。random_split 仍固定 t=0.5。写入主表时请在脚注说明此点。"
    )
out_csv = os.environ.get("ICD_OUTPUT_CSV", os.path.join(OUT, f"icd_row_{DATASET}_{SPLIT_TAG}.csv"))
out_csv = os.path.abspath(out_csv)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
pd.DataFrame([row]).to_csv(out_csv, index=False)
# ready-to-append line (TMD column = RD) for all_methods_{DATASET}_random_split.csv
print("\nappend to all_methods_%s_%s.csv (last col=TMD):" % (DATASET, SPLIT_TAG))
print(
    ",".join(
        str(row[k])
        for k in [
            "Method",
            "AUC_old",
            "AUC_new",
            "RMSE_old",
            "RMSE_new",
            "ACC_old",
            "ACC_new",
            "F1_old",
            "F1_new",
            "RD",
        ]
    )
)
print("row ->", out_csv)
