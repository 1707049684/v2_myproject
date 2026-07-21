"""Approach A: ICD as an incremental baseline aligned to the project's old/new口径.

Protocol (user-specified):
  Stage 1: train ICD on OLD-item interactions (stream)
  Stage 2: train ICD on NEW-item interactions (stream)  <- ICD's incremental step
  Final eval (one-shot): evaluate the final model on OLD-item test and NEW-item test
                         SEPARATELY -> AUC/RMSE/ACC/F1 _old and _new.

Same strict_bipartition as run_incremental_math1_random_split.py (NEW_CONCEPTS=[0,1,3,6],
13 old / 7 new items, 7 old / 4 new concepts) so the row aligns column-wise with
all_methods_math1_random_split.csv. Official ICD hparams (alpha=0.2, tolerance=0.2,
epoch=1, beta=0.9, warmup=0.1). Metrics via sklearn to match the project's definitions.
"""

import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score


def _patch_groupby():
    """Keep EduCDM's interaction transforms compatible with pandas >= 2.2."""

    import EduCDM.ICD.ICD as _icd
    import EduCDM.ICD.etl as _etl

    def user2items(df, dict2=None):
        users = {}
        if dict2:
            dict2.u2i = users
        for uid, group in df.groupby("user_id"):
            if dict2:
                dict2.add_user_items_responses(uid, group["item_id"], group["score"])
            users[uid] = [int(value) for value in (group["item_id"] * 2 + group["score"] + 1)]
        return users

    def item2users(df, dict2=None):
        items = {}
        for iid, group in df.groupby("item_id"):
            if dict2:
                dict2.add_item_users_responses(iid, group["user_id"], group["score"])
            items[iid] = [int(value) for value in (group["user_id"] * 2 + group["score"] + 1)]
        return items

    _etl.user2items = _icd.user2items = user2items
    _etl.item2users = _icd.item2users = item2users
    return user2items, item2users


user2items, item2users = _patch_groupby()
from EduCDM.ICD.ICD import ICD  # noqa: E402
from EduCDM.ICD.etl import dict_etl, inc_stream, transform  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "icd_out_A")
os.makedirs(OUT, exist_ok=True)
# Resolve before os.chdir(OUT): a relative ICD_OUTPUT_CSV would otherwise land under OUT/.
if os.environ.get("ICD_OUTPUT_CSV"):
    os.environ["ICD_OUTPUT_CSV"] = os.path.abspath(os.environ["ICD_OUTPUT_CSV"])
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("icd_A")

NEW_CONCEPTS = [0, 1, 3, 6]
USER_N, ITEM_N, KNOW_N = 4209, 20, 11
ALPHA, TOLERANCE, BETA, WARMUP, EPOCH, STREAM_PER_STAGE = 0.2, 0.2, 0.9, 0.1, 1, 25
SEED = int(os.environ.get("ICD_TRAIN_SEED", "0"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# --- copied verbatim from experiments/_core/run_incremental_math1.py (pure numpy/pandas) ---
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


# --- build bipartitioned data + ICD item.csv ---
Q = np.load(os.path.join(DATA, "math1_Q_matrix.npy"))
Q_re, item_id_map, n_item_old, n_know_old = strict_bipartition(Q, NEW_CONCEPTS)
print(
    f"strict_bipartition: old items={n_item_old} new={ITEM_N - n_item_old}, "
    f"old concepts={n_know_old} new={KNOW_N - n_know_old}"
)

tr = remap_items(pd.read_csv(os.path.join(DATA, "math1_train_0.8_0.2.csv")), item_id_map)
te = remap_items(pd.read_csv(os.path.join(DATA, "math1_test_0.8_0.2.csv")), item_id_map)

item_csv = os.path.join(OUT, "item.csv")
pd.DataFrame(
    [
        {"item_id": i, "knowledge_code": str([int(k) + 1 for k in np.where(Q_re[i] > 0)[0]])}
        for i in range(ITEM_N)
    ]
).to_csv(item_csv, index=False)
from EduCDM.ICD.etl import item2knowledge

i2k = item2knowledge(item_csv)

# split by remapped item_id: old = [0, n_item_old), new = [n_item_old, ITEM_N)
old_tr, new_tr = tr[tr.item_id < n_item_old], tr[tr.item_id >= n_item_old]
old_te, new_te = te[te.item_id < n_item_old], te[te.item_id >= n_item_old]
print(
    f"train old/new rows: {len(old_tr)}/{len(new_tr)}; test old/new rows: {len(old_te)}/{len(new_te)}"
)


def chunks(df, n):
    return list(inc_stream(df, max(1, int(len(df) // n))))


# Stage 1 (old) then Stage 2 (new) is modeled as ONE stream ordered old->new (ICD manages
# warmup/turning-points/dual-momentum across it; splitting into two train() calls collapses it).
old_chunks = chunks(old_tr, STREAM_PER_STAGE)
new_chunks = chunks(new_tr, STREAM_PER_STAGE)
print(f"stream: stage1 old={len(old_chunks)} chunks, stage2 new={len(new_chunks)} chunks")

old_users = sorted(old_tr.user_id.unique().tolist())
u2i_old = user2items(old_tr)


def old_user_traits(_net):
    out = _net.get_user_profiles(dict_etl(old_users, u2i_old, batch_size=256))
    return out["u_trait"].detach().cpu().numpy()


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
        ctx="cpu",
    )


os.chdir(OUT)

# main model: single stream old->new (= stage1 then stage2); used for eval + "after new" snapshot
model = make_icd()
net = model.net
model.train(old_chunks + new_chunks, i2k, beta=BETA, warmup_ratio=WARMUP, tolerance=TOLERANCE)
trait_after = old_user_traits(net)

# reference model: old-only (= state after stage 1) as the RD baseline (project baseline_tmd b0 口径)
ref = make_icd()
ref.train(old_chunks, i2k, beta=BETA, warmup_ratio=WARMUP, tolerance=TOLERANCE)
trait_before = old_user_traits(ref.net)

# RD = mean L2 drift of OLD-user traits caused by the new-item stage
# (same spirit as project baseline_tmd; embedding/trait space, NOT comparable to Ours θ-RD)
RD = float(np.linalg.norm(trait_after - trait_before, axis=1).mean())
print(f"RD (old-user trait drift, L2): {RD:.6f}")

net.eval()

# post-hoc eval: dict2 neighborhoods from FULL train (old+new)
u2i = user2items(tr)
i2u = item2users(tr)


def eval_subset(df_subset):
    yt, yp = [], []
    data = transform(
        df_subset, u2i, i2u, i2k, KNOW_N, batch_size=256, silent=True, allow_missing="skip"
    )
    with torch.no_grad():
        for uid, U, um, iid, I, im, IK, r in data:
            pred, *_ = net(U, um, I, im, IK)
            yp.extend(pred.tolist())
            yt.extend(r.tolist())
    yt, yp = np.array(yt), np.array(yp)
    yl = (yp >= 0.5).astype(int)
    auc = roc_auc_score(yt, yp) if len(set(yt.tolist())) > 1 else float("nan")
    return auc, mean_squared_error(yt, yp) ** 0.5, accuracy_score(yt, yl), f1_score(yt, yl), len(yt)


auc_o, rmse_o, acc_o, f1_o, n_o = eval_subset(old_te)
auc_n, rmse_n, acc_n, f1_n, n_n = eval_subset(new_te)

print("\n===== ICD (cdm=ncd) on math1_random_split — Approach A (old/new split) =====")
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
print(f"  old-test n={n_o}: AUC={auc_o:.4f} ACC={acc_o:.4f} F1={f1_o:.4f} RMSE={rmse_o:.4f}")
print(f"  new-test n={n_n}: AUC={auc_n:.4f} ACC={acc_n:.4f} F1={f1_n:.4f} RMSE={rmse_n:.4f}")
out_csv = os.environ.get("ICD_OUTPUT_CSV", os.path.join(OUT, "icd_row_math1_random_split.csv"))
os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
pd.DataFrame([row]).to_csv(out_csv, index=False)
print("row ->", out_csv)
