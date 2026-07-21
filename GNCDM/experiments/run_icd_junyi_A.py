"""ICD baseline on junyi random_split — Approach A (old/new split).

junyi dense version (dims read from files: ~1000 users x 712 items x 39 concepts). Aligned
column-wise to all_methods_junyi_random_split.csv: same strict_bipartition + auto_new_concepts(Q, 0.34),
ICD official hparams (cdm=ncd, alpha=0.2, tolerance=0.2, beta=0.9, epoch=1, warmup=0.1),
neighborhood caps max_u2i=128/max_i2u=64. Protocol: one stream old->new (stage1 old items
then stage2 new items), final one-shot eval on OLD-item test and NEW-item test separately;
RD = old-user NCD-trait L2 drift (old-only reference model = state after stage 1).

ENV (isolated venv; see GNCDM/docs/ICD_baseline_repro.md):
  pip install torch EduCDM  (+ longling baize fire); any pandas — this script monkeypatches
  EduCDM's groupby(['col']) pandas>=2.2 tuple-key bug at runtime, so NO source edit needed.

Usage:
  python run_icd_a0910_A.py [DATA_DIR] [CTX] [STREAM_PER_STAGE]
    DATA_DIR         dir containing Q_matrix.npy and new_random_split/{train,valid,test}.csv
                     (default: <repo>/data/a0910)
    CTX              'cuda:0' (default if available) or 'cpu'
    STREAM_PER_STAGE chunks per stage (default 25)
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
DATASET = "junyi"
_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", DATASET)
DATA_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(_DEFAULT_DATA)
CTX = sys.argv[2] if len(sys.argv) > 2 else ("cuda:0" if torch.cuda.is_available() else "cpu")
STREAM_PER_STAGE = int(sys.argv[3]) if len(sys.argv) > 3 else 25

NEW_ITEM_FRAC = 0.34
ALPHA, TOLERANCE, BETA, WARMUP, EPOCH = 0.2, 0.2, 0.9, 0.1, 1  # official ICD main() defaults
MAX_U2I, MAX_I2U = 128, 64  # neighborhood caps (dense junyi: ~204 answers/user)
EVAL_BS = 256
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


# ---- load + bipartition ----
rnd = os.path.join(DATA_DIR, "new_random_split")
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
u2i, i2u = user2items(tr), item2users(tr)


def eval_subset(df_subset):
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
    yt, yp = np.array(yt), np.array(yp)
    yl = (yp >= 0.5).astype(int)
    print(
        f"    [diag] n={len(yt)} pos_rate(true)={yt.mean():.4f} "
        f"pred: mean={yp.mean():.4f} std={yp.std():.4f} min={yp.min():.4f} max={yp.max():.4f}"
    )
    auc = roc_auc_score(yt, yp) if len(set(yt.tolist())) > 1 else float("nan")
    return auc, mean_squared_error(yt, yp) ** 0.5, accuracy_score(yt, yl), f1_score(yt, yl), len(yt)


auc_o, rmse_o, acc_o, f1_o, n_o = eval_subset(old_te)
auc_n, rmse_n, acc_n, f1_n, n_n = eval_subset(new_te)

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
print(f"\n===== ICD (cdm=ncd) on {DATASET}_random_split — Approach A =====")
print(f"  old-test n={n_o}: AUC={auc_o:.4f} ACC={acc_o:.4f} F1={f1_o:.4f} RMSE={rmse_o:.4f}")
print(f"  new-test n={n_n}: AUC={auc_n:.4f} ACC={acc_n:.4f} F1={f1_n:.4f} RMSE={rmse_n:.4f}")
out_csv = os.environ.get("ICD_OUTPUT_CSV", os.path.join(OUT, f"icd_row_{DATASET}_random_split.csv"))
os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
pd.DataFrame([row]).to_csv(out_csv, index=False)
# ready-to-append line (TMD column = RD) for all_methods_{DATASET}_random_split.csv
print("\nappend to all_methods_%s_random_split.csv (last col=TMD):" % DATASET)
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
