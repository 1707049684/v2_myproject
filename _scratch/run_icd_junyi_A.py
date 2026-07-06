"""ICD baseline on junyi random_split — Approach A (old/new split), GPU-enabled.

GPU fix: ICD is initialized with ctx='cpu' (avoids DataParallel complications),
then the model is explicitly moved to the target device. A monkey-patch replaces
dual_fit_f and eval_f with device-aware versions that move every batch tensor to
the device before forward(). This guarantees GPU utilisation regardless of baize/
DataParallel behaviour.

Protocol: one stream old->new (stage1 old items then stage2 new items), final
one-shot eval on OLD-item test and NEW-item test separately; RD = old-user
NCD-trait L2 drift (old-only reference model = state after stage 1).

Usage:
  python run_icd_a0910_A.py [DATA_DIR] [CTX] [STREAM_PER_STAGE]
    DATA_DIR         dir containing Q_matrix.npy and new_random_split/{train,valid,test}.csv
                     (default: ../data/a0910 relative to this script)
    CTX              'cuda:0' (default if GPU available) or 'cpu'
    STREAM_PER_STAGE chunks per stage (default 25)
"""

import functools
import logging
import os
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
MAX_U2I, MAX_I2U = 128, 64  # neighborhood caps (junyi dense: ~204 answers/user)
EVAL_BS = 256

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"icd_out_{DATASET}")
os.makedirs(OUT, exist_ok=True)
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(f"icd_{DATASET}")
DEV = torch.device(CTX)
print(f"[{DATASET}] device = {DEV}")


# ===========================================================================
# GPU patch: must be applied BEFORE importing ICD (replaces module-level refs)
# ===========================================================================
def _gpu_patch(dev: torch.device):
    """Monkey-patch EduCDM's dual_fit_f and eval_f to move batch tensors to dev."""
    import EduCDM.ICD.ICD as _icd_mod
    import EduCDM.ICD.sym.fit_eval as _fe
    from baize.metrics import POrderedDict, classification_report
    from baize.torch import fit_wrapper
    from EduCDM.ICD.metrics import doa_report

    # ---- patched dual_fit_f: moves ALL batch tensors to dev ----
    def _dual_fit_gpu(_net, batch_data, loss_function, *args, **kwargs):
        batch_data = tuple(x.to(dev) if isinstance(x, torch.Tensor) else x for x in batch_data)
        _, u2i, u_mask, _, i2u, i_mask, i2k, r = batch_data
        out, theta, a, b, stat_theta, stat_a, stat_b = _net(u2i, u_mask, i2u, i_mask, i2k)
        loss_function["BCE"](out, r)
        loss_function["DTL"](theta, a, b, stat_theta, stat_a, stat_b)
        return loss_function["Loss"](out, r, theta, a, b, stat_theta, stat_a, stat_b)

    _icd_mod.dual_fit_f = fit_wrapper(_dual_fit_gpu)

    # ---- patched eval_f: moves batch tensors to dev, used by turning_point ----
    def _eval_gpu_inner(_net, test_data, *args, **kwargs):
        y_true, y_pred, y_label = [], [], []
        user_id, item_id, user_theta, item_knowledge = [], [], [], []
        for uid, u2i, u_mask, iid, i2u, i_mask, i2k, r in test_data:
            pred, theta, *_ = _net(
                u2i.to(dev),
                u_mask.to(dev),
                i2u.to(dev),
                i_mask.to(dev),
                i2k.to(dev),
            )
            pred_cpu = pred.detach().cpu()
            y_pred.extend(pred_cpu.tolist())
            y_label.extend([0 if p < 0.5 else 1 for p in pred_cpu])
            y_true.extend(r.tolist())
            user_id.extend(uid.tolist())
            item_id.extend(iid.tolist())
            user_theta.extend(theta.detach().cpu().tolist())
            item_knowledge.extend(i2k.cpu().tolist())
        try:
            if not y_true:
                raise ValueError()
            ret = classification_report(y_true, y_label, y_pred)
        except ValueError:
            ret = POrderedDict()
        ret.update(doa_report(user_id, item_id, item_knowledge, y_true, user_theta))
        return ret

    def _eval_wrapper(fn):
        @functools.wraps(fn)
        def wrapped(_net, *args, **kwargs):
            _net.eval()
            result = fn(_net, *args, **kwargs)
            _net.train()
            return result

        return wrapped

    patched_eval_f = _eval_wrapper(_eval_gpu_inner)
    _icd_mod.eval_f = patched_eval_f
    _fe.eval_f = patched_eval_f


_gpu_patch(DEV)

# ---- pandas groupby patch (pandas>=2.2 tuple-key bug) ----
import EduCDM.ICD.ICD as _icd_mod  # noqa: E402
import EduCDM.ICD.etl as _etl  # noqa: E402


def _u2i(df, d2=None):
    ul = {}
    if d2:
        d2.u2i = ul
    for uid, g in df.groupby("user_id"):
        if d2:
            d2.add_user_items_responses(uid, g["item_id"], g["score"])
        ul[uid] = [int(x) for x in (g["item_id"] * 2 + g["score"] + 1)]
    return ul


def _i2u(df, d2=None):
    il = {}
    for iid, g in df.groupby("item_id"):
        if d2:
            d2.add_item_users_responses(iid, g["user_id"], g["score"])
        il[iid] = [int(x) for x in (g["user_id"] * 2 + g["score"] + 1)]
    return il


_etl.user2items = _icd_mod.user2items = _u2i
_etl.item2users = _icd_mod.item2users = _i2u

from EduCDM.ICD.ICD import ICD  # noqa: E402
from EduCDM.ICD.etl import dict_etl, inc_stream, item2knowledge, transform  # noqa: E402

user2items, item2users = _u2i, _i2u


# ---- project-aligned bipartition ----
def auto_new_concepts(Q, frac=0.34):
    freq = (Q > 0).sum(axis=0)
    touched = np.zeros(Q.shape[0], dtype=bool)
    new_set = []
    for k in np.argsort(freq):
        new_set.append(int(k))
        touched |= Q[:, k] > 0
        if touched.sum() >= frac * Q.shape[0]:
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
print(f"dims user={USER_N} item={ITEM_N} know={KNOW_N}")
print(
    f"bipartition: new_concepts={len(new_concepts)}/{KNOW_N}, "
    f"old items={n_item_old} new={ITEM_N - n_item_old}, old concepts={n_know_old}"
)

item_csv = os.path.join(OUT, "item.csv")
pd.DataFrame(
    [
        {"item_id": i, "knowledge_code": str([int(k) + 1 for k in np.where(Q_re[i] > 0)[0]])}
        for i in range(ITEM_N)
    ]
).to_csv(item_csv, index=False)
i2k = item2knowledge(item_csv)

old_tr = tr[tr.item_id < n_item_old]
new_tr = tr[tr.item_id >= n_item_old]
old_te = te[te.item_id < n_item_old]
new_te = te[te.item_id >= n_item_old]
print(f"train old/new rows: {len(old_tr)}/{len(new_tr)}; test old/new: {len(old_te)}/{len(new_te)}")


def chunks(df, n):
    return list(inc_stream(df, max(1, int(len(df) // n))))


def make_icd():
    # ctx='cpu': avoid set_device/DataParallel; we move to DEV explicitly below
    m = ICD(
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
    m.net.to(DEV)
    m.dual_net.to(DEV)
    m.cfg.ctx = CTX  # so get_dual_loss creates loss tensors on DEV
    return m


old_users = sorted(old_tr.user_id.unique().tolist())
u2i_old = user2items(old_tr)


def old_user_traits(_net):
    _net.eval()
    out = _net.get_user_profiles(dict_etl(old_users, u2i_old, batch_size=EVAL_BS))
    _net.train()
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
            pred, *_ = net(U.to(DEV), um.to(DEV), I.to(DEV), im.to(DEV), IK.to(DEV))
            yp.extend(pred.detach().cpu().tolist())
            yt.extend(r.tolist())
    if not yt:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    yt, yp = np.array(yt), np.array(yp)
    yl = (yp >= 0.5).astype(int)
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
out_csv = os.path.join(OUT, f"icd_row_{DATASET}_random_split.csv")
pd.DataFrame([row]).to_csv(out_csv, index=False)
print("\nappend to all_methods_%s_random_split.csv (last col=TMD/RD):" % DATASET)
cols = [
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
print(",".join(str(row[k]) for k in cols))
print("row ->", out_csv)
