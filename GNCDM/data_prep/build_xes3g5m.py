# -*- coding: utf-8 -*-
"""Build XES3G5M -> CD 三元组 + 稀疏 Q（适配 GNCDM 增量管线），带学生子采样开关。

输入（XES3G5M 标准 pykt question-level 分发，放到 SRC_DIR）：
  train_valid_sequences_quelevel.csv  (列含 uid, questions, concepts, responses, selectmasks)
  test_quelevel.csv                    (同上)
  序列内：questions/responses/selectmasks 逗号分隔；concepts 逗号分隔、单题多 KC 用 "_" 连接；
  padding 位 selectmasks=-1（或 response<0）需丢弃。

输出（写到 OUT_DIR=GNCDM/data，命名照搬 nips34_*）：
  xes_{train,valid,test}_0.8_0.1_0.1.csv  列 user_id,item_id,score(0/1)
  xes_Q_matrix.npy                         (n_item, n_know) 多热稀疏 Q
  xes_user_map.csv / xes_item_map.csv / xes_concept_map.csv

★ 子采样开关：SUBSAMPLE_USERS
    None        -> 用全量学生（~18066，5.5M 交互，重）
    int N(如4000)-> 按学生随机抽 N 人（SUBSAMPLE_SEED 固定），把规模压到 a0910/NIPS34 档，
                    抽样后自动只保留被抽中学生触及的题与 KC 并重映射。
"""

import os
import json
import numpy as np
import pandas as pd

# ── 配置 ───────────────────────────────────────────────────────────────
SRC_DIR = r"D:\CD_continue\data\XES3G5M\question_level"  # quelevel csv 所在目录
OUT_DIR = r"D:\CD_continue\GNCDM\data"
PREFIX = "xes"
# ── 子采样开关（二选一；TARGET_INTERACTIONS 优先）─────────────────────────
TARGET_INTERACTIONS = 1_000_000  # 目标交互量；按学生累加到此值自动定人数。None=不按此模式
SUBSAMPLE_USERS = None  # 直接指定学生数；None=全量。仅当 TARGET_INTERACTIONS=None 时生效
SUBSAMPLE_SEED = 42
SPLIT_SEED = 42
DEDUP = "first"  # 每个 (user,item) 多次作答时取 'first'/'last'，CD 单作答口径


def _iter_quelevel(path):
    """逐行摊平 pykt question-level 序列 -> (uid, qid, resp, kc_token) 生成器。"""
    df = pd.read_csv(path, usecols=lambda c: c in ("uid", "questions", "concepts", "responses", "selectmasks"))
    has_mask = "selectmasks" in df.columns
    cols = zip(
        df["uid"].to_numpy(),
        df["questions"].astype(str).to_numpy(),
        df["concepts"].astype(str).to_numpy(),
        df["responses"].astype(str).to_numpy(),
        (df["selectmasks"].astype(str).to_numpy() if has_mask else [None] * len(df)),
    )
    for uid_v, q_str, c_str, r_str, m_str in cols:
        uid = int(uid_v)
        qs = q_str.split(",")
        rs = r_str.split(",")
        cs = c_str.split(",")
        ms = m_str.split(",") if m_str is not None else ["1"] * len(qs)
        for q, resp, c, m in zip(qs, rs, cs, ms):
            if m.strip() in ("-1", "") or resp.strip() in ("-1", ""):
                continue  # padding
            try:
                qid, y = int(q), int(resp)
            except ValueError:
                continue
            if y not in (0, 1):
                continue
            yield uid, qid, y, c.strip()


def main():
    # 1) 摊平 train+test 全部交互
    recs, q2kc = [], {}
    for fn in ("train_valid_sequences_quelevel.csv", "test_quelevel.csv"):
        path = os.path.join(SRC_DIR, fn)
        if not os.path.isfile(path):
            print(f"⚠️ 缺文件 {path}，跳过")
            continue
        for uid, qid, y, kc_tok in _iter_quelevel(path):
            recs.append((uid, qid, y))
            if qid not in q2kc and kc_tok not in ("", "-1"):
                q2kc[qid] = [int(k) for k in kc_tok.split("_") if k not in ("", "-1")]
    df = pd.DataFrame(recs, columns=["uid", "qid", "score"]).drop_duplicates(
        ["uid", "qid"], keep=DEDUP
    )
    print(f"flatten: {len(df)} 交互 | 学生={df.uid.nunique()} 题={df.qid.nunique()}")

    # 2) 子采样（按学生）—— TARGET_INTERACTIONS 优先：累加学生直到达目标交互量
    if TARGET_INTERACTIONS is not None and len(df) > TARGET_INTERACTIONS:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        per_user = df.groupby("uid").size()
        order = rng.permutation(per_user.index.to_numpy())
        cum = per_user.loc[order].cumsum().to_numpy()
        n_keep = int(np.searchsorted(cum, TARGET_INTERACTIONS) + 1)
        keep = order[:n_keep]
        df = df[df.uid.isin(keep)].copy()
        print(
            f"subsample(target≈{TARGET_INTERACTIONS}) -> {n_keep} 学生：{len(df)} 交互、题={df.qid.nunique()}"
        )
    elif SUBSAMPLE_USERS is not None and SUBSAMPLE_USERS < df.uid.nunique():
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        keep = rng.choice(df.uid.unique(), size=SUBSAMPLE_USERS, replace=False)
        df = df[df.uid.isin(keep)].copy()
        print(f"subsample -> {SUBSAMPLE_USERS} 学生：{len(df)} 交互、题={df.qid.nunique()}")

    # 3) 连续重映射（只保留出现过的 user/item/kc）
    users = sorted(df.uid.unique())
    items = sorted(df.qid.unique())
    kcs = sorted({k for q in items for k in q2kc.get(q, [])})
    u_map = {o: i for i, o in enumerate(users)}
    i_map = {o: i for i, o in enumerate(items)}
    k_map = {o: i for i, o in enumerate(kcs)}
    n_user, n_item, n_know = len(users), len(items), len(kcs)
    df["user_id"] = df.uid.map(u_map)
    df["item_id"] = df.qid.map(i_map)
    df = df[["user_id", "item_id", "score"]].reset_index(drop=True)
    print(f"dims: n_user={n_user} n_item={n_item} n_know={n_know} pos_rate={df.score.mean():.4f}")

    # 4) 稀疏 Q（多热）
    Q = np.zeros((n_item, n_know), dtype=np.float32)
    for q in items:
        for k in q2kc.get(q, []):
            Q[i_map[q], k_map[k]] = 1.0
    miss = int((Q.sum(1) == 0).sum())
    print(f"Q: {Q.shape} mean KC/题={Q.sum(1).mean():.3f} 无KC题={miss}")

    # 5) per-user 分层随机 0.8/0.1/0.1（保证每个学生进 train）
    rng = np.random.default_rng(SPLIT_SEED)
    parts = {"train": [], "valid": [], "test": []}
    for _, g in df.groupby("user_id"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(n * 0.8)))
        n_va = min(int(round(n * 0.1)), n - n_tr)
        parts["train"].append(idx[:n_tr])
        parts["valid"].append(idx[n_tr : n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va :])
    os.makedirs(OUT_DIR, exist_ok=True)
    for k in parts:
        v = df.loc[np.concatenate(parts[k])].sample(frac=1, random_state=SPLIT_SEED).reset_index(
            drop=True
        )
        v.to_csv(os.path.join(OUT_DIR, f"{PREFIX}_{k}_0.8_0.1_0.1.csv"), index=False)
        print(f"{k}: {len(v)} 行 users={v.user_id.nunique()} items={v.item_id.nunique()}")

    np.save(os.path.join(OUT_DIR, f"{PREFIX}_Q_matrix.npy"), Q)
    pd.DataFrame({"orig_uid": users, "user_id": range(n_user)}).to_csv(
        os.path.join(OUT_DIR, f"{PREFIX}_user_map.csv"), index=False
    )
    pd.DataFrame({"orig_qid": items, "item_id": range(n_item)}).to_csv(
        os.path.join(OUT_DIR, f"{PREFIX}_item_map.csv"), index=False
    )
    pd.DataFrame({"orig_kc": kcs, "concept_id": range(n_know)}).to_csv(
        os.path.join(OUT_DIR, f"{PREFIX}_concept_map.csv"), index=False
    )
    print(f"\nSaved {PREFIX}_* -> {OUT_DIR}  (SUBSAMPLE_USERS={SUBSAMPLE_USERS})")


if __name__ == "__main__":
    main()
