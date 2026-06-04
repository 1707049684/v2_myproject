# -*- coding: utf-8 -*-
"""
增量学习主实验：6 策略对比，Math1，严格拓扑二分（13 旧题 + 7 新题，ΔK={0,1,3,6}）。
random_split（预测口径）+ user_split（重构口径）两个划分各跑一遍。

设计要点：
1. 训练/评测**喂真实作答日志**（diagnose_theta/psi 需要作答向量，否则 θ/ψ 恒定、不区分学习者）。
2. random_split 评测走 **forward_using_buf**：训练后用训练集作答填充 Theta_buf/Psi_buf，
   查 buffer 预测 → **无信息泄漏**（论文 RQ2 score prediction，test 用户与训练共享）。
3. user_split 评测走 **forward 重构**：test/valid 用户互斥，喂其自身作答现场诊断
   （论文 RQ1 score reconstruction）。base 用旧题空间(13)、策略用扩展空间(20)的评测 log。
4. 输入作答向量在增量阶段跨完整 item 空间（用 df_train 全量构建），新旧分支都拿到真实作答。

策略：Base / Ours-Ablated / Ours(DNA) / Ours(LoRA) / Full-Replay-Oracle / Naive-FT。
结果写 incremental_result/incremental_results_{random_split,user_split}.csv。
"""
import math
import os
import sys

gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

from core.model import GNCDM
from core.train import IDCDataset, calculate_tmd

DATA_DIR = os.path.join(gncdm_dir, "data")
SAVE_DIR = os.path.join(gncdm_dir, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SampleSet(Dataset):
    """只提供 (user_id, item_id, score) 整数三元组；真实作答日志由 full_log_mat 索引得到。"""

    def __init__(self, df):
        self.u = df["user_id"].values.astype(np.int64)
        self.i = df["item_id"].values.astype(np.int64)
        self.s = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, k):
        return self.u[k], self.i[k], self.s[k]


def build_log_mat(df, n_user, n_item):
    """构建 (n_user, n_item) 作答矩阵，值域 {-1,0,1}（与 core/train.py 的 IDCDataset 一致）。"""
    return IDCDataset(df, n_user, n_item).log_mat.astype(np.float32)


def strict_bipartition(Q, new_concepts):
    """严格拓扑二分 + 重排，保证旧题绝不依赖新概念。

    规则：new_concepts 为新知识集 ΔK；旧题=Q 向量在 ΔK 上全 0 的题；新题=触及任一 ΔK 的题。
    重排后：旧概念列在前、新概念列在后；旧题行在前、新题行在后。
    返回 (Q_reordered, item_id_map, n_item_old, n_know_old)。
    """
    K = Q.shape[1]
    new_concepts = list(new_concepts)
    old_concepts = [k for k in range(K) if k not in new_concepts]
    concept_perm = old_concepts + new_concepts  # 旧概念在前

    touches_new = Q[:, new_concepts].sum(axis=1) > 0
    old_items = np.where(~touches_new)[0].tolist()
    new_items = np.where(touches_new)[0].tolist()
    item_perm = old_items + new_items  # 旧题在前

    Q_re = Q[np.ix_(item_perm, concept_perm)].astype(np.float32)
    item_id_map = {old: new for new, old in enumerate(item_perm)}
    return Q_re, item_id_map, len(old_items), len(old_concepts)


def remap_items(df, item_id_map):
    df = df.copy()
    df["item_id"] = df["item_id"].map(item_id_map).astype(int)
    return df


def make_hybrid(train_new, train_old, ratio_old_to_new, seed=42):
    """非对称混态数据流：以 train_new 为主，按 old:new = 1:ratio 采样 train_old 补充。
    例如 ratio_old_to_new=3 -> old 样本数 = |new|/3（old:new=1:3）。"""
    n_new = len(train_new)
    n_old = max(1, n_new // ratio_old_to_new)
    old_s = train_old.sample(n=min(n_old, len(train_old)), random_state=seed)
    mixed = pd.concat([train_new, old_s], ignore_index=True)
    return mixed


def train_real(model, samples_df, full_log_mat, params, device,
               batch_size=256, lr=1e-3, n_epoch=10, desc="train",
               valid_df=None, buffer_log=None, select_metric="acc", eval_fn=None):
    """喂真实作答日志训练。params=要优化的参数列表（增量策略只传 'new' 参数）。

    模型选优（保留验证集 select_metric 最高的快照）支持两种口径：
    - eval_fn 给定：每 epoch 调 eval_fn(model)->metrics（用于 user-split 重构评测）。
    - 否则若 valid_df+buffer_log 给定：buffer 无泄漏评测（random-split 预测口径）。
    """
    loader = DataLoader(SampleSet(samples_df), batch_size=batch_size, shuffle=True)
    log_t = torch.tensor(full_log_mat, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(params, lr=lr)
    do_select = eval_fn is not None or (valid_df is not None and buffer_log is not None)
    best_metric, best_state = -1.0, None
    for epoch in range(n_epoch):
        model.train()
        total = 0.0
        for user_ids, item_ids, score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            score = score.to(device).unsqueeze(1)
            user_log = log_t[user_ids]            # (B, n_item) 真实作答
            item_log = log_t[:, item_ids].T       # (B, n_user) 真实作答
            pred = model(user_log, item_log, user_ids, item_ids)
            loss = F.binary_cross_entropy(pred, score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if do_select:
            if eval_fn is not None:
                vr = eval_fn(model)
            else:
                populate_buffers(model, buffer_log, device)
                vr = evaluate_buf(model, valid_df, device)
            vm = vr[select_metric]
            if vm > best_metric:
                best_metric = vm
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    [{desc}] epoch {epoch + 1}/{n_epoch} loss={total / len(loader):.4f} "
                  f"valid_{select_metric}={vm:.4f} best={best_metric:.4f}")
        else:
            print(f"    [{desc}] epoch {epoch + 1}/{n_epoch} loss={total / len(loader):.4f}")

    if do_select and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def populate_buffers(model, log_mat, device, batch_size=256):
    """用训练集作答填充 Theta_buf / Psi_buf（论文 offline diagnosis 口径）。"""
    model.eval()
    log_t = torch.tensor(log_mat, dtype=torch.float32, device=device)
    n_user, n_item = log_t.shape
    with torch.no_grad():
        for i in range(math.ceil(n_user / batch_size)):
            idx = np.arange(i * batch_size, min(n_user, (i + 1) * batch_size))
            theta = model.diagnose_theta(log_t[idx, :])
            model.update_Theta_buf(theta.detach(), torch.LongTensor(idx).to(device))
        for i in range(math.ceil(n_item / batch_size)):
            idx = np.arange(i * batch_size, min(n_item, (i + 1) * batch_size))
            psi = model.diagnose_psi(log_t[:, idx].T)
            model.update_Psi_buf(psi.detach(), torch.LongTensor(idx).to(device))


def evaluate_buf(model, test_df, device, batch_size=256):
    """无泄漏评测：直接用 buffer 里的 θ/ψ 预测（forward_using_buf）。"""
    model.eval()
    loader = DataLoader(SampleSet(test_df), batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for user_ids, item_ids, score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            pred = model.forward_using_buf(user_ids, item_ids)
            preds.extend(pred.cpu().numpy().reshape(-1).tolist())
            labels.extend(score.numpy().reshape(-1).tolist())
    preds, labels = np.array(preds), np.array(labels)
    pb = (preds >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "acc": accuracy_score(labels, pb),
        "f1": f1_score(labels, pb),
    }


def evaluate_recon(model, eval_df, eval_log_mat, device, batch_size=256):
    """重构评测（user-split 口径）：user_log 与 item_log 同取评测 split 的 log_mat。
    对标论文 RQ1 / 原作者 IDCDataset(eval_data) 的 score reconstruction。"""
    model.eval()
    log_t = torch.tensor(eval_log_mat, dtype=torch.float32, device=device)
    loader = DataLoader(SampleSet(eval_df), batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for user_ids, item_ids, score in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_log = log_t[user_ids]
            item_log = log_t[:, item_ids].T
            pred = model(user_log, item_log, user_ids, item_ids)
            preds.extend(pred.cpu().numpy().reshape(-1).tolist())
            labels.extend(score.numpy().reshape(-1).tolist())
    preds, labels = np.array(preds), np.array(labels)
    pb = (preds >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, preds),
        "rmse": np.sqrt(mean_squared_error(labels, preds)),
        "acc": accuracy_score(labels, pb),
        "f1": f1_score(labels, pb),
    }


def new_params(model):
    return [p for n, p in model.named_parameters() if "new" in n.lower()]


def lora_params(model):
    """LoRA 策略训练的参数：所有低秩矩阵（A_*/B_*），含诊断分支(A_new_f/B_new_f/A_new_g/B_new_g)
    和新概念聚合(A_theta_agg/B_theta_agg/A_psi_agg/B_psi_agg)。
    自动排除整张量 theta_agg_mat/psi_agg_mat（其旧列走旧路，须冻结以保旧）。"""
    return [p for n, p in model.named_parameters() if n.startswith("A_") or n.startswith("B_")]


def fresh_base(base_model):
    """从已训练 base 克隆一个新模型，避免策略间互相污染。"""
    m = GNCDM(n_user=base_model.n_user, n_item=base_model.n_item, n_know=base_model.n_know,
              user_dim=base_model.user_dim, item_dim=base_model.item_dim, alpha=base_model.alpha,
              Q_mat=base_model.Q_mat.cpu().numpy(), monotonicity_assumption=True, device=base_model.device)
    m.load_state_dict(base_model.state_dict())
    return m.to(base_model.device)


def run_experiment(split_name, mode, train_path, valid_path, test_path, Q_path, device,
                   n_user, n_item_total, n_know_total, new_concepts, alpha=0.8):
    """在一个数据划分上跑 6 策略。数据集维度/新概念由参数传入（math1 与 a0910 共用）。
    mode='buf'：random-split → forward_using_buf 无泄漏预测口径（论文 RQ2）。
    mode='recon'：user-split → forward 重构口径，test/valid 用户互斥（论文 RQ1）。
    new_concepts：作为「新知识」ΔK 的概念列索引；严格拓扑二分保证旧题不依赖这些概念。
    """
    print(f"\n{'#' * 70}\n# {split_name}  (mode={mode})\n{'#' * 70}")
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)
    Q_mat = np.load(Q_path)

    # 严格拓扑二分：new_concepts 为新概念 ΔK，重排后旧题/旧概念在前。
    NEW_CONCEPTS = list(new_concepts)
    Q_mat, item_id_map, n_item_old, n_know_old = strict_bipartition(Q_mat, NEW_CONCEPTS)
    df_train = remap_items(df_train, item_id_map)
    df_valid = remap_items(df_valid, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    n_item_new, n_know_new = n_item_total - n_item_old, n_know_total - n_know_old
    Q_old = Q_mat[:n_item_old, :n_know_old].copy()
    Q_expanded = Q_mat.copy()
    assert Q_mat[:n_item_old, n_know_old:].sum() == 0, "旧题依赖了新概念，二分失败！"
    print(f"严格拓扑二分 dK={NEW_CONCEPTS}: 旧题={n_item_old} 新题={n_item_new}, "
          f"旧概念={n_know_old} 新概念={n_know_new}（旧题不依赖新概念）")

    train_old = df_train[df_train["item_id"] < n_item_old].copy()
    train_new = df_train[df_train["item_id"] >= n_item_old].copy()
    test_old = df_test[df_test["item_id"] < n_item_old].copy()
    test_new = df_test[df_test["item_id"] >= n_item_old].copy()
    valid_old = df_valid[df_valid["item_id"] < n_item_old].copy()
    valid_new = df_valid[df_valid["item_id"] >= n_item_old].copy()

    # 训练用作答矩阵（来自训练用户）：base 用旧题空间(13)，增量用完整空间(20)
    log_old = build_log_mat(train_old, n_user, n_item_old)
    log_full = build_log_mat(df_train, n_user, n_item_total)

    # ---- 评测口径分派 ----
    if mode == "buf":
        # random-split：buffer 无泄漏预测（test 用户与训练共享）
        def base_eval_fn(m):
            populate_buffers(m, log_old, device)
            return evaluate_buf(m, valid_old, device)

        def base_final():
            populate_buffers(base, log_old, device)
            return evaluate_buf(base, test_old, device)

        def strat_eval_fn(valid_df):
            def fn(m):
                populate_buffers(m, log_full, device)
                return evaluate_buf(m, valid_df, device)
            return fn

        def final_old(m):
            return evaluate_buf(m, test_old, device)

        def final_new(m):
            return evaluate_buf(m, test_new, device)
    else:
        # user-split：重构口径（test/valid 用户训练时未见 → 喂其自身作答现场诊断）
        log_valid_old = build_log_mat(valid_old, n_user, n_item_old)   # base 旧题空间(13)
        log_test_old = build_log_mat(test_old, n_user, n_item_old)
        log_valid_full = build_log_mat(df_valid, n_user, n_item_total)  # 扩展空间(20)
        log_test_full = build_log_mat(df_test, n_user, n_item_total)

        def base_eval_fn(m):
            return evaluate_recon(m, valid_old, log_valid_old, device)

        def base_final():
            return evaluate_recon(base, test_old, log_test_old, device)

        def strat_eval_fn(valid_df):
            return lambda m: evaluate_recon(m, valid_df, log_valid_full, device)

        def final_old(m):
            return evaluate_recon(m, test_old, log_test_full, device)

        def final_new(m):
            return evaluate_recon(m, test_new, log_test_full, device)

    results = []

    def record(name, r_old, r_new, tmd):
        results.append({
            "Model": name,
            "AUC_old": r_old["auc"], "AUC_new": r_new["auc"] if r_new else "-",
            "RMSE_old": r_old["rmse"], "RMSE_new": r_new["rmse"] if r_new else "-",
            "ACC_old": r_old["acc"], "ACC_new": r_new["acc"] if r_new else "-",
            "F1_old": r_old["f1"], "F1_new": r_new["f1"] if r_new else "-",
            "TMD": tmd if tmd is not None else "",
        })
        new_str = f" | 新: AUC={r_new['auc']:.4f} ACC={r_new['acc']:.4f} F1={r_new['f1']:.4f}" if r_new else ""
        print(f"  [{name}] 旧: AUC={r_old['auc']:.4f} ACC={r_old['acc']:.4f} F1={r_old['f1']:.4f}{new_str}")

    # ---- 1. Base（旧题，全参数）----
    print("\n=== 1. Base ===")
    base = GNCDM(n_user=n_user, n_item=n_item_old, n_know=n_know_old, user_dim=32, item_dim=32,
                 alpha=alpha, Q_mat=Q_old, monotonicity_assumption=True, device=device).to(device)
    train_real(base, train_old, log_old, list(base.parameters()), device, n_epoch=25, desc="Base",
               eval_fn=base_eval_fn)
    populate_buffers(base, log_old, device)              # 取 Theta_buf 供 TMD 参照
    base_theta_old = base.get_Theta_buf().clone()
    record("Base", base_final(), None, None)

    def run_strategy(name, expand_fn, params_fn, train_df, valid_df, n_epoch=25, mask_agg_old=False):
        m = fresh_base(base)
        expand_fn(m)
        populate_buffers(m, log_full, device)

        handles = []
        if mask_agg_old:
            def make_col_mask(k_old):
                def hook(grad):
                    g = grad.clone()
                    g[:, :k_old] = 0.0
                    return g
                return hook
            handles.append(m.theta_agg_mat.weight.register_hook(make_col_mask(n_know_old)))
            handles.append(m.psi_agg_mat.weight.register_hook(make_col_mask(n_know_old)))

        train_real(m, train_df, log_full, params_fn(m), device, n_epoch=n_epoch, desc=name,
                   eval_fn=strat_eval_fn(valid_df))
        for h in handles:
            h.remove()

        populate_buffers(m, log_full, device)            # TMD 参照（+ buf 模式下供最终评测）
        tmd = calculate_tmd(base_theta_old.to(device), m.get_Theta_buf().to(device), n_know_old)
        record(name, final_old(m), final_new(m), tmd)
        return m

    print("\n=== 2. Ours-Ablated ===")
    run_strategy(
        "Ours-Ablated",
        lambda m: m.expand_topology(n_item_new, n_know_new, Q_expanded),
        lambda m: list(m.parameters()),
        train_new, valid_new)

    print("\n=== 3. Ours (Dynamic DNA) ===")
    run_strategy(
        "Ours (Dynamic DNA)",
        lambda m: m.expand_topology(n_item_new, n_know_new, Q_expanded),
        lambda m: new_params(m) + [m.theta_agg_mat.weight, m.psi_agg_mat.weight],
        train_new, valid_new,
        mask_agg_old=True)

    print("\n=== 4. Ours (LoRA) ===")
    run_strategy(
        "Ours (LoRA)",
        lambda m: m.expand_topology_lora(delta_M=n_item_new, delta_K=n_know_new,
                                         Q_expanded=Q_expanded, M_old=n_item_old, rank=16),
        lora_params,
        train_new, valid_new)

    print("\n=== 5. Full Replay Oracle ===")
    run_strategy(
        "Full Replay Oracle",
        lambda m: m.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded),
        lambda m: list(m.parameters()),
        pd.concat([train_old, train_new], ignore_index=True),
        pd.concat([valid_old, valid_new], ignore_index=True))

    print("\n=== 6. Naive FT (NFT) ===")
    run_strategy(
        "Naive FT (NFT)",
        lambda m: m.full_replay_oracle_expand_topology(n_item_new, n_know_new, Q_expanded),
        lambda m: list(m.parameters()),
        train_new, valid_new)

    out = os.path.join(SAVE_DIR, f"incremental_results_{split_name}.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n[{split_name}] 结果已写入 {out}")
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    repo_root = os.path.dirname(gncdm_dir)
    Q_path = os.path.join(DATA_DIR, "math1_Q_matrix.npy")

    # 每个划分用 alpha 扫描得到的最优值（findings.md 第十八轮）：
    #   random_split（预测口径）Base ACC_old 顶点 → alpha=0.20
    #   （0.05 步长扫描 sweep_base_alpha_random.py：Base test ACC_old 在 0.20 见顶 0.7293）
    #   user_split（重构口径）名义最优 → alpha=0.70
    splits = [
        ("math1_random_split", "buf", 0.20,
         os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"),
         os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"),
         os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv")),
        ("math1_user_split", "recon", 0.70,
         os.path.join(repo_root, "data", "math1", "user_split", "train.csv"),
         os.path.join(repo_root, "data", "math1", "user_split", "valid.csv"),
         os.path.join(repo_root, "data", "math1", "user_split", "test.csv")),
    ]
    # math1：4209 users × 20 items × 11 concepts；ΔK={0,1,3,6}（冷门概念，旧题/新题=13/7）
    for split_name, mode, alpha, tr, va, te in splits:
        set_seed(42)  # 每个划分重置随机种子，保证可复现
        run_experiment(split_name, mode, tr, va, te, Q_path, device,
                       n_user=4209, n_item_total=20, n_know_total=11,
                       new_concepts=[0, 1, 3, 6], alpha=alpha)


if __name__ == "__main__":
    main()
