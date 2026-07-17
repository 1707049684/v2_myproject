"""random_split 三 CL 基线（EWC / DER++ / C-LoRA）+ 合并 Ours → 九方法总表。【单文件，覆盖 math1 + a0910】

统一覆盖型脚本（取代原 a0910_{ewc,der,clora}_baseline.py / math1_der_baseline.py 等单基线脚本）：
一次跑完三基线的 random_split，并与 Ours 六策略（incremental_results_{ds}_random_split.csv）
合并成 all_methods_{ds}_random_split.{csv,md}。与 user_split 的 eval_all_methods_user_split.py 配套。

口径（random_split）：test 用户与训练**共享**（按交互划分）→ CognitiveBackbone 的 student_emb
已训练，**无需冷启动**，直接预测 test_old / test_new。与 Ours 的 forward_using_buf 预测口径同属
"预测"（都无自信息），AUC/ACC/F1/RMSE 可逐行对比。严格拓扑二分（math1 用 ΔK=[0,1,3,6]；
a0910 用 auto_new_concepts 选最冷门概念），测试行与 Ours 主表一致。

可比性红线：骨干 CognitiveBackbone≠G-NCDM（只说"同划分/同口径下"，勿称纯策略胜出）；RD* 在
embedding 空间，量级**不可**与 Ours 概念 θ RD 直接比，仅看是否>0。

运行（需 avalanche 给 EWC/DER）：
    pip install avalanche-lib
    cd GNCDM
    python cl_baselines_random_split.py          # 默认只跑 math1；RUN_A0910=True 也跑 a0910（17746 题，重）
"""

import copy
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # GNCDM/
REPO_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(THIS_DIR, "data")
SAVE_DIR = os.path.join(THIS_DIR, "incremental_result")
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_A0910 = False  # 默认只跑 math1；置 True 也跑 a0910（题量大、较慢）
RUN_JUNYI = True  # 也跑 junyi（5000×707×39）；合并 incremental_results_junyi_random_split.csv

# ---- 骨干 / 训练超参（与原 a0910/math1 基线脚本一致）----
EMBED_DIM = 64
LR = 1e-3
TRAIN_MB_SIZE = 128
TRAIN_EPOCHS = 25  # DER 早停最大轮数
EARLY_STOP_PATIENCE = 5
DER_EPOCHS_INNER = 1
MEM_SIZE = 5000
DER_ALPHA = 0.5
DER_BETA = 0.5
EWC_EPOCHS = 15
EWC_MODE = "separate"
EWC_LAMBDA_SWEEP = [0, 1, 10, 100, 1000, 10000]
BASE_EPOCHS = 15
CLORA_EPOCHS = 15
LORA_RANK = 8
LORA_ALPHA = 16
CLORA_LAMBDA_SWEEP = [0, 1, 10, 100, 1000, 10000]


def set_seed(seed=42, *, deterministic=False):
    """Seed all RNGs used by the random-split CL baselines."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================================
# 工具 + 骨干（内联自包含）
# ==========================================================================
def strict_bipartition(Q, new_concepts):
    K = Q.shape[1]
    new_concepts = list(new_concepts)
    old_concepts = [k for k in range(K) if k not in new_concepts]
    concept_perm = old_concepts + new_concepts
    touches_new = Q[:, new_concepts].sum(axis=1) > 0
    old_items = np.where(~touches_new)[0].tolist()
    new_items = np.where(touches_new)[0].tolist()
    item_perm = old_items + new_items
    Q_re = Q[np.ix_(item_perm, concept_perm)].astype(np.float32)
    item_id_map = {old: new for new, old in enumerate(item_perm)}
    return Q_re, item_id_map, len(old_items), len(old_concepts)


def remap_items(df, item_id_map):
    df = df.copy()
    df["item_id"] = df["item_id"].map(item_id_map).astype(int)
    return df


def auto_new_concepts(Q, new_item_frac=0.34):
    """按触及题目数升序累加最冷门概念，直到触及题占比达 new_item_frac（a0910 用）。"""
    n_item = Q.shape[0]
    freq = (Q > 0).sum(axis=0)
    touched = np.zeros(n_item, dtype=bool)
    new_set = []
    for k in np.argsort(freq):
        new_set.append(int(k))
        touched |= Q[:, k] > 0
        if touched.sum() >= new_item_frac * n_item:
            break
    return sorted(new_set)


class CognitiveBackbone(nn.Module):
    def __init__(self, num_students, num_items, embed_dim=64):
        super().__init__()
        self.student_emb = nn.Embedding(num_students, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        u = self.student_emb(x[:, 0])
        v = self.item_emb(x[:, 1])
        return self.mlp(torch.cat([u, v], dim=-1))


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1e-2)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def delta_w(self):
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA):
    for idx in (0, 2, 4):
        assert isinstance(model.mlp[idx], nn.Linear)
        model.mlp[idx] = LoRALinear(model.mlp[idx], rank=rank, alpha=alpha)
    return model


def orthogonal_penalty(model):
    loss_ortho = model.student_emb.weight.new_zeros(())
    for m in model.modules():
        if isinstance(m, LoRALinear):
            W_base = m.base_layer.weight.detach()
            loss_ortho = loss_ortho + torch.sum((W_base @ m.delta_w().t()) ** 2)
    return loss_ortho


def evaluate_cd_metrics(model, eval_dataset, device):
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    y_trues, y_probs, y_preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)
            y_trues.extend(y.cpu().numpy())
            y_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            y_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except ValueError:
        auc = 0.5
    return (
        auc,
        mean_squared_error(y_trues, y_probs) ** 0.5,
        accuracy_score(y_trues, y_preds),
        f1_score(y_trues, y_preds, zero_division=0),
    )


def baseline_tmd(b0, model, old_user_ids, embed_dim):
    final = model.student_emb.weight.data.cpu()
    drift = torch.norm(b0[old_user_ids] - final[old_user_ids], p=2, dim=1)
    return (drift / math.sqrt(embed_dim)).mean().item()


# ==========================================================================
# 数据装载：random_split + 严格拓扑二分（test 用户与训练共享 → 直接预测）
# ==========================================================================
def load_random(cfg):
    Q = np.load(cfg["Q"])
    df_train = pd.read_csv(cfg["train"])
    df_valid = pd.read_csv(cfg["valid"])
    df_test = pd.read_csv(cfg["test"])
    n_item_total = cfg["n_item"]

    new_concepts = (
        auto_new_concepts(Q, 0.34) if cfg["new_concepts"] == "auto" else list(cfg["new_concepts"])
    )
    Q, item_id_map, n_item_old, n_know_old = strict_bipartition(Q, new_concepts)
    df_train = remap_items(df_train, item_id_map)
    df_valid = remap_items(df_valid, item_id_map)
    df_test = remap_items(df_test, item_id_map)
    assert Q[:n_item_old, n_know_old:].sum() == 0, "旧题依赖新概念，二分失败！"
    print(
        f">>> [{cfg['name']}] 二分: 新概念={len(new_concepts)}/{cfg['n_know']}, "
        f"旧题(Task0)={n_item_old} 新题(Task1)={n_item_total - n_item_old}, 旧概念={n_know_old}"
    )

    def _ds(df, lo, hi):
        sub = df[(df["item_id"] >= lo) & (df["item_id"] < hi)]
        x = torch.tensor(sub[["user_id", "item_id"]].values, dtype=torch.long)
        y = torch.tensor(sub["score"].values, dtype=torch.long)
        return TensorDataset(x, y)

    return {
        "n_item_old": n_item_old,
        "num_students": int(
            max(df_train["user_id"].max(), df_valid["user_id"].max(), df_test["user_id"].max())
        )
        + 1,
        "num_items": n_item_total,
        "old_user_ids": torch.tensor(
            df_train[df_train["item_id"] < n_item_old]["user_id"].unique(), dtype=torch.long
        ),
        "train_old_ds": _ds(df_train, 0, n_item_old),
        "train_new_ds": _ds(df_train, n_item_old, n_item_total),
        "valid_old_ds": _ds(df_valid, 0, n_item_old),
        "valid_new_ds": _ds(df_valid, n_item_old, n_item_total),
        "test_old_ds": _ds(df_test, 0, n_item_old),
        "test_new_ds": _ds(df_test, n_item_old, n_item_total),
    }


def _bench(meta):
    from avalanche.benchmarks import dataset_benchmark

    return dataset_benchmark(
        train_datasets=[meta["train_old_ds"], meta["train_new_ds"]],
        test_datasets=[meta["test_old_ds"], meta["test_new_ds"]],
    )


def _row(method, old_m, new_m, tmd):
    return {
        "Method": method,
        "AUC_old": old_m[0],
        "AUC_new": new_m[0],
        "RMSE_old": old_m[1],
        "RMSE_new": new_m[1],
        "ACC_old": old_m[2],
        "ACC_new": new_m[2],
        "F1_old": old_m[3],
        "F1_new": new_m[3],
        "RD": tmd,
    }


def _eval_both(model, meta, device, *, split="test"):
    if split not in {"valid", "test"}:
        raise ValueError(f"split must be 'valid' or 'test', got {split!r}")
    return (
        evaluate_cd_metrics(model, meta[f"{split}_old_ds"], device),
        evaluate_cd_metrics(model, meta[f"{split}_new_ds"], device),
    )


# ==========================================================================
# EWC λ 扫描 / DER++（早停）/ C-LoRA λ 扫描
# ==========================================================================
def run_ewc(meta, device, *, seed=42, lambdas=None, eval_split="test"):
    from avalanche.training.supervised import EWC

    print("\n=== EWC λ 扫描 ===")
    rows = []
    lambdas = EWC_LAMBDA_SWEEP if lambdas is None else list(lambdas)
    for lam in lambdas:
        set_seed(seed)
        model = CognitiveBackbone(meta["num_students"], meta["num_items"], EMBED_DIM).to(device)
        strat = EWC(
            model,
            optim.Adam(model.parameters(), lr=LR),
            nn.CrossEntropyLoss(),
            ewc_lambda=lam,
            mode=EWC_MODE,
            train_mb_size=TRAIN_MB_SIZE,
            train_epochs=EWC_EPOCHS,
            eval_mb_size=256,
            device=device,
        )
        b0 = None
        for exp in _bench(meta).train_stream:
            strat.train(exp)
            if exp.current_experience == 0:
                b0 = model.student_emb.weight.data.clone().cpu()
        old_m, new_m = _eval_both(model, meta, device, split=eval_split)
        r = _row(
            f"EWC (lambda={lam})",
            old_m,
            new_m,
            baseline_tmd(b0, model, meta["old_user_ids"], EMBED_DIM),
        )
        r["lambda"] = lam
        rows.append(r)
        print(f"  λ={lam}: AUC_old={r['AUC_old']:.4f} AUC_new={r['AUC_new']:.4f}")
    return rows


def run_der(meta, device, *, seed=42, eval_split="test", mem_size=MEM_SIZE):
    from avalanche.training.supervised import DER

    print("\n=== DER++（早停）===")
    set_seed(seed)
    model = CognitiveBackbone(meta["num_students"], meta["num_items"], EMBED_DIM).to(device)
    strat = DER(
        model,
        optim.Adam(model.parameters(), lr=LR),
        nn.CrossEntropyLoss(),
        mem_size=mem_size,
        alpha=DER_ALPHA,
        beta=DER_BETA,
        train_mb_size=TRAIN_MB_SIZE,
        train_epochs=DER_EPOCHS_INNER,
        eval_mb_size=256,
        device=device,
    )
    b0 = None
    for exp in _bench(meta).train_stream:
        tid = exp.current_experience
        val_ds = (
            meta["valid_old_ds"]
            if tid == 0
            else ConcatDataset([meta["valid_old_ds"], meta["valid_new_ds"]])
        )
        best_acc, best_state, wait = -1.0, None, 0
        for _ in range(TRAIN_EPOCHS):
            strat.train(exp)
            _, _, val_acc, _ = evaluate_cd_metrics(model, val_ds, device)
            if val_acc > best_acc:
                best_acc, wait = val_acc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                break
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if tid == 0:
            b0 = model.student_emb.weight.data.clone().cpu()
    old_m, new_m = _eval_both(model, meta, device, split=eval_split)
    r = _row(
        f"DER++ (mem={mem_size})",
        old_m,
        new_m,
        baseline_tmd(b0, model, meta["old_user_ids"], EMBED_DIM),
    )
    r["mem_size"] = mem_size
    print(f"  DER++: AUC_old={r['AUC_old']:.4f} AUC_new={r['AUC_new']:.4f}")
    return r


def _train_phase(model, dataset, epochs, device, lambda_ortho=None):
    model.train()
    loader = DataLoader(dataset, batch_size=TRAIN_MB_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            if lambda_ortho is not None and lambda_ortho > 0:
                loss = loss + lambda_ortho * orthogonal_penalty(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_clora(meta, device, *, seed=42, lambdas=None, eval_split="test"):
    print("\n=== C-LoRA λ_ortho 扫描 ===")
    set_seed(seed)
    base_model = CognitiveBackbone(meta["num_students"], meta["num_items"], EMBED_DIM).to(device)
    _train_phase(base_model, meta["train_old_ds"], BASE_EPOCHS, device, lambda_ortho=None)
    base_state = copy.deepcopy(base_model.state_dict())
    rows = []
    lambdas = CLORA_LAMBDA_SWEEP if lambdas is None else list(lambdas)
    for lam in lambdas:
        set_seed(seed)
        model = CognitiveBackbone(meta["num_students"], meta["num_items"], EMBED_DIM).to(device)
        model.load_state_dict(base_state)
        b0 = model.student_emb.weight.data.clone().cpu()
        inject_lora(model)
        model.to(device)
        _train_phase(model, meta["train_new_ds"], CLORA_EPOCHS, device, lambda_ortho=float(lam))
        old_m, new_m = _eval_both(model, meta, device, split=eval_split)
        r = _row(
            f"C-LoRA (lambda={lam})",
            old_m,
            new_m,
            baseline_tmd(b0, model, meta["old_user_ids"], EMBED_DIM),
        )
        r["lambda"] = lam
        rows.append(r)
        print(f"  λ={lam}: AUC_old={r['AUC_old']:.4f} AUC_new={r['AUC_new']:.4f}")
    return rows


def _pick_balanced(rows):
    return max(rows, key=lambda r: (r["AUC_old"] + r["AUC_new"]) / 2.0)


def select_baselines_on_validation(meta, device, *, seed=42):
    """Tune EWC/C-LoRA on validation data and evaluate each choice once on test.

    This is intentionally separate from the legacy ``run_one`` path, whose
    historic wide table selected lambda from test metrics. The returned rows
    are safe to place in a formal per-seed statistical analysis table.
    """

    ewc_valid = run_ewc(meta, device, seed=seed, eval_split="valid")
    ewc_lambda = _pick_balanced(ewc_valid)["lambda"]
    ewc_test = run_ewc(meta, device, seed=seed, lambdas=[ewc_lambda], eval_split="test")[0]
    ewc_test["selected_lambda"] = ewc_lambda
    ewc_test["selection_source"] = "validation_balanced_auc"

    clora_valid = run_clora(meta, device, seed=seed, eval_split="valid")
    clora_lambda = _pick_balanced(clora_valid)["lambda"]
    clora_test = run_clora(meta, device, seed=seed, lambdas=[clora_lambda], eval_split="test")[0]
    clora_test["selected_lambda"] = clora_lambda
    clora_test["selection_source"] = "validation_balanced_auc"

    der_test = run_der(meta, device, seed=seed, eval_split="test")
    der_test["selection_source"] = "validation_early_stopping"
    return [ewc_test, der_test, clora_test]


def run_fixed_baselines(
    meta,
    device,
    *,
    seed=42,
    ewc_lambda,
    der_mem_size=MEM_SIZE,
    clora_lambda=None,
):
    """Run only preselected baseline hyperparameters without any lambda sweep."""

    rows = []
    if ewc_lambda is not None:
        ewc_row = run_ewc(
            meta,
            device,
            seed=seed,
            lambdas=[ewc_lambda],
            eval_split="test",
        )[0]
        ewc_row["selected_lambda"] = ewc_lambda
        ewc_row["selection_source"] = "fixed_from_existing_result"
        rows.append(ewc_row)

    if der_mem_size is not None:
        der_row = run_der(meta, device, seed=seed, eval_split="test", mem_size=der_mem_size)
        der_row["selection_source"] = "fixed_from_existing_result"
        rows.append(der_row)

    if clora_lambda is not None:
        clora_row = run_clora(
            meta,
            device,
            seed=seed,
            lambdas=[clora_lambda],
            eval_split="test",
        )[0]
        clora_row["selected_lambda"] = clora_lambda
        clora_row["selection_source"] = "fixed_from_existing_result"
        rows.append(clora_row)
    return rows


# ==========================================================================
# 合并 Ours 六策略 + 三基线 → 总表
# ==========================================================================
COLS = [
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


def _fmt(x):
    if isinstance(x, str):
        return x
    return "-" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x:.4f}"


def merge_and_write(cfg, ewc_rows, der_row, clora_rows):
    ours = pd.read_csv(os.path.join(SAVE_DIR, cfg["ours_csv"])).rename(
        columns={"Model": "Method", "TMD": "RD"}
    )
    ours_rows = ours.to_dict("records")
    ewc_best, clora_best = _pick_balanced(ewc_rows), _pick_balanced(clora_rows)
    merged = ours_rows + [
        {k: ewc_best[k] for k in COLS},
        {k: der_row[k] for k in COLS},
        {k: clora_best[k] for k in COLS},
    ]
    df = pd.DataFrame(merged, columns=COLS)
    csv_path = os.path.join(SAVE_DIR, f"all_methods_{cfg['name']}_random_split.csv")
    df.to_csv(csv_path, index=False)

    lines = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in merged:
        lines.append("| " + " | ".join([str(r["Method"])] + [_fmt(r[c]) for c in COLS[1:]]) + " |")
    note = (
        f"\n*口径*：{cfg['name']} random_split（test 用户与训练共享，预测口径）。Ours 走 "
        "forward_using_buf 无泄漏预测，基线（CognitiveBackbone）直接预测；均无自信息，可逐行对比。\n"
        f"*均衡点*：EWC λ={ewc_best['lambda']}、C-LoRA λ={clora_best['lambda']}（各取 avg(AUC_old,AUC_new) 最大）。\n"
        "*RD 红线*：Ours 行 RD 在 G-NCDM 概念 θ 空间；基线行 RD 在 embedding 空间，量级不可与 "
        "Ours 直接比，仅看是否>0。骨干不同，勿称纯策略胜出。\n"
    )
    with open(
        os.path.join(SAVE_DIR, f"all_methods_{cfg['name']}_random_split.md"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(lines) + "\n" + note)
    pd.DataFrame(ewc_rows).to_csv(
        os.path.join(SAVE_DIR, f"ewc_lambda_sweep_{cfg['name']}_random_split.csv"), index=False
    )
    pd.DataFrame(clora_rows).to_csv(
        os.path.join(SAVE_DIR, f"clora_lambda_sweep_{cfg['name']}_random_split.csv"), index=False
    )

    print("\n" + "=" * 70)
    print(f" {cfg['name']} random_split 九方法总表")
    print("=" * 70)
    print("\n".join(lines))
    print(f"\n>>> 写入 {csv_path}")


def run_one(cfg, device):
    print(f"\n{'#' * 72}\n# {cfg['name']} random_split 三基线 + 合并总表\n{'#' * 72}")
    meta = load_random(cfg)
    ewc_rows = run_ewc(meta, device)
    der_row = run_der(meta, device)
    clora_rows = run_clora(meta, device)
    merge_and_write(cfg, ewc_rows, der_row, clora_rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    configs = [
        {
            "name": "math1",
            "train": os.path.join(DATA_DIR, "math1_train_0.8_0.2.csv"),
            "valid": os.path.join(DATA_DIR, "math1_valid_0.8_0.2.csv"),
            "test": os.path.join(DATA_DIR, "math1_test_0.8_0.2.csv"),
            "Q": os.path.join(DATA_DIR, "math1_Q_matrix.npy"),
            "n_item": 20,
            "n_know": 11,
            "new_concepts": [0, 1, 3, 6],
            "ours_csv": "incremental_results_math1_random_split.csv",
        },
    ]
    if RUN_A0910:
        a0910 = os.path.join(REPO_ROOT, "data", "a0910")
        configs.append(
            {
                "name": "a0910",
                "train": os.path.join(a0910, "new_random_split", "train.csv"),
                "valid": os.path.join(a0910, "new_random_split", "valid.csv"),
                "test": os.path.join(a0910, "new_random_split", "test.csv"),
                "Q": os.path.join(a0910, "Q_matrix.npy"),
                "n_item": 17746,
                "n_know": 123,
                "new_concepts": "auto",
                "ours_csv": "incremental_results_a0910_random_split.csv",
            }
        )
    if RUN_JUNYI:
        junyi = os.path.join(REPO_ROOT, "data", "junyi")
        Q = np.load(os.path.join(junyi, "Q_matrix.npy"))
        configs.append(
            {
                "name": "junyi",
                "train": os.path.join(junyi, "new_random_split", "train.csv"),
                "valid": os.path.join(junyi, "new_random_split", "valid.csv"),
                "test": os.path.join(junyi, "new_random_split", "test.csv"),
                "Q": os.path.join(junyi, "Q_matrix.npy"),
                "n_item": int(Q.shape[0]),
                "n_know": int(Q.shape[1]),
                "new_concepts": "auto",
                "ours_csv": "incremental_results_junyi_random_split.csv",
            }
        )
    for cfg in configs:
        run_one(cfg, device)


if __name__ == "__main__":
    main()
