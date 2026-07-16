"""ICDM-WWW'24 adapter for the project's random/user Math1 protocols.

This is a clean PyTorch reimplementation rather than a source copy of the
unlicensed official repository.  It preserves the inductive graph mechanism
and adapts it to the project's two-stage old-item -> new-item experiment.

Training alternates two disjoint per-user folds: one fold supplies response
graph edges and the other supplies prediction targets.  Consequently, a
target response is never visible as a right/wrong graph edge in the same
forward pass.  Validation and test query rows are likewise excluded from all
graph contexts.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from baselines.icdm_ww24 import ICDMWW24, InteractionGraph, concat_graphs
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
GNCDM_DIR = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(GNCDM_DIR, "incremental_result")

METHOD_NAME = "ICDM-WWW24 (adapted)"
SUPPORT_FRAC = 0.5
SPLIT_SEED = 7
TRAIN_FOLD_SEED = 17
DEFAULT_DIM = 64
DEFAULT_LR = 2e-3
DEFAULT_EPOCHS = 25
DEFAULT_PATIENCE = 5
DEFAULT_WEIGHT_REG = 1e-3

RESULT_COLUMNS = [
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


@dataclass(frozen=True)
class ProtocolData:
    q: np.ndarray
    train_old: pd.DataFrame
    train_new: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    n_user: int
    n_item_old: int
    n_know_old: int

    @property
    def train_full(self) -> pd.DataFrame:
        return pd.concat([self.train_old, self.train_new], ignore_index=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strict_bipartition(q: np.ndarray, new_concepts):
    new_concepts = list(new_concepts)
    old_concepts = [k for k in range(q.shape[1]) if k not in new_concepts]
    concept_perm = old_concepts + new_concepts
    touches_new = q[:, new_concepts].sum(axis=1) > 0
    old_items = np.where(~touches_new)[0].tolist()
    new_items = np.where(touches_new)[0].tolist()
    item_perm = old_items + new_items
    q_reordered = q[np.ix_(item_perm, concept_perm)].astype(np.float32)
    item_id_map = {old: new for new, old in enumerate(item_perm)}
    return q_reordered, item_id_map, len(old_items), len(old_concepts)


def auto_new_concepts(q: np.ndarray, new_item_frac: float = 0.34):
    """Select rare concepts until they touch the requested fraction of items."""

    frequency = (q > 0).sum(axis=0)
    touched = np.zeros(q.shape[0], dtype=bool)
    selected = []
    for concept in np.argsort(frequency):
        selected.append(int(concept))
        touched |= q[:, concept] > 0
        if touched.sum() >= new_item_frac * q.shape[0]:
            break
    return sorted(selected)


def _remap_items(frame: pd.DataFrame, item_id_map) -> pd.DataFrame:
    frame = frame.copy()
    frame["item_id"] = frame["item_id"].map(item_id_map).astype(int)
    return frame


def prepare_protocol(cfg) -> ProtocolData:
    q = np.load(cfg["Q"])
    new_concepts = cfg["new_concepts"]
    if new_concepts == "auto":
        new_concepts = auto_new_concepts(q, cfg.get("new_item_frac", 0.34))
    q, item_id_map, n_item_old, n_know_old = strict_bipartition(q, new_concepts)
    train = _remap_items(pd.read_csv(cfg["train"]), item_id_map)
    valid = _remap_items(pd.read_csv(cfg["valid"]), item_id_map)
    test = _remap_items(pd.read_csv(cfg["test"]), item_id_map)
    train_old = train[train["item_id"] < n_item_old].copy()
    train_new = train[train["item_id"] >= n_item_old].copy()
    if q[:n_item_old, n_know_old:].sum() != 0:
        raise ValueError("strict bipartition failed: an old item uses a new concept")
    return ProtocolData(
        q=q,
        train_old=train_old,
        train_new=train_new,
        valid=valid,
        test=test,
        n_user=int(cfg["n_user"]),
        n_item_old=n_item_old,
        n_know_old=n_know_old,
    )


def split_support_query(frame: pd.DataFrame, seed: int = SPLIT_SEED):
    support = frame.groupby("user_id", group_keys=False).sample(
        frac=SUPPORT_FRAC, random_state=seed
    )
    return support.copy(), frame.drop(support.index).copy()


def _two_training_folds(frame: pd.DataFrame, seed: int = TRAIN_FOLD_SEED):
    first = frame.groupby("user_id", group_keys=False).sample(frac=0.5, random_state=seed)
    second = frame.drop(first.index)
    if first.empty or second.empty:
        raise ValueError("both leakage-free training folds must be non-empty")
    return first.copy(), second.copy()


def graph_from_frame(frame: pd.DataFrame, device) -> InteractionGraph:
    return InteractionGraph(
        torch.as_tensor(frame["user_id"].to_numpy(), dtype=torch.long, device=device),
        torch.as_tensor(frame["item_id"].to_numpy(), dtype=torch.long, device=device),
        torch.as_tensor(frame["score"].to_numpy(), dtype=torch.float32, device=device),
    )


def _clone_state_dict(model: torch.nn.Module):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate(model: ICDMWW24, context: InteractionGraph, query: pd.DataFrame, device):
    if query.empty:
        return {"auc": float("nan"), "rmse": float("nan"), "acc": float("nan"), "f1": float("nan")}
    target = graph_from_frame(query, device)
    model.eval()
    with torch.no_grad():
        state = model.encode(context)
        probability = model.predict(target.user, target.item, state).cpu().numpy()
    truth = target.score.cpu().numpy().astype(int)
    prediction = (probability >= 0.5).astype(int)
    auc = roc_auc_score(truth, probability) if np.unique(truth).size > 1 else 0.5
    return {
        "auc": float(auc),
        "rmse": float(math.sqrt(mean_squared_error(truth, probability))),
        "acc": float(accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


def train_stage(
    model: ICDMWW24,
    stage_frame: pd.DataFrame,
    validation_context: InteractionGraph,
    validation_query: pd.DataFrame,
    device,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    weight_reg: float = DEFAULT_WEIGHT_REG,
    label: str,
):
    fold_a, fold_b = _two_training_folds(stage_frame)
    graph_a = graph_from_frame(fold_a, device)
    graph_b = graph_from_frame(fold_b, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = _clone_state_dict(model)
    best_auc = -float("inf")
    stale = 0

    for epoch in range(epochs):
        model.train()
        context, target = (graph_a, graph_b) if epoch % 2 == 0 else (graph_b, graph_a)
        loss = model.loss(target, context, weight_reg=weight_reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clip_monotonic_weights()

        metrics = evaluate(model, validation_context, validation_query, device)
        score = metrics["auc"] if math.isfinite(metrics["auc"]) else metrics["acc"]
        print(
            f"  [{label}] epoch={epoch + 1:02d} loss={loss.item():.5f} "
            f"valid_auc={metrics['auc']:.4f} valid_acc={metrics['acc']:.4f}"
        )
        if score > best_auc + 1e-6:
            best_auc = score
            best_state = _clone_state_dict(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  [{label}] early stop at epoch {epoch + 1}; best AUC={best_auc:.4f}")
                break

    model.load_state_dict(best_state)
    return best_auc


def _metric_row(old_metrics, new_metrics, rd):
    return {
        "Method": METHOD_NAME,
        "AUC_old": old_metrics["auc"],
        "AUC_new": new_metrics["auc"],
        "RMSE_old": old_metrics["rmse"],
        "RMSE_new": new_metrics["rmse"],
        "ACC_old": old_metrics["acc"],
        "ACC_new": new_metrics["acc"],
        "F1_old": old_metrics["f1"],
        "F1_new": new_metrics["f1"],
        "RD": rd,
    }


def _format_cell(value):
    if isinstance(value, str):
        value = value.strip()
        try:
            return f"{float(value):.4f}"
        except ValueError:
            return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.4f}"


def _upsert_main_table(split_name: str, row):
    csv_path = os.path.join(SAVE_DIR, f"all_methods_{split_name}.csv")
    md_path = os.path.join(SAVE_DIR, f"all_methods_{split_name}.md")
    if not os.path.exists(csv_path):
        print(f"  主表不存在，跳过并表：{csv_path}")
        return

    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    table.columns = [column.strip() for column in table.columns]
    drift_column = "TMD" if "TMD" in table.columns else "RD"
    stored = dict(row)
    stored[drift_column] = stored.pop("RD")
    for column in table.columns:
        stored.setdefault(column, "")
    stored = {column: stored[column] for column in table.columns}
    method = table["Method"].astype(str).str.strip()
    table = table[method != METHOD_NAME]
    table = pd.concat([table, pd.DataFrame([stored])], ignore_index=True)
    table.to_csv(csv_path, index=False)

    suffix = ""
    if os.path.exists(md_path):
        lines = open(md_path, encoding="utf-8").read().splitlines()
        first_non_table = next(
            (i for i, line in enumerate(lines) if i > 1 and not line.startswith("|")), len(lines)
        )
        suffix = "\n".join(lines[first_non_table:]).strip()
    md_lines = [
        "| " + " | ".join(table.columns) + " |",
        "|" + "|".join(["---"] * len(table.columns)) + "|",
    ]
    for record in table.to_dict("records"):
        md_lines.append("| " + " | ".join(_format_cell(record[c]) for c in table.columns) + " |")
    note = (
        "*ICDM-WWW24 (adapted)*：无 DGL 的独立 PyTorch 重实现；按 old→new 两阶段顺序训练。"
        "训练、验证和测试均保证 target/query 不作为同次 forward 的 response edge。其 RD 位于 ICDM "
        "mastery 空间，不能与 G-NCDM θ-RD 直接比较。"
    )
    tail = suffix
    if "*ICDM-WWW24 (adapted)*" not in tail:
        tail = (tail + "\n" + note).strip()
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("\n".join(md_lines) + "\n\n" + tail + "\n")
    print(f"  已并入：{csv_path}")


def _save_row(split_name: str, row, append: bool):
    os.makedirs(SAVE_DIR, exist_ok=True)
    row_path = os.path.join(SAVE_DIR, f"icdm_row_{split_name}.csv")
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(row_path, index=False)
    print(f"  ICDM 单行结果：{row_path}")
    if append:
        _upsert_main_table(split_name, row)


def _build_model(data: ProtocolData, device, dim: int):
    known_user_mask = torch.zeros(data.n_user, dtype=torch.bool)
    known_ids = torch.as_tensor(data.train_full["user_id"].unique(), dtype=torch.long)
    known_user_mask[known_ids] = True
    return ICDMWW24(
        n_user=data.n_user,
        n_item=data.q.shape[0],
        n_know=data.q.shape[1],
        q_matrix=data.q,
        known_user_mask=known_user_mask,
        dim=dim,
        k_hop=3,
    ).to(device)


def _train_and_measure(
    data: ProtocolData,
    model: ICDMWW24,
    stage1_validation,
    stage2_validation,
    device,
    *,
    epochs: int,
):
    old_graph = graph_from_frame(data.train_old, device)
    train_full_graph = graph_from_frame(data.train_full, device)
    train_stage(
        model,
        data.train_old,
        stage1_validation[0],
        stage1_validation[1],
        device,
        epochs=epochs,
        label="old",
    )
    model.eval()
    old_users = torch.as_tensor(
        sorted(data.train_old["user_id"].unique()), dtype=torch.long, device=device
    )
    with torch.no_grad():
        mastery_before = model.mastery(old_graph)[old_users].cpu()

    train_stage(
        model,
        data.train_new,
        stage2_validation[0],
        stage2_validation[1],
        device,
        epochs=epochs,
        label="new",
    )
    model.eval()
    with torch.no_grad():
        mastery_after = model.mastery(old_graph)[old_users].cpu()
    rd = (
        (
            torch.linalg.vector_norm(mastery_after - mastery_before, dim=1)
            / math.sqrt(data.q.shape[1])
        )
        .mean()
        .item()
    )
    return train_full_graph, rd


def run_random_split(cfg, device, *, epochs=DEFAULT_EPOCHS, dim=DEFAULT_DIM, append=True):
    """Run the sequential ICDM adapter under the shared-user prediction protocol."""

    split_name = f"{cfg['name']}_random_split"
    print(f"\n=== ICDM-WWW24 adapted · {split_name} ===")
    set_seed(42)
    data = prepare_protocol(cfg)
    model = _build_model(data, device, dim)
    valid_old = data.valid[data.valid["item_id"] < data.n_item_old]
    valid_new = data.valid[data.valid["item_id"] >= data.n_item_old]
    old_context = graph_from_frame(data.train_old, device)
    full_context = graph_from_frame(data.train_full, device)
    train_full_graph, rd = _train_and_measure(
        data,
        model,
        (old_context, valid_old),
        (full_context, valid_new),
        device,
        epochs=epochs,
    )
    test_old = data.test[data.test["item_id"] < data.n_item_old]
    test_new = data.test[data.test["item_id"] >= data.n_item_old]
    row = _metric_row(
        evaluate(model, train_full_graph, test_old, device),
        evaluate(model, train_full_graph, test_new, device),
        rd,
    )
    _save_row(split_name, row, append)
    return row


def run_math1_random_split(cfg, device, *, epochs=DEFAULT_EPOCHS, dim=DEFAULT_DIM, append=True):
    """Backward-compatible Math1 wrapper used by the existing main entry."""

    cfg = dict(cfg)
    cfg.setdefault("name", "math1")
    return run_random_split(cfg, device, epochs=epochs, dim=dim, append=append)


def run_math1_user_split(cfg, device, *, epochs=DEFAULT_EPOCHS, dim=DEFAULT_DIM, append=True):
    """Run ICDM on unseen users with the project's exact support/query split."""

    print("\n=== ICDM-WWW24 adapted · math1_user_split ===")
    set_seed(42)
    data = prepare_protocol(cfg)
    model = _build_model(data, device, dim)

    valid_support, valid_query = split_support_query(data.valid)
    valid_support_old = valid_support[valid_support["item_id"] < data.n_item_old]
    valid_query_old = valid_query[valid_query["item_id"] < data.n_item_old]
    valid_query_new = valid_query[valid_query["item_id"] >= data.n_item_old]
    stage1_context = concat_graphs(
        graph_from_frame(data.train_old, device), graph_from_frame(valid_support_old, device)
    )
    stage2_context = concat_graphs(
        graph_from_frame(data.train_full, device), graph_from_frame(valid_support, device)
    )
    _, rd = _train_and_measure(
        data,
        model,
        (stage1_context, valid_query_old),
        (stage2_context, valid_query_new),
        device,
        epochs=epochs,
    )

    test_support, test_query = split_support_query(data.test)
    test_context = concat_graphs(
        graph_from_frame(data.train_full, device), graph_from_frame(test_support, device)
    )
    test_old = test_query[test_query["item_id"] < data.n_item_old]
    test_new = test_query[test_query["item_id"] >= data.n_item_old]
    row = _metric_row(
        evaluate(model, test_context, test_old, device),
        evaluate(model, test_context, test_new, device),
        rd,
    )
    _save_row("math1_user_split", row, append)
    return row
