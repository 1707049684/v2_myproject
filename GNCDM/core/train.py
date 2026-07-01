# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
#
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# Information and shall use it only in accordance with the terms of
# the license agreement.

from __future__ import absolute_import

import gc
import math
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import GNCDM, UAutoRec, CDAE

torch.set_default_tensor_type(torch.FloatTensor)


def generate_cognitive_biased_mnar_mask(Q_new: np.ndarray, gamma: float = 1.0, tau: float = 0.5):
    """
    Cognitive-Biased MNAR Masking Generator

    Args:
        Q_new: np.ndarray (n_new_items, n_knowledge), incremental Q-matrix for new items
        gamma: float, steepness parameter of the logistic function
        tau: float, threshold parameter (cognitive load threshold)

    Returns:
        mask_prob: np.ndarray (n_new_items,), probability of masking for each new item
        mask: np.ndarray (n_new_items,), binary mask (1=keep, 0=mask)
    """
    # Calculate cognitive load for each new item: C_j = sum(Q_new[j, k])
    cognitive_load = np.sum(Q_new, axis=1)

    # Logistic step function: P(M_ij = 0) = 1 / (1 + exp(-gamma * (C_j - tau)))
    # Higher cognitive load -> higher probability of missing (masking)
    mask_prob = 1.0 / (1.0 + np.exp(-gamma * (cognitive_load - tau)))

    # Generate binary mask based on probability
    # mask = 1 means keep (observed), mask = 0 means mask (missing)
    mask = (np.random.rand(len(mask_prob)) > mask_prob).astype(float)

    return mask_prob, mask


def apply_mnar_mask_to_data(
    log_data: np.ndarray,
    Q_new: np.ndarray,
    new_item_indices: np.ndarray,
    gamma: float = 1.0,
    tau: float = 0.5,
):
    """
    Apply MNAR mask to new data based on cognitive load

    Args:
        log_data: np.ndarray (n_users, n_items), original log matrix
        Q_new: np.ndarray (n_new_items, n_knowledge), Q-matrix for new items
        new_item_indices: np.ndarray (n_new_items,), column indices of new items in log_data
        gamma: float, steepness parameter
        tau: float, threshold parameter

    Returns:
        masked_data: np.ndarray (n_users, n_items), log_data with MNAR masking applied
        mask_matrix: np.ndarray (n_users, n_items), binary mask matrix
    """
    masked_data = log_data.copy()
    mask_matrix = np.ones_like(log_data)

    # Get masking probability for each new item
    _, mask = generate_cognitive_biased_mnar_mask(Q_new, gamma, tau)

    # Apply masking to each new item column
    for idx, item_idx in enumerate(new_item_indices):
        if mask[idx] == 0:
            # Apply probabilistic masking per user
            user_mask = (np.random.rand(log_data.shape[0]) > 0.3).astype(float)  # 30% masking rate
            masked_data[:, item_idx] = masked_data[:, item_idx] * user_mask
            mask_matrix[:, item_idx] = user_mask

    # Replace zeros with -1 (padding/missing marker)
    masked_data[masked_data == 0] = -1

    return masked_data, mask_matrix


class AsymmetricHybridDataLoader:
    """
    Asymmetric Hybrid DataLoader for incremental learning

    Maintains two DataLoaders internally:
    - old_loader: for existing data (70% of batch)
    - new_loader: for incremental data (30% of batch, with repeated oversampling)

    Args:
        old_loader: DataLoader for existing data
        new_loader: DataLoader for new/incremental data
        old_ratio: float, ratio of old samples in batch (default: 0.7)
        device: torch.device, target device for data
    """

    def __init__(
        self, old_loader: DataLoader, new_loader: DataLoader, old_ratio: float = 0.7, device=None
    ):
        self.old_loader = old_loader
        self.new_loader = new_loader
        self.old_ratio = old_ratio
        self.device = device if device is not None else torch.device("cpu")

        # Create infinite iterator for new_loader to handle small datasets
        self.new_iter = itertools.cycle(new_loader)

        # Calculate batch sizes
        self.batch_size = old_loader.batch_size
        self.old_batch_size = int(self.batch_size * self.old_ratio)
        self.new_batch_size = self.batch_size - self.old_batch_size

    def __len__(self):
        # Return length based on old_loader (primary dataset)
        return len(self.old_loader)

    def __iter__(self):
        for old_batch in self.old_loader:
            # Get old samples
            old_samples = old_batch

            # Get new samples (with repeated oversampling if needed)
            try:
                new_samples = next(self.new_iter)
            except StopIteration:
                self.new_iter = itertools.cycle(self.new_loader)
                new_samples = next(self.new_iter)

            # Handle case where new_loader batch_size doesn't match
            if isinstance(old_samples, (list, tuple)):
                # Concatenate each element in the batch tuple
                combined_batch = []
                for old_data, new_data in zip(old_samples, new_samples):
                    # Get actual batch size from old_data
                    actual_batch_size = old_data.shape[0]
                    actual_old_size = int(actual_batch_size * self.old_ratio)
                    actual_new_size = actual_batch_size - actual_old_size

                    # Take required number of samples
                    old_part = old_data[:actual_old_size]
                    new_part = new_data[:actual_new_size]

                    # Pad if necessary (handles batch size mismatch)
                    if len(old_part) < actual_old_size:
                        pad_size = actual_old_size - len(old_part)
                        old_part = torch.cat([old_part, old_part[:pad_size]], dim=0)
                    if len(new_part) < actual_new_size:
                        pad_size = actual_new_size - len(new_part)
                        new_part = torch.cat([new_part, new_part[:pad_size]], dim=0)

                    combined = torch.cat([old_part, new_part], dim=0)
                    combined_batch.append(combined)

                # Create is_new mask with actual batch size
                is_new = torch.zeros(actual_batch_size, dtype=torch.bool)
                is_new[actual_old_size:] = True
            else:
                # Single tensor case
                actual_batch_size = old_samples.shape[0]
                actual_old_size = int(actual_batch_size * self.old_ratio)
                actual_new_size = actual_batch_size - actual_old_size

                old_part = old_samples[:actual_old_size]
                new_part = new_samples[:actual_new_size]
                combined_batch = torch.cat([old_part, new_part], dim=0)

                # Create is_new mask with actual batch size
                is_new = torch.zeros(actual_batch_size, dtype=torch.bool)
                is_new[actual_old_size:] = True
                combined_batch = torch.cat([old_part, new_part], dim=0)

            # Add is_new mask to the batch
            if isinstance(combined_batch, list):
                combined_batch.append(is_new)
            else:
                combined_batch = (combined_batch, is_new)

            yield combined_batch


class IDCDataset(Dataset):
    """
    the dataset of IDCD.
    """

    def __init__(self, df_log: pd.DataFrame, n_user: int, n_item: int):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log["user_id"].values
        self.item_id = df_log["item_id"].values
        self.score = df_log["score"].values
        pbar = tqdm(total=df_log.shape[0], desc="Loading data")
        for i, row in df_log.iterrows():
            self.log_mat[int(row["user_id"]), int(row["item_id"])] = (row["score"] - 0.5) * 2
            pbar.update(1)
        pbar.close()

    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        return (
            torch.Tensor(self.log_mat[user_id, :]),
            torch.Tensor(self.log_mat[:, item_id]),
            torch.LongTensor([user_id]),
            torch.LongTensor([item_id]),
            torch.FloatTensor([self.score[index]]),
        )

    def __len__(self):
        return self.user_id.shape[0]


# 2025.04.21. Add AEDataset
class AEDataset(Dataset):
    def __init__(self, df_log: pd.DataFrame, n_user: int, n_item: int):
        self.log_mat = np.zeros((n_user, n_item))
        self.obs_mat = np.zeros((n_user, n_item))
        self.user_id = np.arange(n_user)
        self.score = df_log["score"].values
        pbar = tqdm(total=df_log.shape[0], desc="Loading data")
        for i, row in df_log.iterrows():
            self.log_mat[int(row["user_id"]), int(row["item_id"])] = row["score"]
            self.obs_mat[int(row["user_id"]), int(row["item_id"])] = 1
            pbar.update(1)
        pbar.close()

    def __getitem__(self, index):
        user_id = self.user_id[index]
        return (
            torch.Tensor(self.log_mat[user_id, :]),
            torch.Tensor(self.obs_mat[user_id, :]),
            torch.LongTensor([user_id]),
        )

    def __len__(self):
        return self.user_id.shape[0]


def train(
    model: GNCDM, train_data: pd.DataFrame, valid_data: pd.DataFrame, batch_size, lr, n_epoch
):
    model.train()
    device = model.device
    dataset = IDCDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(model.Theta_buf.is_leaf, model.Psi_buf.is_leaf)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
    result_per_epoch = []
    for epoch in range(n_epoch):
        result_epoch = {}
        pbar = tqdm(total=len(dataloader), desc="Epoch %d" % epoch)
        score_all = []
        pred_all = []
        Theta_old = model.get_Theta_buf().numpy().copy()
        for i, (user_log, item_log, user_id, item_id, score) in enumerate(dataloader):
            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)
            pred = model(user_log, item_log, user_id, item_id)
            loss = F.binary_cross_entropy(pred, score)
            score_all += (
                score.detach()
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
                .tolist()
            )
            pred_all += (
                pred.detach()
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
                .tolist()
            )

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            optimizer.step()
            pbar.update(1)
        pbar.close()

        model.eval()

        # Update examinee traits
        for i in range(math.ceil(dataset.log_mat.shape[0] / batch_size)):
            idx = np.arange(i * batch_size, min(dataset.log_mat.shape[0], (i + 1) * batch_size))
            model.update_Theta_buf(
                model.diagnose_theta(torch.Tensor(dataset.log_mat[idx, :]).to(device)).detach(),
                torch.LongTensor(idx),
            )

        # Update question features
        for i in range(math.ceil(dataset.log_mat.shape[1] / batch_size)):
            idx = np.arange(i * batch_size, min(dataset.log_mat.shape[1], (i + 1) * batch_size))
            model.update_Psi_buf(
                model.diagnose_psi(torch.Tensor(dataset.log_mat[:, idx].T).to(device)).detach(),
                torch.LongTensor(idx),
            )
        model.train()
        Theta_new = model.get_Theta_buf().numpy().copy()

        score_all = np.array(score_all)
        pred_all = np.array(pred_all)

        Theta_norm = np.sqrt(np.sum(np.abs(Theta_new - Theta_old)))
        acc = accuracy_score(score_all, pred_all > 0.5)
        print("Theta_old.head =", Theta_old[:5, 0])
        print("Theta_new.head =", Theta_new[:5, 0])
        print("epoch = %d, theta_norm = %.6f, acc = %.6f" % (epoch, Theta_norm, acc))
        result_epoch["Theta_old_head"] = Theta_old[:5, :5]
        result_epoch["Theta_new_head"] = Theta_new[:5, :5]
        result_epoch["Theta_norm"] = Theta_norm
        result_epoch["train_eval"] = {"acc": acc}

        if valid_data is not None:
            result_epoch["valid_eval"] = eval(model, valid_data, batch_size=16)
        result_per_epoch.append(result_epoch)
    return result_per_epoch


def get_eval_result(s_true, s_pred, s_pred_label):
    acc = accuracy_score(s_true, s_pred_label)
    f1 = f1_score(s_true, s_pred_label)
    rmse = np.sqrt(mean_squared_error(s_true, s_pred))
    print("acc = %.4f f1 = %.4f rmse = %.4f" % (acc, f1, rmse))
    return {"acc": acc, "f1": f1, "rmse": rmse}


def calculate_rd(theta_old: torch.Tensor, theta_new: torch.Tensor, K_old: int) -> float:
    """
    Calculate Representation Drift (RD) between old and new theta vectors.

    Args:
        theta_old: torch.Tensor (n_users, K_old), original theta from frozen model
        theta_new: torch.Tensor (n_users, K_new), new theta from dynamic model
        K_old: int, original knowledge dimension

    Returns:
        rd: float, normalized mean drift across all users
    """
    theta_new_old_dim = theta_new[:, :K_old]
    per_user_norm = torch.norm(theta_old - theta_new_old_dim, dim=1)
    rd = torch.mean(per_user_norm / math.sqrt(K_old))
    return rd.item()


class LinearWarmupScheduler:
    """
    Linear Warm-up Learning Rate Scheduler.

    Gradually increases learning rate from initial_lr * warmup_factor to initial_lr
    over the first warmup_epochs. After that, maintains constant learning rate.

    Args:
        optimizer: torch.optim.Optimizer
        warmup_epochs: int, number of warmup epochs
        warmup_factor: float, initial learning rate factor (default: 0.01)
    """

    def __init__(self, optimizer, warmup_epochs: int = 5, warmup_factor: float = 0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: int = None):
        """Update learning rates based on current epoch."""
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * progress

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * lr_factor
        else:
            # After warmup, restore base learning rates
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i]

        self.current_epoch += 1


def eval_incremental(
    model_old,
    model_dynamic,
    data: pd.DataFrame,
    original_n_item: int,
    original_n_know: int,
    batch_size: int = 16,
):
    """
    Decoupled evaluation for incremental learning.

    Separately evaluates performance on old items and new items,
    and calculates Representation Drift (RD).

    Args:
        model_old: Frozen teacher model
        model_dynamic: Dynamic student model
        data: pd.DataFrame, validation/test data
        original_n_item: int, number of original items (before expansion)
        original_n_know: int, number of original knowledge concepts
        batch_size: int, batch size for evaluation

    Returns:
        eval_result: dict, containing AUC_old, RMSE_old, AUC_new, RMSE_new, and RD
    """
    model_old.eval()
    model_dynamic.eval()
    device = model_dynamic.device

    eval_result = {}

    # Prepare dataset
    dataset = IDCDataset(data, model_dynamic.n_user, model_dynamic.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Separate predictions for old and new items
    y_true_old = []
    y_pred_old = []
    y_true_new = []
    y_pred_new = []

    with torch.no_grad():
        for user_log, item_log, user_id, item_id, score in dataloader:
            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)

            # Forward pass
            pred = model_dynamic(user_log, item_log, user_id, item_id).detach().cpu()
            score_cpu = score.detach().cpu()
            item_id_cpu = item_id.detach().cpu()

            # Separate old and new items based on item_id
            old_mask = (item_id_cpu < original_n_item).squeeze()
            new_mask = ~old_mask

            if old_mask.any():
                y_true_old.extend(score_cpu[old_mask].tolist())
                y_pred_old.extend(pred[old_mask].tolist())

            if new_mask.any():
                y_true_new.extend(score_cpu[new_mask].tolist())
                y_pred_new.extend(pred[new_mask].tolist())

    # Convert to numpy arrays
    y_true_old = np.array(y_true_old)
    y_pred_old = np.array(y_pred_old)
    y_true_new = np.array(y_true_new)
    y_pred_new = np.array(y_pred_new)

    # Calculate metrics for old items
    if len(y_true_old) > 0:
        y_plab_old = (y_pred_old > 0.5).astype(int)
        print("=== Old Items Evaluation ===")
        eval_result["old"] = get_eval_result(y_true_old, y_pred_old, y_plab_old)
    else:
        eval_result["old"] = {"acc": 0.0, "f1": 0.0, "rmse": 0.0}

    # Calculate metrics for new items
    if len(y_true_new) > 0:
        y_plab_new = (y_pred_new > 0.5).astype(int)
        print("=== New Items Evaluation ===")
        eval_result["new"] = get_eval_result(y_true_new, y_pred_new, y_plab_new)
    else:
        eval_result["new"] = {"acc": 0.0, "f1": 0.0, "rmse": 0.0}

    # Calculate RD
    print("=== Representation Drift (RD) ===")
    with torch.no_grad():
        # Get theta from old model (using dummy user_log to trigger forward)
        sample_user_log = torch.zeros(1, model_old.n_item).to(device)
        theta_old = model_old.diagnose_theta(sample_user_log)

        # Get theta from dynamic model
        sample_user_log_dynamic = torch.zeros(1, model_dynamic.n_item).to(device)
        theta_new = model_dynamic.diagnose_theta(sample_user_log_dynamic)

        rd = calculate_rd(theta_old, theta_new, original_n_know)
        print(f"RD = {rd:.6f}")
        eval_result["rd"] = rd

    return eval_result


def train_incremental(
    model_old,
    model_dynamic,
    train_data_old: pd.DataFrame,
    train_data_new: pd.DataFrame,
    valid_data: pd.DataFrame,
    original_n_know: int,
    V_old: int,
    V_new: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_epoch: int = 50,
):
    """
    Incremental training loop with decoupled loss and LR warmup.

    Args:
        model_old: Frozen teacher model
        model_dynamic: Dynamic student model (already expanded)
        train_data_old: pd.DataFrame, original training data
        train_data_new: pd.DataFrame, new incremental training data
        valid_data: pd.DataFrame, validation data
        original_n_know: int, original knowledge dimension
        V_old: int, number of non-zero elements in old Q-matrix
        V_new: int, number of non-zero elements in new Q-matrix delta
        batch_size: int, batch size
        lr: float, learning rate
        n_epoch: int, number of epochs

    Returns:
        result_per_epoch: list of dicts, training results per epoch
    """
    from loss import TopologyAwareDecoupledLoss

    device = model_dynamic.device

    # Create datasets and dataloaders
    dataset_old = IDCDataset(train_data_old, model_dynamic.n_user, model_dynamic.n_item)
    dataset_new = IDCDataset(train_data_new, model_dynamic.n_user, model_dynamic.n_item)

    old_loader = DataLoader(dataset=dataset_old, batch_size=batch_size, shuffle=True)
    new_loader = DataLoader(dataset=dataset_new, batch_size=batch_size, shuffle=True)

    # Create hybrid dataloader
    hybrid_loader = AsymmetricHybridDataLoader(old_loader, new_loader, old_ratio=0.7, device=device)

    # Filter only active parameters (requires_grad=True)
    active_params = [param for param in model_dynamic.parameters() if param.requires_grad]
    print(f"Number of active parameters: {len(active_params)}")

    # Create optimizer with only active parameters
    optimizer = torch.optim.Adam(active_params, lr=lr)

    # Create LR warmup scheduler
    scheduler = LinearWarmupScheduler(optimizer, warmup_epochs=5, warmup_factor=0.01)

    # Initialize decoupled loss function
    loss_fn = TopologyAwareDecoupledLoss(
        model_old=model_old,
        model_dynamic=model_dynamic,
        original_know_dim=original_n_know,
        device=device,
    )

    result_per_epoch = []

    for epoch in range(n_epoch):
        model_dynamic.train()
        result_epoch = {}
        pbar = tqdm(total=len(hybrid_loader), desc=f"Epoch {epoch}")

        total_loss = 0.0
        count = 0

        for batch in hybrid_loader:
            user_log, item_log, user_id, item_id, score, is_new = batch

            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)
            is_new = is_new.to(device)

            # Compute decoupled loss
            loss, loss_components = loss_fn(
                user_log=user_log,
                item_log=item_log,
                user_id=user_id,
                item_id=item_id,
                score=score,
                is_new=is_new,
                epoch=epoch,
                total_epochs=n_epoch,
                V_old=V_old,
                V_new=V_new,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "L_old": f"{loss_components['L_old']:.6f}",
                    "L_new": f"{loss_components['L_new']:.6f}",
                    "alpha": f"{loss_components['alpha']:.4f}",
                }
            )
            pbar.update(1)

        pbar.close()

        # Update scheduler
        scheduler.step(epoch)

        # Log epoch results
        avg_loss = total_loss / count
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")
        result_epoch["avg_loss"] = avg_loss
        result_epoch["loss_components"] = loss_components

        # Incremental evaluation
        if valid_data is not None:
            eval_result = eval_incremental(
                model_old=model_old,
                model_dynamic=model_dynamic,
                data=valid_data,
                original_n_item=model_old.n_item,
                original_n_know=original_n_know,
                batch_size=16,
            )
            result_epoch["valid_eval"] = eval_result

        result_per_epoch.append(result_epoch)

    return result_per_epoch


def eval(model: GNCDM, data: pd.DataFrame, batch_size):
    model.eval()
    device = model.device
    eval_result = {}
    dataset = IDCDataset(data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    y_pred_1 = []
    y_pred_2 = []
    y_true = []
    for i, (user_log, item_log, user_id, item_id, score) in enumerate(dataloader):
        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        pred_1_batch = model.forward_using_buf(user_id, item_id).detach().cpu().tolist()
        pred_2_batch = model.forward(user_log, item_log, user_id, item_id).detach().cpu().tolist()
        y_pred_1 += pred_1_batch
        y_pred_2 += pred_2_batch
        y_true += score.detach().tolist()
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)
    y_plab_1 = (y_pred_1 > 0.5).astype(int)
    y_plab_2 = (y_pred_2 > 0.5).astype(int)
    # print('True: #pos/all =',np.sum(y_true)/len(y_true))
    print("Score Prediction:")
    eval_result = get_eval_result(y_true, y_pred_1, y_plab_1)
    # print('#pos/all =',np.sum(y_plab_1)/len(y_plab_1))
    print("Score Reconstruction:")
    eval_result["without_buf"] = get_eval_result(y_true, y_pred_2, y_plab_2)
    # print('#pos/all =',np.sum(y_plab_2)/len(y_plab_2))
    return eval_result


# 2025.04.21. Add train_AE and eval_AE
def train_AE(
    model: UAutoRec, train_data: pd.DataFrame, valid_data: pd.DataFrame, batch_size, lr, n_epoch
):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset = AEDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(n_epoch):
        pbar = tqdm(total=len(dataloader), desc="Epoch %d" % epoch)
        avg_loss = 0
        count = 0
        for idx, (log_vec, obs_vec, user_id) in enumerate(dataloader):
            log_vec = log_vec.to(model.device)
            obs_vec = obs_vec.to(model.device)
            user_id = user_id.to(model.device)
            y_pred = model(log_vec, user_id).reshape(
                -1,
            )
            y_true = log_vec.reshape(
                -1,
            )
            obs = obs_vec.reshape(
                -1,
            )

            # loss
            loss = loss_func(y_pred * obs, y_true * obs)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            count += 1
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
        # avg_loss /= count
        print("Epoch %d done. loss = %.6f" % (epoch, avg_loss))
        pbar.close()
        eval_AE(model, train_data, valid_data)


def eval_AE(model: UAutoRec, train_data: pd.DataFrame, test_data: pd.DataFrame):
    y_pred_mat = []
    y_pred_all = []
    y_true_all = []
    train_dataset = AEDataset(train_data, model.n_user, model.n_item)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_dataset = AEDataset(test_data, model.n_user, model.n_item)
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for i, (log_vec, obs_vec, user_id) in enumerate(train_dataloader):
            log_vec = log_vec.to(model.device)
            user_id = user_id.to(model.device)
            y_pred = model(log_vec, user_id).detach().cpu().numpy()
            y_pred_mat += [elem for elem in y_pred]
    y_pred_mat = np.stack(y_pred_mat)
    xs, ys = np.where(test_dataset.obs_mat > 0)
    for i, j in zip(xs, ys):
        y_pred_all.append(y_pred_mat[i, j])
        y_true_all.append(test_dataset.log_mat[i, j])
    y_pred_label = [elem > 0.5 for elem in y_pred_all]
    return get_eval_result(y_true_all, y_pred_all, y_pred_label)
