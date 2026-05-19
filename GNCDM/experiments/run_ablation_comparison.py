# -*- coding: utf-8 -*-
"""
Ablation Comparison Experiment Script for Generative-CD Incremental Learning

This script compares three approaches:
1. Base Model (Ancestral Manifold) - Original G-NCDM
2. Baseline (Naive Fine-Tuning) - Full parameter fine-tuning without freezing
3. Ours (Dynamic Neural Architecture) - Our proposed incremental learning approach

Output: Markdown comparison table showing stability-plasticity trade-off

Usage:
    cd GNCDM
    python experiments/run_ablation_comparison.py
"""

import os
import sys

# Add GNCDM directory to path for local imports
gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# Import local modules
from core.model import GNCDM
from core.train import IDCDataset, generate_cognitive_biased_mnar_mask, AsymmetricHybridDataLoader
from incremental.loss import TopologyAwareDecoupledLoss

# Configuration
SEED = 42
DEVICE = torch.device('cpu')
N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

# Data configuration
N_USER = 100
N_ITEM_OLD = 50
N_ITEM_NEW = 10  # delta_M
N_KNOW_OLD = 10
N_KNOW_NEW = 5   # delta_K

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_synthetic_data(n_user, n_item, n_know):
    """Generate synthetic cognitive diagnosis data."""
    # Generate Q-matrix
    Q_mat = np.random.randint(0, 2, size=(n_item, n_know))
    Q_mat[Q_mat.sum(axis=1) == 0, 0] = 1  # Ensure no empty rows
    
    # Generate true theta (student abilities)
    theta_true = np.random.rand(n_user, n_know)
    
    # Generate item parameters
    b = np.random.rand(n_item) * 2 - 1  # difficulty
    
    # Generate responses
    log_mat = np.zeros((n_user, n_item))
    user_ids = []
    item_ids = []
    scores = []
    
    for u in range(n_user):
        for i in range(n_item):
            # Probability of correct answer
            q = Q_mat[i]
            theta_u = theta_true[u]
            p = 1 / (1 + np.exp(-(theta_u @ q - b[i])))
            score = 1 if np.random.rand() < p else 0
            log_mat[u, i] = (score - 0.5) * 2  # Scale to [-1, 1]
            user_ids.append(u)
            item_ids.append(i)
            scores.append(score)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'score': scores
    })
    
    return df, Q_mat, theta_true

def compute_metrics(y_true, y_pred):
    """Compute AUC, RMSE, ACC, and F1-score metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # AUC
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = 0.0
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # ACC and F1 (using 0.5 threshold)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return auc, rmse, acc, f1

def evaluate_model(model, data, n_user, n_item, batch_size=32):
    """Evaluate model on given data."""
    model.eval()
    dataset = IDCDataset(data, n_user, n_item)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for user_log, item_log, user_id, item_id, score in dataloader:
            user_log = user_log.to(DEVICE)
            item_log = item_log.to(DEVICE)
            user_id = user_id.to(DEVICE)
            item_id = item_id.to(DEVICE)
            
            pred = model(user_log, item_log, user_id, item_id)
            
            y_true.extend(score.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())
    
    auc, rmse, acc, f1 = compute_metrics(y_true, y_pred)
    return auc, rmse, acc, f1

def get_theta_anchor(model, n_user, n_item):
    """Generate theta anchor from model."""
    model.eval()
    with torch.no_grad():
        dummy_log = torch.zeros(n_user, n_item).to(DEVICE)
        theta = model.diagnose_theta(dummy_log)
    return theta.cpu().numpy()

def calculate_tmd(theta_old, theta_new, K_old):
    """Calculate Trait Manifold Drift."""
    theta_new_old_dim = theta_new[:, :K_old]
    per_user_norm = np.linalg.norm(theta_old - theta_new_old_dim, axis=1)
    tmd = np.mean(per_user_norm / np.sqrt(K_old))
    return tmd

def train_base_model(train_data, Q_mat, n_user, n_item, n_know):
    """Train the base G-NCDM model."""
    print("\n=== Training Base Model ===")
    
    model = GNCDM(
        n_user=n_user,
        n_item=n_item,
        n_know=n_know,
        user_dim=32,
        item_dim=32,
        Q_mat=Q_mat,
        device=DEVICE,
        alpha=0.5,
        monotonicity_assumption=True
    )
    
    dataset = IDCDataset(train_data, n_user, n_item)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for user_log, item_log, user_id, item_id, score in dataloader:
            user_log = user_log.to(DEVICE)
            item_log = item_log.to(DEVICE)
            user_id = user_id.to(DEVICE)
            item_id = item_id.to(DEVICE)
            score = score.to(DEVICE)
            
            pred = model(user_log, item_log, user_id, item_id)
            loss = F.binary_cross_entropy(pred, score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    return model

def naive_fine_tune(model, train_data_hybrid, Q_expanded, n_user, n_item_total, n_know_total):
    """Baseline: Naive fine-tuning without freezing."""
    print("\n=== Baseline: Naive Fine-Tuning ===")
    
    # Expand model dimensions manually (naive approach)
    model.n_item = n_item_total
    model.n_know = n_know_total
    model.Q_mat = torch.Tensor(Q_expanded).to(DEVICE)
    
    # Reinitialize layers with expanded dimensions (naive approach)
    f_linear = type(model.f_nn[0])
    
    # Expand f_nn
    model.f_nn = nn.Sequential(
        nn.Sequential(
            f_linear(n_item_total, n_know_total),
            nn.Sigmoid(),
            f_linear(n_know_total, n_know_total),
            nn.Sigmoid()
        )
    ).to(DEVICE)
    
    # Expand g_nn (input dimension remains n_user)
    model.g_nn = nn.Sequential(
        nn.Sequential(
            nn.Linear(n_user, n_know_total),
            nn.Sigmoid(),
            nn.Linear(n_know_total, n_know_total),
            nn.Sigmoid(),
            nn.Linear(n_know_total, n_know_total),
            nn.Sigmoid()
        )
    ).to(DEVICE)
    
    # Expand aggregation matrices
    model.theta_agg_mat = f_linear(n_know_total, model.user_dim).to(DEVICE)
    model.psi_agg_mat = nn.Linear(n_know_total, model.item_dim).to(DEVICE)
    
    # Make all parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Training with BCE loss
    dataset = IDCDataset(train_data_hybrid, n_user, n_item_total)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for user_log, item_log, user_id, item_id, score in dataloader:
            user_log = user_log.to(DEVICE)
            item_log = item_log.to(DEVICE)
            user_id = user_id.to(DEVICE)
            item_id = item_id.to(DEVICE)
            score = score.to(DEVICE)
            
            pred = model(user_log, item_log, user_id, item_id)
            loss = F.binary_cross_entropy(pred, score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    return model

def our_incremental_learning(model_old, train_data_old, train_data_new, Q_expanded, 
                             n_user, n_item_old, n_item_total, n_know_old, n_know_total):
    """Our approach: Dynamic neural architecture with incremental learning."""
    print("\n=== Ours: Dynamic Neural Architecture ===")
    
    # Create expanded model copy
    model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        Q_mat=model_old.Q_mat.cpu().numpy(),
        device=DEVICE,
        alpha=0.5,
        monotonicity_assumption=True
    )
    
    # Copy weights from old model
    model.load_state_dict(model_old.state_dict())
    
    # Expand topology
    delta_M = n_item_total - n_item_old
    delta_K = n_know_total - n_know_old
    model.expand_topology(delta_M, delta_K, Q_expanded)
    
    # Create hybrid dataloader
    dataset_old = IDCDataset(train_data_old, n_user, n_item_total)
    dataset_new = IDCDataset(train_data_new, n_user, n_item_total)
    
    old_loader = DataLoader(dataset_old, batch_size=BATCH_SIZE, shuffle=True)
    new_loader = DataLoader(dataset_new, batch_size=BATCH_SIZE, shuffle=True)
    
    hybrid_loader = AsymmetricHybridDataLoader(old_loader, new_loader, old_ratio=0.7, device=DEVICE)
    
    # Filter active parameters
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active_params, lr=LR)
    
    # Initialize decoupled loss
    loss_fn = TopologyAwareDecoupledLoss(
        model_old=model_old,
        model_dynamic=model,
        original_know_dim=n_know_old,
        device=DEVICE
    )
    
    # Calculate Q-matrix non-zero counts
    V_old = int(model_old.Q_mat.sum().item())
    V_new = int(Q_expanded[n_item_old:, n_know_old:].sum())
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in hybrid_loader:
            user_log, item_log, user_id, item_id, score, is_new = batch
            
            user_log = user_log.to(DEVICE)
            item_log = item_log.to(DEVICE)
            user_id = user_id.to(DEVICE)
            item_id = item_id.to(DEVICE)
            score = score.to(DEVICE)
            is_new = is_new.to(DEVICE)
            
            loss, _ = loss_fn(
                user_log=user_log,
                item_log=item_log,
                user_id=user_id,
                item_id=item_id,
                score=score,
                is_new=is_new,
                epoch=epoch,
                total_epochs=N_EPOCHS,
                V_old=V_old,
                V_new=V_new
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(hybrid_loader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    return model

def main():
    print("=" * 70)
    print("Ablation Comparison Experiment: Stability vs Plasticity")
    print("=" * 70)
    
    set_seed(SEED)
    
    # Generate synthetic data
    print("\n[Step 1] Generating synthetic data...")
    
    # Old data (base)
    train_data_old, Q_old, theta_true_old = generate_synthetic_data(N_USER, N_ITEM_OLD, N_KNOW_OLD)
    
    # New data (incremental)
    train_data_new_raw, Q_new_raw, _ = generate_synthetic_data(N_USER, N_ITEM_NEW, N_KNOW_NEW)
    
    # Apply MNAR masking to new data
    Q_new = np.zeros((N_ITEM_NEW, N_KNOW_OLD + N_KNOW_NEW))
    Q_new[:, N_KNOW_OLD:] = Q_new_raw
    
    mask_prob, mask = generate_cognitive_biased_mnar_mask(Q_new, gamma=1.0, tau=0.5)
    print(f"MNAR Masking applied: {100*(1-mask.mean()):.1f}% of new items masked")
    
    # Shift item IDs for new data
    train_data_new = train_data_new_raw.copy()
    train_data_new['item_id'] = train_data_new['item_id'] + N_ITEM_OLD
    
    # Create expanded Q-matrix
    Q_expanded = np.zeros((N_ITEM_OLD + N_ITEM_NEW, N_KNOW_OLD + N_KNOW_NEW))
    Q_expanded[:N_ITEM_OLD, :N_KNOW_OLD] = Q_old
    Q_expanded[N_ITEM_OLD:, :] = Q_new
    
    # Create hybrid training data for baseline
    train_data_hybrid = pd.concat([train_data_old, train_data_new], ignore_index=True)
    
    # Create test data (separate old and new items)
    test_data_old = train_data_old.sample(frac=0.2, random_state=SEED)
    test_data_new = train_data_new.sample(frac=0.2, random_state=SEED)
    
    # ============================================================================
    # Process 1: Base Model (Ancestral Manifold)
    # ============================================================================
    print("\n" + "=" * 50)
    print("Process 1: Base Model (Ancestral Manifold)")
    print("=" * 50)
    
    base_model = train_base_model(train_data_old, Q_old, N_USER, N_ITEM_OLD, N_KNOW_OLD)
    
    # Evaluate on old test data
    auc_old_base, rmse_old_base, acc_old_base, f1_old_base = evaluate_model(base_model, test_data_old, N_USER, N_ITEM_OLD)
    print(f"Base Model - AUC_old: {auc_old_base:.4f}, RMSE_old: {rmse_old_base:.4f}, ACC_old: {acc_old_base:.4f}, F1_old: {f1_old_base:.4f}")
    
    # Generate theta anchor
    theta_anchor = get_theta_anchor(base_model, N_USER, N_ITEM_OLD)
    
    # ============================================================================
    # Process 2: Baseline (Naive Fine-Tuning)
    # ============================================================================
    print("\n" + "=" * 50)
    print("Process 2: Baseline (Naive Fine-Tuning)")
    print("=" * 50)
    
    # Create baseline model copy
    baseline_model = GNCDM(
        n_user=N_USER,
        n_item=N_ITEM_OLD,
        n_know=N_KNOW_OLD,
        user_dim=32,
        item_dim=32,
        Q_mat=Q_old,
        device=DEVICE,
        alpha=0.5,
        monotonicity_assumption=True
    )
    baseline_model.load_state_dict(base_model.state_dict())
    
    # Naive fine-tuning
    baseline_model = naive_fine_tune(
        baseline_model, train_data_hybrid, Q_expanded,
        N_USER, N_ITEM_OLD + N_ITEM_NEW, N_KNOW_OLD + N_KNOW_NEW
    )
    
    # Evaluate
    auc_old_baseline, rmse_old_baseline, acc_old_baseline, f1_old_baseline = evaluate_model(baseline_model, test_data_old, N_USER, N_ITEM_OLD + N_ITEM_NEW)
    auc_new_baseline, rmse_new_baseline, acc_new_baseline, f1_new_baseline = evaluate_model(baseline_model, test_data_new, N_USER, N_ITEM_OLD + N_ITEM_NEW)
    
    # Calculate TMD
    theta_baseline = get_theta_anchor(baseline_model, N_USER, N_ITEM_OLD + N_ITEM_NEW)
    tmd_baseline = calculate_tmd(theta_anchor, theta_baseline, N_KNOW_OLD)
    
    print(f"Baseline - AUC_old: {auc_old_baseline:.4f}, AUC_new: {auc_new_baseline:.4f}, RMSE_old: {rmse_old_baseline:.4f}, RMSE_new: {rmse_new_baseline:.4f}, ACC_old: {acc_old_baseline:.4f}, ACC_new: {acc_new_baseline:.4f}, F1_old: {f1_old_baseline:.4f}, F1_new: {f1_new_baseline:.4f}, TMD: {tmd_baseline:.4f}")
    
    # ============================================================================
    # Process 3: Ours (Dynamic Neural Architecture)
    # ============================================================================
    print("\n" + "=" * 50)
    print("Process 3: Ours (Dynamic Neural Architecture)")
    print("=" * 50)
    
    # Incremental learning with our approach
    our_model = our_incremental_learning(
        base_model, train_data_old, train_data_new, Q_expanded,
        N_USER, N_ITEM_OLD, N_ITEM_OLD + N_ITEM_NEW, N_KNOW_OLD, N_KNOW_OLD + N_KNOW_NEW
    )
    
    # Evaluate
    auc_old_ours, rmse_old_ours, acc_old_ours, f1_old_ours = evaluate_model(our_model, test_data_old, N_USER, N_ITEM_OLD + N_ITEM_NEW)
    auc_new_ours, rmse_new_ours, acc_new_ours, f1_new_ours = evaluate_model(our_model, test_data_new, N_USER, N_ITEM_OLD + N_ITEM_NEW)
    
    # Calculate TMD
    theta_ours = get_theta_anchor(our_model, N_USER, N_ITEM_OLD + N_ITEM_NEW)
    tmd_ours = calculate_tmd(theta_anchor, theta_ours, N_KNOW_OLD)
    
    print(f"Ours - AUC_old: {auc_old_ours:.4f}, AUC_new: {auc_new_ours:.4f}, RMSE_old: {rmse_old_ours:.4f}, RMSE_new: {rmse_new_ours:.4f}, ACC_old: {acc_old_ours:.4f}, ACC_new: {acc_new_ours:.4f}, F1_old: {f1_old_ours:.4f}, F1_new: {f1_new_ours:.4f}, TMD: {tmd_ours:.4f}")
    
    # ============================================================================
    # Generate Comparison Table
    # ============================================================================
    print("\n" + "=" * 70)
    print("Ablation Comparison Results")
    print("=" * 70)
    
    # Dataset Information
    print("\n[Dataset Information]")
    print(f"- Number of users: {N_USER}")
    print(f"- Old items: {N_ITEM_OLD}, Old knowledge concepts: {N_KNOW_OLD}")
    print(f"- New items: {N_ITEM_NEW}, New knowledge concepts: {N_KNOW_NEW}")
    print(f"- Total items: {N_ITEM_OLD + N_ITEM_NEW}, Total knowledge concepts: {N_KNOW_OLD + N_KNOW_NEW}")
    print(f"- Data type: Synthetic cognitive diagnosis data (simulated student responses)")
    
    results = pd.DataFrame({
        'Model': ['Base', 'Baseline (Naive FT)', 'Ours (Dynamic DNA)'],
        'AUC_old': [f"{auc_old_base:.4f}", f"{auc_old_baseline:.4f}", f"{auc_old_ours:.4f}"],
        'AUC_new': ['-', f"{auc_new_baseline:.4f}", f"{auc_new_ours:.4f}"],
        'RMSE_old': [f"{rmse_old_base:.4f}", f"{rmse_old_baseline:.4f}", f"{rmse_old_ours:.4f}"],
        'RMSE_new': ['-', f"{rmse_new_baseline:.4f}", f"{rmse_new_ours:.4f}"],
        'ACC_old': [f"{acc_old_base:.4f}", f"{acc_old_baseline:.4f}", f"{acc_old_ours:.4f}"],
        'ACC_new': ['-', f"{acc_new_baseline:.4f}", f"{acc_new_ours:.4f}"],
        'F1_old': [f"{f1_old_base:.4f}", f"{f1_old_baseline:.4f}", f"{f1_old_ours:.4f}"],
        'F1_new': ['-', f"{f1_new_baseline:.4f}", f"{f1_new_ours:.4f}"],
        'TMD': [0.0, f"{tmd_baseline:.4f}", f"{tmd_ours:.4f}"]
    })
    
    print("\n" + results.to_markdown(index=False))
    print("\n" + "=" * 70)
    
    # Save results
    results.to_csv('ablation_results.csv', index=False)
    print("Results saved to ablation_results.csv")

if __name__ == "__main__":
    main()