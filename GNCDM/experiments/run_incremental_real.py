# -*- coding: utf-8 -*-
"""
增量学习实验：Ours (Dynamic DNA) vs Ours (LoRA)
使用 Math1 真实数据集，2/3 作为旧数据，1/3 作为新数据
"""

import sys
import os

# Add the GNCDM directory to path
gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# Import our model
from core.model import GNCDM

# Configuration
DATA_DIR = os.path.join(gncdm_dir, 'data')
SAVE_DIR = os.path.join(gncdm_dir, 'incremental_result')

# Create save directory
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Dataset class
class IDCDataset(Dataset):
    def __init__(self, data, n_user, n_item):
        self.data = data.reset_index(drop=True)  # 重置索引
        self.n_user = n_user
        self.n_item = n_item
        
    def __getitem__(self, index):
        user_id = self.data.iloc[index]['user_id']
        item_id = self.data.iloc[index]['item_id']
        label = self.data.iloc[index]['score']
        return user_id, item_id, label
    
    def __len__(self):
        return len(self.data)

# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Evaluation function
def evaluate_model(model, test_data, n_user, n_item, base_model=None):
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        
        test_dataset = IDCDataset(test_data, n_user, n_item)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        for user_ids, item_ids, _labels in test_loader:
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            
            user_log = torch.zeros((len(user_ids), n_item))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = model(user_log, item_log, user_ids, item_ids)
            preds.extend(pred.cpu().numpy())
            labels.extend(_labels.cpu().numpy())
        
        auc = roc_auc_score(labels, preds)
        rmse = np.sqrt(mean_squared_error(labels, preds))
        pred_binary = (np.array(preds) >= 0.5).astype(int)
        acc = accuracy_score(labels, pred_binary)
        f1 = f1_score(labels, pred_binary)
        
        # TMD (Manifold Drift): measure the drift between current model and base model
        tmd = None
        if base_model is not None:
            base_model.eval()
            base_preds = []
            with torch.no_grad():
                # Base model has original dimensions (before expansion)
                # We need to use the original n_item for base_model
                base_n_item = base_model.n_item
                
                for user_ids, item_ids, _ in test_loader:
                    user_ids = user_ids.long()
                    item_ids = item_ids.long()
                    
                    # For base model, use original dimensions
                    user_log_base = torch.zeros((len(user_ids), base_n_item))
                    item_log_base = torch.zeros((len(item_ids), n_user))
                    
                    base_pred = base_model(user_log_base, item_log_base, user_ids, item_ids)
                    base_preds.extend(base_pred.cpu().numpy())
            
            # TMD: Mean Absolute Difference between current and base predictions
            # Lower TMD means less drift (better preservation of old knowledge)
            preds_np = np.array(preds)
            base_preds_np = np.array(base_preds)
            tmd = np.mean(np.abs(preds_np - base_preds_np))
        
        return {
            'auc': auc,
            'rmse': rmse,
            'acc': acc,
            'f1': f1,
            'tmd': tmd
        }

# Train base model
def train_base_model(model, train_data, valid_data, n_user, n_item, batch_size=256, lr=1e-3, n_epoch=15):
    train_dataset = IDCDataset(train_data, n_user, n_item)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_valid_auc = 0.0
    best_model_state = None
    
    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.0
        
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch}"):
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            labels = labels.float()
            
            optimizer.zero_grad()
            
            user_log = torch.zeros((len(user_ids), n_item))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        valid_result = evaluate_model(model, valid_data, n_user, n_item)
        
        print(f"  Train Loss: {train_loss:.4f}, Valid AUC: {valid_result['auc']:.4f}")
        
        if valid_result['auc'] > best_valid_auc:
            best_valid_auc = valid_result['auc']
            best_model_state = model.state_dict().copy()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Full Replay Oracle - Theoretical Upper Bound
def train_incremental_full_replay_oracle(base_model, train_data_new, train_data_old, valid_data, n_user, n_item_old, n_item_new, n_know_old, n_know_new, Q_expanded, batch_size=256, lr=1e-3, n_epoch=10):
    """
    Full Replay Oracle - Theoretical Upper Bound for incremental learning.
    This represents the ideal performance when the model has access to complete replay of old data.
    
    Features:
    - Direct matrix expansion without lateral branching
    - All parameters trainable (global fine-tuning)
    - Training on complete dataset (old + new items combined)
    - Uses original BCELoss without decoupling
    """
    # Expand topology with Full Replay Oracle method
    base_model.full_replay_oracle_expand_topology(delta_M=n_item_new, delta_K=n_know_new, Q_expanded=Q_expanded)
    
    # Prepare MIXED training data (both new AND old items) - critical for inducing forgetting
    train_data_mixed = pd.concat([train_data_old, train_data_new], ignore_index=True)
    train_dataset = IDCDataset(train_data_mixed, n_user, n_item_old + n_item_new)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer - train ALL parameters (no freezing whatsoever)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    
    # Use original BCELoss (NOT the decoupled loss)
    criterion = nn.BCELoss()
    
    best_valid_auc = 0.0
    best_model_state = None
    
    for epoch in range(n_epoch):
        base_model.train()
        train_loss = 0.0
        
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"Oracle Epoch {epoch+1}/{n_epoch}"):
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            labels = labels.float()
            
            optimizer.zero_grad()
            
            user_log = torch.zeros((len(user_ids), n_item_old + n_item_new))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = base_model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        valid_result = evaluate_model(base_model, valid_data, n_user, n_item_old + n_item_new)
        
        print(f"  Train Loss: {train_loss:.4f}, Valid AUC: {valid_result['auc']:.4f}")
        
        if valid_result['auc'] > best_valid_auc:
            best_valid_auc = valid_result['auc']
            best_model_state = base_model.state_dict().copy()
    
    if best_model_state is not None:
        base_model.load_state_dict(best_model_state)
    
    return base_model

# Naive Fine-Tuning (NFT) - Catastrophic Forgetting Control Group
def train_incremental_nft(base_model, train_data_new, valid_data, n_user, n_item_old, n_item_new, n_know_old, n_know_new, Q_expanded, batch_size=256, lr=1e-3, n_epoch=10):
    """
    Naive Fine-Tuning (NFT) - Catastrophic Forgetting Control Group.
    This baseline intentionally induces maximum catastrophic forgetting by:
    - Training ONLY on new data without any replay of old data
    - Fine-tuning ALL parameters globally
    - Using simple BCELoss without any decoupling mechanism
    
    Features:
    - Direct matrix expansion using full_replay_oracle_expand_topology
    - All parameters trainable (global fine-tuning)
    - Training ONLY on new data (NO old data replay)
    - Uses original BCELoss without decoupling
    
    This is designed to measure the worst-case forgetting scenario.
    """
    # Expand topology (same structure as Full Replay Oracle for fair comparison)
    base_model.full_replay_oracle_expand_topology(delta_M=n_item_new, delta_K=n_know_new, Q_expanded=Q_expanded)
    
    # CRITICAL: Train ONLY on NEW data - NO old data replay
    # This forces the model to adapt to new knowledge without remembering old tasks
    train_dataset = IDCDataset(train_data_new, n_user, n_item_old + n_item_new)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer - train ALL parameters (no freezing whatsoever)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    
    # Use original BCELoss (NOT the decoupled loss)
    criterion = nn.BCELoss()
    
    best_valid_auc = 0.0
    best_model_state = None
    
    for epoch in range(n_epoch):
        base_model.train()
        train_loss = 0.0
        
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"NFT Epoch {epoch+1}/{n_epoch}"):
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            labels = labels.float()
            
            optimizer.zero_grad()
            
            user_log = torch.zeros((len(user_ids), n_item_old + n_item_new))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = base_model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        valid_result = evaluate_model(base_model, valid_data, n_user, n_item_old + n_item_new)
        
        print(f"  Train Loss: {train_loss:.4f}, Valid AUC: {valid_result['auc']:.4f}")
        
        if valid_result['auc'] > best_valid_auc:
            best_valid_auc = valid_result['auc']
            best_model_state = base_model.state_dict().copy()
    
    if best_model_state is not None:
        base_model.load_state_dict(best_model_state)
    
    return base_model

# Incremental training with Ours-Ablated (with lateral branching but without mathematical constraints)
def train_incremental_baseline(base_model, train_data_new, valid_data, n_user, n_item_old, n_item_new, n_know_old, n_know_new, Q_expanded, batch_size=256, lr=1e-3, n_epoch=10):
    # Expand topology WITH lateral branching (f_nn_new, g_nn_new)
    base_model.expand_topology(delta_M=n_item_new, delta_K=n_know_new, Q_expanded=Q_expanded)
    
    # Prepare training data (only new items)
    train_dataset = IDCDataset(train_data_new, n_user, n_item_old + n_item_new)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer - train ALL parameters (no freezing)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_valid_auc = 0.0
    best_model_state = None
    
    for epoch in range(n_epoch):
        base_model.train()
        train_loss = 0.0
        
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"Ours-Ablated Epoch {epoch+1}/{n_epoch}"):
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            labels = labels.float()
            
            optimizer.zero_grad()
            
            user_log = torch.zeros((len(user_ids), n_item_old + n_item_new))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = base_model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        valid_result = evaluate_model(base_model, valid_data, n_user, n_item_old + n_item_new)
        
        print(f"  Train Loss: {train_loss:.4f}, Valid AUC: {valid_result['auc']:.4f}")
        
        if valid_result['auc'] > best_valid_auc:
            best_valid_auc = valid_result['auc']
            best_model_state = base_model.state_dict().copy()
    
    if best_model_state is not None:
        base_model.load_state_dict(best_model_state)
    
    return base_model

# Incremental training with Dynamic DNA
def train_incremental_dna(base_model, train_data_new, valid_data, n_user, n_item_old, n_item_new, n_know_old, n_know_new, Q_expanded, batch_size=256, lr=1e-3, n_epoch=10):
    # Expand topology using Dynamic DNA
    base_model.expand_topology(delta_M=n_item_new, delta_K=n_know_new, Q_expanded=Q_expanded)
    
    # Prepare training data (only new items)
    train_dataset = IDCDataset(train_data_new, n_user, n_item_old + n_item_new)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer - only train new parameters
    params_to_train = []
    for name, param in base_model.named_parameters():
        if 'new' in name.lower():
            params_to_train.append(param)
    
    optimizer = torch.optim.Adam(params_to_train, lr=lr)
    criterion = nn.BCELoss()
    
    best_valid_auc = 0.0
    best_model_state = None
    
    for epoch in range(n_epoch):
        base_model.train()
        train_loss = 0.0
        
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"DNA Epoch {epoch+1}/{n_epoch}"):
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            labels = labels.float()
            
            optimizer.zero_grad()
            
            user_log = torch.zeros((len(user_ids), n_item_old + n_item_new))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = base_model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        valid_result = evaluate_model(base_model, valid_data, n_user, n_item_old + n_item_new)
        
        print(f"  Train Loss: {train_loss:.4f}, Valid AUC: {valid_result['auc']:.4f}")
        
        if valid_result['auc'] > best_valid_auc:
            best_valid_auc = valid_result['auc']
            best_model_state = base_model.state_dict().copy()
    
    if best_model_state is not None:
        base_model.load_state_dict(best_model_state)
    
    return base_model

# Incremental training with LoRA
def train_incremental_lora(base_model, train_data_new, valid_data, n_user, n_item_old, n_item_new, n_know_old, n_know_new, Q_expanded, rank=16, batch_size=256, lr=1e-3, n_epoch=10):
    # Expand topology using LoRA
    base_model.expand_topology_lora(delta_M=n_item_new, delta_K=n_know_new, Q_expanded=Q_expanded, M_old=n_item_old, rank=rank)
    
    # Prepare training data (only new items)
    train_dataset = IDCDataset(train_data_new, n_user, n_item_old + n_item_new)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer - separate parameter groups for LoRA (higher learning rate)
    lora_params = []
    other_params = []
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if 'A_new' in name or 'B_new' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': lora_params, 'lr': lr * 10}  # 10x higher LR for LoRA
    ])
    criterion = nn.BCELoss()
    
    best_valid_auc = 0.0
    best_model_state = None
    
    for epoch in range(n_epoch):
        base_model.train()
        train_loss = 0.0
        
        for user_ids, item_ids, labels in tqdm(train_loader, desc=f"LoRA Epoch {epoch+1}/{n_epoch}"):
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            labels = labels.float()
            
            optimizer.zero_grad()
            
            user_log = torch.zeros((len(user_ids), n_item_old + n_item_new))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = base_model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        valid_result = evaluate_model(base_model, valid_data, n_user, n_item_old + n_item_new)
        
        print(f"  Train Loss: {train_loss:.4f}, Valid AUC: {valid_result['auc']:.4f}")
        
        if valid_result['auc'] > best_valid_auc:
            best_valid_auc = valid_result['auc']
            best_model_state = base_model.state_dict().copy()
    
    if best_model_state is not None:
        base_model.load_state_dict(best_model_state)
    
    return base_model

def main():
    set_seed(42)
    
    print("=" * 80)
    print("增量学习实验 - Math1 真实数据集")
    print("2/3 数据作为旧数据，1/3 数据作为新数据")
    print("=" * 80)
    print()
    
    # Load data
    print("加载 Math1 数据集...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'math1_train_0.8_0.2.csv'))
    df_valid = pd.read_csv(os.path.join(DATA_DIR, 'math1_valid_0.8_0.2.csv'))
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'math1_test_0.8_0.2.csv'))
    
    Q_mat = np.load(os.path.join(DATA_DIR, 'math1_Q_matrix.npy'))
    
    n_user = 4209
    n_item_total = 20
    n_know_total = 11
    
    # Split items: 2/3 old, 1/3 new
    n_item_old = int(n_item_total * 2 / 3)  # 13 items
    n_item_new = n_item_total - n_item_old  # 7 items
    
    # Split knowledge: 2/3 old, 1/3 new
    n_know_old = int(n_know_total * 2 / 3)  # 7 concepts
    n_know_new = n_know_total - n_know_old  # 4 concepts
    
    print(f"数据集划分:")
    print(f"  旧题目数: {n_item_old}, 新知识点数: {n_know_old}")
    print(f"  新题目数: {n_item_new}, 新知识点数: {n_know_new}")
    print()
    
    # Old Q matrix (first n_item_old items, first n_know_old concepts)
    Q_old = Q_mat[:n_item_old, :n_know_old].copy()
    
    # Expanded Q matrix (full)
    Q_expanded = Q_mat.copy()
    
    # Split training data
    train_df_old = df_train[df_train['item_id'] < n_item_old].copy()
    train_df_new = df_train[df_train['item_id'] >= n_item_old].copy()
    
    # Split validation data
    valid_df_old = df_valid[df_valid['item_id'] < n_item_old].copy()
    
    # Split test data
    test_df_old = df_test[df_test['item_id'] < n_item_old].copy()
    test_df_new = df_test[df_test['item_id'] >= n_item_old].copy()
    
    print(f"训练数据: 旧题目 {len(train_df_old)} 条, 新题目 {len(train_df_new)} 条")
    print(f"验证数据: 旧题目 {len(valid_df_old)} 条")
    print(f"测试数据: 旧题目 {len(test_df_old)} 条, 新题目 {len(test_df_new)} 条")
    print()
    
    # Results list
    results = []
    
    # 1. Train Base Model on old data
    print("=" * 80)
    print("1. 训练 Base 模型（仅旧数据）")
    print("=" * 80)
    
    device = torch.device('cpu')
    base_model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=0.8,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device
    )
    
    base_model = train_base_model(base_model, train_df_old, valid_df_old, n_user, n_item_old)
    
    # Evaluate base model on old test set
    base_result_old = evaluate_model(base_model, test_df_old, n_user, n_item_old)
    print(f"Base 模型 - 旧测试集: AUC={base_result_old['auc']:.4f}, RMSE={base_result_old['rmse']:.4f}, ACC={base_result_old['acc']:.4f}, F1={base_result_old['f1']:.4f}")
    
    results.append({
        'Model': 'Base',
        'AUC_old': base_result_old['auc'],
        'AUC_new': '-',
        'RMSE_old': base_result_old['rmse'],
        'RMSE_new': '-',
        'ACC_old': base_result_old['acc'],
        'ACC_new': '-',
        'F1_old': base_result_old['f1'],
        'F1_new': '-'
    })
    
    # 2. Ours-Ablated (Naive Fine-Tuning without lateral branching)
    print()
    print("=" * 80)
    print("2. Ours-Ablated (Naive Fine-Tuning)")
    print("=" * 80)
    
    baseline_model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=0.8,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device
    )
    baseline_model.load_state_dict(base_model.state_dict())
    
    baseline_model = train_incremental_baseline(
        baseline_model, train_df_new, df_valid, 
        n_user, n_item_old, n_item_new, 
        n_know_old, n_know_new, Q_expanded
    )
    
    # Evaluate
    baseline_result_old = evaluate_model(baseline_model, test_df_old, n_user, n_item_old + n_item_new, base_model)
    baseline_result_new = evaluate_model(baseline_model, test_df_new, n_user, n_item_old + n_item_new)
    
    print(f"Ours-Ablated - 旧测试集: AUC={baseline_result_old['auc']:.4f}, RMSE={baseline_result_old['rmse']:.4f}, ACC={baseline_result_old['acc']:.4f}, F1={baseline_result_old['f1']:.4f}, TMD={baseline_result_old['tmd']:.4f}")
    print(f"Ours-Ablated - 新测试集: AUC={baseline_result_new['auc']:.4f}, RMSE={baseline_result_new['rmse']:.4f}, ACC={baseline_result_new['acc']:.4f}, F1={baseline_result_new['f1']:.4f}")
    
    results.append({
        'Model': 'Ours-Ablated',
        'AUC_old': baseline_result_old['auc'],
        'AUC_new': baseline_result_new['auc'],
        'RMSE_old': baseline_result_old['rmse'],
        'RMSE_new': baseline_result_new['rmse'],
        'ACC_old': baseline_result_old['acc'],
        'ACC_new': baseline_result_new['acc'],
        'F1_old': baseline_result_old['f1'],
        'F1_new': baseline_result_new['f1'],
        'TMD': baseline_result_old['tmd']
    })
    
    # 3. Ours (Dynamic DNA)
    print()
    print("=" * 80)
    print("3. Ours (Dynamic DNA) 增量学习")
    print("=" * 80)
    
    dna_model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=0.8,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device
    )
    dna_model.load_state_dict(base_model.state_dict())
    
    dna_model = train_incremental_dna(
        dna_model, train_df_new, df_valid, 
        n_user, n_item_old, n_item_new, 
        n_know_old, n_know_new, Q_expanded
    )
    
    # Evaluate
    dna_result_old = evaluate_model(dna_model, test_df_old, n_user, n_item_old + n_item_new, base_model)
    dna_result_new = evaluate_model(dna_model, test_df_new, n_user, n_item_old + n_item_new)
    
    print(f"Ours (DNA) - 旧测试集: AUC={dna_result_old['auc']:.4f}, RMSE={dna_result_old['rmse']:.4f}, ACC={dna_result_old['acc']:.4f}, F1={dna_result_old['f1']:.4f}, TMD={dna_result_old['tmd']:.4f}")
    print(f"Ours (DNA) - 新测试集: AUC={dna_result_new['auc']:.4f}, RMSE={dna_result_new['rmse']:.4f}, ACC={dna_result_new['acc']:.4f}, F1={dna_result_new['f1']:.4f}")
    
    results.append({
        'Model': 'Ours (Dynamic DNA)',
        'AUC_old': dna_result_old['auc'],
        'AUC_new': dna_result_new['auc'],
        'RMSE_old': dna_result_old['rmse'],
        'RMSE_new': dna_result_new['rmse'],
        'ACC_old': dna_result_old['acc'],
        'ACC_new': dna_result_new['acc'],
        'F1_old': dna_result_old['f1'],
        'F1_new': dna_result_new['f1'],
        'TMD': dna_result_old['tmd']
    })
    
    # 4. Ours (LoRA)
    print()
    print("=" * 80)
    print("4. Ours (LoRA) 增量学习")
    print("=" * 80)
    
    lora_model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=0.8,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device
    )
    lora_model.load_state_dict(base_model.state_dict())
    
    lora_model = train_incremental_lora(
        lora_model, train_df_new, df_valid, 
        n_user, n_item_old, n_item_new, 
        n_know_old, n_know_new, Q_expanded,
        rank=16
    )
    
    # Evaluate
    lora_result_old = evaluate_model(lora_model, test_df_old, n_user, n_item_old + n_item_new, base_model)
    lora_result_new = evaluate_model(lora_model, test_df_new, n_user, n_item_old + n_item_new)
    
    print(f"Ours (LoRA) - 旧测试集: AUC={lora_result_old['auc']:.4f}, RMSE={lora_result_old['rmse']:.4f}, ACC={lora_result_old['acc']:.4f}, F1={lora_result_old['f1']:.4f}, TMD={lora_result_old['tmd']:.4f}")
    print(f"Ours (LoRA) - 新测试集: AUC={lora_result_new['auc']:.4f}, RMSE={lora_result_new['rmse']:.4f}, ACC={lora_result_new['acc']:.4f}, F1={lora_result_new['f1']:.4f}")
    
    results.append({
        'Model': 'Ours (LoRA)',
        'AUC_old': lora_result_old['auc'],
        'AUC_new': lora_result_new['auc'],
        'RMSE_old': lora_result_old['rmse'],
        'RMSE_new': lora_result_new['rmse'],
        'ACC_old': lora_result_old['acc'],
        'ACC_new': lora_result_new['acc'],
        'F1_old': lora_result_old['f1'],
        'F1_new': lora_result_new['f1'],
        'TMD': lora_result_old['tmd']
    })

    # 5. Full Replay Oracle (Theoretical Upper Bound)
    print()
    print("=" * 80)
    print("5. Full Replay Oracle (Theoretical Upper Bound)")
    print("=" * 80)
    
    oracle_model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=0.8,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device
    )
    oracle_model.load_state_dict(base_model.state_dict())
    
    oracle_model = train_incremental_full_replay_oracle(
        oracle_model, train_df_new, train_df_old, df_valid, 
        n_user, n_item_old, n_item_new, 
        n_know_old, n_know_new, Q_expanded
    )
    
    # Evaluate
    oracle_result_old = evaluate_model(oracle_model, test_df_old, n_user, n_item_old + n_item_new, base_model)
    oracle_result_new = evaluate_model(oracle_model, test_df_new, n_user, n_item_old + n_item_new)
    
    print(f"Oracle - 旧测试集: AUC={oracle_result_old['auc']:.4f}, RMSE={oracle_result_old['rmse']:.4f}, ACC={oracle_result_old['acc']:.4f}, F1={oracle_result_old['f1']:.4f}, TMD={oracle_result_old['tmd']:.4f}")
    print(f"Oracle - 新测试集: AUC={oracle_result_new['auc']:.4f}, RMSE={oracle_result_new['rmse']:.4f}, ACC={oracle_result_new['acc']:.4f}, F1={oracle_result_new['f1']:.4f}")
    
    results.append({
        'Model': 'Full Replay Oracle',
        'AUC_old': oracle_result_old['auc'],
        'AUC_new': oracle_result_new['auc'],
        'RMSE_old': oracle_result_old['rmse'],
        'RMSE_new': oracle_result_new['rmse'],
        'ACC_old': oracle_result_old['acc'],
        'ACC_new': oracle_result_new['acc'],
        'F1_old': oracle_result_old['f1'],
        'F1_new': oracle_result_new['f1'],
        'TMD': oracle_result_old['tmd']
    })

    # 6. Naive Fine-Tuning (NFT) - Catastrophic Forgetting Control Group
    print()
    print("=" * 80)
    print("6. Naive Fine-Tuning (NFT) - Catastrophic Forgetting Control")
    print("=" * 80)
    
    nft_model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        alpha=0.8,
        Q_mat=Q_old,
        monotonicity_assumption=True,
        device=device
    )
    nft_model.load_state_dict(base_model.state_dict())
    
    nft_model = train_incremental_nft(
        nft_model, train_df_new, df_valid, 
        n_user, n_item_old, n_item_new, 
        n_know_old, n_know_new, Q_expanded
    )
    
    # Evaluate
    nft_result_old = evaluate_model(nft_model, test_df_old, n_user, n_item_old + n_item_new, base_model)
    nft_result_new = evaluate_model(nft_model, test_df_new, n_user, n_item_old + n_item_new)
    
    print(f"NFT - 旧测试集: AUC={nft_result_old['auc']:.4f}, RMSE={nft_result_old['rmse']:.4f}, ACC={nft_result_old['acc']:.4f}, F1={nft_result_old['f1']:.4f}, TMD={nft_result_old['tmd']:.4f}")
    print(f"NFT - 新测试集: AUC={nft_result_new['auc']:.4f}, RMSE={nft_result_new['rmse']:.4f}, ACC={nft_result_new['acc']:.4f}, F1={nft_result_new['f1']:.4f}")
    
    results.append({
        'Model': 'Naive FT (NFT)',
        'AUC_old': nft_result_old['auc'],
        'AUC_new': nft_result_new['auc'],
        'RMSE_old': nft_result_old['rmse'],
        'RMSE_new': nft_result_new['rmse'],
        'ACC_old': nft_result_old['acc'],
        'ACC_new': nft_result_new['acc'],
        'F1_old': nft_result_old['f1'],
        'F1_new': nft_result_new['f1'],
        'TMD': nft_result_old['tmd']
    })

    # Print results table
    print()
    print("=" * 80)
    print("增量学习实验结果")
    print("=" * 80)
    print()
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print()
    
    # Save results
    output_path = os.path.join(SAVE_DIR, 'incremental_results.csv')
    df.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

