# -*- coding: utf-8 -*-
"""
在真实数据集（Math1）上运行消融对比实验 - 简化版本
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# 添加 GNCDM 目录到路径
gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

from core.model import GNCDM
from core.train import IDCDataset, AsymmetricHybridDataLoader
from incremental.loss import TopologyAwareDecoupledLoss

# 配置
SEED = 42
DEVICE = torch.device('cpu')
N_EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-3

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred):
    """计算 AUC、RMSE、ACC 和 F1 指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_pred_binary = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    return auc, rmse, acc, f1

def load_math1_data():
    """加载 Math1 真实数据集"""
    data_dir = os.path.join(gncdm_dir, 'data')
    
    # 加载训练、验证、测试数据
    train_df = pd.read_csv(os.path.join(data_dir, 'math1_train_0.8_0.2.csv'))
    valid_df = pd.read_csv(os.path.join(data_dir, 'math1_valid_0.8_0.2.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'math1_test_0.8_0.2.csv'))
    
    # 加载 Q 矩阵
    Q_mat = np.load(os.path.join(data_dir, 'math1_Q_matrix.npy'))
    
    # 获取基本信息
    n_user = int(train_df['user_id'].max()) + 1
    n_item = int(train_df['item_id'].max()) + 1
    n_know = Q_mat.shape[1]
    
    print(f"[Math1 数据集信息]")
    print(f"- 用户数: {n_user}")
    print(f"- 题目数: {n_item}")
    print(f"- 知识点数: {n_know}")
    print(f"- 训练记录: {len(train_df)}, 验证记录: {len(valid_df)}, 测试记录: {len(test_df)}")
    print(f"- Q矩阵形状: {Q_mat.shape}")
    print()
    
    # 划分旧/新部分
    n_item_old = 15
    n_know_old = 8
    n_item_new = n_item - n_item_old
    n_know_new = n_know - n_know_old
    
    Q_old = Q_mat[:n_item_old, :n_know_old].copy()
    Q_expanded = Q_mat.copy()
    
    # 切分数据
    train_df_old = train_df[train_df['item_id'] < n_item_old].copy()
    train_df_new = train_df[train_df['item_id'] >= n_item_old].copy()
    test_df_old = test_df[test_df['item_id'] < n_item_old].copy()
    test_df_new = test_df[test_df['item_id'] >= n_item_old].copy()
    
    return {
        'n_user': n_user,
        'n_item': n_item,
        'n_know': n_know,
        'n_item_old': n_item_old,
        'n_item_new': n_item_new,
        'n_know_old': n_know_old,
        'n_know_new': n_know_new,
        'train_df': train_df,
        'train_df_old': train_df_old,
        'train_df_new': train_df_new,
        'test_df': test_df,
        'test_df_old': test_df_old,
        'test_df_new': test_df_new,
        'Q_mat': Q_mat,
        'Q_old': Q_old,
        'Q_expanded': Q_expanded
    }

def evaluate_model(model, df_data, n_user, n_item):
    """评估模型"""
    model.eval()
    dataset = IDCDataset(df_data, n_user, n_item)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
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
    
    return compute_metrics(y_true, y_pred)

def evaluate_separate(model, test_df_old, test_df_new, n_user, n_item_total, n_item_old):
    """分别评估旧/新题目"""
    # 评估旧题目
    if len(test_df_old) > 0:
        auc_old, rmse_old, acc_old, f1_old = evaluate_model(model, test_df_old, n_user, n_item_old)
    else:
        auc_old, rmse_old, acc_old, f1_old = 0, 0, 0, 0
    
    # 评估新题目
    if len(test_df_new) > 0:
        auc_new, rmse_new, acc_new, f1_new = evaluate_model(model, test_df_new, n_user, n_item_total)
    else:
        auc_new, rmse_new, acc_new, f1_new = 0, 0, 0, 0
    
    return auc_old, rmse_old, acc_old, f1_old, auc_new, rmse_new, acc_new, f1_new

def get_theta_anchor(model, n_user, n_item):
    """获取旧模型的 theta 作为锚点"""
    model.eval()
    with torch.no_grad():
        dummy_log = torch.zeros(n_user, n_item).to(DEVICE)
        theta = model.diagnose_theta(dummy_log)
    return theta.cpu().numpy()

def calculate_tmd(theta_old, theta_new, K_old):
    """计算特质流形漂移度"""
    theta_new_old_dim = theta_new[:, :K_old]
    per_user_norm = np.linalg.norm(theta_old - theta_new_old_dim, axis=1)
    return np.mean(per_user_norm)

def train_base_model(data_info):
    """训练基础模型"""
    print("=" * 80)
    print("Process 1: Base Model")
    print("=" * 80)
    
    n_user = data_info['n_user']
    n_item_old = data_info['n_item_old']
    n_know_old = data_info['n_know_old']
    Q_old = data_info['Q_old']
    train_df_old = data_info['train_df_old']
    
    model = GNCDM(
        n_user=n_user,
        n_item=n_item_old,
        n_know=n_know_old,
        user_dim=32,
        item_dim=32,
        Q_mat=Q_old,
        device=DEVICE,
        alpha=0.5,
        monotonicity_assumption=True
    )
    
    dataset = IDCDataset(train_df_old, n_user, n_item_old)
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
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # 评估
    auc, rmse, acc, f1 = evaluate_model(model, data_info['test_df_old'], n_user, n_item_old)
    print(f"Base - AUC: {auc:.4f}, RMSE: {rmse:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
    print()
    
    return model

def naive_fine_tune(model, data_info):
    """Baseline: 粗暴微调"""
    print("=" * 80)
    print("Process 2: Baseline (Naive Fine-Tuning)")
    print("=" * 80)
    
    n_user = data_info['n_user']
    n_item_total = data_info['n_item']
    n_know_total = data_info['n_know']
    Q_expanded = data_info['Q_expanded']
    
    # 手动扩展
    model.n_item = n_item_total
    model.n_know = n_know_total
    model.Q_mat = torch.Tensor(Q_expanded).to(DEVICE)
    
    f_linear = type(model.f_nn[0])
    
    # 重初始化层
    model.f_nn = nn.Sequential(
        nn.Sequential(
            f_linear(n_item_total, n_know_total),
            nn.Sigmoid(),
            f_linear(n_know_total, n_know_total),
            nn.Sigmoid()
        )
    ).to(DEVICE)
    
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
    
    model.theta_agg_mat = f_linear(n_know_total, model.user_dim).to(DEVICE)
    model.psi_agg_mat = nn.Linear(n_know_total, model.item_dim).to(DEVICE)
    
    for param in model.parameters():
        param.requires_grad = True
    
    dataset = IDCDataset(data_info['train_df'], n_user, n_item_total)
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
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # 评估
    auc_old, rmse_old, acc_old, f1_old, auc_new, rmse_new, acc_new, f1_new = evaluate_separate(
        model, data_info['test_df_old'], data_info['test_df_new'],
        n_user, n_item_total, data_info['n_item_old']
    )
    
    print(f"Baseline - AUC_old: {auc_old:.4f}, AUC_new: {auc_new:.4f}, RMSE_old: {rmse_old:.4f}, RMSE_new: {rmse_new:.4f}")
    print()
    
    return model, {
        'auc_old': auc_old, 'auc_new': auc_new,
        'rmse_old': rmse_old, 'rmse_new': rmse_new,
        'acc_old': acc_old, 'acc_new': acc_new,
        'f1_old': f1_old, 'f1_new': f1_new
    }

def our_incremental_learning(model_old, data_info):
    """Ours: 动态神经架构"""
    print("=" * 80)
    print("Process 3: Ours (Dynamic Neural Architecture)")
    print("=" * 80)
    
    n_user = data_info['n_user']
    n_item_old = data_info['n_item_old']
    n_item_total = data_info['n_item']
    n_know_old = data_info['n_know_old']
    n_know_total = data_info['n_know']
    Q_expanded = data_info['Q_expanded']
    
    # 创建模型并加载权重
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
    model.load_state_dict(model_old.state_dict())
    
    # 扩展
    delta_M = n_item_total - n_item_old
    delta_K = n_know_total - n_know_old
    model.expand_topology(delta_M, delta_K, Q_expanded)
    
    # 数据加载
    dataset_old = IDCDataset(data_info['train_df_old'], n_user, n_item_total)
    dataset_new = IDCDataset(data_info['train_df_new'], n_user, n_item_total)
    old_loader = DataLoader(dataset_old, batch_size=BATCH_SIZE, shuffle=True)
    new_loader = DataLoader(dataset_new, batch_size=BATCH_SIZE, shuffle=True)
    hybrid_loader = AsymmetricHybridDataLoader(old_loader, new_loader, old_ratio=0.7, device=DEVICE)
    
    # 优化器
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(active_params, lr=LR)
    
    # 损失函数
    loss_fn = TopologyAwareDecoupledLoss(
        model_old=model_old,
        model_dynamic=model,
        original_know_dim=n_know_old,
        device=DEVICE
    )
    
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
                user_log=user_log, item_log=item_log, user_id=user_id, item_id=item_id,
                score=score, is_new=is_new, epoch=epoch, total_epochs=N_EPOCHS,
                V_old=V_old, V_new=V_new
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(hybrid_loader)
            print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # 评估
    auc_old, rmse_old, acc_old, f1_old, auc_new, rmse_new, acc_new, f1_new = evaluate_separate(
        model, data_info['test_df_old'], data_info['test_df_new'],
        n_user, n_item_total, data_info['n_item_old']
    )
    
    print(f"Ours - AUC_old: {auc_old:.4f}, AUC_new: {auc_new:.4f}, RMSE_old: {rmse_old:.4f}, RMSE_new: {rmse_new:.4f}")
    print()
    
    return model, {
        'auc_old': auc_old, 'auc_new': auc_new,
        'rmse_old': rmse_old, 'rmse_new': rmse_new,
        'acc_old': acc_old, 'acc_new': acc_new,
        'f1_old': f1_old, 'f1_new': f1_new
    }

def our_incremental_learning_lora(model_old, data_info, rank=16):
    """Ours: LoRA 动态神经架构"""
    print("=" * 80)
    print("Process 4: Ours (LoRA-based)")
    print("=" * 80)
    
    n_user = data_info['n_user']
    n_item_old = data_info['n_item_old']
    n_item_total = data_info['n_item']
    n_know_old = data_info['n_know_old']
    n_know_total = data_info['n_know']
    Q_expanded = data_info['Q_expanded']
    
    # 创建模型并加载权重
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
    model.load_state_dict(model_old.state_dict())
    
    # 使用 LoRA 扩展
    delta_M = n_item_total - n_item_old
    delta_K = n_know_total - n_know_old
    model.expand_topology_lora(delta_M, delta_K, Q_expanded, M_old=n_item_old, rank=rank)
    
    # 数据加载
    dataset_old = IDCDataset(data_info['train_df_old'], n_user, n_item_total)
    dataset_new = IDCDataset(data_info['train_df_new'], n_user, n_item_total)
    old_loader = DataLoader(dataset_old, batch_size=BATCH_SIZE, shuffle=True)
    new_loader = DataLoader(dataset_new, batch_size=BATCH_SIZE, shuffle=True)
    hybrid_loader = AsymmetricHybridDataLoader(old_loader, new_loader, old_ratio=0.7, device=DEVICE)
    
    # 参数分组
    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'A_new_' in name or 'B_new_' in name or 'A_theta_agg' in name or 'B_theta_agg' in name or 'A_psi_agg' in name or 'B_psi_agg' in name:
            lora_params.append(param)
        elif param.requires_grad:
            other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': LR},
        {'params': lora_params, 'lr': LR * 10}
    ])
    
    # 损失函数
    loss_fn = TopologyAwareDecoupledLoss(
        model_old=model_old,
        model_dynamic=model,
        original_know_dim=n_know_old,
        device=DEVICE
    )
    
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
                user_log=user_log, item_log=item_log, user_id=user_id, item_id=item_id,
                score=score, is_new=is_new, epoch=epoch, total_epochs=N_EPOCHS,
                V_old=V_old, V_new=V_new
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(hybrid_loader)
            print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # 评估
    auc_old, rmse_old, acc_old, f1_old, auc_new, rmse_new, acc_new, f1_new = evaluate_separate(
        model, data_info['test_df_old'], data_info['test_df_new'],
        n_user, n_item_total, data_info['n_item_old']
    )
    
    print(f"Ours (LoRA) - AUC_old: {auc_old:.4f}, AUC_new: {auc_new:.4f}, RMSE_old: {rmse_old:.4f}, RMSE_new: {rmse_new:.4f}")
    print()
    
    return model, {
        'auc_old': auc_old, 'auc_new': auc_new,
        'rmse_old': rmse_old, 'rmse_new': rmse_new,
        'acc_old': acc_old, 'acc_new': acc_new,
        'f1_old': f1_old, 'f1_new': f1_new
    }

def main():
    set_seed(SEED)
    
    print("=" * 80)
    print("Generative-CD Incremental Learning - Math1 Real Data")
    print("=" * 80)
    print()
    
    # 加载数据
    data_info = load_math1_data()
    
    # 结果列表
    results = []
    
    # 1. Base
    base_model = train_base_model(data_info)
    theta_anchor = get_theta_anchor(base_model, data_info['n_user'], data_info['n_item_old'])
    
    auc_base, rmse_base, acc_base, f1_base = evaluate_model(
        base_model, data_info['test_df_old'], data_info['n_user'], data_info['n_item_old']
    )
    results.append({
        'Model': 'Base',
        'AUC_old': auc_base,
        'AUC_new': '-',
        'RMSE_old': rmse_base,
        'RMSE_new': '-',
        'ACC_old': acc_base,
        'ACC_new': '-',
        'F1_old': f1_base,
        'F1_new': '-',
        'TMD': 0.0
    })
    
    # 2. Baseline
    baseline_model = GNCDM(
        n_user=data_info['n_user'], n_item=data_info['n_item_old'], 
        n_know=data_info['n_know_old'], user_dim=32, item_dim=32,
        Q_mat=data_info['Q_old'], device=DEVICE, alpha=0.5, 
        monotonicity_assumption=True
    )
    baseline_model.load_state_dict(base_model.state_dict())
    baseline_model, baseline_res = naive_fine_tune(baseline_model, data_info)
    
    theta_baseline = get_theta_anchor(baseline_model, data_info['n_user'], data_info['n_item'])
    tmd_baseline = calculate_tmd(theta_anchor, theta_baseline, data_info['n_know_old'])
    
    results.append({
        'Model': 'Baseline (Naive FT)',
        'AUC_old': baseline_res['auc_old'],
        'AUC_new': baseline_res['auc_new'],
        'RMSE_old': baseline_res['rmse_old'],
        'RMSE_new': baseline_res['rmse_new'],
        'ACC_old': baseline_res['acc_old'],
        'ACC_new': baseline_res['acc_new'],
        'F1_old': baseline_res['f1_old'],
        'F1_new': baseline_res['f1_new'],
        'TMD': tmd_baseline
    })
    
    # 3. Ours (DNA)
    our_model = GNCDM(
        n_user=data_info['n_user'], n_item=data_info['n_item_old'], 
        n_know=data_info['n_know_old'], user_dim=32, item_dim=32,
        Q_mat=data_info['Q_old'], device=DEVICE, alpha=0.5, 
        monotonicity_assumption=True
    )
    our_model.load_state_dict(base_model.state_dict())
    our_model, our_res = our_incremental_learning(our_model, data_info)
    
    theta_our = get_theta_anchor(our_model, data_info['n_user'], data_info['n_item'])
    tmd_our = calculate_tmd(theta_anchor, theta_our, data_info['n_know_old'])
    
    results.append({
        'Model': 'Ours (Dynamic DNA)',
        'AUC_old': our_res['auc_old'],
        'AUC_new': our_res['auc_new'],
        'RMSE_old': our_res['rmse_old'],
        'RMSE_new': our_res['rmse_new'],
        'ACC_old': our_res['acc_old'],
        'ACC_new': our_res['acc_new'],
        'F1_old': our_res['f1_old'],
        'F1_new': our_res['f1_new'],
        'TMD': tmd_our
    })
    
    # 4. Ours (LoRA)
    our_lora_model = GNCDM(
        n_user=data_info['n_user'], n_item=data_info['n_item_old'], 
        n_know=data_info['n_know_old'], user_dim=32, item_dim=32,
        Q_mat=data_info['Q_old'], device=DEVICE, alpha=0.5, 
        monotonicity_assumption=True
    )
    our_lora_model.load_state_dict(base_model.state_dict())
    our_lora_model, our_lora_res = our_incremental_learning_lora(our_lora_model, data_info, rank=16)
    
    theta_lora = get_theta_anchor(our_lora_model, data_info['n_user'], data_info['n_item'])
    tmd_lora = calculate_tmd(theta_anchor, theta_lora, data_info['n_know_old'])
    
    results.append({
        'Model': 'Ours (LoRA)',
        'AUC_old': our_lora_res['auc_old'],
        'AUC_new': our_lora_res['auc_new'],
        'RMSE_old': our_lora_res['rmse_old'],
        'RMSE_new': our_lora_res['rmse_new'],
        'ACC_old': our_lora_res['acc_old'],
        'ACC_new': our_lora_res['acc_new'],
        'F1_old': our_lora_res['f1_old'],
        'F1_new': our_lora_res['f1_new'],
        'TMD': tmd_lora
    })
    
    # 打印结果
    print("=" * 80)
    print("Math1 Real Data - Ablation Results")
    print("=" * 80)
    print()
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print()
    
    output_path = os.path.join(gncdm_dir, 'experiments', 'math1_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
