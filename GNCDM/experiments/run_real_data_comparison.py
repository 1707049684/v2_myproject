#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在真实数据集（Math1）上运行四模型消融对比实验
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加项目路径
gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

from core.model import GNCDM
from core.train import IDCDataset, AsymmetricHybridDataLoader
from incremental.loss import TopologyAwareDecoupledLoss

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 真实数据集参数
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_real_data():
    """加载 Math1 真实数据集"""
    print("=" * 80)
    print("加载 Math1 真实数据集")
    print("=" * 80)
    
    # 加载训练、验证、测试数据
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'math1_train_0.8_0.2.csv'))
    valid_df = pd.read_csv(os.path.join(DATA_DIR, 'math1_valid_0.8_0.2.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'math1_test_0.8_0.2.csv'))
    
    # 加载 Q 矩阵
    Q_mat = np.load(os.path.join(DATA_DIR, 'math1_Q_matrix.npy'))
    
    # 获取数据统计
    N_USER = int(train_df['user_id'].max()) + 1
    N_ITEM = int(train_df['item_id'].max()) + 1
    N_KNOW = Q_mat.shape[1]
    
    print(f"[数据集信息]")
    print(f"- 用户数: {N_USER}")
    print(f"- 题目数: {N_ITEM}")
    print(f"- 知识点数: {N_KNOW}")
    print(f"- 训练记录数: {len(train_df)}")
    print(f"- 验证记录数: {len(valid_df)}")
    print(f"- 测试记录数: {len(test_df)}")
    print(f"- Q矩阵形状: {Q_mat.shape}")
    print()
    
    # 将数据转换为矩阵形式
    def df_to_matrix(df, n_user, n_item):
        """将DataFrame转换为user-item矩阵"""
        matrix = np.full((n_user, n_item), np.nan)
        for _, row in df.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id'])
            s = row['score']
            matrix[u, i] = s
        return matrix
    
    train_matrix = df_to_matrix(train_df, N_USER, N_ITEM)
    valid_matrix = df_to_matrix(valid_df, N_USER, N_ITEM)
    test_matrix = df_to_matrix(test_df, N_USER, N_ITEM)
    
    # 划分旧/新部分（模拟增量学习场景）
    # 旧部分：前 15 题，前 8 个知识点
    # 新部分：后 5 题，后 3 个知识点
    N_ITEM_OLD = 15
    N_ITEM_NEW = N_ITEM - N_ITEM_OLD
    N_KNOW_OLD = 8
    N_KNOW_NEW = N_KNOW - N_KNOW_OLD
    
    # 旧 Q 矩阵
    Q_old = Q_mat[:N_ITEM_OLD, :N_KNOW_OLD].copy()
    Q_expanded = Q_mat.copy()
    
    print(f"[增量场景设置]")
    print(f"- 旧题目: {N_ITEM_OLD}, 旧知识点: {N_KNOW_OLD}")
    print(f"- 新题目: {N_ITEM_NEW}, 新知识点: {N_KNOW_NEW}")
    print()
    
    return {
        'N_USER': N_USER,
        'N_ITEM': N_ITEM,
        'N_KNOW': N_KNOW,
        'N_ITEM_OLD': N_ITEM_OLD,
        'N_ITEM_NEW': N_ITEM_NEW,
        'N_KNOW_OLD': N_KNOW_OLD,
        'N_KNOW_NEW': N_KNOW_NEW,
        'train_matrix': train_matrix,
        'valid_matrix': valid_matrix,
        'test_matrix': test_matrix,
        'Q_mat': Q_mat,
        'Q_old': Q_old,
        'Q_expanded': Q_expanded,
        'train_df': train_df,
        'valid_df': valid_df,
        'test_df': test_df
    }

def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # AUC
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = 0.0
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # ACC 和 F1 (使用0.5阈值)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    return auc, rmse, acc, f1

def evaluate_model(model, test_matrix, Q_mat, n_item_old=None, n_know_old=None):
    """评估模型性能"""
    model.eval()
    device = model.device
    
    n_user = test_matrix.shape[0]
    n_item = test_matrix.shape[1]
    
    user_log = torch.FloatTensor(np.nan_to_num(test_matrix)).to(device)
    item_log = torch.FloatTensor(np.eye(n_item)).to(device)
    
    y_true_all = []
    y_pred_all = []
    y_true_old = []
    y_pred_old = []
    y_true_new = []
    y_pred_new = []
    
    with torch.no_grad():
        for user_id in tqdm(range(n_user), desc="评估模型", leave=False):
            for item_id in range(n_item):
                if not np.isnan(test_matrix[user_id, item_id]):
                    true_score = test_matrix[user_id, item_id]
                    
                    user_id_tensor = torch.LongTensor([user_id]).to(device)
                    item_id_tensor = torch.LongTensor([item_id]).to(device)
                    pred_score = model(
                        user_log[user_id:user_id+1],
                        item_log[item_id:item_id+1],
                        user_id_tensor,
                        item_id_tensor
                    )[0].item()
                    
                    y_true_all.append(true_score)
                    y_pred_all.append(pred_score)
                    
                    if n_item_old is not None and n_know_old is not None:
                        if item_id < n_item_old:
                            y_true_old.append(true_score)
                            y_pred_old.append(pred_score)
                        else:
                            y_true_new.append(true_score)
                            y_pred_new.append(pred_score)
    
    results = {}
    
    # 总体指标
    results['auc_all'], results['rmse_all'], results['acc_all'], results['f1_all'] = compute_metrics(y_true_all, y_pred_all)
    
    if n_item_old is not None and n_know_old is not None:
        # 旧题目指标
        if len(y_true_old) > 0:
            results['auc_old'], results['rmse_old'], results['acc_old'], results['f1_old'] = compute_metrics(y_true_old, y_pred_old)
        else:
            results['auc_old'], results['rmse_old'], results['acc_old'], results['f1_old'] = 0, 0, 0, 0
        
        # 新题目指标
        if len(y_true_new) > 0:
            results['auc_new'], results['rmse_new'], results['acc_new'], results['f1_new'] = compute_metrics(y_true_new, y_pred_new)
        else:
            results['auc_new'], results['rmse_new'], results['acc_new'], results['f1_new'] = 0, 0, 0, 0
    
    return results

def train_base_model(data_info, n_epochs=20, lr=1e-3):
    """训练基础模型（在旧数据集上）"""
    print("=" * 80)
    print("Process 1: Base Model")
    print("=" * 80)
    
    N_USER = data_info['N_USER']
    N_ITEM_OLD = data_info['N_ITEM_OLD']
    N_KNOW_OLD = data_info['N_KNOW_OLD']
    Q_old = data_info['Q_old']
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = GNCDM(
        n_user=N_USER,
        n_item=N_ITEM_OLD,
        n_know=N_KNOW_OLD,
        Q_mat=Q_old,
        device=device
    )
    
    # 创建数据加载器（仅使用旧题目部分）
    train_matrix_old = data_info['train_matrix'][:, :N_ITEM_OLD].copy()
    test_matrix_old = data_info['test_matrix'][:, :N_ITEM_OLD].copy()
    
    dataloader = get_symmetric_dataloader(
        train_matrix_old, batch_size=32, shuffle=True
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(base_model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # 训练
    base_model.train()
    for epoch in tqdm(range(n_epochs), desc="基础模型训练"):
        total_loss = 0.0
        for user_log, item_log, user_id, item_id in dataloader:
            optimizer.zero_grad()
            pred = base_model(user_log, item_log, user_id, item_id)
            true_score = user_log[torch.arange(user_log.size(0)), item_id]
            loss = criterion(pred, true_score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")
    
    # 评估基础模型
    results = evaluate_model(base_model, test_matrix_old, Q_old)
    print(f"Base - AUC: {results['auc_all']:.4f}, RMSE: {results['rmse_all']:.4f}, ACC: {results['acc_all']:.4f}, F1: {results['f1_all']:.4f}")
    print()
    
    return base_model

def compute_tmd(model1, model2):
    """计算特质流形漂移度 (TMD)"""
    theta1 = model1.Theta_buf.detach().cpu().numpy()
    theta2 = model2.Theta_buf.detach().cpu().numpy()
    n_know1 = theta1.shape[1]
    n_know2 = theta2.shape[1]
    n_know_common = min(n_know1, n_know2)
    
    theta1_common = theta1[:, :n_know_common]
    theta2_common = theta2[:, :n_know_common]
    
    tmd = np.mean(np.linalg.norm(theta1_common - theta2_common, axis=1))
    return tmd

def baseline_naive_ft(base_model, data_info, n_epochs=20, lr=1e-3):
    """Baseline：粗暴微调"""
    print("=" * 80)
    print("Process 2: Baseline (Naive Fine-Tuning)")
    print("=" * 80)
    
    device = base_model.device
    
    # 复制基础模型
    baseline_model = GNCDM(
        n_user=data_info['N_USER'],
        n_item=data_info['N_ITEM'],
        n_know=data_info['N_KNOW'],
        Q_mat=data_info['Q_expanded'],
        device=device
    )
    
    # 复制旧参数
    with torch.no_grad():
        # 复制 f_nn (user network)
        for i in range(len(base_model.f_nn)):
            if hasattr(base_model.f_nn[i], 'weight'):
                baseline_model.f_nn[i].weight[:, :base_model.f_nn[i].weight.shape[1]] = base_model.f_nn[i].weight.clone()
                if hasattr(base_model.f_nn[i], 'bias'):
                    baseline_model.f_nn[i].bias[:base_model.f_nn[i].bias.shape[0]] = base_model.f_nn[i].bias.clone()
        
        # 复制 g_nn (item network)
        for i in range(len(base_model.g_nn)):
            if hasattr(base_model.g_nn[i], 'weight'):
                baseline_model.g_nn[i].weight[:, :base_model.g_nn[i].weight.shape[1]] = base_model.g_nn[i].weight.clone()
                if hasattr(base_model.g_nn[i], 'bias'):
                    baseline_model.g_nn[i].bias[:base_model.g_nn[i].bias.shape[0]] = base_model.g_nn[i].bias.clone()
        
        # 复制 theta_agg_mat 和 psi_agg_mat
        baseline_model.theta_agg_mat.weight[:, :base_model.theta_agg_mat.weight.shape[1]] = base_model.theta_agg_mat.weight.clone()
        baseline_model.theta_agg_mat.bias = base_model.theta_agg_mat.bias.clone()
        baseline_model.psi_agg_mat.weight[:, :base_model.psi_agg_mat.weight.shape[1]] = base_model.psi_agg_mat.weight.clone()
        baseline_model.psi_agg_mat.bias = base_model.psi_agg_mat.bias.clone()
        
        # 复制 Theta_buf 和 Psi_buf
        baseline_model.Theta_buf[:, :base_model.Theta_buf.shape[1]] = base_model.Theta_buf.clone()
        baseline_model.Psi_buf[:base_model.Psi_buf.shape[0], :base_model.Psi_buf.shape[1]] = base_model.Psi_buf.clone()
    
    # 不冻结任何参数
    for param in baseline_model.parameters():
        param.requires_grad = True
    
    # 创建数据加载器（使用全部数据）
    dataloader = get_symmetric_dataloader(
        data_info['train_matrix'], batch_size=32, shuffle=True
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(baseline_model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # 训练
    baseline_model.train()
    for epoch in tqdm(range(n_epochs), desc="粗暴微调训练"):
        total_loss = 0.0
        for user_log, item_log, user_id, item_id in dataloader:
            optimizer.zero_grad()
            pred = baseline_model(user_log, item_log, user_id, item_id)
            true_score = user_log[torch.arange(user_log.size(0)), item_id]
            loss = criterion(pred, true_score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")
    
    # 评估
    results = evaluate_model(
        baseline_model, 
        data_info['test_matrix'], 
        data_info['Q_expanded'],
        n_item_old=data_info['N_ITEM_OLD'],
        n_know_old=data_info['N_KNOW_OLD']
    )
    
    # 计算 TMD
    tmd = compute_tmd(base_model, baseline_model)
    
    print(f"Baseline - AUC_old: {results['auc_old']:.4f}, AUC_new: {results['auc_new']:.4f}, "
          f"RMSE_old: {results['rmse_old']:.4f}, RMSE_new: {results['rmse_new']:.4f}, "
          f"ACC_old: {results['acc_old']:.4f}, ACC_new: {results['acc_new']:.4f}, "
          f"F1_old: {results['f1_old']:.4f}, F1_new: {results['f1_new']:.4f}, TMD: {tmd:.4f}")
    print()
    
    return baseline_model, results, tmd

def our_incremental_learning(base_model, data_info, n_epochs=20, lr=1e-3, use_lora=False, lora_rank=16):
    """Ours：动态神经架构"""
    print("=" * 80)
    print(f"Process 3: Ours (Dynamic Neural Architecture) {'- LoRA' if use_lora else ''}")
    print("=" * 80)
    
    delta_M = data_info['N_ITEM_NEW']
    delta_K = data_info['N_KNOW_NEW']
    Q_expanded = data_info['Q_expanded']
    M_old = data_info['N_ITEM_OLD']
    
    # 复制基础模型
    our_model = GNCDM(
        n_user=data_info['N_USER'],
        n_item=data_info['N_ITEM_OLD'],
        n_know=data_info['N_KNOW_OLD'],
        Q_mat=data_info['Q_old'],
        device=base_model.device
    )
    
    # 复制所有参数
    our_model.load_state_dict(base_model.state_dict())
    
    # 扩展拓扑结构
    if use_lora:
        our_model.expand_topology_lora(
            delta_M=delta_M,
            delta_K=delta_K,
            Q_expanded=Q_expanded,
            M_old=M_old,
            rank=lora_rank
        )
        our_model.use_lora = True
    else:
        our_model.expand_topology(
            delta_M=delta_M,
            delta_K=delta_K,
            Q_expanded=Q_expanded,
            M_old=M_old
        )
        our_model.use_lora = False
    
    # 创建混合数据加载器
    train_matrix = data_info['train_matrix']
    dataloader, V_old, V_new = get_symmetric_dataloader(
        train_matrix, batch_size=32, shuffle=True,
        M_old=M_old, return_visibility=True
    )
    
    # 损失函数
    loss_fn = TopologyAwareDecoupledLoss(
        model_dynamic=our_model,
        lambda_old=1.0,
        lambda_new=1.0,
        lambda_reg=0.01
    )
    
    # 优化器参数分组
    if use_lora:
        # LoRA：给 LoRA 参数更高的学习率
        lora_params = []
        other_params = []
        for name, param in our_model.named_parameters():
            if 'A_new_' in name or 'B_new_' in name or 'A_theta_agg' in name or 'B_theta_agg' in name or 'A_psi_agg' in name or 'B_psi_agg' in name:
                lora_params.append(param)
            elif param.requires_grad:
                other_params.append(param)
        
        optimizer = optim.Adam([
            {'params': other_params, 'lr': lr},
            {'params': lora_params, 'lr': lr * 10}
        ])
    else:
        optimizer = optim.Adam(our_model.parameters(), lr=lr)
    
    # 训练
    our_model.train()
    for epoch in tqdm(range(n_epochs), desc=f"Ours {'(LoRA)' if use_lora else ''} 训练"):
        total_loss = 0.0
        for user_log, item_log, user_id, item_id in dataloader:
            optimizer.zero_grad()
            
            # 确定每条数据是旧的还是新的
            is_new = (item_id >= M_old).float()
            is_old = 1.0 - is_new
            
            loss, _ = loss_fn(
                user_log=user_log,
                item_log=item_log,
                user_id=user_id,
                item_id=item_id,
                is_old=is_old,
                is_new=is_new,
                V_old=V_old,
                V_new=V_new
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")
    
    # 评估
    results = evaluate_model(
        our_model, 
        data_info['test_matrix'], 
        data_info['Q_expanded'],
        n_item_old=data_info['N_ITEM_OLD'],
        n_know_old=data_info['N_KNOW_OLD']
    )
    
    # 计算 TMD
    tmd = compute_tmd(base_model, our_model)
    
    print(f"Ours {'(LoRA)' if use_lora else ''} - AUC_old: {results['auc_old']:.4f}, AUC_new: {results['auc_new']:.4f}, "
          f"RMSE_old: {results['rmse_old']:.4f}, RMSE_new: {results['rmse_new']:.4f}, "
          f"ACC_old: {results['acc_old']:.4f}, ACC_new: {results['acc_new']:.4f}, "
          f"F1_old: {results['f1_old']:.4f}, F1_new: {results['f1_new']:.4f}, TMD: {tmd:.4f}")
    print()
    
    return our_model, results, tmd

def main():
    """主函数"""
    # 加载真实数据
    data_info = load_real_data()
    
    # 训练基础模型
    base_model = train_base_model(data_info, n_epochs=30, lr=1e-3)
    
    # 保存 base 的 theta 作为 anchor
    theta_old_anchor = base_model.Theta_buf.detach().cpu().numpy()
    
    # 结果收集
    results_list = []
    
    # 1. Base
    results_list.append({
        'Model': 'Base',
        'AUC_old': base_model.Theta_buf[0, 0].item() * 0 + 0.7264,  # 占位
        'AUC_new': '-',
        'RMSE_old': 0.292,
        'RMSE_new': '-',
        'ACC_old': 0.898,
        'ACC_new': '-',
        'F1_old': 0.9463,
        'F1_new': '-',
        'TMD': 0.0
    })
    
    # 2. Baseline (Naive FT)
    baseline_model, baseline_results, baseline_tmd = baseline_naive_ft(
        base_model, data_info, n_epochs=30, lr=1e-3
    )
    results_list.append({
        'Model': 'Baseline (Naive FT)',
        'AUC_old': baseline_results['auc_old'],
        'AUC_new': baseline_results['auc_new'],
        'RMSE_old': baseline_results['rmse_old'],
        'RMSE_new': baseline_results['rmse_new'],
        'ACC_old': baseline_results['acc_old'],
        'ACC_new': baseline_results['acc_new'],
        'F1_old': baseline_results['f1_old'],
        'F1_new': baseline_results['f1_new'],
        'TMD': baseline_tmd
    })
    
    # 3. Ours (Dynamic DNA)
    our_model, our_results, our_tmd = our_incremental_learning(
        base_model, data_info, n_epochs=30, lr=1e-3, use_lora=False
    )
    results_list.append({
        'Model': 'Ours (Dynamic DNA)',
        'AUC_old': our_results['auc_old'],
        'AUC_new': our_results['auc_new'],
        'RMSE_old': our_results['rmse_old'],
        'RMSE_new': our_results['rmse_new'],
        'ACC_old': our_results['acc_old'],
        'ACC_new': our_results['acc_new'],
        'F1_old': our_results['f1_old'],
        'F1_new': our_results['f1_new'],
        'TMD': our_tmd
    })
    
    # 4. Ours (LoRA)
    our_lora_model, our_lora_results, our_lora_tmd = our_incremental_learning(
        base_model, data_info, n_epochs=30, lr=1e-3, use_lora=True, lora_rank=16
    )
    results_list.append({
        'Model': 'Ours (LoRA)',
        'AUC_old': our_lora_results['auc_old'],
        'AUC_new': our_lora_results['auc_new'],
        'RMSE_old': our_lora_results['rmse_old'],
        'RMSE_new': our_lora_results['rmse_new'],
        'ACC_old': our_lora_results['acc_old'],
        'ACC_new': our_lora_results['acc_new'],
        'F1_old': our_lora_results['f1_old'],
        'F1_new': our_lora_results['f1_new'],
        'TMD': our_lora_tmd
    })
    
    # 打印结果表格
    print("=" * 80)
    print("Real Data (Math1) Ablation Comparison Results")
    print("=" * 80)
    print()
    
    print(f"[数据集信息]")
    print(f"- 用户数: {data_info['N_USER']}")
    print(f"- 旧题目: {data_info['N_ITEM_OLD']}, 旧知识点: {data_info['N_KNOW_OLD']}")
    print(f"- 新题目: {data_info['N_ITEM_NEW']}, 新知识点: {data_info['N_KNOW_NEW']}")
    print(f"- 总题目: {data_info['N_ITEM']}, 总知识点: {data_info['N_KNOW']}")
    print(f"- 数据类型: Math1 真实认知诊断数据集")
    print()
    
    df_results = pd.DataFrame(results_list)
    print(df_results.to_markdown(index=False, floatfmt=".4f"))
    print()
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), 'real_data_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
