# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
# 
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.


from collections import OrderedDict
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as F


def truncated_normal_init(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Micro-Variance Truncated Normal Initialization
    
    Args:
        tensor: torch.Tensor, the tensor to initialize
        mean: float, mean of the truncated normal distribution
        std: float, standard deviation (variance = std^2)
        a: float, lower bound of truncation
        b: float, upper bound of truncation
    """
    # Sample from normal distribution
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    
    # Apply truncation
    tensor.data = torch.clamp(tensor.data, min=a*std + mean, max=b*std + mean)
    
    # Ensure non-negativity by taking absolute value
    tensor.data = torch.abs(tensor.data)


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
        
class GNCDM(nn.Module):
    def __init__(self, n_user:int, n_item:int, n_know:int, \
        user_dim:int, item_dim:int, alpha:float, \
        Q_mat: np.array = None, \
        monotonicity_assumption: bool = True,\
        device = torch.device('cpu')):
        '''
        Args:
            n_user:int, the number of learners
            n_item:int, the number of test items
            n_know:int, the number of knowledge concepts,
                which equals to the dimension of diagnostic results.
            user_dim:int, the dimension of aggregated user representations.
            item_dim:int, the dimension of aggregated item representations.
            Q_mat:np.array((n_item,n_know)), the binary Q-matrix.
            monotonicity_assumption:bool (default False) whether to apply
                the monotonicity assumption to the diagnostic module. If True,
                the monotonicity assumption is applied.
            device:torch.device
        '''
        super(GNCDM,self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.n_know = n_know
        self.user_dim = user_dim 
        self.item_dim = item_dim
        self.itf = self.ncd_func
        self.device = device

        self.Q_mat = torch.Tensor(Q_mat) \
            if Q_mat is not None else torch.ones((n_item, n_know))

        self.K_diff_mat = nn.Parameter(torch.zeros((n_know, user_dim)),\
            requires_grad=False).to(device)
        self.K_diff_mat.requires_grad = True

        self.Q_mat = self.Q_mat.to(device)

        self.alpha = alpha

        # Buffer of examinee traits
        self.Theta_buf = nn.Parameter(torch.zeros((n_user, n_know))\
            , requires_grad=False).to(device)

        # Buffer of question feature traits
        self.Psi_buf = nn.Parameter(torch.zeros((n_item, n_know))\
            , requires_grad=False).to(device)
        
        # Track expansion state
        self.is_expanded = False
        self.original_n_item = n_item
        self.original_n_know = n_know
        
        f_linear = nn.Linear if monotonicity_assumption is False else PosLinear


        self.f_nn = nn.Sequential(
            OrderedDict(
                [
                    ('f_layer_1', f_linear(n_item, n_know)),
                    ('f_activate_1', nn.Sigmoid()),
                    ('f_layer_2', f_linear(n_know, n_know)),
                    ('f_activate_2', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.g_nn = nn.Sequential(
            OrderedDict(
                [
                    ('g_layer_1', nn.Linear(n_user, n_know)),
                    ('g_activate_1', nn.Sigmoid()),
                    ('g_layer_2', nn.Linear(n_know, n_know)),
                    ('g_activate_2', nn.Sigmoid()),
                    ('g_layer_3', nn.Linear(n_know, n_know)),
                    ('g_activate_3', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.theta_agg_mat = f_linear(n_know, user_dim).to(device)
        self.psi_agg_mat = nn.Linear(n_know, item_dim).to(device)

        self.ncd = nn.Sequential(
            OrderedDict([
                ('pred_layer_1', nn.Linear(user_dim, 64)),
                ('pred_activate_1', nn.Sigmoid()),
                ('pred_dropout_1', nn.Dropout(p=0.5)),
                ('pred_layer_2', nn.Linear(64, 32)),
                ('pred_activate_2', nn.Sigmoid()),
                ('pred_dropout_2', nn.Dropout(p=0.5)),
                ('pred_layer_3', nn.Linear(32, 1)),
                ('pred_activate_3', nn.Sigmoid()),

            ])
        ).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def _freeze_parameters(self):
        """
        Freeze all existing parameters by setting requires_grad = False
        This preserves the old manifold phi_old
        """
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    def _initialize_new_params_with_micro_variance(self, new_params, reference_params):
        """
        Initialize new parameters with micro-variance truncated normal distribution.
        The variance is 1e-3 times the variance of the reference parameters.
        
        Args:
            new_params: list of new parameter tensors to initialize
            reference_params: list of reference parameter tensors for variance estimation
        """
        for new_param, ref_param in zip(new_params, reference_params):
            # Calculate reference variance
            ref_var = ref_param.data.var().item()
            micro_std = np.sqrt(ref_var * 1e-3)
            
            # Apply truncated normal initialization with micro variance
            truncated_normal_init(new_param.data, mean=0.0, std=micro_std, a=-2.0, b=2.0)
    
    def expand_topology(self, delta_M: int, delta_K: int, Q_expanded: np.ndarray):
        """
        Expand the neural network topology for incremental learning.
        
        Args:
            delta_M: int, number of new items to add
            delta_K: int, number of new knowledge concepts to add
            Q_expanded: np.ndarray, the expanded Q-matrix (M+delta_M) x (K+delta_K)
        """
        # Stage 1: Freeze all existing parameters
        self._freeze_parameters()
        
        # Update dimensions
        new_n_item = self.n_item + delta_M
        new_n_know = self.n_know + delta_K
        
        # Get reference parameters for variance estimation
        ref_weights = [p for name, p in self.named_parameters() if 'weight' in name]
        
        # Stage 2: Expand Theta_buf and Psi_buf
        with torch.no_grad():
            # Expand Theta_buf
            new_theta_buf = torch.zeros((self.n_user, new_n_know), device=self.device)
            new_theta_buf[:, :self.n_know] = self.Theta_buf.data
            self.Theta_buf = nn.Parameter(new_theta_buf, requires_grad=False)
            
            # Expand Psi_buf
            new_psi_buf = torch.zeros((new_n_item, new_n_know), device=self.device)
            new_psi_buf[:self.n_item, :self.n_know] = self.Psi_buf.data
            self.Psi_buf = nn.Parameter(new_psi_buf, requires_grad=False)
            
            # Update Q_mat
            self.Q_mat = torch.Tensor(Q_expanded).to(self.device)
        
        # Stage 3: Stage 1 - Lateral neural splitting for f_nn (diagnose_theta)
        f_linear = type(self.f_nn[0])  # Get PosLinear or nn.Linear
        # f_nn_new takes delta_M (new items) as input, outputs delta_K (new knowledge)
        self.f_nn_new = nn.Sequential(
            OrderedDict([
                ('f_new_layer_1', f_linear(delta_M, delta_K)),
                ('f_new_activate_1', nn.Sigmoid()),
                ('f_new_layer_2', f_linear(delta_K, delta_K)),
                ('f_new_activate_2', nn.Sigmoid())
            ])
        ).to(self.device)
        
        # Initialize new f_nn parameters with micro variance
        new_f_params = [self.f_nn_new[0].weight, self.f_nn_new[0].bias,
                        self.f_nn_new[2].weight, self.f_nn_new[2].bias]
        if len(ref_weights) >= 2:
            self._initialize_new_params_with_micro_variance(
                new_f_params, [ref_weights[0], ref_weights[0], ref_weights[1], ref_weights[1]]
            )
        
        # Stage 4: Stage 1 - Lateral neural splitting for g_nn (diagnose_psi)
        # Note: g_nn takes n_user as input (from item_log which is user x item matrix)
        self.g_nn_new = nn.Sequential(
            OrderedDict([
                ('g_new_layer_1', nn.Linear(self.n_user, delta_K)),
                ('g_new_activate_1', nn.Sigmoid()),
                ('g_new_layer_2', nn.Linear(delta_K, delta_K)),
                ('g_new_activate_2', nn.Sigmoid()),
                ('g_new_layer_3', nn.Linear(delta_K, delta_K)),
                ('g_new_activate_3', nn.Sigmoid())
            ])
        ).to(self.device)
        
        # Initialize new g_nn parameters with micro variance
        new_g_params = [self.g_nn_new[0].weight, self.g_nn_new[0].bias,
                        self.g_nn_new[2].weight, self.g_nn_new[2].bias,
                        self.g_nn_new[4].weight, self.g_nn_new[4].bias]
        if len(ref_weights) >= 3:
            self._initialize_new_params_with_micro_variance(
                new_g_params, [ref_weights[2], ref_weights[2], 
                              ref_weights[3], ref_weights[3],
                              ref_weights[4], ref_weights[4]]
            )
        
        # Stage 5: Stage 2 - Orthogonal mask expansion for aggregation matrices
        # Note: nn.Linear weight shape is (out_features, in_features)
        # Expand theta_agg_mat: input=n_know, output=user_dim
        new_theta_agg_weight = nn.Parameter(
            torch.zeros(self.user_dim, new_n_know, device=self.device)
        )
        new_theta_agg_bias = nn.Parameter(
            torch.zeros(self.user_dim, device=self.device)
        )
        with torch.no_grad():
            new_theta_agg_weight.data[:, :self.n_know] = self.theta_agg_mat.weight.data
            new_theta_agg_bias.data[:] = self.theta_agg_mat.bias.data
        
        # Initialize new weights with micro variance
        if len(ref_weights) >= 5:
            self._initialize_new_params_with_micro_variance(
                [new_theta_agg_weight.data[:, self.n_know:]], 
                [ref_weights[5]]
            )
        
        # Replace old theta_agg_mat
        self.theta_agg_mat = f_linear(new_n_know, self.user_dim, device=self.device)
        self.theta_agg_mat.weight = new_theta_agg_weight
        self.theta_agg_mat.bias = new_theta_agg_bias
        
        # Expand psi_agg_mat: input=n_know, output=item_dim
        new_psi_agg_weight = nn.Parameter(
            torch.zeros(self.item_dim, new_n_know, device=self.device)
        )
        new_psi_agg_bias = nn.Parameter(
            torch.zeros(self.item_dim, device=self.device)
        )
        with torch.no_grad():
            new_psi_agg_weight.data[:, :self.n_know] = self.psi_agg_mat.weight.data
            new_psi_agg_bias.data[:] = self.psi_agg_mat.bias.data
        
        # Initialize new weights with micro variance
        if len(ref_weights) >= 6:
            self._initialize_new_params_with_micro_variance(
                [new_psi_agg_weight.data[:, self.n_know:]], 
                [ref_weights[6]]
            )
        
        # Replace old psi_agg_mat
        self.psi_agg_mat = nn.Linear(new_n_know, self.item_dim, device=self.device)
        self.psi_agg_mat.weight = new_psi_agg_weight
        self.psi_agg_mat.bias = new_psi_agg_bias
        
        # Update dimensions
        self.n_item = new_n_item
        self.n_know = new_n_know
        self.is_expanded = True
        
        print(f"Topology expanded: {self.original_n_item} -> {self.n_item} items, "
              f"{self.original_n_know} -> {self.n_know} knowledge concepts")

    def expand_topology_lora(self, delta_M: int, delta_K: int, Q_expanded: np.ndarray, 
                            M_old: int, rank: int = 4):
        """
        Expand the neural network topology using Low-Rank Adaptation (LoRA) for incremental learning.
        
        Args:
            delta_M: int, number of new items to add
            delta_K: int, number of new knowledge concepts to add
            Q_expanded: np.ndarray, the expanded Q-matrix (M+delta_M) x (K+delta_K)
            M_old: int, number of original items (for slicing)
            rank: int, the rank of low-rank decomposition (default: 4)
        """
        # Stage 1: Freeze all existing parameters
        self._freeze_parameters()
        
        # Update dimensions
        new_n_item = self.n_item + delta_M
        new_n_know = self.n_know + delta_K
        
        # Stage 2: Expand Theta_buf and Psi_buf
        with torch.no_grad():
            # Expand Theta_buf
            new_theta_buf = torch.zeros((self.n_user, new_n_know), device=self.device)
            new_theta_buf[:, :self.n_know] = self.Theta_buf.data
            self.Theta_buf = nn.Parameter(new_theta_buf, requires_grad=False)
            
            # Expand Psi_buf
            new_psi_buf = torch.zeros((new_n_item, new_n_know), device=self.device)
            new_psi_buf[:self.n_item, :self.n_know] = self.Psi_buf.data
            self.Psi_buf = nn.Parameter(new_psi_buf, requires_grad=False)
            
            # Update Q_mat
            self.Q_mat = torch.Tensor(Q_expanded).to(self.device)
        
        # Stage 3: Stage 1 - LoRA-based lateral growth for g_nn (diagnose_psi)
        # g_nn takes n_user as input, outputs n_know
        # LoRA is applied after the first hidden layer (output dimension = n_know)
        # So: A: n_know x rank, B: rank x delta_K, W_new = A @ B: n_know x delta_K
        self.lora_rank = rank
        
        # MV-NN-LoRA: 使用极小方差正态分布初始化 (打破零梯度死亡)
        # 初始化: torch.randn(...) * 1e-3, 保证 A @ B 初始值为 ~1e-6 量级
        # LoRA parameters for g_nn (psi diagnosis)
        self.A_new_g = nn.Parameter(torch.randn(self.n_know, rank, device=self.device) * 1e-3)
        self.B_new_g = nn.Parameter(torch.randn(rank, delta_K, device=self.device) * 1e-3)
        
      
# Stage 4: LoRA-based lateral growth for f_nn (theta诊断)
    # 【修复1】：A_new_f 必须接受新题目(delta_M)，而不是旧隐藏层 H！
        self.A_new_f = nn.Parameter(torch.randn(delta_M, rank, device=self.device) * 1e-3)
        self.B_new_f = nn.Parameter(torch.randn(rank, delta_K, device=self.device) * 1e-3)
        # Stage 5: Expand aggregation matrices with LoRA
        f_linear = type(self.f_nn[0])
        
        # Expand theta_agg_mat: input=n_know, output=user_dim
        new_theta_agg_weight = nn.Parameter(
            torch.zeros(self.user_dim, new_n_know, device=self.device)
        )
        new_theta_agg_bias = nn.Parameter(
            torch.zeros(self.user_dim, device=self.device)
        )
        with torch.no_grad():
            new_theta_agg_weight.data[:, :self.n_know] = self.theta_agg_mat.weight.data
            new_theta_agg_bias.data[:] = self.theta_agg_mat.bias.data
        
        # LoRA for theta_agg_mat new part (MV-NN-LoRA)
        self.A_theta_agg = nn.Parameter(torch.randn(self.user_dim, rank, device=self.device) * 1e-3)
        self.B_theta_agg = nn.Parameter(torch.randn(rank, delta_K, device=self.device) * 1e-3)
        
        # Replace old theta_agg_mat
        self.theta_agg_mat = f_linear(new_n_know, self.user_dim, device=self.device)
        self.theta_agg_mat.weight = new_theta_agg_weight
        self.theta_agg_mat.bias = new_theta_agg_bias
        
        # Expand psi_agg_mat: input=n_know, output=item_dim
        new_psi_agg_weight = nn.Parameter(
            torch.zeros(self.item_dim, new_n_know, device=self.device)
        )
        new_psi_agg_bias = nn.Parameter(
            torch.zeros(self.item_dim, device=self.device)
        )
        with torch.no_grad():
            new_psi_agg_weight.data[:, :self.n_know] = self.psi_agg_mat.weight.data
            new_psi_agg_bias.data[:] = self.psi_agg_mat.bias.data
        
        # LoRA for psi_agg_mat new part (MV-NN-LoRA)
        self.A_psi_agg = nn.Parameter(torch.randn(self.item_dim, rank, device=self.device) * 1e-3)
        self.B_psi_agg = nn.Parameter(torch.randn(rank, delta_K, device=self.device) * 1e-3)
        
        # Replace old psi_agg_mat
        self.psi_agg_mat = nn.Linear(new_n_know, self.item_dim, device=self.device)
        self.psi_agg_mat.weight = new_psi_agg_weight
        self.psi_agg_mat.bias = new_psi_agg_bias
        
        # Update dimensions
        self.n_item = new_n_item
        self.n_know = new_n_know
        self.is_expanded = True
        self.use_lora = True  # Flag to indicate using LoRA expansion
        
        print(f"LoRA Topology expanded: {self.original_n_item} -> {self.n_item} items, "
              f"{self.original_n_know} -> {self.n_know} knowledge concepts (rank={rank})")

    def full_replay_oracle_expand_topology(self, delta_M: int, delta_K: int, Q_expanded: np.ndarray):
        """
        Full Replay Oracle - Theoretical Upper Bound for incremental learning.
        Uses all available data (old + new) for training with globally expanded network.
        This represents the ideal performance when the model has access to complete replay.
        
        Args:
            delta_M: int, number of new items to add
            delta_K: int, number of new knowledge concepts to add
            Q_expanded: np.ndarray, the expanded Q-matrix (M+delta_M) x (K+delta_K)
        """
        # Stage 1: Update topology attributes
        self.n_item_old = self.n_item
        self.n_item += delta_M
        self.n_know += delta_K
        
        # Update Q matrix
        self.Q_mat = torch.Tensor(Q_expanded).to(self.device)
        
        # Stage 2: Expand Theta_buf and Psi_buf
        with torch.no_grad():
            # Expand Theta_buf: [n_user, n_know_old] -> [n_user, n_know_new]
            old_theta = self.Theta_buf.data
            new_theta = torch.zeros((self.n_user, self.n_know), device=self.device)
            new_theta[:, :self.n_know - delta_K] = old_theta
            self.Theta_buf = nn.Parameter(new_theta, requires_grad=False)
            
            # Expand Psi_buf: [n_item_old, n_know_old] -> [n_item_new, n_know_new]
            old_psi = self.Psi_buf.data
            new_psi = torch.zeros((self.n_item, self.n_know), device=self.device)
            new_psi[:self.n_item - delta_M, :self.n_know - delta_K] = old_psi
            self.Psi_buf = nn.Parameter(new_psi, requires_grad=False)
        
        # Stage 3: Expand f_nn (diagnose_theta)
        # f_nn structure: [Linear(n_item, n_know) -> Sigmoid -> Linear(n_know, n_know) -> Sigmoid]
        f_linear = type(self.f_nn[0])
        n_know_old = self.n_know - delta_K
        n_item_old = self.n_item - delta_M
        
        # Expand f_layer_1: Linear(n_item, n_know) -> Linear(n_item+delta_M, n_know+delta_K)
        old_weight_f1 = self.f_nn[0].weight.data
        new_weight_f1 = torch.cat([
            torch.cat([old_weight_f1, torch.randn(n_know_old, delta_M, device=self.device) * 0.01], dim=1),
            torch.randn(delta_K, self.n_item, device=self.device) * 0.01
        ], dim=0)
        old_bias_f1 = self.f_nn[0].bias.data
        new_bias_f1 = torch.cat([old_bias_f1, torch.randn(delta_K, device=self.device) * 0.01], dim=0)
        
        self.f_nn[0] = f_linear(self.n_item, self.n_know, device=self.device)
        self.f_nn[0].weight = nn.Parameter(new_weight_f1)
        self.f_nn[0].bias = nn.Parameter(new_bias_f1)
        
        # Expand f_layer_2: Linear(n_know, n_know) -> Linear(n_know+delta_K, n_know+delta_K)
        old_weight_f2 = self.f_nn[2].weight.data
        new_weight_f2 = torch.cat([
            torch.cat([old_weight_f2, torch.randn(n_know_old, delta_K, device=self.device) * 0.01], dim=1),
            torch.randn(delta_K, self.n_know, device=self.device) * 0.01
        ], dim=0)
        old_bias_f2 = self.f_nn[2].bias.data
        new_bias_f2 = torch.cat([old_bias_f2, torch.randn(delta_K, device=self.device) * 0.01], dim=0)
        
        self.f_nn[2] = f_linear(self.n_know, self.n_know, device=self.device)
        self.f_nn[2].weight = nn.Parameter(new_weight_f2)
        self.f_nn[2].bias = nn.Parameter(new_bias_f2)
        
        # Stage 4: Expand g_nn (diagnose_psi)
        # g_nn structure: [Linear(n_user, n_know) -> Sigmoid -> Linear(n_know, n_know) -> Sigmoid -> Linear(n_know, n_know) -> Sigmoid]
        
        # Expand g_layer_1: Linear(n_user, n_know) -> Linear(n_user, n_know+delta_K)
        old_weight_g1 = self.g_nn[0].weight.data
        new_weight_g1 = torch.cat([old_weight_g1, torch.randn(delta_K, self.n_user, device=self.device) * 0.01], dim=0)
        old_bias_g1 = self.g_nn[0].bias.data
        new_bias_g1 = torch.cat([old_bias_g1, torch.randn(delta_K, device=self.device) * 0.01], dim=0)
        
        self.g_nn[0] = nn.Linear(self.n_user, self.n_know, device=self.device)
        self.g_nn[0].weight = nn.Parameter(new_weight_g1)
        self.g_nn[0].bias = nn.Parameter(new_bias_g1)
        
        # Expand g_layer_2: Linear(n_know, n_know) -> Linear(n_know+delta_K, n_know+delta_K)
        old_weight_g2 = self.g_nn[2].weight.data
        new_weight_g2 = torch.cat([
            torch.cat([old_weight_g2, torch.randn(n_know_old, delta_K, device=self.device) * 0.01], dim=1),
            torch.randn(delta_K, self.n_know, device=self.device) * 0.01
        ], dim=0)
        old_bias_g2 = self.g_nn[2].bias.data
        new_bias_g2 = torch.cat([old_bias_g2, torch.randn(delta_K, device=self.device) * 0.01], dim=0)
        
        self.g_nn[2] = nn.Linear(self.n_know, self.n_know, device=self.device)
        self.g_nn[2].weight = nn.Parameter(new_weight_g2)
        self.g_nn[2].bias = nn.Parameter(new_bias_g2)
        
        # Expand g_layer_3: Linear(n_know, n_know) -> Linear(n_know+delta_K, n_know+delta_K)
        old_weight_g3 = self.g_nn[4].weight.data
        new_weight_g3 = torch.cat([
            torch.cat([old_weight_g3, torch.randn(n_know_old, delta_K, device=self.device) * 0.01], dim=1),
            torch.randn(delta_K, self.n_know, device=self.device) * 0.01
        ], dim=0)
        old_bias_g3 = self.g_nn[4].bias.data
        new_bias_g3 = torch.cat([old_bias_g3, torch.randn(delta_K, device=self.device) * 0.01], dim=0)
        
        self.g_nn[4] = nn.Linear(self.n_know, self.n_know, device=self.device)
        self.g_nn[4].weight = nn.Parameter(new_weight_g3)
        self.g_nn[4].bias = nn.Parameter(new_bias_g3)
        
        # Stage 5: Expand aggregation matrices
        
        # Expand theta_agg_mat: Linear(n_know, user_dim) -> Linear(n_know+delta_K, user_dim)
        old_weight_theta = self.theta_agg_mat.weight.data
        new_weight_theta = torch.cat([old_weight_theta, torch.randn(self.user_dim, delta_K, device=self.device) * 0.01], dim=1)
        
        self.theta_agg_mat = f_linear(self.n_know, self.user_dim, device=self.device)
        self.theta_agg_mat.weight = nn.Parameter(new_weight_theta)
        # bias remains the same size (user_dim)
        
        # Expand psi_agg_mat: Linear(n_know, item_dim) -> Linear(n_know+delta_K, item_dim)
        old_weight_psi = self.psi_agg_mat.weight.data
        new_weight_psi = torch.cat([old_weight_psi, torch.randn(self.item_dim, delta_K, device=self.device) * 0.01], dim=1)
        
        self.psi_agg_mat = nn.Linear(self.n_know, self.item_dim, device=self.device)
        self.psi_agg_mat.weight = nn.Parameter(new_weight_psi)
        # bias remains the same size (item_dim)
        
        # Stage 6: Global unfreezing - all parameters trainable
        for name, param in self.named_parameters():
            param.requires_grad = True
        
        self.is_expanded = True
        self.use_full_replay_oracle = True  # Flag for Full Replay Oracle mode
        
        print(f"Full Replay Oracle Topology expanded: {self.original_n_item} -> {self.n_item} items, "
              f"{self.original_n_know} -> {self.n_know} knowledge concepts")
        print("INFO: Full Replay Oracle - Theoretical Upper Bound with complete replay access")

    def ncd_func(self, theta, psi):
        assert(self.user_dim == self.item_dim)
        y_pred = self.ncd(theta - psi)
        return y_pred

    def diagnose_theta(self, user_log: torch.Tensor): 
        if not self.is_expanded: 
            theta = self.f_nn(user_log) * (1-self.alpha) + torch.sigmoid(user_log @ self.Q_mat/(self.n_know**0.5)) * self.alpha 
            return theta 
        
        # Full Replay Oracle mode: use expanded network directly without lateral branching
        if getattr(self, 'use_full_replay_oracle', False):
            theta = self.f_nn(user_log) * (1-self.alpha) + torch.sigmoid(user_log @ self.Q_mat/(self.n_know**0.5)) * self.alpha 
            return theta 

        # ===== 1. 主路特征流形生成 (完全解耦) ===== 
        # 旧特征：由冻结的旧网络纯净解码旧题目 
        theta_old = self.f_nn(user_log[:, :self.original_n_item]) 
        
        if getattr(self, 'use_lora', False): 
            # Ours (LoRA): 直接让新网络看新题作答 x_new，彻底斩断对旧网络 h_old 的污染！ 
            x_new = user_log[:, self.original_n_item:] 
            W_new_f = torch.abs(torch.matmul(self.A_new_f, self.B_new_f)) 
            theta_new = torch.sigmoid(torch.matmul(x_new, W_new_f)) 
        else: 
            # Ours (Dense): 密集子网络看新题 
            theta_new = self.f_nn_new(user_log[:, self.original_n_item:]) 
            
        theta_concat = torch.cat([theta_old, theta_new], dim=-1) 

        # ===== 2. 统一的残差解耦与流形尺度对齐 ===== 
        # 旧残差：严格使用 original_n_know 尺度，历史信号绝不缩水 
        residual_old = torch.sigmoid( 
            user_log[:, :self.original_n_item] @ self.Q_mat[:self.original_n_item, :self.original_n_know] / (self.original_n_know**0.5) 
        ) 
        # 新残差：独立使用增量 delta_K 尺度 
        delta_K = self.n_know - self.original_n_know 
        residual_new = torch.sigmoid( 
            user_log @ self.Q_mat[:, self.original_n_know:] / (delta_K**0.5) 
        ) 
        residual_concat = torch.cat([residual_old, residual_new], dim=-1) 

        # 混合输出 
        theta = theta_concat * (1 - self.alpha) + residual_concat * self.alpha 
        return theta
    def diagnose_psi(self, item_log: torch.Tensor): 
        if not self.is_expanded: 
            return self.g_nn(item_log) 
        
        # Full Replay Oracle mode: use expanded network directly without lateral branching
        if getattr(self, 'use_full_replay_oracle', False):
            return self.g_nn(item_log)
             
        psi_old = self.g_nn(item_log) 
         
        if getattr(self, 'use_lora', False): 
            # 获取原网络倒数第二层特征用于 LoRA 侧向生长 
            h_old_g = self.g_nn[0](item_log) 
            h_old_g = self.g_nn[1](h_old_g) 
            h_old_g = self.g_nn[2](h_old_g) 
            h_old_g = self.g_nn[3](h_old_g) 
            W_new_g = torch.abs(torch.matmul(self.A_new_g, self.B_new_g)) 
            psi_new = torch.sigmoid(torch.matmul(h_old_g, W_new_g)) 
        else: 
            psi_new = self.g_nn_new(item_log) 
             
        return torch.cat([psi_old, psi_new], dim=-1)

    def diagnose_theta_psi(self,  user_log: torch.Tensor, item_log: torch.Tensor):
        '''
        For convenience, simultaneously diagnose learners' and items' traits.
        Args:
            user_log:torch.Tensor((batch_size, n_items)), the user logs.
                for each element, -1 = incorrect; 0 = skip; 1 = correct
            item_log:torch.Tensor((batch_size, n_items)), the user logs.
                for each element, -1 = incorrect; 0 = skip; 1 = correct
        Return:
            theta:torch.Tensor((batch_size, n_know)), diagnostic results
                of each learner.
            psi:torch.Tensor((batch_size, n_know)), diagnostic results
                of each item.
        '''
        theta = self.diagnose_theta(user_log)
        psi = self.diagnose_psi(item_log)
        return theta, psi
    
    def update_Theta_buf(self, theta_new, user_id):
        self.Theta_buf[user_id] = theta_new
    
    def update_Psi_buf(self, psi_new, item_id):
        self.Psi_buf[item_id] = psi_new

    def predict_response(self, theta, psi, Q_batch): 
        if not self.is_expanded: 
            theta_agg = F.linear(theta * Q_batch, self.theta_agg_mat.weight, self.theta_agg_mat.bias) 
            psi_agg = F.linear(psi * Q_batch, self.psi_agg_mat.weight, self.psi_agg_mat.bias) 
            return self.itf(theta_agg, psi_agg) 
        
        # Full Replay Oracle mode: use expanded network directly without decoupling
        if getattr(self, 'use_full_replay_oracle', False):
            theta_agg = F.linear(theta * Q_batch, self.theta_agg_mat.weight, self.theta_agg_mat.bias) 
            psi_agg = F.linear(psi * Q_batch, self.psi_agg_mat.weight, self.psi_agg_mat.bias) 
            return self.itf(theta_agg, psi_agg)

        # ===== 强制张量切片解耦 (修复 Q_new 广播坍缩 Bug) ===== 
        theta_old, theta_new = theta[:, :self.original_n_know], theta[:, self.original_n_know:] 
        psi_old, psi_new = psi[:, :self.original_n_know], psi[:, self.original_n_know:] 
         
        # 这里的冒号 [:, self.original_n_know:] 绝对不能丢！ 
        Q_old = Q_batch[:, :self.original_n_know] 
        Q_new = Q_batch[:, self.original_n_know:] 

        # 聚合历史空间 (带 Bias) 
        theta_agg_old = F.linear(theta_old * Q_old, self.theta_agg_mat.weight[:, :self.original_n_know], self.theta_agg_mat.bias) 
        psi_agg_old = F.linear(psi_old * Q_old, self.psi_agg_mat.weight[:, :self.original_n_know], self.psi_agg_mat.bias) 

        # 聚合增量空间 (无 Bias) - 使用扩展后的聚合矩阵的增量部分 
        if getattr(self, 'use_lora', False): 
            W_theta_new = torch.abs(torch.matmul(self.A_theta_agg, self.B_theta_agg)) 
            W_psi_new = torch.abs(torch.matmul(self.A_psi_agg, self.B_psi_agg)) 
            theta_agg_new = F.linear(theta_new * Q_new, W_theta_new, None) 
            psi_agg_new = F.linear(psi_new * Q_new, W_psi_new, None) 
        else: 
            # Dense 模式：使用聚合矩阵的增量部分 
            theta_agg_new = F.linear(theta_new * Q_new, self.theta_agg_mat.weight[:, self.original_n_know:], None) 
            psi_agg_new = F.linear(psi_new * Q_new, self.psi_agg_mat.weight[:, self.original_n_know:], None) 

        theta_agg = theta_agg_old + theta_agg_new 
        psi_agg = psi_agg_old + psi_agg_new 

        return self.itf(theta_agg, psi_agg)

    def forward(self, user_log: torch.Tensor, item_log: torch.Tensor, \
        user_id: torch.LongTensor, item_id: torch.LongTensor):
        theta, psi = self.diagnose_theta_psi(user_log, item_log)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.predict_response(theta, psi, Q_batch)
        return output

    def forward_using_buf(self, user_id: torch.LongTensor, \
        item_id: torch.LongTensor):
        ''' 
        Unlike forward(), this method predict response using thetas
        and psis from bufferes rather than from outputs of diagnostic modules
        given response logs.
        '''
        theta = self.Theta_buf[user_id].squeeze(dim=1)
        psi = self.Psi_buf[item_id].squeeze(dim=1)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.predict_response(theta, psi, Q_batch)
        return output

    def get_Theta_buf(self):
        return self.Theta_buf.detach().cpu()

    def get_Psi_buf(self):
        return self.Psi_buf.detach().cpu()

# 2025.04.21. Add UAutoRec and CDAE
class UAutoRec(nn.Module):
    def __init__(self, n_user: int, n_item: int, \
        hidden_dim: int, device = torch.device('cpu')):
        super(UAutoRec, self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.hidden_dim = hidden_dim 
        self.device = device 
        self.f_enc = nn.Linear(n_item, \
            hidden_dim).to(device)
        self.f_dec = nn.Linear(hidden_dim, \
            n_item).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, x_input: torch.Tensor, \
        user_id: torch.LongTensor):
        h = torch.sigmoid(self.f_enc(x_input))
        x_output = torch.sigmoid(self.f_dec(h))
        return x_output

class CDAE(nn.Module):
    def __init__(self, n_user: int, n_item: int, \
        hidden_dim: int, device = torch.device('cpu')):
        super(CDAE, self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.hidden_dim = hidden_dim 
        self.device = device 
        self.f_enc = nn.Linear(n_item, \
            hidden_dim).to(device)
        self.user_emb = nn.Embedding(n_user, \
            n_item).to(device)
        self.dropout = nn.Dropout(p=0.5).to(device)
        self.f_dec = nn.Linear(hidden_dim, \
            n_item).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, x_input: torch.Tensor, \
        user_id: torch.LongTensor):
        # print(x_input.size(),self.user_emb(user_id).size())
        h = torch.sigmoid(self.dropout(\
            self.f_enc(x_input+self.user_emb(user_id).squeeze(dim=1))))
        x_output = torch.sigmoid(self.f_dec(h))
        return x_output