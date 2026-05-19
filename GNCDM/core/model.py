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
        self.f_nn_new = nn.Sequential(
            OrderedDict([
                ('f_new_layer_1', f_linear(new_n_item, delta_K)),
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

    def ncd_func(self, theta, psi):
        assert(self.user_dim == self.item_dim)
        y_pred = self.ncd(theta - psi)
        return y_pred

    def diagnose_theta(self, user_log: torch.Tensor):
        '''
        Directly diagnose learner cognitive states from their logs.
        This method is recommended to use after training the model.
        Args:
            user_log:torch.Tensor((batch_size, n_items)), the user logs.
                for each element, -1 = incorrect; 0 = skip; 1 = correct
        Return:
            theta:torch.Tensor((batch_size, n_know)), diagnostic results
                of each learner.
        '''
        if not self.is_expanded:
            # Original behavior
            theta = self.f_nn(user_log) * (1-self.alpha) + torch.sigmoid(user_log @ self.Q_mat/(self.n_know**0.5)) * self.alpha
            return theta
        else:
            # Expanded behavior: concatenate old and new features
            # Get old features from frozen f_nn
            theta_old = self.f_nn(user_log[:, :self.original_n_item])
            
            # Get new features from new sub-network
            theta_new = self.f_nn_new(user_log)
            
            # Concatenate along knowledge dimension
            theta_concat = torch.cat([theta_old, theta_new], dim=-1)
            
            # Mix with Q-matrix projection
            theta = theta_concat * (1-self.alpha) + torch.sigmoid(user_log @ self.Q_mat/(self.n_know**0.5)) * self.alpha
            return theta

    def diagnose_psi(self, item_log: torch.Tensor):
        '''
        Args:
            item_log:torch.Tensor((batch_size, n_users)), the item logs.
                for each element, -1 = incorrect; 0 = skip; 1 = correct
        Return:
            psi:torch.Tensor((batch_size, n_know)), diagnostic results
                of each item.
        '''
        if not self.is_expanded:
            # Original behavior
            psi = self.g_nn(item_log)
            return psi
        else:
            # Expanded behavior: concatenate old and new features
            # Get old features from frozen g_nn (takes n_user as input)
            psi_old = self.g_nn(item_log)
            
            # Get new features from new sub-network
            psi_new = self.g_nn_new(item_log)
            
            # Concatenate along knowledge dimension
            psi_concat = torch.cat([psi_old, psi_new], dim=-1)
            return psi_concat

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
        '''
        Predict response scores given a batch of theta (learners' cognitive states),
        psi (items' features), and Q-vectors of these items
        Args:
            theta:torch.Tensor((batch_size, n_know)), learners' cognitive states
            psi:torch.Tensor((batch_size, n_know)), items' cognitive states
            Q_batch:torch.Tensor((batch_size, n_know)), Q-vectors. Q_batch[i] is
                the Q-vector of the item with feature psi[i]
        Return:
            output:torch.Tensor((batch_size,1)), the predicted correct probability
                of each pair of learner and item.
        '''
        theta_agg = self.theta_agg_mat(theta * Q_batch)
        psi_agg = self.psi_agg_mat(psi * Q_batch)
        output = self.itf(theta_agg, psi_agg)
        return output

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