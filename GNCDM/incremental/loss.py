# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyAwareDecoupledLoss(nn.Module):
    """
    Topology-Aware Decoupled Loss for incremental learning in G-NCDM.
    
    This loss function handles hybrid batches with is_new flags, combining:
    - Knowledge distillation loss for preserving old knowledge (L_old)
    - BCE loss for learning new knowledge (L_new)
    - Spatio-temporal adaptive weighting with cosine annealing
    
    Args:
        model_old: Frozen teacher model (pre-trained on old data)
        model_dynamic: Dynamic student model (being trained incrementally)
        original_know_dim: Original knowledge dimension K before expansion
        device: torch.device
    """
    
    def __init__(self, model_old, model_dynamic, original_know_dim: int, 
                 device=torch.device('cpu')):
        super(TopologyAwareDecoupledLoss, self).__init__()
        self.model_old = model_old
        self.model_dynamic = model_dynamic
        self.original_know_dim = original_know_dim
        self.device = device
        
        # Freeze the old model
        self.model_old.eval()
        for param in self.model_old.parameters():
            param.requires_grad = False
        
        # BCE loss for score reconstruction
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, user_log: torch.Tensor, item_log: torch.Tensor,
                user_id: torch.LongTensor, item_id: torch.LongTensor,
                score: torch.Tensor, is_new: torch.Tensor,
                epoch: int, total_epochs: int,
                V_old: int, V_new: int):
        """
        Compute the topology-aware decoupled loss.
        
        Args:
            user_log: torch.Tensor (batch_size, n_items), user response logs
            item_log: torch.Tensor (batch_size, n_items), item response logs
            user_id: torch.LongTensor (batch_size,), user indices
            item_id: torch.LongTensor (batch_size,), item indices
            score: torch.Tensor (batch_size,), true scores (0 or 1)
            is_new: torch.Tensor (batch_size,), boolean mask indicating new samples
            epoch: int, current training epoch
            total_epochs: int, total number of training epochs
            V_old: int, number of non-zero elements in old Q-matrix
            V_new: int, number of non-zero elements in new Q-matrix (delta)
        
        Returns:
            total_loss: torch.Tensor, combined loss with adaptive weighting
            loss_components: dict, containing individual loss components
        """
        # ==============================================
        # Part 1: Calculate spatial-temporal adaptive weights
        # ==============================================
        # Spatial base weight: proportional to new knowledge volume
        alpha_base = V_new / (V_old + V_new) if (V_old + V_new) > 0 else 0.5
        
        # Temporal cosine annealing
        cos_anneal = 0.5 * (1 + math.cos(epoch * math.pi / total_epochs))
        alpha = alpha_base * cos_anneal  # Weight for new loss
        beta = 1.0 - alpha  # Weight for old loss
        
        # ==============================================
        # Part 2: L_old - Knowledge distillation for old knowledge preservation
        # ==============================================
        old_mask = ~is_new  # is_new == False
        L_old = torch.tensor(0.0, device=self.device)
        
        if old_mask.any():
            # Extract old samples
            user_log_old = user_log[old_mask]
            item_log_old = item_log[old_mask]
            
            # Get old model's theta (frozen, no gradients)
            # Need to slice to original item dimension for model_old
            user_log_old_sliced = user_log_old[:, :self.model_old.n_item]
            with torch.no_grad():
                theta_old_target = self.model_old.diagnose_theta(user_log_old_sliced)
            
            # Get dynamic model's theta
            theta_dynamic = self.model_dynamic.diagnose_theta(user_log_old)
            
            # Compute L2 loss between first K dimensions (original knowledge)
            theta_old_dim = theta_dynamic[:, :self.original_know_dim]
            L_old = F.mse_loss(theta_old_dim, theta_old_target)
        
        # ==============================================
        # Part 3: L_new - BCE loss for new knowledge learning
        # ==============================================
        new_mask = is_new  # is_new == True
        L_new = torch.tensor(0.0, device=self.device)
        
        if new_mask.any():
            # Extract new samples
            user_log_new = user_log[new_mask]
            item_log_new = item_log[new_mask]
            item_id_new = item_id[new_mask]
            score_new = score[new_mask]
            
            # Forward pass through dynamic model
            pred = self.model_dynamic(user_log_new, item_log_new, 
                                      torch.zeros_like(item_id_new), item_id_new)
            
            # Compute BCE loss only for non-missing responses
            # Filter out padding/missing markers (-1)
            valid_mask = (score_new != -1).float()
            
            if valid_mask.sum() > 0:
                bce_raw = self.bce_loss(pred.squeeze(), score_new.squeeze())
                L_new = (bce_raw * valid_mask).sum() / valid_mask.sum()
        
        # ==============================================
        # Part 4: Combine losses with adaptive weights
        # ==============================================
        total_loss = alpha * L_new + beta * L_old
        
        # Return loss components for monitoring
        loss_components = {
            'total_loss': total_loss.item(),
            'L_old': L_old.item(),
            'L_new': L_new.item(),
            'alpha': alpha,
            'beta': beta,
            'alpha_base': alpha_base,
            'cos_anneal': cos_anneal
        }
        
        return total_loss, loss_components


class IncrementalDecoupledLoss(nn.Module):
    """
    Simplified interface for the incremental decoupled loss.
    Wraps TopologyAwareDecoupledLoss with additional utilities.
    """
    
    def __init__(self, model_old, model_dynamic, original_know_dim: int,
                 device=torch.device('cpu')):
        super(IncrementalDecoupledLoss, self).__init__()
        self.loss_fn = TopologyAwareDecoupledLoss(
            model_old=model_old,
            model_dynamic=model_dynamic,
            original_know_dim=original_know_dim,
            device=device
        )
    
    def forward(self, batch, epoch: int, total_epochs: int,
                V_old: int, V_new: int):
        """
        Args:
            batch: tuple containing (user_log, item_log, user_id, item_id, score, is_new)
            epoch: int, current epoch
            total_epochs: int, total epochs
            V_old: int, non-zero elements in old Q-matrix
            V_new: int, non-zero elements in new Q-matrix delta
        """
        user_log, item_log, user_id, item_id, score, is_new = batch
        return self.loss_fn(
            user_log=user_log,
            item_log=item_log,
            user_id=user_id,
            item_id=item_id,
            score=score,
            is_new=is_new,
            epoch=epoch,
            total_epochs=total_epochs,
            V_old=V_old,
            V_new=V_new
        )
