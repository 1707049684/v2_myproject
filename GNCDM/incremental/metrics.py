# -*- coding: utf-8 -*-
"""
Incremental Learning Metrics and Utilities

This module contains evaluation metrics and schedulers for
incremental learning experiments.
"""

import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


def calculate_rd(theta_old: np.ndarray, theta_new: np.ndarray, K_old: int) -> float:
    """
    Calculate Representation Drift (RD) metric.
    
    RD measures the average root mean square deviation of the latent
    trait vectors in the old knowledge dimension before and after
    incremental learning.
    
    Formula:
        RD = mean(||theta_old - theta_new[:K_old]||_2 / sqrt(K_old))
    
    Args:
        theta_old: np.ndarray, original theta matrix (n_user, n_know)
        theta_new: np.ndarray, evolved theta matrix (n_user, n_know_total)
        K_old: int, original knowledge dimension
    
    Returns:
        float, the Representation Drift value
    
    Example:
        >>> theta_old = np.random.randn(100, 10)
        >>> theta_new = np.random.randn(100, 15)
        >>> rd = calculate_rd(theta_old, theta_new, 10)
        >>> print(f"RD: {rd:.4f}")
    """
    theta_new_old_dim = theta_new[:, :K_old]
    per_user_norm = np.linalg.norm(theta_old - theta_new_old_dim, axis=1)
    rd = np.mean(per_user_norm / np.sqrt(K_old))
    return float(rd)


def calculate_rd_torch(theta_old: torch.Tensor, theta_new: torch.Tensor, K_old: int) -> torch.Tensor:
    """
    PyTorch version of RD calculation.
    
    Args:
        theta_old: torch.Tensor, original theta matrix (n_user, n_know)
        theta_new: torch.Tensor, evolved theta matrix (n_user, n_know_total)
        K_old: int, original knowledge dimension
    
    Returns:
        torch.Tensor, the Representation Drift value
    """
    theta_new_old_dim = theta_new[:, :K_old]
    per_user_norm = torch.norm(theta_old - theta_new_old_dim, dim=1)
    rd = torch.mean(per_user_norm / math.sqrt(K_old))
    return rd


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear Learning Rate Warm-up Scheduler for incremental learning.
    
    This scheduler increases the learning rate linearly from a small value
    to the target learning rate during the warm-up period, then keeps it
    constant. This helps stabilize training in the early stages of
    incremental learning.
    
    Args:
        optimizer: torch.optim.Optimizer, the optimizer to schedule
        warmup_epochs: int, number of warm-up epochs (default: 5)
        initial_lr: float, initial learning rate during warm-up (default: 1e-5)
        last_epoch: int, the index of the last epoch (default: -1)
    
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = LinearWarmupScheduler(optimizer, warmup_epochs=5)
        >>> for epoch in range(20):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 5, 
                 initial_lr: float = 1e-5, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate current learning rate based on epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up phase
            progress = self.last_epoch / self.warmup_epochs
            # Linear interpolation from initial_lr to base_lr
            warmup_factor = self.initial_lr / self.base_lr if self.base_lr > 0 else 0
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * progress
            return [lr for _ in self.base_lrs]
        else:
            # After warm-up, use base learning rate
            return [self.base_lr for _ in self.base_lrs]
    
    def _get_closed_form_lr(self):
        """Get learning rate from closed form (for state_dict)."""
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * progress
            return [lr for _ in self.base_lrs]
        else:
            return [self.base_lr for _ in self.base_lrs]


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine Annealing with Linear Warm-up Scheduler.
    
    Combines linear warm-up with cosine annealing decay for
    smooth learning rate scheduling in incremental learning.
    
    Args:
        optimizer: torch.optim.Optimizer
        warmup_epochs: int, number of warm-up epochs
        total_epochs: int, total number of training epochs
        min_lr: float, minimum learning rate after annealing (default: 1e-6)
        last_epoch: int
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 5,
                 total_epochs: int = 100, min_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        super(CosineAnnealingWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up
            progress = self.last_epoch / self.warmup_epochs
            return [self.base_lr * progress for _ in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
            return [lr for _ in self.base_lrs]
