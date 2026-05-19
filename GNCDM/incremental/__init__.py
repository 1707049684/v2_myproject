# -*- coding: utf-8 -*-
"""
Incremental Learning Module for Generative-CD

This module contains the core components for dynamic neural architecture
incremental learning in the G-NCDM model.

Components:
    - TopologyAwareDecoupledLoss: Decoupled loss for knowledge distillation
    - IncrementalDecoupledLoss: Simplified loss interface
    - calculate_tmd: Trait Manifold Drift calculator
    - LinearWarmupScheduler: Learning rate warm-up scheduler
    - CosineAnnealingWarmup: Combined warm-up and annealing scheduler
"""

from .loss import TopologyAwareDecoupledLoss, IncrementalDecoupledLoss
from .metrics import (
    calculate_tmd, 
    calculate_tmd_torch,
    LinearWarmupScheduler, 
    CosineAnnealingWarmup
)

__all__ = [
    'TopologyAwareDecoupledLoss',
    'IncrementalDecoupledLoss',
    'calculate_tmd',
    'calculate_tmd_torch',
    'LinearWarmupScheduler',
    'CosineAnnealingWarmup'
]
