# -*- coding: utf-8 -*-
"""
Run GNCDM on Math1 Real Dataset
"""

import sys
import os

# Add the core directory to path
core_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core')
sys.path.insert(0, core_dir)

import json
import pandas as pd
import numpy as np
import torch
from model import GNCDM
from train import train, eval

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'math1_real_result')

# Create save directory
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load data
print("=" * 80)
print("Loading Math1 Real Dataset")
print("=" * 80)

df_train = pd.read_csv(os.path.join(DATA_DIR, 'math1_train_0.8_0.2.csv'))
df_valid = pd.read_csv(os.path.join(DATA_DIR, 'math1_valid_0.8_0.2.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'math1_test_0.8_0.2.csv'))

Q_mat = np.load(os.path.join(DATA_DIR, 'math1_Q_matrix.npy'))

n_user = 4209
n_item = 20
n_know = 11

print(f"  Users: {n_user}")
print(f"  Items: {n_item}")
print(f"  Knowledge concepts: {n_know}")
print(f"  Train records: {len(df_train)}")
print(f"  Valid records: {len(df_valid)}")
print(f"  Test records: {len(df_test)}")
print()

# Create model
print("=" * 80)
print("Creating and Training Model")
print("=" * 80)

device = torch.device('cpu')
net = GNCDM(
    n_user=n_user,
    n_item=n_item,
    n_know=n_know,
    user_dim=32,
    item_dim=32,
    alpha=0.8,
    Q_mat=Q_mat,
    monotonicity_assumption=True,
    device=device
)

# Train model
print("Starting training...")
result_all = train(
    net,
    df_train,
    df_valid,
    batch_size=32,
    lr=1e-3,
    n_epoch=20
)

# Save training results
np.save(os.path.join(SAVE_DIR, 'result_all.npy'), result_all)

# Evaluate on test set
print()
print("=" * 80)
print("Evaluating on Test Set")
print("=" * 80)

test_result = eval(net, df_test, batch_size=256)
print(f"  AUC: {test_result['auc']:.4f}")
print(f"  RMSE: {test_result['rmse']:.4f}")
print(f"  ACC: {test_result['acc']:.4f}")
print(f"  F1: {test_result['f1']:.4f}")

# Save test result
with open(os.path.join(SAVE_DIR, 'test_result.json'), 'w') as fp:
    json.dump(test_result, fp)

# Save model
torch.save(net, os.path.join(SAVE_DIR, 'model.pt'))
print()
print(f"Results saved to {SAVE_DIR}")
print("=" * 80)
print("Done!")

