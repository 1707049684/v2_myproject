# -*- coding: utf-8 -*-
"""
Simple script to run GNCDM on Math1 Real Dataset
"""

import sys
import os

# Add the GNCDM directory to path
gncdm_dir = os.path.dirname(os.path.abspath(__file__))
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
SAVE_DIR = os.path.join(gncdm_dir, 'math1_result')

# Create save directory
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Dataset class (from core/train.py)
class IDCDataset(Dataset):
    def __init__(self, data, n_user, n_item):
        self.data = data
        self.n_user = n_user
        self.n_item = n_item
        
    def __getitem__(self, index):
        user_id = self.data.loc[index, 'user_id']
        item_id = self.data.loc[index, 'item_id']
        label = self.data.loc[index, 'score']
        return user_id, item_id, label
    
    def __len__(self):
        return len(self.data)

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

# Training function
def train_model(model, train_data, valid_data, batch_size=32, lr=1e-3, n_epoch=20):
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
            
            # Create user log and item log (simple version for this case)
            user_log = torch.zeros((len(user_ids), n_item))
            item_log = torch.zeros((len(item_ids), n_user))
            
            pred = model(user_log, item_log, user_ids, item_ids)
            loss = criterion(pred, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            valid_preds = []
            valid_labels = []
            
            valid_dataset = IDCDataset(valid_data, n_user, n_item)
            valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
            
            for user_ids, item_ids, labels in valid_loader:
                user_ids = user_ids.long()
                item_ids = item_ids.long()
                
                user_log = torch.zeros((len(user_ids), n_item))
                item_log = torch.zeros((len(item_ids), n_user))
                
                pred = model(user_log, item_log, user_ids, item_ids)
                valid_preds.extend(pred.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
            
            valid_auc = roc_auc_score(valid_labels, valid_preds)
            valid_rmse = np.sqrt(mean_squared_error(valid_labels, valid_preds))
            valid_pred_binary = (np.array(valid_preds) >= 0.5).astype(int)
            valid_acc = accuracy_score(valid_labels, valid_pred_binary)
            valid_f1 = f1_score(valid_labels, valid_pred_binary)
        
        print(f"  Epoch {epoch+1}")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Valid AUC: {valid_auc:.4f}")
        print(f"    Valid RMSE: {valid_rmse:.4f}")
        print(f"    Valid ACC: {valid_acc:.4f}")
        print(f"    Valid F1: {valid_f1:.4f}")
        
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Train the model
print("Starting training...")
net = train_model(net, df_train, df_valid, batch_size=256, lr=1e-3, n_epoch=15)

# Evaluate on test set
print()
print("=" * 80)
print("Evaluating on Test Set")
print("=" * 80)

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        
        test_dataset = IDCDataset(test_data, n_user, n_item)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        for user_ids, item_ids, _labels in tqdm(test_loader, desc="Evaluating"):
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
        
        return {
            'auc': auc,
            'rmse': rmse,
            'acc': acc,
            'f1': f1
        }

test_result = evaluate_model(net, df_test)
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

