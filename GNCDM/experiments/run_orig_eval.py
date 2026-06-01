# -*- coding: utf-8 -*-
"""Authentic original-path driver: calls core.train.train() + core.train.eval()
directly (no reimplementation), to read the paper's own 'Score Reconstruction'
number on the math1 user split. No seed set (mirrors original run.py)."""
import os
import sys

gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import numpy as np
import pandas as pd
import torch

from core.model import GNCDM
from core import train as train_mod

repo_root = os.path.dirname(gncdm_dir)
DATA = os.path.join(repo_root, "data", "math1", "user_split")
Q = np.load(os.path.join(gncdm_dir, "data", "math1_Q_matrix.npy"))

device = torch.device("cpu")
df_train = pd.read_csv(os.path.join(DATA, "train.csv"))
df_valid = pd.read_csv(os.path.join(DATA, "valid.csv"))
df_test = pd.read_csv(os.path.join(DATA, "test.csv"))

net = GNCDM(4209, 20, 11, 32, 32, 0.95, Q_mat=Q,
            monotonicity_assumption=True, device=device)
train_mod.train(net, df_train, df_valid, batch_size=16, lr=1e-3, n_epoch=3)
res = train_mod.eval(net, df_test, batch_size=256)
print("\n=== original core.train.eval on math1 user-split test ===")
print("Score Prediction   :", res)
print("Score Reconstruction:", res.get("without_buf"))
