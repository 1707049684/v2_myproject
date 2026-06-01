"""Minimal smoke tests for the GNCDM model.

These do not check learning quality — they only confirm the model builds and a
forward pass runs end to end, so Claude/CI can catch import or shape breakage
in seconds without running a full experiment.
"""
import numpy as np
import torch

from core.model import GNCDM
from core.train import IDCDataset
import pandas as pd

N_USER, N_ITEM, N_KNOW = 5, 4, 3
DIM = 8


def _build_model():
    rng = np.random.default_rng(0)
    q_mat = (rng.random((N_ITEM, N_KNOW)) > 0.5).astype("float32")
    q_mat[:, 0] = 1.0  # ensure no all-zero rows
    return GNCDM(
        n_user=N_USER,
        n_item=N_ITEM,
        n_know=N_KNOW,
        user_dim=DIM,
        item_dim=DIM,
        alpha=0.5,
        Q_mat=q_mat,
        device=torch.device("cpu"),
    )


def test_model_builds():
    model = _build_model()
    assert model.n_user == N_USER
    assert model.n_item == N_ITEM
    assert model.Q_mat.shape == (N_ITEM, N_KNOW)


def test_forward_using_buf_runs():
    model = _build_model()
    model.eval()
    user_id = torch.tensor([[0], [1]], dtype=torch.long)
    item_id = torch.tensor([[0], [1]], dtype=torch.long)
    with torch.no_grad():
        out = model.forward_using_buf(user_id, item_id)
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_idc_dataset_loads():
    df = pd.DataFrame(
        {"user_id": [0, 1, 2], "item_id": [0, 1, 2], "score": [1, 0, 1]}
    )
    ds = IDCDataset(df, n_user=N_USER, n_item=N_ITEM)
    assert len(ds) == 3
    user_log, item_log, user_id, item_id, score = ds[0]
    assert user_log.shape == (N_ITEM,)
    assert item_log.shape == (N_USER,)
    assert int(user_id) == 0
    assert int(score) == 1
