"""Smoke and protocol-safety tests for the ICDM-WWW'24 adapter."""

import numpy as np
import pandas as pd
import torch
from baselines.icdm_ww24 import ICDMWW24, InteractionGraph
from experiments._core.run_icdm_ww24 import split_support_query


def _model():
    q = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    known = torch.tensor([True, True, False, False])
    return ICDMWW24(
        n_user=4,
        n_item=3,
        n_know=2,
        q_matrix=q,
        known_user_mask=known,
        dim=8,
        k_hop=2,
        dropout=0.0,
    )


def _graph():
    return InteractionGraph(
        user=torch.tensor([0, 0, 1, 2]),
        item=torch.tensor([0, 1, 1, 2]),
        score=torch.tensor([1.0, 0.0, 1.0, 1.0]),
    )


def test_icdm_forward_and_backward():
    model = _model()
    graph = _graph()
    state = model.encode(graph)
    probability = model.predict(torch.tensor([0, 2]), torch.tensor([2, 0]), state)
    assert probability.shape == (2,)
    assert torch.isfinite(probability).all()
    loss = model.loss(graph, graph)
    loss.backward()
    assert torch.isfinite(loss)


def test_unseen_user_does_not_use_untrained_id_embedding():
    model = _model().eval()
    graph = _graph()
    with torch.no_grad():
        before = model.encode(graph).student[2].clone()
        model.student_embedding.weight[2].fill_(1000.0)
        after = model.encode(graph).student[2]
    assert torch.allclose(before, after)


def test_support_query_indices_are_disjoint():
    frame = pd.DataFrame(
        {
            "user_id": np.repeat([10, 11], 4),
            "item_id": np.tile(np.arange(4), 2),
            "score": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    support, query = split_support_query(frame)
    assert set(support.index).isdisjoint(query.index)
    assert len(support) + len(query) == len(frame)
    assert support.groupby("user_id").size().to_dict() == {10: 2, 11: 2}


def test_zero_q_item_uses_diagnostic_fallback():
    q = np.array([[1, 0], [0, 0]], dtype=np.float32)
    model = ICDMWW24(2, 2, 2, q, torch.tensor([True, True]), dim=4, dropout=0.0)
    graph = InteractionGraph(
        user=torch.tensor([0, 1]),
        item=torch.tensor([0, 1]),
        score=torch.tensor([1.0, 0.0]),
    )
    probability = model.predict(torch.tensor([1]), torch.tensor([1]), model.encode(graph))
    assert torch.isfinite(probability).all()
    assert torch.equal(model.diagnostic_q[1], torch.ones(2))
