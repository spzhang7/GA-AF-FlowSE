import importlib.util

import pytest


torch_missing = importlib.util.find_spec("torch") is None
pytestmark = pytest.mark.skipif(torch_missing, reason="PyTorch is only installed on server")


def test_padding_does_not_change_loss_or_gradient():
    import torch

    from rl.common.flow_matching import (
        completed_square_loss,
    )

    current = torch.tensor([[[1.0], [3.0], [99.0]]], requires_grad=True)
    target = torch.zeros_like(current)
    mask = torch.tensor([[True, True, False]])
    loss = completed_square_loss(current, target, torch.tensor([2.0]), mask)
    gradient = torch.autograd.grad(loss, current)[0]
    assert loss.item() == pytest.approx(10.0)
    assert gradient[0, 2, 0].item() == 0.0


