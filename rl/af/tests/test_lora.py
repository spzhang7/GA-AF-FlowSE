import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is only installed on server")
nn = torch.nn

from rl.common.lora import (  # noqa: E402
    add_direction,
    capture_runtime_state,
    direction_norm,
    inject_lora,
    lora_enabled,
    lora_parameters,
    per_tensor_norms,
    random_direction_like,
    restore_runtime_state,
    snapshot_lora,
    subtract_states,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = nn.Linear(3, 4)
        self.block = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))

    def forward(self, value):
        return self.block(self.keep(value))


def adapted_model():
    torch.manual_seed(7)
    model = TinyModel()
    baseline = copy.deepcopy(model)
    report = inject_lora(
        model,
        target_patterns=[r"block\.0", r"block\.2"],
        rank=2,
        alpha=2.0,
        expected_modules=2,
    )
    return model, baseline, report


def test_zero_lora_is_exact_frozen_baseline():
    model, baseline, report = adapted_model()
    inputs = torch.randn(6, 3)
    torch.testing.assert_close(model(inputs), baseline(inputs), rtol=0, atol=0)
    assert report.module_names == ("block.0", "block.2")
    assert all(not parameter.requires_grad for parameter in model.keep.parameters())
    assert len(lora_parameters(model)) == 4


def test_plus_minus_and_random_are_tensor_norm_matched():
    model, _, _ = adapted_model()
    before = snapshot_lora(model)
    after = {name: value + torch.randn_like(value) * 0.1 for name, value in before.items()}
    direction = subtract_states(after, before)
    plus = add_direction(before, direction, 0.5)
    minus = add_direction(before, direction, -0.5)
    for name in before:
        torch.testing.assert_close(plus[name] - before[name], -(minus[name] - before[name]))
    random_direction = random_direction_like(direction, seed=123)
    assert per_tensor_norms(random_direction) == pytest.approx(per_tensor_norms(direction))
    assert direction_norm(random_direction) == pytest.approx(direction_norm(direction))


def test_lora_disable_and_runtime_restore_are_exact():
    model, baseline, _ = adapted_model()
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    state = capture_runtime_state(model, optimizer)
    inputs = torch.randn(4, 3)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(value, state.adapter[name])
        for name, value in snapshot_lora(model).items()
    )
    restore_runtime_state(model, optimizer, state)
    for name, value in snapshot_lora(model).items():
        torch.testing.assert_close(value, state.adapter[name], rtol=0, atol=0)
    with lora_enabled(model, False):
        torch.testing.assert_close(model(inputs), baseline(inputs), rtol=0, atol=0)
    assert np.random.get_state()[0] == state.numpy_rng[0]

