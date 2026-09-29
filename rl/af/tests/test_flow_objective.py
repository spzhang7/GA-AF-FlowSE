import pytest

torch = pytest.importorskip("torch", reason="PyTorch is only installed on server")

from rl.common.conditioning import (  # noqa: E402
    ConditioningProtocol,
)
from rl.common.flow_objective import (  # noqa: E402
    PairedTrainingCondition,
    RolloutTrainingCondition,
    advantageflow_loss,
    paired_anchor_loss,
    sample_times,
)


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, *, x, cond, text, time, mask, **_):
        return self.scale * x + 0.1 * cond


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyTransformer()


class TinyBundle:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = TinyModel()

    def prepare_policy_text(self, transcript, conditioning):
        return transcript

    def _tokenize(self, transcript, batch):
        return torch.zeros(batch, 1, dtype=torch.long)


def _conditioning():
    return ConditioningProtocol(mode="text", use_text=True, drop_text=False)


def _rollout_condition():
    return RolloutTrainingCondition(
        utterance="p001_001",
        transcript="hello",
        condition_mel=torch.randn(1, 5, 3),
        terminal_mels=torch.randn(4, 5, 3),
        advantages=torch.tensor([-1.0, -0.25, 0.5, 1.0]),
    )


def test_af_microbatch_does_not_change_loss_or_gradient():
    first = TinyBundle()
    second = TinyBundle()
    second.model.load_state_dict(first.model.state_dict())
    condition = _rollout_condition()

    def evaluate(bundle, microbatch):
        generator = torch.Generator().manual_seed(17)
        loss, diagnostics = advantageflow_loss(
            bundle,
            [condition],
            conditioning=_conditioning(),
            generator=generator,
            lambda_reference=0.1,
            curvature_margin=1e-4,
            time_minimum=0.0,
            time_maximum=1.0,
            microbatch_size=microbatch,
        )
        loss.backward()
        return loss.detach(), bundle.model.transformer.scale.grad.detach(), diagnostics

    loss_one, grad_one, diagnostics = evaluate(first, 1)
    loss_two, grad_two, _ = evaluate(second, 2)
    torch.testing.assert_close(loss_one, loss_two)
    torch.testing.assert_close(grad_one, grad_two)
    assert diagnostics["logical_endpoints"] == 4
    assert diagnostics["curvature_min"] == pytest.approx(1.1)


def test_paired_clean_prediction_anchor_gradient_is_finite_nonzero():
    bundle = TinyBundle()
    condition = PairedTrainingCondition(
        utterance="p001_002",
        transcript="world",
        condition_mel=torch.randn(1, 6, 3),
        clean_mel=torch.randn(1, 6, 3),
    )
    loss, diagnostics = paired_anchor_loss(
        bundle,
        [condition],
        conditioning=_conditioning(),
        generator=torch.Generator().manual_seed(23),
        time_minimum=0.0,
        time_maximum=1.0,
    )
    loss.backward()
    gradient = bundle.model.transformer.scale.grad
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient)
    assert gradient.abs().item() > 0
    assert diagnostics["paired_conditions"] == 1


def test_time_sampler_never_returns_exact_endpoint():
    values = sample_times(
        10000,
        generator=torch.Generator().manual_seed(31),
        device=torch.device("cpu"),
        minimum=0.0,
        maximum=1.0,
    )
    assert values.min().item() >= 0.0
    assert values.max().item() < 1.0


