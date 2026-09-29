from types import SimpleNamespace

import torch

from rl.common.conditioning import ConditioningProtocol
from rl.grpo.policy import policy_velocity


class FakeTransformer:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        x,
        cond,
        text,
        time,
        mask,
        drop_audio_cond,
        drop_text,
    ):
        del text, time, mask
        self.calls.append((drop_audio_cond, drop_text))
        return x - cond if drop_audio_cond else x + 2.0 * cond


class FakeBundle:
    def __init__(self):
        self.model = SimpleNamespace(transformer=FakeTransformer())

    def prepare_policy_text(self, transcript, conditioning):
        assert transcript == ""
        return conditioning.policy_text(transcript)

    def _tokenize(self, text, batch):
        del text
        return torch.zeros(batch, 1, dtype=torch.long)


def test_cfg_uses_released_flowse_parameterization_and_audio_only_branches():
    bundle = FakeBundle()
    state = torch.ones(2, 3, 1)
    condition = torch.full((1, 3, 1), 2.0)
    protocol = ConditioningProtocol(mode="wotext", use_text=False, drop_text=True)
    output = policy_velocity(
        bundle,
        state=state,
        condition_mel=condition,
        time=0.5,
        frame_mask=torch.ones(2, 3, dtype=torch.bool),
        conditioning=protocol,
        cfg_strength=0.5,
    )
    conditional = state + 2.0 * condition
    unconditional = state - condition
    assert torch.equal(output, conditional + 0.5 * (conditional - unconditional))
    assert bundle.model.transformer.calls == [(False, True), (True, True)]


def test_cfg_zero_only_executes_conditional_forward():
    bundle = FakeBundle()
    protocol = ConditioningProtocol(mode="wotext", use_text=False, drop_text=True)
    policy_velocity(
        bundle,
        state=torch.zeros(1, 2, 1),
        condition_mel=torch.ones(1, 2, 1),
        time=0.25,
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
        conditioning=protocol,
        cfg_strength=0.0,
    )
    assert bundle.model.transformer.calls == [(False, True)]
