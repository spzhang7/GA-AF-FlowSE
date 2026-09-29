import importlib.util

import pytest


torch_missing = importlib.util.find_spec("torch") is None
pytestmark = pytest.mark.skipif(torch_missing, reason="PyTorch is only installed on server")


def test_transcript_change_cannot_change_wotext_terminal_mel():
    import torch

    from rl.common.conditioning import (
        ConditioningProtocol,
    )
    from rl.common.flowse_interface import FlowSEBundle

    class FakeTransformer:
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
            del cond, time, mask, drop_audio_cond
            text_value = 0.0 if drop_text else text.float().sum(dim=1)[:, None, None]
            return 0.1 * x + text_value

    class FakeModel:
        transformer = FakeTransformer()

    bundle = object.__new__(FlowSEBundle)
    bundle.model = FakeModel()
    bundle.tokenizer_name = "char"
    bundle._tokenize = lambda text, batch: torch.full(
        (batch, 1), sum(ord(character) for character in text), dtype=torch.long
    )
    condition = torch.zeros(2, 3, 1)
    latents = torch.ones_like(condition)
    wotext = ConditioningProtocol("wotext", False, True)
    first = bundle.sample_from_latents(
        condition,
        "FIRST REAL TRANSCRIPT",
        latents,
        nfe=4,
        cfg_strength=0.0,
        conditioning=wotext,
    )
    second = bundle.sample_from_latents(
        condition,
        "COMPLETELY DIFFERENT TRANSCRIPT",
        latents,
        nfe=4,
        cfg_strength=0.0,
        conditioning=wotext,
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_text_conditioned_terminal_uses_transcript():
    import torch

    from rl.common.conditioning import (
        ConditioningProtocol,
    )
    from rl.common.flowse_interface import FlowSEBundle

    class FakeTransformer:
        def __call__(self, *, x, text, drop_text, **kwargs):
            del kwargs
            assert not drop_text
            return 0.01 * text.float().sum(dim=1)[:, None, None].expand_as(x)

    class FakeModel:
        transformer = FakeTransformer()

    bundle = object.__new__(FlowSEBundle)
    bundle.model = FakeModel()
    bundle.tokenizer_name = "char"
    bundle._tokenize = lambda text, batch: torch.full(
        (batch, 1), sum(ord(character) for character in text), dtype=torch.long
    )
    condition = torch.zeros(1, 2, 1)
    latents = torch.zeros_like(condition)
    protocol = ConditioningProtocol("text", True, False)
    first = bundle.sample_from_latents(
        condition,
        "A",
        latents,
        nfe=1,
        cfg_strength=0.0,
        conditioning=protocol,
    )
    second = bundle.sample_from_latents(
        condition,
        "B",
        latents,
        nfe=1,
        cfg_strength=0.0,
        conditioning=protocol,
    )
    assert not torch.equal(first, second)

