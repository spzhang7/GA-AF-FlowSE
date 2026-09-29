import pytest

from rl.common.conditioning import (
    ConditioningProtocol,
)


def test_text_conditioned_policy_receives_transcript():
    protocol = ConditioningProtocol.from_config(
        {"mode": "text", "use_text": True, "drop_text": False}
    )
    assert protocol.policy_text("REAL TRANSCRIPT") == "REAL TRANSCRIPT"


def test_wotext_policy_never_receives_real_transcript():
    protocol = ConditioningProtocol.from_config(
        {"mode": "wotext", "use_text": False, "drop_text": True}
    )
    assert protocol.policy_text("FIRST SECRET TRANSCRIPT") == " "
    assert protocol.policy_text("DIFFERENT TRANSCRIPT") == " "


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "text", "use_text": False, "drop_text": False},
        {"mode": "text", "use_text": True, "drop_text": True},
        {"mode": "wotext", "use_text": True, "drop_text": True},
        {"mode": "wotext", "use_text": False, "drop_text": False},
    ],
)
def test_inconsistent_conditioning_is_rejected(config):
    with pytest.raises(ValueError):
        ConditioningProtocol.from_config(config)


