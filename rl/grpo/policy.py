"""Single audio-only FlowSE policy interface shared by every GRPO phase."""

from __future__ import annotations

import math

import torch

from rl.common.conditioning import ConditioningProtocol


def policy_velocity(
    bundle,
    *,
    state: torch.Tensor,
    condition_mel: torch.Tensor,
    time: float | torch.Tensor,
    frame_mask: torch.Tensor | None,
    conditioning: ConditioningProtocol,
    cfg_strength: float,
) -> torch.Tensor:
    """Evaluate the online, old, or reference policy with identical semantics.

    State selection is deliberately external: callers use the existing LoRA
    context managers to select online/old/reference tensors and always call
    this function.  Training never consumes a transcript.
    """

    conditioning.validate()
    if conditioning.fingerprint() != {
        "mode": "wotext",
        "use_text": False,
        "drop_text": True,
    }:
        raise ValueError("FlowSE-GRPO is permanently audio-only")
    if state.ndim != 3:
        raise ValueError("state must have shape [batch, frames, mel_channels]")
    if condition_mel.ndim != 3 or condition_mel.shape[1:] != state.shape[1:]:
        raise ValueError("condition_mel must match the state frame/channel shape")
    if condition_mel.shape[0] not in {1, state.shape[0]}:
        raise ValueError("condition batch must be one or equal to the state batch")
    if not math.isfinite(float(cfg_strength)) or cfg_strength < 0.0:
        raise ValueError("cfg_strength must be finite and non-negative")

    batch = state.shape[0]
    condition = condition_mel.to(device=state.device, dtype=state.dtype)
    if condition.shape[0] == 1 and batch != 1:
        condition = condition.expand(batch, -1, -1).contiguous()
    text = bundle._tokenize(bundle.prepare_policy_text("", conditioning), batch)
    mask = None
    if frame_mask is not None:
        mask = torch.as_tensor(frame_mask, device=state.device, dtype=torch.bool)
        if mask.shape != state.shape[:2]:
            raise ValueError("frame_mask must have shape [batch, frames]")
    time_tensor = torch.as_tensor(time, device=state.device, dtype=state.dtype)
    if time_tensor.ndim == 0:
        time_tensor = time_tensor.expand(batch)
    elif time_tensor.shape != (batch,):
        raise ValueError("time must be scalar or have shape [batch]")

    conditional = bundle.model.transformer(
        x=state,
        cond=condition,
        text=text,
        time=time_tensor,
        mask=mask,
        drop_audio_cond=False,
        drop_text=True,
    )
    if cfg_strength < 1.0e-5:
        return conditional
    unconditional = bundle.model.transformer(
        x=state,
        cond=condition,
        text=text,
        time=time_tensor,
        mask=mask,
        drop_audio_cond=True,
        drop_text=True,
    )
    return conditional + float(cfg_strength) * (conditional - unconditional)
