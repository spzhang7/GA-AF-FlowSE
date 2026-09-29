"""Method-neutral waveform and velocity primitives for speech training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .conditioning import ConditioningProtocol
from .flow_matching import (
    build_completed_square_target,
    clean_prediction,
    completed_square_loss,
    flow_interpolate,
    paired_clean_prediction_loss,
)
from .lora import lora_enabled


@dataclass(frozen=True)
class RolloutTrainingCondition:
    utterance: str
    transcript: str
    condition_mel: torch.Tensor
    terminal_mels: torch.Tensor
    advantages: torch.Tensor


@dataclass(frozen=True)
class PairedTrainingCondition:
    utterance: str
    transcript: str
    condition_mel: torch.Tensor
    clean_mel: torch.Tensor


def sample_times(
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    if count < 1:
        raise ValueError("time sample count must be positive")
    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError("time interval must satisfy 0 <= min < max <= 1")
    return minimum + (maximum - minimum) * torch.rand(
        count, generator=generator, device=device, dtype=torch.float32
    )


def _velocity(
    bundle,
    *,
    state: torch.Tensor,
    condition_mel: torch.Tensor,
    transcript: str,
    time: torch.Tensor,
    conditioning: ConditioningProtocol,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch = state.shape[0]
    if condition_mel.ndim != 3 or condition_mel.shape[0] != 1:
        raise ValueError("condition mel must have shape [1, frames, channels]")
    if condition_mel.shape[1:] != state.shape[1:]:
        raise ValueError("condition and state mel shapes differ")
    condition = condition_mel.expand(batch, -1, -1).contiguous()
    text = bundle._tokenize(
        bundle.prepare_policy_text(transcript, conditioning), batch
    )
    return bundle.model.transformer(
        x=state,
        cond=condition,
        text=text,
        time=time,
        mask=mask,
        drop_audio_cond=False,
        drop_text=conditioning.drop_text,
    )


def advantageflow_loss(
    bundle,
    conditions: Sequence[RolloutTrainingCondition],
    *,
    conditioning: ConditioningProtocol,
    generator: torch.Generator,
    lambda_reference: float,
    curvature_margin: float,
    time_minimum: float,
    time_maximum: float,
    microbatch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one logical-batch AF loss without changing its pooled advantages."""

    if not conditions:
        raise ValueError("AF loss requires rollout conditions")
    if microbatch_size < 1:
        raise ValueError("microbatch size must be positive")
    total_endpoints = sum(int(item.terminal_mels.shape[0]) for item in conditions)
    loss = torch.zeros((), device=bundle.device, dtype=torch.float32)
    curvature_min = float("inf")
    target_norm_sum = 0.0
    target_elements = 0
    for item in conditions:
        terminals = item.terminal_mels.detach().to(bundle.device)
        advantages = item.advantages.detach().float().to(bundle.device)
        if terminals.ndim != 3 or advantages.shape != (terminals.shape[0],):
            raise ValueError("terminal/advantage shapes are inconsistent")
        if not torch.isfinite(advantages).all():
            raise ValueError(f"non-finite advantage for {item.utterance}")
        condition = item.condition_mel.to(
            device=bundle.device, dtype=terminals.dtype
        )
        group = terminals.shape[0]
        noise = torch.randn(
            terminals.shape,
            generator=generator,
            device=bundle.device,
            dtype=terminals.dtype,
        )
        times = sample_times(
            group,
            generator=generator,
            device=bundle.device,
            minimum=time_minimum,
            maximum=time_maximum,
        )
        states = flow_interpolate(noise, terminals, times)
        for start in range(0, group, microbatch_size):
            stop = min(group, start + microbatch_size)
            state = states[start:stop]
            terminal = terminals[start:stop]
            time = times[start:stop]
            advantage = advantages[start:stop]
            mask = torch.ones(
                state.shape[:2], device=bundle.device, dtype=torch.bool
            )
            with torch.no_grad(), lora_enabled(bundle.model.transformer, False):
                base_velocity = _velocity(
                    bundle,
                    state=state,
                    condition_mel=condition,
                    transcript=item.transcript,
                    time=time,
                    conditioning=conditioning,
                    mask=mask,
                )
                base_prediction = clean_prediction(state, base_velocity, time)
            target, curvature = build_completed_square_target(
                terminal,
                base_prediction,
                base_prediction,
                advantage,
                lambda_reference,
                curvature_margin=curvature_margin,
            )
            with lora_enabled(bundle.model.transformer, True):
                current_velocity = _velocity(
                    bundle,
                    state=state,
                    condition_mel=condition,
                    transcript=item.transcript,
                    time=time,
                    conditioning=conditioning,
                    mask=mask,
                )
                current_prediction = clean_prediction(state, current_velocity, time)
            micro_loss = completed_square_loss(
                current_prediction, target, curvature, mask
            )
            weight = (stop - start) / total_endpoints
            loss = loss + micro_loss * weight
            curvature_min = min(curvature_min, float(curvature.min().item()))
            target_norm_sum += float(target.detach().double().square().sum().item())
            target_elements += int(target.numel())
    return loss, {
        "af_loss": float(loss.detach().item()),
        "curvature_min": float(curvature_min),
        "completed_target_rms": float((target_norm_sum / target_elements) ** 0.5),
        "logical_endpoints": int(total_endpoints),
    }


def paired_anchor_loss(
    bundle,
    conditions: Sequence[PairedTrainingCondition],
    *,
    conditioning: ConditioningProtocol,
    generator: torch.Generator,
    time_minimum: float,
    time_maximum: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not conditions:
        raise ValueError("paired anchor requires conditions")
    total = torch.zeros((), device=bundle.device, dtype=torch.float32)
    for item in conditions:
        frames = min(item.condition_mel.shape[1], item.clean_mel.shape[1])
        if frames < 1:
            raise ValueError(f"empty paired mel for {item.utterance}")
        condition = item.condition_mel[:, :frames].to(bundle.device)
        clean = item.clean_mel[:, :frames].to(
            device=bundle.device, dtype=condition.dtype
        )
        noise = torch.randn(
            clean.shape,
            generator=generator,
            device=bundle.device,
            dtype=clean.dtype,
        )
        time = sample_times(
            1,
            generator=generator,
            device=bundle.device,
            minimum=time_minimum,
            maximum=time_maximum,
        )
        state = flow_interpolate(noise, clean, time)
        mask = torch.ones(state.shape[:2], device=bundle.device, dtype=torch.bool)
        with lora_enabled(bundle.model.transformer, True):
            velocity = _velocity(
                bundle,
                state=state,
                condition_mel=condition,
                transcript=item.transcript,
                time=time,
                conditioning=conditioning,
                mask=mask,
            )
            item_loss = paired_clean_prediction_loss(
                state, velocity, clean, time, mask
            )
        total = total + item_loss / len(conditions)
    return total, {
        "paired_anchor_loss": float(total.detach().item()),
        "paired_conditions": int(len(conditions)),
    }


def waveform_to_mel(bundle, path: str) -> torch.Tensor:
    waveform = bundle.load_condition(path)
    with torch.no_grad():
        mel = bundle.model.mel_spec(waveform).permute(0, 2, 1)
    return mel.to(dtype=next(bundle.model.parameters()).dtype)
