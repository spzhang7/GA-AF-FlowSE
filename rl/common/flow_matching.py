"""Method-neutral torch flow math for controlled speech experiments.

This module deliberately contains no FlowSE model construction.  It isolates the
time convention, masking and completed-square objective so they can be tested
before an optimizer or LoRA implementation is introduced.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised on the local editing machine
    torch = None


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("flow_matching requires PyTorch; run this code on the server")
    return torch


def _broadcast_per_example(value, reference):
    """Convert a scalar or ``[batch]`` tensor to reference-broadcastable shape."""
    th = _require_torch()
    value = th.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value
    if value.shape[0] != reference.shape[0]:
        raise ValueError(
            f"batch mismatch: value has {value.shape[0]}, reference has "
            f"{reference.shape[0]}"
        )
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def flow_interpolate(noise, clean, time):
    """FlowSE noise-to-clean interpolation ``x_s=(1-s)eps+s*y``."""
    time = _broadcast_per_example(time, clean)
    return (1.0 - time) * noise + time * clean


def clean_prediction(state, velocity, time):
    """Convert FlowSE velocity to its clean endpoint prediction.

    FlowSE integrates from Gaussian noise at ``s=0`` to clean speech at ``s=1``.
    Its correct clean prediction is therefore ``x_s + (1-s) u_theta``.
    """
    time = _broadcast_per_example(time, state)
    return state + (1.0 - time) * velocity


def euler_terminal(vector_field, initial_state, nfe: int):
    """Integrate ``s=0 -> 1`` with exactly ``nfe`` explicit Euler calls."""
    th = _require_torch()
    if nfe < 1:
        raise ValueError("nfe must be positive")
    state = initial_state
    step_size = 1.0 / nfe
    for step in range(nfe):
        time = th.as_tensor(
            step * step_size, device=state.device, dtype=state.dtype
        )
        state = state + step_size * vector_field(time, state)
    return state


def euler_trajectory(vector_field, initial_state, nfe: int):
    """Return all ``nfe + 1`` explicit-Euler states from ``s=0`` to ``1``."""
    th = _require_torch()
    if nfe < 1:
        raise ValueError("nfe must be positive")
    state = initial_state
    states = [state]
    step_size = 1.0 / nfe
    for step in range(nfe):
        time = th.as_tensor(
            step * step_size, device=state.device, dtype=state.dtype
        )
        state = state + step_size * vector_field(time, state)
        states.append(state)
    return th.stack(states, dim=0)


def frame_mask(lengths, max_frames: int | None = None):
    """Return a boolean ``[batch, frames]`` mask from frame lengths."""
    th = _require_torch()
    lengths = th.as_tensor(lengths)
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if (lengths < 0).any():
        raise ValueError("lengths cannot be negative")
    if max_frames is None:
        max_frames = int(lengths.max().item()) if lengths.numel() else 0
    if (lengths > max_frames).any():
        raise ValueError("a length exceeds max_frames")
    return th.arange(max_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def masked_per_example_mean(values, mask=None):
    """Mean over valid frames * channels, then return one value per example."""
    th = _require_torch()
    if values.ndim < 2:
        raise ValueError("values must include batch and at least one feature dimension")
    if mask is None:
        return values.mean(dim=tuple(range(1, values.ndim)))
    mask = th.as_tensor(mask, device=values.device, dtype=th.bool)
    if mask.ndim != 2 or mask.shape != values.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(values.shape[:2])}, got {tuple(mask.shape)}"
        )
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    counts = expanded.sum(dim=tuple(range(1, values.ndim)))
    if (counts == 0).any():
        raise ValueError("every example must contain at least one valid element")
    masked = th.where(expanded, values, th.zeros((), dtype=values.dtype, device=values.device))
    totals = masked.sum(dim=tuple(range(1, values.ndim)))
    return totals / counts.to(values.dtype)


def build_completed_square_target(
    rollout_target,
    old_prediction,
    reference_prediction,
    advantage,
    lambda_reference: float,
    *,
    gamma=None,
    curvature_margin: float = 1e-4,
):
    """Build the detached target for the completed-square AF objective.

    The default ``gamma=1-A`` makes curvature equal to
    ``1 + lambda_reference`` for every endpoint, including negative-advantage
    endpoints.  All target-side arithmetic is intentionally float32 and detached.
    """
    th = _require_torch()
    if lambda_reference < 0:
        raise ValueError("lambda_reference must be non-negative")
    rollout_target = rollout_target.detach().float()
    old_prediction = old_prediction.detach().float()
    reference_prediction = reference_prediction.detach().float()
    if not (
        rollout_target.shape == old_prediction.shape == reference_prediction.shape
    ):
        raise ValueError("rollout, old and reference predictions must have equal shapes")
    advantage = _broadcast_per_example(
        th.as_tensor(advantage, device=rollout_target.device, dtype=th.float32),
        rollout_target,
    ).detach()
    if gamma is None:
        gamma = 1.0 - advantage
    else:
        gamma = _broadcast_per_example(
            th.as_tensor(gamma, device=rollout_target.device, dtype=th.float32),
            rollout_target,
        ).detach()
    curvature = (advantage + gamma + float(lambda_reference)).detach()
    if not th.isfinite(curvature).all():
        raise ValueError("curvature contains NaN or Inf")
    if float(curvature.min().item()) < curvature_margin:
        raise ValueError(
            f"curvature below margin: {curvature.min().item():.6g} < "
            f"{curvature_margin:.6g}"
        )
    if gamma is not None:
        expected = th.full_like(curvature, 1.0 + float(lambda_reference))
        default_gamma = 1.0 - advantage
        if th.allclose(gamma, default_gamma) and not th.allclose(
            curvature, expected, rtol=1e-5, atol=1e-6
        ):
            raise AssertionError("gamma=1-A must yield curvature=1+lambda_reference")
    target = (
        advantage * rollout_target
        + gamma * old_prediction
        + float(lambda_reference) * reference_prediction
    ) / curvature
    if not th.isfinite(target).all():
        raise ValueError("completed-square target contains NaN or Inf")
    return target.detach(), curvature.detach()


def completed_square_loss(current_prediction, target, curvature, mask=None):
    """Mask-aware AF loss with per-utterance then batch reduction."""
    current = current_prediction.float()
    target = target.detach().to(device=current.device, dtype=th_float32())
    curvature = _broadcast_per_example(
        curvature.detach().to(device=current.device, dtype=th_float32()), current
    )
    losses = curvature * (current - target).square()
    return masked_per_example_mean(losses, mask).mean()


def paired_clean_prediction_loss(state, velocity, clean, time, mask=None):
    """Mask-aware fidelity anchor in clean-prediction space."""
    th = _require_torch()
    time_tensor = th.as_tensor(time, device=state.device)
    if not th.isfinite(time_tensor).all() or (time_tensor < 0).any() or (
        time_tensor >= 1
    ).any():
        raise ValueError("paired-anchor time must satisfy 0 <= s < 1")
    prediction = clean_prediction(state.float(), velocity.float(), time)
    losses = (prediction - clean.float()).square()
    return masked_per_example_mean(losses, mask).mean()


def th_float32():
    """Keep import-time behavior friendly when PyTorch is absent locally."""
    return _require_torch().float32
