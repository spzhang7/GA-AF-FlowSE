"""Paper-aligned mathematical/runtime core for audio-only AdvantageFlow.

This module is intentionally independent of the historical one-step audits.
It implements the complete-batch advantage normalization, the three-policy
completed-square loss, LoRA EMA updates, and exact optimizer/RNG checkpoints
needed by an iterative speech AdvantageFlow trainer.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from rl.common.conditioning import ConditioningProtocol
from rl.common.flow_matching import (
    build_completed_square_target,
    clean_prediction,
    completed_square_loss,
    flow_interpolate,
)
from rl.common.flow_objective import RolloutTrainingCondition, _velocity
from rl.common.lora import AdapterState, load_lora, lora_enabled, snapshot_lora


@dataclass(frozen=True)
class GlobalAdvantageResult:
    """Condition-centered advantages from one complete ``L x K`` batch."""

    advantages: np.ndarray
    centered_rewards: np.ndarray
    group_means: np.ndarray
    pooled_scale: float
    clipped_fraction: float
    mapping: str
    temperature: float | None


def validate_reward_constraints(config: Mapping | None) -> None:
    """Validate the optional fixed-Lagrangian reward constraints.

    The constraint layer is deliberately separate from the AdvantageFlow loss:
    it only changes the scalar terminal reward before group-relative advantages
    are computed.  ``fixed_lagrangian`` means that the multipliers are frozen
    for a pilot; this avoids introducing a second mutable optimizer state while
    still allowing fidelity floors to be tested safely.
    """

    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ValueError("reward_constraints must be a mapping")
    required = {"enabled", "mode", "constraints"}
    if set(config) != required:
        raise ValueError(
            "reward_constraints must contain exactly enabled, mode, constraints"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("reward_constraints.enabled must be boolean")
    if str(config["mode"]) != "fixed_lagrangian":
        raise ValueError("reward_constraints.mode must be fixed_lagrangian")
    constraints = config["constraints"]
    if not isinstance(constraints, Mapping) or not constraints:
        raise ValueError("reward_constraints.constraints must be a non-empty mapping")
    allowed_metrics = {
        "dnsmos_ovrl",
        "eres2net_speaker_similarity",
        "speechbertscore",
    }
    required_spec = {"metric", "direction", "threshold", "scale", "multiplier", "max_penalty"}
    for name, raw in constraints.items():
        if not isinstance(name, str) or not name:
            raise ValueError("reward constraint names must be non-empty strings")
        if not isinstance(raw, Mapping) or set(raw) != required_spec:
            raise ValueError(
                f"reward constraint {name!r} must contain exactly "
                f"{sorted(required_spec)}"
            )
        metric = str(raw["metric"])
        if metric not in allowed_metrics:
            raise ValueError(f"unsupported reward constraint metric: {metric!r}")
        if str(raw["direction"]) not in {"min", "max"}:
            raise ValueError(f"reward constraint {name!r} direction must be min or max")
        for field in ("threshold", "scale", "multiplier", "max_penalty"):
            value = float(raw[field])
            if not math.isfinite(value):
                raise ValueError(f"reward constraint {name!r} {field} must be finite")
        if float(raw["scale"]) <= 0.0:
            raise ValueError(f"reward constraint {name!r} scale must be positive")
        if float(raw["multiplier"]) < 0.0:
            raise ValueError(f"reward constraint {name!r} multiplier must be non-negative")
        if float(raw["max_penalty"]) <= 0.0:
            raise ValueError(f"reward constraint {name!r} max_penalty must be positive")


def apply_reward_constraints(rows: list[dict], config: Mapping | None) -> dict:
    """Apply fixed soft fidelity floors/ceilings to scored rollout rows.

    The returned penalty is in the same scalar space as the base composite
    reward.  Violations are normalized by the configured raw-metric scale and
    capped per constraint, so one pathological evaluator value cannot dominate
    an entire AF update.  The base reward and every violation are retained in
    the rows for transparent ablations.
    """

    if config is None or not bool(config.get("enabled", False)):
        return {
            "enabled": False,
            "mode": None,
            "constraints": {},
            "base_reward_mean": float(np.mean([float(row["reward"]) for row in rows]))
            if rows
            else 0.0,
            "adjusted_reward_mean": float(np.mean([float(row["reward"]) for row in rows]))
            if rows
            else 0.0,
            "total_penalty_mean": 0.0,
            "penalized_fraction": 0.0,
        }
    validate_reward_constraints(config)
    constraints = config["constraints"]
    per_constraint = {}
    total_penalties = np.zeros(len(rows), dtype=np.float64)
    for name, raw in constraints.items():
        metric = str(raw["metric"])
        direction = str(raw["direction"])
        threshold = float(raw["threshold"])
        scale = float(raw["scale"])
        multiplier = float(raw["multiplier"])
        max_penalty = float(raw["max_penalty"])
        violations = []
        penalties = []
        for index, row in enumerate(rows):
            if metric not in row or not math.isfinite(float(row[metric])):
                raise ValueError(
                    f"rollout row lacks a finite metric for constraint {name!r}: {metric}"
                )
            value = float(row[metric])
            violation = max(0.0, threshold - value) if direction == "min" else max(0.0, value - threshold)
            penalty = min(max_penalty, multiplier * violation / scale)
            violations.append(violation)
            penalties.append(penalty)
            total_penalties[index] += penalty
            row.setdefault("constraint_violations", {})[name] = float(violation)
            row.setdefault("constraint_penalties", {})[name] = float(penalty)
        per_constraint[name] = {
            "metric": metric,
            "direction": direction,
            "threshold": threshold,
            "scale": scale,
            "multiplier": multiplier,
            "max_penalty": max_penalty,
            "mean_metric": float(np.mean([float(row[metric]) for row in rows])),
            "violation_mean": float(np.mean(violations)),
            "violation_rate": float(np.mean(np.asarray(violations) > 0.0)),
            "penalty_mean": float(np.mean(penalties)),
            "penalty_max": float(max(penalties, default=0.0)),
        }
    base_rewards = []
    adjusted_rewards = []
    for index, row in enumerate(rows):
        base = float(row.get("unconstrained_reward", row["reward"]))
        adjusted = base - float(total_penalties[index])
        if not math.isfinite(adjusted):
            raise ValueError("constraint-adjusted reward is not finite")
        row["unconstrained_reward"] = base
        row["constraint_penalty_total"] = float(total_penalties[index])
        row["reward"] = adjusted
        base_rewards.append(base)
        adjusted_rewards.append(adjusted)
    return {
        "enabled": True,
        "mode": str(config["mode"]),
        "constraints": per_constraint,
        "base_reward_mean": float(np.mean(base_rewards)) if base_rewards else 0.0,
        "adjusted_reward_mean": float(np.mean(adjusted_rewards)) if adjusted_rewards else 0.0,
        "total_penalty_mean": float(np.mean(total_penalties)) if rows else 0.0,
        "penalized_fraction": float(np.mean(total_penalties > 0.0)) if rows else 0.0,
    }


def compute_paper_global_advantages(
    rewards: np.ndarray | Sequence[Sequence[float]],
    *,
    clip: float = 1.0,
    minimum_scale: float = 1.0e-8,
    mapping: str = "linear_clipped",
    temperature: float = 1.0,
) -> GlobalAdvantageResult:
    """Compute condition-centered linear or FlowAWR-style advantages.

    For rewards ``[L, K]``, rewards are centered independently for each of the
    ``L`` conditions, while ``Z`` is the RMS of *all* ``L*K`` centered rewards.
    Splitting a logical batch and computing multiple scales is deliberately not
    supported here.  ``linear_clipped`` preserves the original AdvantageFlow
    mapping exactly.  ``exponential_awr`` implements

    ``A_lk = K * softmax_k(centered_reward_lk / (temperature * Z)) - 1``.

    The latter is Eq. 11 of FlowAWR with an explicit temperature multiplier and
    the same complete-batch, condition-centered scale used by this experiment.
    """

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("rewards must have shape [L, K] with L>=1 and K>=2")
    if not np.isfinite(values).all():
        raise ValueError("rewards contain NaN or Inf")
    mapping = str(mapping)
    if mapping not in {"linear_clipped", "exponential_awr"}:
        raise ValueError(
            "advantage mapping must be linear_clipped or exponential_awr"
        )
    if clip <= 0 or minimum_scale < 0:
        raise ValueError("clip must be positive and minimum_scale non-negative")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("advantage temperature must be finite and positive")
    means = values.mean(axis=1, keepdims=True)
    centered = values - means
    pooled_scale = float(np.sqrt(np.mean(np.square(centered))))
    if not math.isfinite(pooled_scale) or pooled_scale <= minimum_scale:
        raise ValueError(
            f"complete AdvantageFlow batch has no reward signal: Z={pooled_scale:.9g}"
        )
    if mapping == "linear_clipped":
        unclipped = centered / pooled_scale
        advantages = np.clip(unclipped, -float(clip), float(clip))
        clipped_fraction = float(np.mean(np.abs(unclipped) > float(clip)))
        active_temperature = None
    else:
        logits = centered / (float(temperature) * pooled_scale)
        # Row-wise max subtraction makes the exponential stable without
        # changing K*softmax(logits)-1.  At least one shifted logit is zero in
        # every row, so the normalizer cannot underflow to zero.
        shifted = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        advantages = weights / weights.mean(axis=1, keepdims=True) - 1.0
        clipped_fraction = 0.0
        active_temperature = float(temperature)
    if not np.isfinite(advantages).all():
        raise ValueError("computed advantages contain NaN or Inf")
    return GlobalAdvantageResult(
        advantages=advantages,
        centered_rewards=centered,
        group_means=means[:, 0],
        pooled_scale=pooled_scale,
        clipped_fraction=clipped_fraction,
        mapping=mapping,
        temperature=active_temperature,
    )


def gamma_from_advantage(advantage: torch.Tensor, mode: str) -> torch.Tensor:
    """Return either paper-tested ``gamma=1.1`` or ``gamma(A)=1-A``."""

    if mode == "constant_1p1":
        return torch.full_like(advantage, 1.1)
    if mode == "one_minus_advantage":
        return 1.0 - advantage
    raise ValueError("gamma mode must be constant_1p1 or one_minus_advantage")


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def length_adaptive_microbatch_size(
    mel_frames: int,
    *,
    maximum_size: int,
    config: Mapping | None,
) -> int:
    """Bound the approximate ``batch * frames**2`` attention workload.

    The logical K group is unchanged.  This only selects how many endpoints
    are executed together before gradient accumulation or endpoint assembly.
    A missing config preserves the historical fixed-microbatch behavior.
    """

    if mel_frames < 1 or maximum_size < 1:
        raise ValueError("mel_frames and maximum_size must be positive")
    if config is None:
        return int(maximum_size)
    if str(config.get("mode")) != "quadratic_mel_frame_budget":
        raise ValueError("unsupported length-adaptive microbatch mode")
    reference_frames = int(config.get("reference_mel_frames", 0))
    reference_batch = int(config.get("reference_batch_size", 0))
    minimum_size = int(config.get("minimum_size", 0))
    if reference_frames < 1 or reference_batch < 1:
        raise ValueError("adaptive reference frames/batch must be positive")
    if minimum_size < 1 or minimum_size > maximum_size:
        raise ValueError("adaptive minimum_size must lie in [1, maximum_size]")
    quadratic_budget = reference_batch * reference_frames * reference_frames
    selected = quadratic_budget // (int(mel_frames) * int(mel_frames))
    return int(max(minimum_size, min(maximum_size, selected)))


def _fresh_noise_and_time(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    time_minimum: float,
    time_maximum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= time_minimum < time_maximum <= 1.0:
        raise ValueError("time interval must satisfy 0 <= min < max <= 1")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    time = time_minimum + (time_maximum - time_minimum) * torch.rand(
        1, generator=generator, device=device, dtype=torch.float32
    )
    return noise, time


def ema_adapter_state(
    old_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
    decay: float,
    *,
    device: str | torch.device = "cpu",
) -> AdapterState:
    """Compute ``theta_old <- rho*theta_old + (1-rho)*theta``."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must satisfy 0 <= decay < 1")
    if set(old_state) != set(current_state):
        raise ValueError("EMA states have different parameter keys")
    result: AdapterState = {}
    for name in sorted(old_state):
        old = old_state[name].detach().to(device=device, dtype=torch.float32)
        current = current_state[name].detach().to(device=device, dtype=torch.float32)
        if old.shape != current.shape:
            raise ValueError(f"EMA state shape mismatch for {name}")
        value = old.mul(float(decay)).add(current, alpha=1.0 - float(decay))
        result[name] = value.to(dtype=current_state[name].dtype)
    return result


def adapter_distance(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    if set(left) != set(right):
        raise ValueError("adapter states have different keys")
    squared = 0.0
    for name in left:
        delta = left[name].detach().double().cpu() - right[name].detach().double().cpu()
        squared += float(delta.square().sum().item())
    return float(math.sqrt(squared))


def paper_advantageflow_loss(
    bundle,
    conditions: Sequence[RolloutTrainingCondition],
    *,
    current_state: Mapping[str, torch.Tensor],
    rollout_state: Mapping[str, torch.Tensor],
    conditioning: ConditioningProtocol,
    draw_seed_base: int,
    optimizer_step: int,
    lambda_reference: float,
    gamma_mode: str,
    curvature_margin: float,
    time_minimum: float,
    time_maximum: float,
    microbatch_size: int,
    length_adaptive_microbatching: Mapping | None = None,
    accumulate_gradients: bool = True,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Backpropagate one complete-batch, three-policy AdvantageFlow loss.

    ``current_state`` receives gradients, ``rollout_state`` supplies ``f_old``,
    and the upstream model with LoRA disabled supplies frozen ``f_ref``.  Fresh
    interpolation noise/time is drawn for every generated endpoint.  The model
    is restored to ``current_state`` before returning.
    """

    if not conditions:
        raise ValueError("AdvantageFlow loss requires conditions")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    conditioning.validate()
    if conditioning.mode != "wotext":
        raise ValueError("speech AdvantageFlow requires audio-only conditioning")
    total_endpoints = sum(int(item.terminal_mels.shape[0]) for item in conditions)
    if total_endpoints < 2:
        raise ValueError("logical batch must contain at least two endpoints")
    total_loss = torch.zeros((), device=bundle.device, dtype=torch.float32)
    curvature_min = float("inf")
    curvature_max = float("-inf")
    target_squared = 0.0
    target_elements = 0
    effective_microbatch_sizes = []
    try:
        for item in conditions:
            if item.transcript.strip():
                raise ValueError(
                    "audio-only AdvantageFlow training conditions must not carry transcripts"
                )
            terminals = item.terminal_mels.detach().to(bundle.device)
            advantages = item.advantages.detach().float().to(bundle.device)
            if terminals.ndim != 3 or advantages.shape != (terminals.shape[0],):
                raise ValueError(f"invalid endpoint tensors for {item.utterance}")
            if not torch.isfinite(terminals).all() or not torch.isfinite(advantages).all():
                raise ValueError(f"non-finite endpoint tensors for {item.utterance}")
            condition_mel = item.condition_mel.detach().to(
                device=bundle.device, dtype=terminals.dtype
            )
            item_microbatch_size = length_adaptive_microbatch_size(
                int(terminals.shape[1]),
                maximum_size=microbatch_size,
                config=length_adaptive_microbatching,
            )
            effective_microbatch_sizes.append(item_microbatch_size)
            noises, times = [], []
            for endpoint_index in range(terminals.shape[0]):
                noise, time = _fresh_noise_and_time(
                    tuple(terminals[endpoint_index].shape),
                    device=bundle.device,
                    dtype=terminals.dtype,
                    seed=stable_seed(
                        draw_seed_base,
                        "loss",
                        optimizer_step,
                        item.utterance,
                        endpoint_index,
                    ),
                    time_minimum=time_minimum,
                    time_maximum=time_maximum,
                )
                noises.append(noise)
                times.append(time)
            states = flow_interpolate(
                torch.stack(noises), terminals, torch.cat(times)
            )
            times_tensor = torch.cat(times)

            # Evaluate each frozen policy once per condition.  This preserves
            # microbatch memory bounds without copying all LoRA tensors three
            # times for every single endpoint.
            old_predictions = []
            load_lora(bundle.model.transformer, rollout_state)
            for start in range(0, terminals.shape[0], item_microbatch_size):
                stop = min(terminals.shape[0], start + item_microbatch_size)
                state = states[start:stop]
                time = times_tensor[start:stop]
                mask = torch.ones(state.shape[:2], device=bundle.device, dtype=torch.bool)
                with torch.no_grad(), lora_enabled(bundle.model.transformer, True):
                    old_velocity = _velocity(
                        bundle,
                        state=state,
                        condition_mel=condition_mel,
                        transcript=item.transcript,
                        time=time,
                        conditioning=conditioning,
                        mask=mask,
                    )
                    old_predictions.append(
                        clean_prediction(state, old_velocity, time).detach()
                    )

            reference_predictions = []
            for start in range(0, terminals.shape[0], item_microbatch_size):
                stop = min(terminals.shape[0], start + item_microbatch_size)
                state = states[start:stop]
                time = times_tensor[start:stop]
                mask = torch.ones(state.shape[:2], device=bundle.device, dtype=torch.bool)
                with torch.no_grad(), lora_enabled(bundle.model.transformer, False):
                    reference_velocity = _velocity(
                        bundle,
                        state=state,
                        condition_mel=condition_mel,
                        transcript=item.transcript,
                        time=time,
                        conditioning=conditioning,
                        mask=mask,
                    )
                    reference_predictions.append(
                        clean_prediction(state, reference_velocity, time).detach()
                    )

            targets = []
            curvatures = []
            chunk_index = 0
            for start in range(0, terminals.shape[0], item_microbatch_size):
                stop = min(terminals.shape[0], start + item_microbatch_size)
                terminal = terminals[start:stop]
                advantage = advantages[start:stop]
                gamma = gamma_from_advantage(advantage, gamma_mode)
                target, curvature = build_completed_square_target(
                    terminal,
                    old_predictions[chunk_index],
                    reference_predictions[chunk_index],
                    advantage,
                    lambda_reference,
                    gamma=gamma,
                    curvature_margin=curvature_margin,
                )
                targets.append(target)
                curvatures.append(curvature)
                chunk_index += 1

            load_lora(bundle.model.transformer, current_state)
            chunk_index = 0
            for start in range(0, terminals.shape[0], item_microbatch_size):
                stop = min(terminals.shape[0], start + item_microbatch_size)
                state = states[start:stop]
                time = times_tensor[start:stop]
                mask = torch.ones(state.shape[:2], device=bundle.device, dtype=torch.bool)
                target = targets[chunk_index]
                curvature = curvatures[chunk_index]
                with lora_enabled(bundle.model.transformer, True):
                    current_velocity = _velocity(
                        bundle,
                        state=state,
                        condition_mel=condition_mel,
                        transcript=item.transcript,
                        time=time,
                        conditioning=conditioning,
                        mask=mask,
                    )
                    current_prediction = clean_prediction(state, current_velocity, time)
                    micro_loss = completed_square_loss(
                        current_prediction, target, curvature, mask
                    )
                weighted = micro_loss * ((stop - start) / total_endpoints)
                if accumulate_gradients:
                    weighted.backward()
                    total_loss = total_loss + weighted.detach()
                else:
                    total_loss = total_loss + weighted
                curvature_min = min(curvature_min, float(curvature.min().item()))
                curvature_max = max(curvature_max, float(curvature.max().item()))
                target_squared += float(target.double().square().sum().item())
                target_elements += int(target.numel())
                chunk_index += 1
    finally:
        load_lora(bundle.model.transformer, current_state)

    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "logical_endpoints": float(total_endpoints),
        "curvature_min": curvature_min,
        "curvature_max": curvature_max,
        "completed_target_rms": float(math.sqrt(target_squared / target_elements)),
        "gamma_mode": gamma_mode,
        "policy_roles": "current_trainable__ema_rollout__lora_disabled_reference",
        "configured_microbatch_size": float(microbatch_size),
        "effective_microbatch_size_min": float(min(effective_microbatch_sizes)),
        "effective_microbatch_size_max": float(max(effective_microbatch_sizes)),
        "length_adapted_conditions": float(
            sum(value < microbatch_size for value in effective_microbatch_sizes)
        ),
    }


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    transformer: torch.nn.Module,
    rollout_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    protocol_hash: str,
    extra: Mapping | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol_hash": str(protocol_hash),
        "completed_step": int(step),
        "current_lora": snapshot_lora(transformer, device="cpu"),
        "rollout_lora": {
            name: value.detach().cpu().clone() for name, value in rollout_state.items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler else None,
        "rng": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: str | Path,
    *,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    expected_protocol_hash: str | None = None,
) -> tuple[int, AdapterState, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported speech AdvantageFlow checkpoint schema")
    if (
        expected_protocol_hash is not None
        and payload.get("protocol_hash") != expected_protocol_hash
    ):
        raise ValueError("checkpoint protocol hash does not match active run")
    load_lora(transformer, payload["current_lora"])
    optimizer.load_state_dict(payload["optimizer"])
    saved_scheduler = payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("scheduler presence differs from checkpoint")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    restore_rng_state(payload["rng"])
    rollout = {
        name: value.detach().cpu().clone()
        for name, value in payload["rollout_lora"].items()
    }
    if set(rollout) != set(snapshot_lora(transformer)):
        raise ValueError("rollout LoRA state does not match current model")
    return int(payload["completed_step"]), rollout, dict(payload.get("extra") or {})
