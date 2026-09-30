"""Gradient-space fusion primitives for speech AdvantageFlow.

The module contains the original OVRL-gated GA-AF path, an OVRL-primary
asymmetric projection path, and the OVRL-preferred MARBLE simplex solver.
Keeping them together makes their shared component-advantage construction
explicit while preserving checkpoint compatibility for existing runs.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch

from rl.common.flow_objective import RolloutTrainingCondition
from rl.af.advantage_flow import compute_paper_global_advantages


PRIMARY = "dnsmos"
AUXILIARIES = ("speaker", "speechbertscore")
COMPONENTS = (PRIMARY, *AUXILIARIES)


def _flatten_gradient(value: Mapping[str, torch.Tensor]) -> np.ndarray:
    """Return a deterministic CPU vector for a LoRA gradient mapping."""

    if any(not isinstance(name, str) for name in value):
        raise ValueError("gradient mapping keys must be strings")
    chunks = []
    for name in sorted(value):
        tensor = value[name].detach().float().cpu()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite gradient values for {name}")
        chunks.append(tensor.reshape(-1).numpy().astype(np.float64, copy=False))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def _simplex_qp_solution(
    gram: np.ndarray,
    linear: np.ndarray,
    *,
    active_indices: Sequence[int],
) -> np.ndarray | None:
    """Solve a tiny equality-constrained simplex QP on one active set.

    The objective is ``0.5 * a^T G a - b^T a`` with ``sum(a)=1``.  Enumerating
    the (three) active sets is exact for our speech reward count and avoids a
    dependency on a numerical QP package in the training process.
    """

    active = np.asarray(list(active_indices), dtype=np.int64)
    if active.size == 0:
        return None
    sub = np.asarray(gram[np.ix_(active, active)], dtype=np.float64)
    rhs = np.asarray(linear[active], dtype=np.float64)
    # KKT stationarity: G a - b + nu*1 = 0, 1^T a = 1.
    try:
        inv_rhs = np.linalg.solve(sub, rhs)
        inv_one = np.linalg.solve(sub, np.ones(active.size, dtype=np.float64))
    except np.linalg.LinAlgError:
        inv_rhs = np.linalg.pinv(sub, rcond=1.0e-12) @ rhs
        inv_one = np.linalg.pinv(sub, rcond=1.0e-12) @ np.ones(
            active.size, dtype=np.float64
        )
    denominator = float(np.dot(np.ones(active.size), inv_one))
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None
    nu = (float(np.dot(np.ones(active.size), inv_rhs)) - 1.0) / denominator
    solution = inv_rhs - nu * inv_one
    if not np.isfinite(solution).all() or np.min(solution) < -1.0e-8:
        return None
    solution = np.maximum(solution, 0.0)
    total = float(solution.sum())
    if total <= 1.0e-12:
        return None
    output = np.zeros(gram.shape[0], dtype=np.float64)
    output[active] = solution / total
    return output


def marble_simplex_weights(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    primary_preference: float = 0.2,
    norm_epsilon: float = 1.0e-12,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict]:
    """Compute OVRL-preferred MARBLE coefficients from reward-induced gradients.

    Each non-zero component gradient is normalized before solving the simplex
    minimum-norm problem.  ``primary_preference`` adds ``-lambda * alpha_ovrl``
    to that objective, making the method an explicitly asymmetric, OVRL-
    preferred MARBLE variant while retaining all non-conflicting rewards.
    """

    if set(gradients) != set(COMPONENTS):
        raise ValueError("MARBLE calibration requires exactly three component gradients")
    preference = float(primary_preference)
    epsilon = float(norm_epsilon)
    if not math.isfinite(preference) or preference < 0.0 or preference > 1.0:
        raise ValueError("MARBLE primary preference must lie in [0,1]")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("MARBLE norm_epsilon must be finite and positive")

    norms = {name: gradient_norm(gradients[name]) for name in COMPONENTS}
    vectors = []
    active_names = []
    for name in COMPONENTS:
        if norms[name] > epsilon:
            vector = _flatten_gradient(gradients[name])
            vectors.append(vector / norms[name])
            active_names.append(name)
        else:
            vectors.append(None)
    if PRIMARY not in active_names:
        raise ValueError("MARBLE primary OVRL gradient has zero norm")
    gram = np.zeros((len(COMPONENTS), len(COMPONENTS)), dtype=np.float64)
    for i, left in enumerate(vectors):
        if left is None:
            continue
        for j, right in enumerate(vectors):
            if right is not None:
                gram[i, j] = float(np.dot(left, right))
    linear = np.zeros(len(COMPONENTS), dtype=np.float64)
    linear[COMPONENTS.index(PRIMARY)] = preference
    active_indices = [COMPONENTS.index(name) for name in active_names]
    candidates = []
    for mask in range(1, 1 << len(active_indices)):
        subset = [
            active_indices[index]
            for index in range(len(active_indices))
            if mask & (1 << index)
        ]
        solution = _simplex_qp_solution(gram, linear, active_indices=subset)
        if solution is None:
            continue
        objective = 0.5 * float(solution @ gram @ solution) - float(
            linear @ solution
        )
        candidates.append((objective, tuple(solution.tolist()), solution))
    if not candidates:
        # The primary-only point is always feasible, even if the Gram matrix is
        # numerically singular.
        solution = np.zeros(len(COMPONENTS), dtype=np.float64)
        solution[COMPONENTS.index(PRIMARY)] = 1.0
    else:
        solution = min(candidates, key=lambda item: (item[0], item[1]))[2]
    solution = np.maximum(solution, 0.0)
    solution /= float(solution.sum())
    weights = {name: float(solution[index]) for index, name in enumerate(COMPONENTS)}
    direction = sum(
        solution[index] * vectors[index]
        for index in range(len(COMPONENTS))
        if vectors[index] is not None
    )
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= epsilon:
        alignments = {name: 0.0 for name in COMPONENTS}
    else:
        alignments = {
            name: (
                float(np.dot(direction / direction_norm, vectors[index]))
                if vectors[index] is not None
                else 0.0
            )
            for index, name in enumerate(COMPONENTS)
        }
    pairwise_cosines = {}
    for i, left_name in enumerate(COMPONENTS):
        for j in range(i + 1, len(COMPONENTS)):
            right_name = COMPONENTS[j]
            pairwise_cosines[f"{left_name}_to_{right_name}"] = float(gram[i, j])
    diagnostics = {
        "method": "ovrl_preferred_marble_simplex_qp",
        "objective": float(
            0.5 * (solution @ gram @ solution) - (linear @ solution)
        ),
        "primary_preference": preference,
        "normalized_direction_norm": direction_norm,
        "pairwise_gradient_cosines": pairwise_cosines,
        "direction_alignments": alignments,
        "zero_gradient_components": [
            name for name in COMPONENTS if norms[name] <= epsilon
        ],
    }
    return weights, alignments, norms, diagnostics


def update_marble_state(
    previous: Mapping | None,
    *,
    observed_weights: Mapping[str, float],
    alignments: Mapping[str, float],
    gradient_norms: Mapping[str, float],
    diagnostics: Mapping,
    ema_decay: float,
    local_step: int,
    global_step: int,
) -> dict:
    """Apply MARBLE's coefficient amortization (EMA) and persist diagnostics."""

    decay = float(ema_decay)
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("MARBLE coefficient EMA decay must lie in [0,1)")
    if set(observed_weights) != set(COMPONENTS):
        raise ValueError("MARBLE observed coefficients are incomplete")
    observed = {name: max(0.0, float(observed_weights[name])) for name in COMPONENTS}
    if previous is None:
        coefficients = observed
        calibration_index = 1
    else:
        old = previous.get("coefficients")
        if not isinstance(old, Mapping) or set(old) != set(COMPONENTS):
            raise ValueError("resumed MARBLE state has invalid coefficients")
        coefficients = {
            name: decay * float(old[name]) + (1.0 - decay) * observed[name]
            for name in COMPONENTS
        }
        calibration_index = int(previous.get("calibration_index", 0)) + 1
    total = float(sum(coefficients.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("MARBLE coefficients cannot all be zero")
    convex = {name: float(coefficients[name] / total) for name in COMPONENTS}
    return {
        "schema_version": 1,
        "calibration_index": calibration_index,
        "last_calibration_local_step": int(local_step),
        "last_calibration_global_step": int(global_step),
        "coefficients": {name: float(coefficients[name]) for name in COMPONENTS},
        "convex_weights": convex,
        "last_observed_weights": observed,
        "last_direction_alignments": {
            name: float(alignments.get(name, 0.0)) for name in COMPONENTS
        },
        "last_reward_induced_gradient_norms": {
            name: float(gradient_norms[name]) for name in COMPONENTS
        },
        "last_qp_diagnostics": json_safe_mapping(diagnostics),
        "coefficient_ema_decay": decay,
    }


def json_safe_mapping(value: Mapping) -> dict:
    """Convert the small numerical MARBLE diagnostic mapping to JSON values."""

    output = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            output[str(key)] = json_safe_mapping(item)
        elif isinstance(item, (list, tuple)):
            output[str(key)] = [json_safe_mapping(v) if isinstance(v, Mapping) else v for v in item]
        elif isinstance(item, (np.floating, np.integer)):
            output[str(key)] = item.item()
        else:
            output[str(key)] = item
    return output


def validate_marble_state(state: Mapping) -> None:
    required = {
        "schema_version",
        "calibration_index",
        "last_calibration_local_step",
        "last_calibration_global_step",
        "coefficients",
        "convex_weights",
        "last_observed_weights",
        "last_direction_alignments",
        "last_reward_induced_gradient_norms",
        "last_qp_diagnostics",
        "coefficient_ema_decay",
    }
    if set(state) != required or int(state["schema_version"]) != 1:
        raise ValueError("invalid resumed MARBLE state schema")
    for key in ("coefficients", "convex_weights", "last_observed_weights"):
        value = state[key]
        if not isinstance(value, Mapping) or set(value) != set(COMPONENTS):
            raise ValueError(f"invalid resumed MARBLE {key}")
        if any(not math.isfinite(float(item)) or float(item) < 0.0 for item in value.values()):
            raise ValueError(f"invalid resumed MARBLE {key} values")
    for key in ("last_direction_alignments", "last_reward_induced_gradient_norms"):
        value = state[key]
        if not isinstance(value, Mapping) or set(value) != set(COMPONENTS):
            raise ValueError(f"invalid resumed MARBLE {key}")
        if any(not math.isfinite(float(item)) for item in value.values()):
            raise ValueError(f"invalid resumed MARBLE {key} values")
    if not isinstance(state["last_qp_diagnostics"], Mapping):
        raise ValueError("invalid resumed MARBLE QP diagnostics")
    decay = float(state["coefficient_ema_decay"])
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("invalid resumed MARBLE coefficient EMA decay")
    if abs(sum(float(v) for v in state["convex_weights"].values()) - 1.0) > 1.0e-5:
        raise ValueError("MARBLE convex weights must sum to one")


def component_advantage_streams(
    rows: Sequence[Mapping],
    *,
    conditions: int,
    candidates: int,
    advantage_config: Mapping,
) -> tuple[dict[str, list[torch.Tensor]], dict]:
    """Build one independently normalized AF advantage per reward component."""

    raw = np.empty((conditions, candidates, len(COMPONENTS)), dtype=np.float64)
    observed = set()
    for row in rows:
        condition = int(row["condition_index"])
        candidate = int(row["candidate_index"])
        key = (condition, candidate)
        if key in observed:
            raise ValueError(f"duplicate GA-AF rollout endpoint: {key}")
        observed.add(key)
        components = row.get("reward_components", {}).get("raw")
        if not isinstance(components, Mapping) or not set(COMPONENTS).issubset(
            components
        ):
            raise ValueError("GA-AF rollout lacks three raw composite components")
        raw[condition, candidate] = [float(components[name]) for name in COMPONENTS]
    expected = {
        (condition, candidate)
        for condition in range(conditions)
        for candidate in range(candidates)
    }
    if observed != expected or not np.isfinite(raw).all():
        raise ValueError("GA-AF rollout does not exactly cover a finite LxK batch")

    arrays = {}
    pooled_scales = {}
    clipped_fractions = {}
    mappings = {}
    for index, name in enumerate(COMPONENTS):
        rewards = raw[:, :, index]
        centered = rewards - rewards.mean(axis=1, keepdims=True)
        scale = float(np.sqrt(np.mean(np.square(centered))))
        minimum = float(advantage_config["minimum_global_scale"])
        if scale <= minimum:
            if name == PRIMARY:
                raise ValueError("GA-AF primary OVRL has no complete-batch signal")
            arrays[name] = np.zeros_like(rewards)
            pooled_scales[name] = scale
            clipped_fractions[name] = 0.0
            mappings[name] = "zero_no_batch_signal"
            continue
        result = compute_paper_global_advantages(
            rewards,
            clip=float(advantage_config["clip"]),
            minimum_scale=minimum,
            mapping=str(advantage_config.get("mapping", "linear_clipped")),
            temperature=float(advantage_config.get("temperature", 1.0)),
        )
        arrays[name] = result.advantages
        pooled_scales[name] = float(result.pooled_scale)
        clipped_fractions[name] = float(result.clipped_fraction)
        mappings[name] = result.mapping
    streams = {
        name: [torch.from_numpy(arrays[name][index]).float() for index in range(conditions)]
        for name in COMPONENTS
    }
    diagnostics = {
        "construction": (
            "per-component condition centering, complete-LxK RMS normalization, "
            "then frozen AF clipping"
        ),
        "pooled_scales": pooled_scales,
        "clipped_fractions": clipped_fractions,
        "mapping": mappings,
    }
    return streams, diagnostics


def replace_advantages(
    conditions: Sequence[RolloutTrainingCondition],
    stream: Sequence[torch.Tensor],
) -> list[RolloutTrainingCondition]:
    if len(conditions) != len(stream):
        raise ValueError("GA-AF advantage stream length differs from conditions")
    output = []
    for index, item in enumerate(conditions):
        advantage = stream[index].detach().float().cpu()
        if advantage.shape != item.advantages.shape or not torch.isfinite(advantage).all():
            raise ValueError(f"invalid GA-AF advantage stream for {item.utterance}")
        output.append(
            RolloutTrainingCondition(
                utterance=item.utterance,
                transcript=item.transcript,
                condition_mel=item.condition_mel,
                terminal_mels=item.terminal_mels,
                advantages=advantage,
            )
        )
    return output


def convex_fuse_streams(
    streams: Mapping[str, Sequence[torch.Tensor]], weights: Mapping[str, float]
) -> list[torch.Tensor]:
    if set(streams) != set(COMPONENTS) or set(weights) != set(COMPONENTS):
        raise ValueError("GA-AF fusion requires exactly three component streams")
    parsed = {name: float(weights[name]) for name in COMPONENTS}
    if any(not math.isfinite(value) or value < 0.0 for value in parsed.values()):
        raise ValueError("GA-AF fusion weights must be finite and non-negative")
    total = float(sum(parsed.values()))
    if total <= 0.0:
        raise ValueError("GA-AF fusion weights cannot all be zero")
    length = len(streams[PRIMARY])
    if any(len(streams[name]) != length for name in COMPONENTS):
        raise ValueError("GA-AF component stream lengths differ")
    output = []
    for index in range(length):
        fused = sum(
            parsed[name] * streams[name][index] for name in COMPONENTS
        ) / total
        if not torch.isfinite(fused).all() or float(fused.abs().max()) > 1.0 + 1.0e-6:
            raise ValueError("convex GA-AF advantage escaped the frozen [-1,1] range")
        output.append(fused.detach().float().cpu())
    return output


def marble_fuse_streams(
    streams: Mapping[str, Sequence[torch.Tensor]],
    weights: Mapping[str, float],
    gradient_norms: Mapping[str, float],
    *,
    clip: float = 1.0,
    norm_epsilon: float = 1.0e-12,
) -> tuple[list[torch.Tensor], dict]:
    """Fuse component advantages to realize MARBLE's normalized-gradient direction.

    Since the AdvantageFlow gradient is linear in the supplied advantage, the
    coefficient for component ``m`` is ``alpha_m * mean(||q||) / ||q_m||``.
    A single positive post-scale keeps the fused stream inside the configured
    advantage range without changing its direction.
    """

    if set(streams) != set(COMPONENTS) or set(weights) != set(COMPONENTS):
        raise ValueError("MARBLE fusion requires exactly three component streams")
    if set(gradient_norms) != set(COMPONENTS):
        raise ValueError("MARBLE fusion requires all component gradient norms")
    clip = float(clip)
    epsilon = float(norm_epsilon)
    if not math.isfinite(clip) or clip <= 0.0 or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("invalid MARBLE fusion clip or norm_epsilon")
    parsed = {name: float(weights[name]) for name in COMPONENTS}
    if any(not math.isfinite(value) or value < 0.0 for value in parsed.values()):
        raise ValueError("MARBLE fusion weights must be finite and non-negative")
    total = float(sum(parsed.values()))
    if total <= 0.0:
        raise ValueError("MARBLE fusion weights cannot all be zero")
    parsed = {name: value / total for name, value in parsed.items()}
    all_norms = [float(gradient_norms[name]) for name in COMPONENTS]
    nonzero_norms = [value for value in all_norms if value > epsilon]
    if not nonzero_norms:
        raise ValueError("MARBLE fusion has no non-zero gradient norms")
    # MARBLE restores the mean *original* gradient norm after unit-gradient
    # harmonization.  Keep zero-norm components in that mean, while assigning
    # them a zero stream coefficient below.
    target_norm = float(np.mean(all_norms))
    if target_norm <= epsilon:
        target_norm = float(np.mean(nonzero_norms))
    raw_coefficients = {
        name: (
            parsed[name] * target_norm / float(gradient_norms[name])
            if float(gradient_norms[name]) > epsilon
            else 0.0
        )
        for name in COMPONENTS
    }
    length = len(streams[PRIMARY])
    if any(len(streams[name]) != length for name in COMPONENTS):
        raise ValueError("MARBLE component stream lengths differ")
    raw_stream = []
    peak = 0.0
    for index in range(length):
        fused = sum(
            raw_coefficients[name] * streams[name][index] for name in COMPONENTS
        )
        if not torch.isfinite(fused).all():
            raise ValueError("non-finite MARBLE fused advantage")
        raw_stream.append(fused.detach().float().cpu())
        peak = max(peak, float(fused.abs().max().item()))
    post_scale = min(1.0, clip / peak) if peak > epsilon else 1.0
    output = [value.mul(post_scale) for value in raw_stream]
    if any(float(value.abs().max()) > clip + 1.0e-5 for value in output):
        raise ValueError("MARBLE fused advantage escaped configured range")
    return output, {
        "target_gradient_norm": target_norm,
        "raw_coefficients": raw_coefficients,
        "post_scale": float(post_scale),
        "effective_coefficients": {
            name: float(value * post_scale)
            for name, value in raw_coefficients.items()
        },
        "raw_peak": peak,
    }


def reward_induced_gradients(
    *,
    loss_function,
    streams: Mapping[str, Sequence[torch.Tensor]],
    zero_grad,
    named_parameters,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float], int]:
    """Compute ``q_m = grad L_AF(A_m) - grad L_AF(0)`` for every component."""

    if set(streams) != set(COMPONENTS):
        raise ValueError("GA-AF calibration requires exactly three component streams")
    parameters = list(named_parameters)
    if not parameters:
        raise ValueError("GA-AF calibration found no LoRA parameters")
    names = [str(name) for name, _ in parameters]
    if len(names) != len(set(names)):
        raise ValueError("GA-AF LoRA parameter names are not unique")

    def capture(stream: Sequence[torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        zero_grad()
        loss = loss_function(stream)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("GA-AF loss callback must return a scalar tensor")
        gradients = {}
        for name, parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                gradients[name] = torch.zeros_like(parameter, device="cpu")
            else:
                if not torch.isfinite(gradient).all():
                    raise ValueError(f"non-finite GA-AF gradient for {name}")
                gradients[name] = gradient.detach().float().cpu().clone()
        return gradients, float(loss.detach().item())

    first = streams[PRIMARY]
    if not first:
        raise ValueError("GA-AF calibration stream is empty")
    shapes = [tuple(value.shape) for value in first]
    if any(
        len(streams[name]) != len(first)
        or [tuple(value.shape) for value in streams[name]] != shapes
        for name in COMPONENTS
    ):
        raise ValueError("GA-AF calibration stream geometry differs by component")
    zero_stream = [torch.zeros_like(value) for value in first]
    baseline, baseline_loss = capture(zero_stream)
    outputs = {}
    losses = {"zero_advantage": baseline_loss}
    for name in COMPONENTS:
        full, loss = capture(streams[name])
        outputs[name] = {
            key: full[key] - baseline[key] for key in names
        }
        losses[name] = loss
    zero_grad()
    return outputs, losses, 1 + len(COMPONENTS)


def gradient_dot(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    if set(left) != set(right):
        raise ValueError("GA-AF gradient states have different keys")
    return float(
        sum(
            torch.sum(left[name].double() * right[name].double()).item()
            for name in left
        )
    )


def gradient_norm(value: Mapping[str, torch.Tensor]) -> float:
    return float(math.sqrt(max(0.0, gradient_dot(value, value))))


def gradient_cosine(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    denominator = gradient_norm(left) * gradient_norm(right)
    if denominator <= 1.0e-15:
        raise ValueError("GA-AF reward-induced gradient has zero norm")
    cosine = float(gradient_dot(left, right) / denominator)
    if not math.isfinite(cosine):
        raise ValueError("GA-AF gradient cosine is non-finite")
    return max(-1.0, min(1.0, cosine))


def observed_gate_weights(
    gradients: Mapping[str, Mapping[str, torch.Tensor]], *, auxiliary_cap: float
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if not 0.0 <= float(auxiliary_cap) <= 1.0:
        raise ValueError("GA-AF auxiliary cap must lie in [0,1]")
    primary_norm = gradient_norm(gradients[PRIMARY])
    if primary_norm <= 1.0e-15:
        raise ValueError("GA-AF primary reward-induced gradient has zero norm")
    norms = {PRIMARY: primary_norm} | {
        name: gradient_norm(gradients[name]) for name in AUXILIARIES
    }
    cosines = {
        name: (
            gradient_cosine(gradients[PRIMARY], gradients[name])
            if norms[name] > 1.0e-15
            else 0.0
        )
        for name in AUXILIARIES
    }
    weights = {PRIMARY: 1.0} | {
        name: float(auxiliary_cap) * max(0.0, cosines[name])
        for name in AUXILIARIES
    }
    return weights, cosines, norms


def projected_gaaf_observed_weights(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    auxiliary_target_norm_ratio: float,
    auxiliary_coefficient_cap: float,
    projection_epsilon: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict]:
    """Map OVRL-primary projected gradients back to AF stream coefficients.

    This is a one-sided, PCGrad-style operation: for every auxiliary gradient
    ``g_i``, remove only the component that conflicts with the OVRL gradient
    ``g_o``::

        g_i_perp = g_i - min(0, <g_i,g_o>) / (||g_o||^2 + eps) * g_o

    The retained auxiliary direction is scaled to a small target fraction of
    ``||g_o||`` and capped.  AdvantageFlow is linear in the supplied advantage
    stream, so ``g_o + beta_i * g_i_perp`` can be realized without changing the
    optimizer by assigning ``beta_i`` to the auxiliary stream and adding the
    projection compensation to the OVRL stream coefficient.
    """

    if set(gradients) != set(COMPONENTS):
        raise ValueError(
            "projected GA-AF calibration requires exactly three component gradients"
        )
    target_ratio = float(auxiliary_target_norm_ratio)
    coefficient_cap = float(auxiliary_coefficient_cap)
    epsilon = float(projection_epsilon)
    if not math.isfinite(target_ratio) or not 0.0 <= target_ratio <= 1.0:
        raise ValueError(
            "projected GA-AF auxiliary_target_norm_ratio must lie in [0,1]"
        )
    if not math.isfinite(coefficient_cap) or not 0.0 <= coefficient_cap <= 1.0:
        raise ValueError(
            "projected GA-AF auxiliary_coefficient_cap must lie in [0,1]"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("projected GA-AF projection_epsilon must be positive")

    norms = {name: gradient_norm(gradients[name]) for name in COMPONENTS}
    primary_norm = norms[PRIMARY]
    if primary_norm <= 1.0e-15:
        raise ValueError("projected GA-AF primary OVRL gradient has zero norm")
    primary_squared_norm = gradient_dot(gradients[PRIMARY], gradients[PRIMARY])

    cosines: dict[str, float] = {}
    projection_boosts: dict[str, float] = {}
    projected_norms: dict[str, float] = {}
    projected_primary_dots: dict[str, float] = {}
    auxiliary_coefficients: dict[str, float] = {}
    target_auxiliary_norm = target_ratio * primary_norm
    for name in AUXILIARIES:
        auxiliary_norm = norms[name]
        dot = gradient_dot(gradients[name], gradients[PRIMARY])
        cosine = (
            max(-1.0, min(1.0, dot / (auxiliary_norm * primary_norm)))
            if auxiliary_norm > 1.0e-15
            else 0.0
        )
        boost = max(0.0, -dot / (primary_squared_norm + epsilon))
        projected_squared_norm = max(
            0.0,
            auxiliary_norm * auxiliary_norm
            + 2.0 * boost * dot
            + boost * boost * primary_squared_norm,
        )
        projected_norm = math.sqrt(projected_squared_norm)
        projected_primary_dot = dot + boost * primary_squared_norm
        if projected_norm <= epsilon or target_auxiliary_norm == 0.0:
            coefficient = 0.0
        else:
            coefficient = min(
                coefficient_cap,
                target_auxiliary_norm / (projected_norm + epsilon),
            )
        cosines[name] = float(cosine)
        projection_boosts[name] = float(boost)
        projected_norms[name] = float(projected_norm)
        projected_primary_dots[name] = float(projected_primary_dot)
        auxiliary_coefficients[name] = float(coefficient)

    weights = {
        PRIMARY: float(
            1.0
            + sum(
                auxiliary_coefficients[name] * projection_boosts[name]
                for name in AUXILIARIES
            )
        ),
        **auxiliary_coefficients,
    }
    diagnostics = {
        "method": "ovrl_primary_asymmetric_gradient_projection",
        "auxiliary_target_norm_ratio": target_ratio,
        "auxiliary_coefficient_cap": coefficient_cap,
        "projection_epsilon": epsilon,
        "primary_squared_gradient_norm": float(primary_squared_norm),
        "target_auxiliary_gradient_norm": float(target_auxiliary_norm),
        "gradient_dots_with_primary": {
            name: float(gradient_dot(gradients[name], gradients[PRIMARY]))
            for name in AUXILIARIES
        },
        "gradient_cosines_with_primary": dict(cosines),
        "projection_boosts": projection_boosts,
        "projected_gradient_norms": projected_norms,
        "projected_gradient_dots_with_primary": projected_primary_dots,
        "auxiliary_coefficients": auxiliary_coefficients,
        "stream_coefficients": dict(weights),
    }
    return weights, cosines, norms, diagnostics


def update_projected_gaaf_state(
    previous: Mapping | None,
    *,
    observed_weights: Mapping[str, float],
    cosines: Mapping[str, float],
    gradient_norms: Mapping[str, float],
    diagnostics: Mapping,
    ema_decay: float,
    local_step: int,
    global_step: int,
) -> dict:
    """EMA-amortize projected coefficients without hard-zeroing conflicts."""

    decay = float(ema_decay)
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError(
            "projected GA-AF coefficient EMA decay must lie in [0,1)"
        )
    if set(observed_weights) != set(COMPONENTS):
        raise ValueError("projected GA-AF observed weights are incomplete")
    observed = {name: float(observed_weights[name]) for name in COMPONENTS}
    if any(
        not math.isfinite(value) or value < 0.0 for value in observed.values()
    ) or observed[PRIMARY] <= 0.0:
        raise ValueError("invalid projected GA-AF observed weights")
    if set(cosines) != set(AUXILIARIES) or any(
        not math.isfinite(float(value)) for value in cosines.values()
    ):
        raise ValueError("invalid projected GA-AF gradient cosines")
    if set(gradient_norms) != set(COMPONENTS) or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in gradient_norms.values()
    ):
        raise ValueError("invalid projected GA-AF gradient norms")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("invalid projected GA-AF projection diagnostics")
    projection_boosts = diagnostics.get("projection_boosts")
    if not isinstance(projection_boosts, Mapping) or set(
        projection_boosts
    ) != set(AUXILIARIES):
        raise ValueError("projected GA-AF diagnostics lack projection boosts")
    projection_boosts = {
        name: float(projection_boosts[name]) for name in AUXILIARIES
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in projection_boosts.values()
    ):
        raise ValueError("invalid projected GA-AF projection boosts")

    if previous is None:
        auxiliary_weights = {
            name: observed[name] for name in AUXILIARIES
        }
        calibration_index = 1
    else:
        validate_projected_gaaf_state(previous)
        old = previous["weights"]
        auxiliary_weights = {
            name: decay * float(old[name]) + (1.0 - decay) * observed[name]
            for name in AUXILIARIES
        }
        calibration_index = int(previous["calibration_index"]) + 1
    # Smooth the retained auxiliary magnitudes, then reconstruct the *current*
    # OVRL compensation exactly.  EMA-ing the primary coefficient itself would
    # mix old projection geometry into the new calibration and could reintroduce
    # a negative dot product with the current OVRL gradient.
    weights = {
        PRIMARY: float(
            1.0
            + sum(
                auxiliary_weights[name] * projection_boosts[name]
                for name in AUXILIARIES
            )
        ),
        **auxiliary_weights,
    }
    total = float(sum(weights.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("projected GA-AF weights cannot all be zero")
    convex = {name: float(weights[name] / total) for name in COMPONENTS}
    return {
        "schema_version": 1,
        "calibration_index": calibration_index,
        "last_calibration_local_step": int(local_step),
        "last_calibration_global_step": int(global_step),
        "weights": {name: float(weights[name]) for name in COMPONENTS},
        "convex_weights": convex,
        "last_observed_weights": observed,
        "last_gradient_cosines": {
            name: float(cosines[name]) for name in AUXILIARIES
        },
        "last_reward_induced_gradient_norms": {
            name: float(gradient_norms[name]) for name in COMPONENTS
        },
        "last_projection_diagnostics": json_safe_mapping(diagnostics),
        "coefficient_ema_decay": decay,
    }


def validate_projected_gaaf_state(state: Mapping) -> None:
    required = {
        "schema_version",
        "calibration_index",
        "last_calibration_local_step",
        "last_calibration_global_step",
        "weights",
        "convex_weights",
        "last_observed_weights",
        "last_gradient_cosines",
        "last_reward_induced_gradient_norms",
        "last_projection_diagnostics",
        "coefficient_ema_decay",
    }
    if set(state) != required or int(state["schema_version"]) != 1:
        raise ValueError("invalid resumed projected GA-AF state schema")
    for key in ("weights", "convex_weights", "last_observed_weights"):
        value = state[key]
        if not isinstance(value, Mapping) or set(value) != set(COMPONENTS):
            raise ValueError(f"invalid resumed projected GA-AF {key}")
        if any(
            not math.isfinite(float(item)) or float(item) < 0.0
            for item in value.values()
        ):
            raise ValueError(f"invalid resumed projected GA-AF {key} values")
    if float(state["weights"][PRIMARY]) <= 0.0:
        raise ValueError("projected GA-AF primary weight must be positive")
    if abs(sum(float(v) for v in state["convex_weights"].values()) - 1.0) > 1.0e-5:
        raise ValueError("projected GA-AF convex weights must sum to one")
    cosines = state["last_gradient_cosines"]
    if not isinstance(cosines, Mapping) or set(cosines) != set(AUXILIARIES):
        raise ValueError("invalid resumed projected GA-AF gradient cosines")
    if any(not math.isfinite(float(value)) for value in cosines.values()):
        raise ValueError("invalid resumed projected GA-AF gradient cosine values")
    norms = state["last_reward_induced_gradient_norms"]
    if not isinstance(norms, Mapping) or set(norms) != set(COMPONENTS):
        raise ValueError("invalid resumed projected GA-AF gradient norms")
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in norms.values()
    ):
        raise ValueError("invalid resumed projected GA-AF gradient norm values")
    diagnostics = state["last_projection_diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError("invalid resumed projected GA-AF diagnostics")
    projection_boosts = diagnostics.get("projection_boosts")
    if not isinstance(projection_boosts, Mapping) or set(
        projection_boosts
    ) != set(AUXILIARIES):
        raise ValueError("invalid resumed projected GA-AF projection boosts")
    expected_primary = 1.0 + sum(
        float(state["weights"][name]) * float(projection_boosts[name])
        for name in AUXILIARIES
    )
    if not math.isclose(
        float(state["weights"][PRIMARY]),
        expected_primary,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "projected GA-AF primary weight does not reconstruct the projection"
        )
    decay = float(state["coefficient_ema_decay"])
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("invalid resumed projected GA-AF EMA decay")


def projected_gaaf_calibration_due(
    state: Mapping | None, *, local_step: int, refresh_interval: int
) -> bool:
    """Resume-invariant calibration cadence for projected GA-AF."""

    if local_step < 1 or refresh_interval < 1:
        raise ValueError(
            "projected GA-AF calibration step and interval must be positive"
        )
    if state is None:
        if local_step != 1:
            raise ValueError("missing projected GA-AF state after the first local step")
        return True
    validate_projected_gaaf_state(state)
    return (local_step - 1) % refresh_interval == 0


def update_gate_state(
    previous: Mapping | None,
    *,
    observed_weights: Mapping[str, float],
    cosines: Mapping[str, float],
    gradient_norms: Mapping[str, float],
    ema_decay: float,
    local_step: int,
    global_step: int,
) -> dict:
    if not 0.0 <= float(ema_decay) < 1.0:
        raise ValueError("GA-AF coefficient EMA decay must lie in [0,1)")
    if set(observed_weights) != set(COMPONENTS):
        raise ValueError("GA-AF observed weights are incomplete")
    if previous is None:
        weights = {name: float(observed_weights[name]) for name in COMPONENTS}
        calibration_index = 1
    else:
        old = previous.get("weights")
        if not isinstance(old, Mapping) or set(old) != set(COMPONENTS):
            raise ValueError("resumed GA-AF gate state has invalid weights")
        weights = {
            PRIMARY: 1.0,
            **{
                name: (
                    0.0
                    if float(observed_weights[name]) == 0.0
                    else float(ema_decay) * float(old[name])
                    + (1.0 - float(ema_decay)) * float(observed_weights[name])
                )
                for name in AUXILIARIES
            },
        }
        calibration_index = int(previous.get("calibration_index", 0)) + 1
    total = float(sum(weights.values()))
    convex = {name: float(weights[name] / total) for name in COMPONENTS}
    return {
        "schema_version": 1,
        "calibration_index": calibration_index,
        "last_calibration_local_step": int(local_step),
        "last_calibration_global_step": int(global_step),
        "weights": weights,
        "convex_weights": convex,
        "last_observed_weights": {
            name: float(observed_weights[name]) for name in COMPONENTS
        },
        "last_gradient_cosines": {
            name: float(cosines[name]) for name in AUXILIARIES
        },
        "last_reward_induced_gradient_norms": {
            name: float(gradient_norms[name]) for name in COMPONENTS
        },
        "coefficient_ema_decay": float(ema_decay),
    }


def validate_gate_state(state: Mapping) -> None:
    required = {
        "schema_version",
        "calibration_index",
        "last_calibration_local_step",
        "last_calibration_global_step",
        "weights",
        "convex_weights",
        "last_observed_weights",
        "last_gradient_cosines",
        "last_reward_induced_gradient_norms",
        "coefficient_ema_decay",
    }
    if set(state) != required or int(state["schema_version"]) != 1:
        raise ValueError("invalid resumed GA-AF gate-state schema")
    weights = state["weights"]
    if not isinstance(weights, Mapping) or set(weights) != set(COMPONENTS):
        raise ValueError("invalid resumed GA-AF weights")
    if float(weights[PRIMARY]) != 1.0 or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in weights.values()
    ):
        raise ValueError("invalid resumed GA-AF weight values")


def calibration_due(
    state: Mapping | None, *, local_step: int, refresh_interval: int
) -> bool:
    """Use a global local-step cadence that is invariant to resume boundaries."""

    if local_step < 1 or refresh_interval < 1:
        raise ValueError("GA-AF calibration step and interval must be positive")
    if state is None:
        if local_step != 1:
            raise ValueError("missing GA-AF gate state after the first local step")
        return True
    validate_gate_state(state)
    return (local_step - 1) % refresh_interval == 0


def marble_calibration_due(
    state: Mapping | None, *, local_step: int, refresh_interval: int
) -> bool:
    """Resume-invariant calibration cadence shared by the MARBLE variant."""

    if local_step < 1 or refresh_interval < 1:
        raise ValueError("MARBLE calibration step and interval must be positive")
    if state is None:
        if local_step != 1:
            raise ValueError("missing MARBLE state after the first local step")
        return True
    validate_marble_state(state)
    return (local_step - 1) % refresh_interval == 0
