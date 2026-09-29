"""Group-relative advantage and ranking utilities for Gate A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


@dataclass(frozen=True)
class AdvantageResult:
    """Result of pooled group-relative normalization."""

    advantages: np.ndarray
    centered_rewards: np.ndarray
    valid_groups: np.ndarray
    within_group_std: np.ndarray
    pooled_scale: float


def _as_groups(values: np.ndarray | list[list[float]]) -> np.ndarray:
    groups = np.asarray(values, dtype=np.float64)
    if groups.ndim != 2 or groups.shape[1] < 2:
        raise ValueError("expected [conditions, group_size] with group_size >= 2")
    return groups


def compute_group_advantages(
    rewards: np.ndarray | list[list[float]],
    *,
    epsilon: float = 1e-6,
    zero_signal_threshold: float = 1e-4,
    clip: float = 1.0,
) -> AdvantageResult:
    """Compute condition-centered advantages with one pooled batch scale.

    Groups containing invalid rewards or with population standard deviation below
    ``zero_signal_threshold`` are rejected and represented by NaNs.  This makes it
    impossible to silently turn a zero-signal group into a training example.
    """
    if epsilon <= 0 or zero_signal_threshold < 0 or clip <= 0:
        raise ValueError("epsilon and clip must be positive; threshold cannot be negative")
    groups = _as_groups(rewards)
    finite = np.isfinite(groups).all(axis=1)
    means = np.mean(groups, axis=1, keepdims=True)
    centered = groups - means
    within_std = np.std(groups, axis=1, ddof=0)
    valid = finite & (within_std >= zero_signal_threshold)
    advantages = np.full_like(groups, np.nan)
    if valid.any():
        pooled_scale = float(np.sqrt(np.mean(np.square(centered[valid]))))
        advantages[valid] = np.clip(
            centered[valid] / (pooled_scale + epsilon), -clip, clip
        )
    else:
        pooled_scale = 0.0
    centered[~finite] = np.nan
    return AdvantageResult(
        advantages=advantages,
        centered_rewards=centered,
        valid_groups=valid,
        within_group_std=within_std,
        pooled_scale=pooled_scale,
    )


def group_reward_statistics(
    rewards: np.ndarray | list[list[float]],
    *,
    zero_signal_threshold: float = 1e-4,
) -> list[dict[str, float | bool]]:
    """Return the preregistered per-condition reward-geometry statistics."""
    groups = _as_groups(rewards)
    records = []
    for group in groups:
        if not np.isfinite(group).all():
            records.append({"valid": False})
            continue
        ordered = np.sort(group)
        width = float(ordered[-1] - ordered[0])
        median = float(np.median(ordered))
        max_deviation = float(np.max(np.abs(ordered - median)))
        records.append(
            {
                "valid": True,
                "range": width,
                "std": float(np.std(group, ddof=0)),
                "iqr": float(np.percentile(group, 75) - np.percentile(group, 25)),
                "top2_minus_bottom2": float(
                    np.mean(ordered[-2:]) - np.mean(ordered[:2])
                ),
                "largest_median_deviation_fraction": (
                    max_deviation / width if width > 0 else 0.0
                ),
                "zero_signal": bool(np.std(group, ddof=0) < zero_signal_threshold),
            }
        )
    return records


def center_within_groups(values: np.ndarray | list[list[float]]) -> np.ndarray:
    groups = _as_groups(values)
    return groups - np.nanmean(groups, axis=1, keepdims=True)


def per_group_rank_correlations(
    left: np.ndarray | list[list[float]],
    right: np.ndarray | list[list[float]],
) -> list[dict[str, float]]:
    """Compute correlations separately per condition, never across utterances."""
    left = _as_groups(left)
    right = _as_groups(right)
    if left.shape != right.shape:
        raise ValueError("metric matrices must have equal shapes")
    output = []
    for left_group, right_group in zip(left, right, strict=True):
        valid = np.isfinite(left_group) & np.isfinite(right_group)
        if valid.sum() < 3:
            output.append({"spearman": np.nan, "kendall": np.nan})
            continue
        left_valid = left_group[valid]
        right_valid = right_group[valid]
        if np.ptp(left_valid) <= 1e-12 or np.ptp(right_valid) <= 1e-12:
            output.append({"spearman": np.nan, "kendall": np.nan})
            continue
        output.append(
            {
                "spearman": float(spearmanr(left_valid, right_valid).statistic),
                "kendall": float(kendalltau(left_valid, right_valid).statistic),
            }
        )
    return output


def rank_consistency(fast: np.ndarray, deployment: np.ndarray) -> dict[str, float]:
    """Compare two NFE settings for one condition with identical latent seeds."""
    fast = np.asarray(fast, dtype=np.float64)
    deployment = np.asarray(deployment, dtype=np.float64)
    if fast.ndim != 1 or fast.shape != deployment.shape or fast.size < 2:
        raise ValueError("rank vectors must be equal one-dimensional arrays")
    valid = np.isfinite(fast) & np.isfinite(deployment)
    fast = fast[valid]
    deployment = deployment[valid]
    if fast.size < 2:
        return {
            "spearman": np.nan,
            "top_half_overlap": np.nan,
            "top2_overlap": np.nan,
            "top1_agreement": np.nan,
            "mean_absolute_difference": np.nan,
        }
    half = max(1, fast.size // 2)
    top_fast = set(np.argsort(rankdata(fast))[-half:])
    top_deployment = set(np.argsort(rankdata(deployment))[-half:])
    top2_count = min(2, fast.size)
    top2_fast = set(np.argsort(fast)[-top2_count:])
    top2_deployment = set(np.argsort(deployment)[-top2_count:])
    return {
        "spearman": float(spearmanr(fast, deployment).statistic),
        "top_half_overlap": len(top_fast & top_deployment) / half,
        "top2_overlap": len(top2_fast & top2_deployment) / top2_count,
        "top1_agreement": float(int(np.argmax(fast) == np.argmax(deployment))),
        "mean_absolute_difference": float(np.mean(np.abs(fast - deployment))),
    }


def cluster_bootstrap_mean(
    values: np.ndarray | list[float],
    *,
    seed: int = 1234,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Utterance-cluster bootstrap interval for one paired value per condition."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    boot = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def shortcut_permutation_test(
    reward_groups: np.ndarray,
    shortcut_groups: np.ndarray,
    *,
    seed: int,
    samples: int = 5000,
    rho_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calibrate within-condition absolute Spearman statistics under permutation.

    Each null draw independently permutes the shortcut ranks inside every
    condition, preserving both marginal distributions and the group size.
    Constant shortcut groups contribute rho=0 because they cannot rank endpoints.
    """
    reward_groups = _as_groups(reward_groups)
    shortcut_groups = _as_groups(shortcut_groups)
    if reward_groups.shape != shortcut_groups.shape:
        raise ValueError("reward and shortcut matrices must have equal shapes")
    if samples < 100:
        raise ValueError("permutation test requires at least 100 samples")
    if not 0 < rho_threshold < 1:
        raise ValueError("rho_threshold must lie in (0, 1)")
    complete = np.isfinite(reward_groups).all(axis=1) & np.isfinite(
        shortcut_groups
    ).all(axis=1)
    varying_reward = np.ptp(reward_groups, axis=1) > 1e-12
    valid = complete & varying_reward
    reward_groups = reward_groups[valid]
    shortcut_groups = shortcut_groups[valid]
    if reward_groups.shape[0] == 0:
        return {
            "valid_utterances": 0,
            "observed_median_absolute_rho": np.nan,
            "observed_fraction_abs_ge_threshold": np.nan,
            "null_median_absolute_rho_mean": np.nan,
            "null_median_absolute_rho_p95": np.nan,
            "null_fraction_mean": np.nan,
            "null_fraction_p95": np.nan,
            "median_absolute_rho_pvalue": np.nan,
            "fraction_pvalue": np.nan,
        }

    left = rankdata(reward_groups, axis=1)
    right = rankdata(shortcut_groups, axis=1)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    left_norm = np.sqrt(np.sum(left**2, axis=1))
    right_norm = np.sqrt(np.sum(right**2, axis=1))
    denominator = left_norm * right_norm

    def correlations(permuted_right):
        numerator = np.sum(left * permuted_right, axis=1)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )

    observed = correlations(right)
    observed_median = float(np.median(np.abs(observed)))
    observed_fraction = float(np.mean(np.abs(observed) >= rho_threshold))
    generator = np.random.default_rng(seed)
    null_medians = np.empty(samples, dtype=np.float64)
    null_fractions = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        order = np.argsort(generator.random(right.shape), axis=1)
        permuted = np.take_along_axis(right, order, axis=1)
        null = correlations(permuted)
        null_medians[index] = np.median(np.abs(null))
        null_fractions[index] = np.mean(np.abs(null) >= rho_threshold)
    return {
        "valid_utterances": int(reward_groups.shape[0]),
        "rho_threshold": float(rho_threshold),
        "permutation_samples": int(samples),
        "observed_median_absolute_rho": observed_median,
        "observed_fraction_abs_ge_threshold": observed_fraction,
        "null_median_absolute_rho_mean": float(np.mean(null_medians)),
        "null_median_absolute_rho_p95": float(np.quantile(null_medians, 0.95)),
        "null_fraction_mean": float(np.mean(null_fractions)),
        "null_fraction_p95": float(np.quantile(null_fractions, 0.95)),
        "median_absolute_rho_pvalue": float(
            (1 + np.count_nonzero(null_medians >= observed_median)) / (samples + 1)
        ),
        "fraction_pvalue": float(
            (1 + np.count_nonzero(null_fractions >= observed_fraction)) / (samples + 1)
        ),
    }


def holm_adjusted_pvalues(values: list[float]) -> list[float]:
    """Holm family-wise adjusted p-values in original input order."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values")
    order = np.argsort(array)
    adjusted_sorted = np.empty_like(array)
    running = 0.0
    count = array.size
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * array[original_index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(array)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return [float(value) for value in adjusted]
