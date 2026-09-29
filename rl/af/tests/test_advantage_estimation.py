import numpy as np

from rl.af.advantage_estimation import (
    cluster_bootstrap_mean,
    compute_group_advantages,
    group_reward_statistics,
    holm_adjusted_pvalues,
    rank_consistency,
    shortcut_permutation_test,
)
def test_group_centering_uses_one_pooled_scale():
    rewards = np.asarray([[1.0, 2.0, 3.0, 4.0], [10.0, 10.5, 11.0, 11.5]])
    result = compute_group_advantages(rewards, clip=100.0)
    expected_centered = rewards - rewards.mean(axis=1, keepdims=True)
    expected_scale = np.sqrt(np.mean(expected_centered**2))
    np.testing.assert_allclose(result.centered_rewards, expected_centered)
    np.testing.assert_allclose(result.pooled_scale, expected_scale)
    np.testing.assert_allclose(result.advantages.sum(axis=1), 0.0, atol=1e-12)


def test_zero_signal_and_invalid_groups_are_rejected():
    rewards = [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, np.nan, 2.0]]
    result = compute_group_advantages(rewards)
    assert result.valid_groups.tolist() == [False, True, False]
    assert np.isnan(result.advantages[0]).all()
    assert np.isfinite(result.advantages[1]).all()
    assert np.isnan(result.advantages[2]).all()


def test_reward_geometry_and_rank_consistency():
    stats = group_reward_statistics([[0.0, 1.0, 2.0, 3.0]])[0]
    assert stats["range"] == 3.0
    assert stats["top2_minus_bottom2"] == 2.0
    ranking = rank_consistency(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1.1, 2.1, 3.1, 4.1]),
    )
    assert ranking["spearman"] == 1.0
    assert ranking["top_half_overlap"] == 1.0
    assert ranking["top1_agreement"] == 1.0


def test_cluster_bootstrap_is_deterministic():
    first = cluster_bootstrap_mean([1.0, 2.0, 3.0], seed=7, samples=100)
    second = cluster_bootstrap_mean([1.0, 2.0, 3.0], seed=7, samples=100)
    assert first == second
    assert first["mean"] == 2.0


def test_shortcut_permutation_detects_consistent_rank_copy():
    rewards = np.tile(np.arange(8, dtype=np.float64), (64, 1))
    result = shortcut_permutation_test(
        rewards,
        rewards.copy(),
        seed=7,
        samples=500,
        rho_threshold=0.5,
    )
    assert result["observed_median_absolute_rho"] == 1.0
    assert result["observed_fraction_abs_ge_threshold"] == 1.0
    assert result["fraction_pvalue"] < 0.01


def test_constant_shortcut_has_no_permutation_signal():
    rewards = np.tile(np.arange(8, dtype=np.float64), (16, 1))
    constant = np.ones_like(rewards)
    result = shortcut_permutation_test(
        rewards,
        constant,
        seed=7,
        samples=100,
        rho_threshold=0.5,
    )
    assert result["observed_median_absolute_rho"] == 0.0
    assert result["observed_fraction_abs_ge_threshold"] == 0.0
    assert result["fraction_pvalue"] == 1.0


def test_holm_adjustment_preserves_order_and_controls_family():
    adjusted = holm_adjusted_pvalues([0.01, 0.04, 0.20, 0.90])
    np.testing.assert_allclose(adjusted, [0.04, 0.12, 0.40, 0.90])

