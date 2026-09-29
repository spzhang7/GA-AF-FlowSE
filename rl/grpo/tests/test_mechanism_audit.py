import json

import torch

from rl.grpo.math import compute_group_advantages
from rl.grpo.mechanism_audit import (
    AUDIT_REVISION,
    audit_logged_group_advantages,
    reaudit_run,
)


FLOAT32_OFFSET_REWARDS = [
    20.01468849182129,
    19.992389678955078,
    19.989282608032227,
    19.999732971191406,
    19.980628967285156,
    20.005298614501953,
    20.002538681030273,
    20.007556915283203,
    20.021760940551758,
    20.007665634155273,
]


def _rows():
    rewards = torch.tensor(FLOAT32_OFFSET_REWARDS, dtype=torch.float32)
    result = compute_group_advantages(rewards[None, :], correction=0, epsilon=1.0e-8)
    return [
        {
            "group_id": "group-0",
            "candidate_index": index,
            "reward": float(reward),
            "advantage": float(result.advantages[0, index]),
            "eligible": bool(result.eligible_candidates[0, index]),
        }
        for index, reward in enumerate(FLOAT32_OFFSET_REWARDS)
    ]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_float32_eq8_replay_accepts_conditioned_nonzero_mean_and_rejects_tamper():
    rows = _rows()
    audit = audit_logged_group_advantages(
        rows,
        group_size=10,
        expected_groups=1,
        correction=0,
        epsilon=1.0e-8,
    )
    assert audit["max_abs_group_advantage_mean"] > 1.0e-5
    assert (
        audit["max_abs_group_advantage_mean"] < audit["max_conditioned_centering_bound"]
    )
    assert audit["max_abs_replay_error"] == 0.0
    assert audit["passed"] is True

    rows[0]["advantage"] += 0.2
    tampered = audit_logged_group_advantages(
        rows,
        group_size=10,
        expected_groups=1,
        correction=0,
        epsilon=1.0e-8,
    )
    assert tampered["checks"]["logged_advantages_match_float32_eq8"] is False
    assert tampered["passed"] is False


def test_reaudit_preserves_v1_and_writes_provenance_linked_v2(tmp_path):
    _write_json(
        tmp_path / "frozen_config.json",
        {
            "collection": {
                "group_size": 10,
                "prompts_per_mini_batch": 1,
                "mini_batch_repeats": 1,
            },
            "advantage": {"std_correction": 0, "epsilon": 1.0e-8},
            "mechanism_audit": {"checkpoint_collections": [0, 1]},
        },
    )
    source_path = tmp_path / "mechanism_audits" / "mechanism_audit_report.json"
    source = {
        "schema_version": 1,
        "status": "GRPO-MECHANISM-AUDIT-FAIL",
        "checks": {
            "all_registered_points_present": True,
            "all_point_invariants_passed": False,
            "nonzero_gradients_observed": True,
        },
        "invariant_failures_by_collection": {"1": ["group_advantages_zero_mean"]},
        "trends": [],
    }
    _write_json(source_path, source)
    source_bytes = source_path.read_bytes()
    rollout_path = tmp_path / "collection_artifacts" / "rollout_collection_000001.jsonl"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n", encoding="utf-8"
    )

    report, output_path = reaudit_run(tmp_path)
    assert report["status"] == "GRPO-MECHANISM-AUDIT-PASS"
    assert report["audit_revision"] == AUDIT_REVISION
    assert report["invariant_failures_by_collection"] == {}
    assert report["source_report"]["preserved_unmodified"] is True
    assert source_path.read_bytes() == source_bytes
    assert output_path.is_file()
    assert output_path != source_path
