import json

import pytest

from rl.grpo.run_summary import (
    build_acceptance_summary,
    format_acceptance_summary,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _completed_run(tmp_path):
    _write_json(
        tmp_path / "frozen_config.json",
        {
            "run": {"mode": "pilot", "collections": 75},
            "comparison": {"budget_role": "implementation_correctness_pilot_2gpu"},
            "mechanism_audit": {"enabled": True},
            "collection": {
                "prompts_per_mini_batch": 6,
                "mini_batch_repeats": 12,
                "group_size": 10,
                "optimizer_updates": 4,
            },
        },
    )
    selection = {
        "status": "CHECKPOINT-SELECTION-COMPLETE",
        "candidates": [
            {
                "percentage": 0,
                "collection_index": 0,
                "composite_mean": 1.0,
                "ovrl_mean": 3.0,
            },
            {
                "percentage": 50,
                "collection_index": 38,
                "composite_mean": 1.2,
                "ovrl_mean": 3.1,
            },
            {
                "percentage": 100,
                "collection_index": 75,
                "composite_mean": 1.1,
                "ovrl_mean": 3.05,
            },
        ],
        "comparison_checkpoint": {"percentage": 50},
        "fixed_budget_endpoint": {"percentage": 100},
        "best_safe_checkpoint": {"percentage": 50},
    }
    mechanism = {
        "status": "GRPO-MECHANISM-AUDIT-PASS",
        "checks": {"all_points": True, "nonzero_gradients": True},
        "invariant_failures_by_collection": {},
    }
    _write_json(tmp_path / "checkpoint_selection.json", selection)
    _write_json(
        tmp_path / "mechanism_audits" / "mechanism_audit_report.json",
        mechanism,
    )
    _write_json(
        tmp_path / "training_report.json",
        {
            "status": "GRPO-TRAINING-COMPLETE",
            "completed_collections": 75,
            "completed_optimizer_steps": 300,
            "released_checkpoint_sha256_before": "base-hash",
            "released_checkpoint_sha256_after": "base-hash",
            "evaluation_status": "milestone_validation_and_selection_complete",
            "collection_statistics": {
                "scored_trajectories": 54000,
                "eligible_trajectories": 53990,
                "used_trajectories": 53990,
                "unused_eligible_trajectories": 0,
                "reward_calls": 54000,
                "compute_accounting": {"optimizer_updates": 300},
            },
        },
    )


def test_completed_run_has_compact_three_level_verdict(tmp_path):
    _completed_run(tmp_path)
    summary = build_acceptance_summary(tmp_path)
    assert summary["verdict"]["run_integrity"] == "PASS"
    assert summary["verdict"]["implementation_correctness"] == "PASS"
    assert summary["verdict"]["descriptive_validation_learning_signal"] == "OBSERVED"
    assert summary["verdict"]["best_safe_validation_checkpoint"] == "FOUND"
    assert summary["verdict"]["paper_result_reproduction"].startswith("NOT_ESTABLISHED")
    assert summary["trajectory_accounting"]["eligible_utilization"] == 1.0
    assert summary["validation"]["rows"][1]["composite_delta_vs_0pct"] == pytest.approx(
        0.2
    )
    rendered = format_acceptance_summary(summary)
    assert "Implementation correctness : PASS" in rendered
    assert "Paper result reproduction  : NOT ESTABLISHED" in rendered


def test_missing_eligible_trajectory_fails_implementation_acceptance(tmp_path):
    _completed_run(tmp_path)
    report_path = tmp_path / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["collection_statistics"]["used_trajectories"] -= 1
    _write_json(report_path, report)
    summary = build_acceptance_summary(tmp_path)
    assert summary["verdict"]["implementation_correctness"] == "FAIL"
    assert "all_eligible_trajectories_used" in summary["failed_checks"]
    assert "FAIL blocks implementation acceptance" in format_acceptance_summary(summary)


def test_mechanism_failure_is_expanded_in_compact_output(tmp_path):
    _completed_run(tmp_path)
    mechanism_path = tmp_path / "mechanism_audits" / "mechanism_audit_report.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    mechanism["status"] = "GRPO-MECHANISM-AUDIT-FAIL"
    mechanism["checks"]["nonzero_gradients"] = False
    mechanism["invariant_failures_by_collection"] = {"5": ["update_metrics_finite"]}
    _write_json(mechanism_path, mechanism)
    report_path = tmp_path / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "GRPO-TRAINING-COMPLETE-MECHANISM-AUDIT-FAILED"
    _write_json(report_path, report)

    summary = build_acceptance_summary(tmp_path)
    rendered = format_acceptance_summary(summary)
    assert summary["mechanism"]["failed_checks"] == ["nonzero_gradients"]
    assert "Mechanism failed checks     : nonzero_gradients" in rendered
    assert "Collection 5 failures : update_metrics_finite" in rendered


def test_corrected_v2_audit_supersedes_only_derivative_training_failure(tmp_path):
    _completed_run(tmp_path)
    report_path = tmp_path / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "GRPO-TRAINING-COMPLETE-MECHANISM-AUDIT-FAILED"
    _write_json(report_path, report)
    original_path = tmp_path / "mechanism_audits" / "mechanism_audit_report.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original["status"] = "GRPO-MECHANISM-AUDIT-FAIL"
    original["checks"]["nonzero_gradients"] = False
    _write_json(original_path, original)
    _write_json(
        tmp_path / "mechanism_audits" / "mechanism_audit_report_float32_replay_v2.json",
        {
            "status": "GRPO-MECHANISM-AUDIT-PASS",
            "audit_revision": "float32_eq8_replay_v2",
            "source_report": {
                "original_status": "GRPO-MECHANISM-AUDIT-FAIL",
                "preserved_unmodified": True,
            },
            "checks": {"all_points": True, "float32_replay": True},
            "invariant_failures_by_collection": {},
        },
    )
    summary = build_acceptance_summary(tmp_path)
    assert summary["verdict"]["run_integrity"] == "PASS"
    assert summary["verdict"]["implementation_correctness"] == "PASS"
    assert summary["mechanism"]["audit_revision"] == "float32_eq8_replay_v2"


def test_formal_run_does_not_require_correctness_pilot_mechanism_artifact(tmp_path):
    _completed_run(tmp_path)
    config_path = tmp_path / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run"] = {"mode": "secondary", "collections": 1250}
    config["comparison"] = {"budget_role": "grpo_5000_update_scaleup"}
    config.pop("mechanism_audit")
    _write_json(config_path, config)

    report_path = tmp_path / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["completed_collections"] = 1250
    report["completed_optimizer_steps"] = 5000
    report["mechanism_audit"] = None
    report["collection_statistics"] = {
        "scored_trajectories": 900000,
        "eligible_trajectories": 900000,
        "used_trajectories": 900000,
        "unused_eligible_trajectories": 0,
        "reward_calls": 900000,
        "compute_accounting": {"optimizer_updates": 5000},
    }
    _write_json(report_path, report)
    (tmp_path / "mechanism_audits" / "mechanism_audit_report.json").unlink()

    summary = build_acceptance_summary(tmp_path)
    rendered = format_acceptance_summary(summary)
    assert summary["verdict"]["run_integrity"] == "PASS"
    assert (
        summary["verdict"]["implementation_correctness"]
        == "NOT_REAUDITED_THIS_RUN"
    )
    assert summary["mechanism"]["required_for_this_run"] is False
    assert (
        summary["mechanism"]["status"]
        == "GRPO-MECHANISM-AUDIT-NOT-RUN-NOT-REQUIRED"
    )
    assert "mechanism_audit_not_required_for_run_mode" in summary["checks"]
    assert "intentionally not repeated" in rendered
    assert "paired A/B and official test pending" in rendered


def test_correctness_pilot_still_fails_closed_when_mechanism_is_missing(tmp_path):
    _completed_run(tmp_path)
    config_path = tmp_path / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mechanism_audit"] = {"enabled": True}
    _write_json(config_path, config)
    (tmp_path / "mechanism_audits" / "mechanism_audit_report.json").unlink()

    with pytest.raises(FileNotFoundError):
        build_acceptance_summary(tmp_path)
