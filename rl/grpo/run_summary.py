"""Compact, offline acceptance summary for a completed FlowSE-GRPO run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _read_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _selection_artifact(run_dir: Path, training_report: Mapping) -> dict:
    path = run_dir / "checkpoint_selection.json"
    if path.is_file():
        return _read_mapping(path)
    value = training_report.get("checkpoint_selection")
    if not isinstance(value, Mapping):
        raise FileNotFoundError(path)
    return dict(value)


def _mechanism_artifact(
    run_dir: Path, training_report: Mapping, *, required: bool
) -> dict:
    corrected_path = (
        run_dir / "mechanism_audits" / "mechanism_audit_report_float32_replay_v2.json"
    )
    if corrected_path.is_file():
        return _read_mapping(corrected_path)
    path = run_dir / "mechanism_audits" / "mechanism_audit_report.json"
    if path.is_file():
        return _read_mapping(path)
    value = training_report.get("mechanism_audit")
    if isinstance(value, Mapping):
        return dict(value)
    if required:
        raise FileNotFoundError(path)
    return {
        "status": "GRPO-MECHANISM-AUDIT-NOT-RUN-NOT-REQUIRED",
        "audit_revision": None,
        "checks": {},
        "invariant_failures_by_collection": {},
        "not_required_reason": (
            "mechanism audit is reserved for the isolated correctness pilot and "
            "is intentionally disabled for formal/scale-up training"
        ),
    }


def _validation_rows(selection: Mapping) -> list[dict]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("checkpoint selection contains no candidate list")
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("checkpoint selection candidate must be a mapping")
        rows.append(
            {
                "percentage": int(candidate["percentage"]),
                "collection_index": int(candidate["collection_index"]),
                "composite_mean": float(candidate["composite_mean"]),
                "ovrl_mean": float(candidate["ovrl_mean"]),
            }
        )
    rows.sort(key=lambda value: value["percentage"])
    if not rows or rows[0]["percentage"] != 0:
        raise ValueError("checkpoint selection lacks the registered 0% state")
    zero = rows[0]
    for row in rows:
        row["composite_delta_vs_0pct"] = float(
            row["composite_mean"] - zero["composite_mean"]
        )
        row["ovrl_delta_vs_0pct"] = float(row["ovrl_mean"] - zero["ovrl_mean"])
    return rows


def build_acceptance_summary(run_dir: str | Path) -> dict:
    """Validate completed artifacts and separate mechanism from outcome claims."""

    run_dir = Path(run_dir)
    config = _read_mapping(run_dir / "frozen_config.json")
    training = _read_mapping(run_dir / "training_report.json")
    selection = _selection_artifact(run_dir, training)
    mechanism_required = isinstance(config.get("mechanism_audit"), Mapping)
    mechanism = _mechanism_artifact(
        run_dir, training, required=mechanism_required
    )

    run = config["run"]
    collection = config["collection"]
    expected_collections = int(run["collections"])
    updates_per_collection = int(collection["optimizer_updates"])
    expected_optimizer_steps = expected_collections * updates_per_collection
    candidates_per_collection = int(
        collection["prompts_per_mini_batch"]
        * collection["mini_batch_repeats"]
        * collection["group_size"]
    )
    expected_scored = expected_collections * candidates_per_collection

    statistics = training.get("collection_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("training report lacks collection statistics")
    compute = statistics.get("compute_accounting")
    if not isinstance(compute, Mapping):
        raise ValueError("collection statistics lack compute accounting")
    mechanism_checks = mechanism.get("checks")
    if not isinstance(mechanism_checks, Mapping):
        raise ValueError("mechanism report lacks checks")
    if mechanism_required and not mechanism_checks:
        raise ValueError("required mechanism report contains no checks")

    scored = int(statistics.get("scored_trajectories", -1))
    eligible = int(statistics.get("eligible_trajectories", -1))
    used = int(statistics.get("used_trajectories", -1))
    unused_eligible = int(statistics.get("unused_eligible_trajectories", -1))
    reward_calls = int(statistics.get("reward_calls", -1))
    accounted_updates = int(compute.get("optimizer_updates", -1))
    released_hash_before = training.get("released_checkpoint_sha256_before")
    released_hash_after = training.get("released_checkpoint_sha256_after")
    mechanism_passed = mechanism.get("status") == "GRPO-MECHANISM-AUDIT-PASS"
    corrected_audit = mechanism.get("audit_revision") == "float32_eq8_replay_v2"
    training_status = training.get("status")
    training_status_accepted = training_status == "GRPO-TRAINING-COMPLETE" or (
        training_status == "GRPO-TRAINING-COMPLETE-MECHANISM-AUDIT-FAILED"
        and corrected_audit
        and mechanism_passed
    )
    checks = {
        "training_completion_status_accepted": training_status_accepted,
        "all_collections_complete": int(training.get("completed_collections", -1))
        == expected_collections,
        "all_optimizer_steps_complete": int(
            training.get("completed_optimizer_steps", -1)
        )
        == expected_optimizer_steps,
        "released_base_unchanged": bool(released_hash_before)
        and released_hash_before == released_hash_after,
        "milestone_validation_complete": training.get("evaluation_status")
        == "milestone_validation_and_selection_complete",
        "checkpoint_selection_complete": selection.get("status")
        == "CHECKPOINT-SELECTION-COMPLETE",
        "all_candidates_scored": scored == expected_scored,
        "reward_calls_match_scored_candidates": reward_calls == scored,
        "all_eligible_trajectories_used": eligible >= 0
        and used == eligible
        and unused_eligible == 0,
        "optimizer_updates_accounted": accounted_updates == expected_optimizer_steps,
    }
    if mechanism_required:
        checks["mechanism_audit_passed"] = mechanism_passed
        checks["all_mechanism_checks_passed"] = all(
            bool(value) for value in mechanism_checks.values()
        )
    else:
        checks["mechanism_audit_not_required_for_run_mode"] = (
            mechanism.get("status")
            == "GRPO-MECHANISM-AUDIT-NOT-RUN-NOT-REQUIRED"
        )
    run_integrity_passed = all(checks.values())
    implementation_verdict = (
        "PASS" if mechanism_required and run_integrity_passed else (
            "FAIL" if mechanism_required else "NOT_REAUDITED_THIS_RUN"
        )
    )

    validation_rows = _validation_rows(selection)
    nonzero_rows = [row for row in validation_rows if row["percentage"] > 0]
    best_nonzero = (
        max(
            nonzero_rows,
            key=lambda row: (
                row["composite_mean"],
                row["ovrl_mean"],
                -row["percentage"],
            ),
        )
        if nonzero_rows
        else None
    )
    descriptive_gain = bool(
        best_nonzero is not None
        and float(best_nonzero["composite_delta_vs_0pct"]) > 0.0
    )
    best_safe = selection.get("best_safe_checkpoint")
    best_safe_found = isinstance(best_safe, Mapping)

    budget_role = str(config.get("comparison", {}).get("budget_role", ""))
    correctness_pilot = budget_role == "implementation_correctness_pilot_2gpu"
    paper_reproduction_status = (
        "NOT_ESTABLISHED_BY_CORRECTNESS_PILOT"
        if correctness_pilot
        else "NOT_ESTABLISHED_UNTIL_PAIRED_AB_AND_OFFICIAL_TEST"
    )
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "run_mode": str(run["mode"]),
        "budget_role": budget_role,
        "verdict": {
            "run_integrity": "PASS" if run_integrity_passed else "FAIL",
            "implementation_correctness": implementation_verdict,
            "descriptive_validation_learning_signal": (
                "OBSERVED" if descriptive_gain else "NOT_OBSERVED"
            ),
            "best_safe_validation_checkpoint": (
                "FOUND" if best_safe_found else "NOT_FOUND"
            ),
            "paper_result_reproduction": paper_reproduction_status,
        },
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "training": {
            "status": training.get("status"),
            "completed_collections": int(training.get("completed_collections", -1)),
            "expected_collections": expected_collections,
            "completed_optimizer_steps": int(
                training.get("completed_optimizer_steps", -1)
            ),
            "expected_optimizer_steps": expected_optimizer_steps,
        },
        "trajectory_accounting": {
            "candidates_per_collection": candidates_per_collection,
            "expected_scored_trajectories": expected_scored,
            "scored_trajectories": scored,
            "eligible_trajectories": eligible,
            "used_trajectories": used,
            "unused_eligible_trajectories": unused_eligible,
            "eligible_utilization": float(used / eligible) if eligible > 0 else None,
            "reward_calls": reward_calls,
        },
        "mechanism": {
            "required_for_this_run": mechanism_required,
            "status": mechanism.get("status"),
            "audit_revision": (
                mechanism.get("audit_revision", "original_v1")
                if mechanism_required
                else None
            ),
            "not_required_reason": mechanism.get("not_required_reason"),
            "original_status": mechanism.get("source_report", {}).get(
                "original_status"
            ),
            "checks": dict(mechanism_checks),
            "failed_checks": sorted(
                name for name, passed in mechanism_checks.items() if not bool(passed)
            ),
            "invariant_failures_by_collection": mechanism.get(
                "invariant_failures_by_collection", {}
            ),
        },
        "validation": {
            "rows": validation_rows,
            "comparison_checkpoint": selection.get("comparison_checkpoint"),
            "fixed_budget_endpoint": selection.get("fixed_budget_endpoint"),
            "best_nonzero_checkpoint": best_nonzero,
            "best_safe_checkpoint": best_safe,
            "interpretation": (
                "descriptive validation change only; not an official-test or "
                "paper-reproduction claim"
            ),
        },
    }
    return summary


def _format_float(value) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"


def format_acceptance_summary(summary: Mapping) -> str:
    verdict = summary["verdict"]
    training = summary["training"]
    trajectories = summary["trajectory_accounting"]
    mechanism = summary["mechanism"]
    validation = summary["validation"]
    utilization = trajectories["eligible_utilization"]
    utilization_text = "n/a" if utilization is None else f"{utilization:.2%}"
    lines = [
        "GRPO ACCEPTANCE SUMMARY",
        "=" * 72,
        f"Run integrity              : {verdict['run_integrity']}",
        f"Implementation correctness : {verdict['implementation_correctness']}",
        "Validation learning signal : "
        f"{verdict['descriptive_validation_learning_signal']} (descriptive)",
        f"Best-safe checkpoint       : {verdict['best_safe_validation_checkpoint']}",
        "Paper result reproduction  : "
        + (
            "NOT ESTABLISHED (correctness pilot only)"
            if summary["verdict"]["paper_result_reproduction"]
            == "NOT_ESTABLISHED_BY_CORRECTNESS_PILOT"
            else "NOT ESTABLISHED (paired A/B and official test pending)"
        ),
        "-" * 72,
        "Training                    : "
        f"{training['completed_collections']}/{training['expected_collections']} "
        "collections, "
        f"{training['completed_optimizer_steps']}/{training['expected_optimizer_steps']} "
        "optimizer steps",
        "Trajectory use              : "
        f"scored={trajectories['scored_trajectories']} "
        f"eligible={trajectories['eligible_trajectories']} "
        f"used={trajectories['used_trajectories']} "
        f"utilization={utilization_text}",
        f"Mechanism audit             : {mechanism['status']}",
    ]
    if mechanism.get("audit_revision") is not None:
        lines.append(f"Mechanism audit revision    : {mechanism['audit_revision']}")
    if mechanism.get("not_required_reason") is not None:
        lines.append("Mechanism audit scope       : intentionally not repeated")
    if mechanism.get("original_status") is not None:
        lines.append(
            f"Original mechanism status   : {mechanism['original_status']} (preserved)"
        )
    mechanism_failed = mechanism.get("failed_checks", [])
    if mechanism_failed:
        lines.append("Mechanism failed checks     : " + ", ".join(mechanism_failed))
    invariant_failures = mechanism.get("invariant_failures_by_collection", {})
    if invariant_failures:
        for collection_index, names in sorted(
            invariant_failures.items(), key=lambda value: int(value[0])
        ):
            lines.append(
                f"Collection {collection_index} failures : " + ", ".join(names)
            )
    failed = summary.get("failed_checks", [])
    if failed:
        lines.append("Failed checks               : " + ", ".join(failed))
    lines.extend(
        [
            "-" * 72,
            "Validation milestones (delta is relative to the registered 0% state)",
            "budget  collection  composite       delta        OVRL        delta",
        ]
    )
    for row in validation["rows"]:
        lines.append(
            f"{row['percentage']:>3}%    {row['collection_index']:>6}      "
            f"{row['composite_mean']:>+10.6f}  "
            f"{_format_float(row['composite_delta_vs_0pct']):>10}  "
            f"{row['ovrl_mean']:>+10.6f}  "
            f"{_format_float(row['ovrl_delta_vs_0pct']):>10}"
        )
    comparison = validation.get("comparison_checkpoint")
    endpoint = validation.get("fixed_budget_endpoint")
    best_nonzero = validation.get("best_nonzero_checkpoint")
    best_safe = validation.get("best_safe_checkpoint")
    if isinstance(comparison, Mapping):
        lines.append(f"Comparison checkpoint       : {comparison['percentage']}%")
    if isinstance(endpoint, Mapping):
        endpoint_row = next(
            (
                row
                for row in validation["rows"]
                if row["percentage"] == int(endpoint["percentage"])
            ),
            None,
        )
        if endpoint_row is not None:
            lines.append(
                "100% endpoint delta       : composite="
                f"{_format_float(endpoint_row['composite_delta_vs_0pct'])}, "
                f"OVRL={_format_float(endpoint_row['ovrl_delta_vs_0pct'])}"
            )
    if isinstance(best_nonzero, Mapping):
        lines.append(
            f"Best nonzero delta          : {best_nonzero['percentage']}% composite="
            f"{_format_float(best_nonzero['composite_delta_vs_0pct'])}"
        )
    lines.append(
        "Best-safe checkpoint       : "
        + (f"{best_safe['percentage']}%" if isinstance(best_safe, Mapping) else "none")
    )
    lines.append("-" * 72)
    if (
        verdict["implementation_correctness"] == "PASS"
        and mechanism.get("required_for_this_run")
    ):
        lines.append(
            "Interpretation: PASS establishes the implemented 300-step GRPO mechanism."
        )
    elif verdict["run_integrity"] == "PASS":
        lines.extend(
            [
                "Interpretation: PASS establishes completion and integrity of this",
                "formal/scale-up run. The mechanism audit was intentionally not",
                "repeated; cite the separate correctness-pilot audit for mechanism",
                "implementation evidence.",
            ]
        )
    else:
        lines.extend(
            [
                "Interpretation: FAIL blocks implementation acceptance until the",
                "reported mechanism failure is explained or corrected.",
                "Validation gains do not override a mechanism-audit failure.",
            ]
        )
    lines.append(
        "Paper reproduction still requires the paired A/B selection and official test."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a compact acceptance summary for a completed GRPO run"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    summary = build_acceptance_summary(args.run_dir)
    print(format_acceptance_summary(summary))
    if not args.no_write:
        output = args.output or args.run_dir / "acceptance_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Summary JSON: {output}")
    mechanism_required = bool(summary["mechanism"]["required_for_this_run"])
    verdict = summary["verdict"]
    if verdict["run_integrity"] != "PASS" or (
        mechanism_required and verdict["implementation_correctness"] != "PASS"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
