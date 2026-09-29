"""Method-neutral registered milestone checkpoint selection."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence


COMMON_MILESTONES = (0, 20, 40, 60, 80, 100)


def select_registered_milestones(
    *,
    policy_kind: str,
    milestone_reports: Sequence[Mapping],
    checkpoint_sources: Mapping[int, Mapping],
    base_report: Mapping | None = None,
    safety_config: Mapping | None = None,
    best_safe_selector: Callable | None = None,
) -> dict:
    """Apply the shared composite/OVRL/earlier validation-selection rule."""

    observed = sorted(int(report["percentage"]) for report in milestone_reports)
    if observed != list(COMMON_MILESTONES):
        raise ValueError(
            "selection must use exactly the six registered milestones: "
            f"observed={observed}"
        )
    rows = []
    report_by_percentage = {}
    for report in milestone_reports:
        percentage = int(report["percentage"])
        report_by_percentage[percentage] = report
        source = dict(checkpoint_sources[percentage])
        row = {
            "percentage": percentage,
            "budget_position": int(
                report.get(
                    "optimizer_step", report.get("collection_index", percentage)
                )
            ),
            "checkpoint": str(source["checkpoint"]),
            "checkpoint_sha256": str(source["checkpoint_sha256"]),
            "payload_key": str(source["payload_key"]),
            "composite_mean": float(
                report["summary"]["flowse_grpo_composite_reward"]["mean"]
            ),
            "ovrl_mean": float(report["summary"]["dnsmos_ovrl"]["mean"]),
        }
        if "collection_index" in report:
            row["collection_index"] = int(report["collection_index"])
        if report.get("milestone_metadata") is not None:
            row["milestone_metadata"] = dict(report["milestone_metadata"])
        rows.append(row)
    comparison = max(
        rows,
        key=lambda row: (
            row["composite_mean"],
            row["ovrl_mean"],
            -row["percentage"],
        ),
    )
    best_safe = None
    safety_candidates = []
    if base_report is not None or safety_config is not None:
        if base_report is None or safety_config is None:
            raise ValueError("best-safe selection needs both base report and config")
        if best_safe_selector is None:
            raise ValueError("best-safe selection needs an explicit neutral callback")
        normalized = [report_by_percentage[value] for value in COMMON_MILESTONES]
        selected, safety_candidates = best_safe_selector(
            base_report, normalized, config=safety_config
        )
        if selected is not None:
            selected_row = next(
                row
                for row in rows
                if int(row["percentage"]) == int(selected["percentage"])
            )
            best_safe = {**selected, **selected_row}
    return {
        "schema_version": 1,
        "status": "CHECKPOINT-SELECTION-COMPLETE",
        "policy_kind": policy_kind,
        "selection_opportunities": list(COMMON_MILESTONES),
        "selection_input": "validation_only",
        "comparison_rule": (
            "max validation composite mean, then OVRL mean, then earlier budget percentage"
        ),
        "comparison_checkpoint": comparison,
        "best_safe_checkpoint": best_safe,
        "fixed_budget_endpoint": next(
            row for row in rows if int(row["percentage"]) == 100
        ),
        "candidates": rows,
        "best_safe_candidates": safety_candidates,
        "official_test_selection_forbidden": True,
    }
