"""Offline v2 audit for float32 group-advantage centering residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .math import compute_group_advantages


AUDIT_REVISION = "float32_eq8_replay_v2"
OUTPUT_NAME = "mechanism_audit_report_float32_replay_v2.json"


def _read_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL artifact: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def audit_logged_group_advantages(
    rows: Sequence[Mapping],
    *,
    group_size: int,
    expected_groups: int,
    correction: int,
    epsilon: float,
) -> dict:
    """Replay the exact float32 Eq. (8) path and check conditioned error bounds."""

    grouped: dict[str, list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    complete_geometry = len(grouped) == int(expected_groups) and all(
        len(group) == int(group_size) for group in grouped.values()
    )
    float32_epsilon = float(torch.finfo(torch.float32).eps)
    replay_errors = []
    replay_tolerances = []
    centering_residuals = []
    centering_bounds = []
    std_errors = []
    valid_group_count = 0
    eligible_match = True
    finite = True

    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["candidate_index"]))
        if len(ordered) != int(group_size):
            continue
        rewards = torch.tensor(
            [float(row["reward"]) for row in ordered], dtype=torch.float32
        )
        observed = torch.tensor(
            [float(row["advantage"]) for row in ordered], dtype=torch.float32
        )
        replay = compute_group_advantages(
            rewards[None, :], correction=int(correction), epsilon=float(epsilon)
        )
        expected = replay.advantages[0]
        expected_eligible = replay.eligible_candidates[0]
        observed_eligible = torch.tensor(
            [bool(row["eligible"]) for row in ordered], dtype=torch.bool
        )
        eligible_match = eligible_match and bool(
            torch.equal(expected_eligible, observed_eligible)
        )
        finite = finite and bool(torch.isfinite(observed).all().item())
        replay_error = float((observed - expected).abs().max().item())
        replay_scale = max(1.0, float(expected.abs().max().item()))
        replay_tolerance = 8.0 * float32_epsilon * replay_scale
        replay_errors.append(replay_error)
        replay_tolerances.append(replay_tolerance)

        if bool(replay.valid_groups[0].item()):
            valid_group_count += 1
            reward_std = float(replay.group_std[0].item())
            reward_scale = float(rewards.abs().max().item())
            condition_scale = max(1.0, reward_scale / reward_std)
            # Conservative first-order bound for float32 mean/subtract/divide.
            centering_bound = 8.0 * float32_epsilon * condition_scale
            centering_residuals.append(float(abs(observed.double().mean().item())))
            centering_bounds.append(centering_bound)
            std_errors.append(
                float(abs(observed.double().std(correction=0).item() - 1.0))
            )

    checks = {
        "complete_group_geometry": complete_geometry,
        "logged_advantages_finite": finite,
        "eligible_flags_match_float32_eq8": eligible_match,
        "logged_advantages_match_float32_eq8": bool(replay_errors)
        and all(
            error <= tolerance
            for error, tolerance in zip(replay_errors, replay_tolerances, strict=True)
        ),
        "float32_centering_residual_within_conditioned_bound": bool(centering_residuals)
        and all(
            residual <= bound
            for residual, bound in zip(
                centering_residuals, centering_bounds, strict=True
            )
        ),
        "population_std_within_float32_bound": bool(std_errors)
        and max(std_errors) <= 32.0 * float32_epsilon,
    }
    return {
        "audit_revision": AUDIT_REVISION,
        "passed": all(checks.values()),
        "checks": checks,
        "groups": len(grouped),
        "valid_groups": valid_group_count,
        "max_abs_replay_error": max(replay_errors, default=None),
        "max_replay_tolerance": max(replay_tolerances, default=None),
        "max_abs_group_advantage_mean": max(centering_residuals, default=None),
        "max_conditioned_centering_bound": max(centering_bounds, default=None),
        "max_abs_group_advantage_std_error": max(std_errors, default=None),
        "float32_epsilon": float32_epsilon,
    }


def reaudit_run(
    run_dir: str | Path, *, output: str | Path | None = None
) -> tuple[dict, Path]:
    """Create a provenance-linked v2 report without modifying the v1 artifacts."""

    run_dir = Path(run_dir)
    config = _read_mapping(run_dir / "frozen_config.json")
    source_path = run_dir / "mechanism_audits" / "mechanism_audit_report.json"
    source = _read_mapping(source_path)
    source_checks = source.get("checks")
    if not isinstance(source_checks, Mapping):
        raise ValueError("source mechanism report lacks checks")
    source_failures = source.get("invariant_failures_by_collection", {})
    if not isinstance(source_failures, Mapping):
        raise ValueError("source mechanism report has invalid invariant failures")

    collection = config["collection"]
    group_size = int(collection["group_size"])
    expected_groups = int(
        collection["prompts_per_mini_batch"] * collection["mini_batch_repeats"]
    )
    registered = [
        int(value)
        for value in config["mechanism_audit"]["checkpoint_collections"]
        if int(value) > 0
    ]
    advantage_audits = {}
    corrected_failures = {}
    collection_sources = {}
    for collection_index in registered:
        rollout_path = (
            run_dir
            / "collection_artifacts"
            / (f"rollout_collection_{collection_index:06d}.jsonl")
        )
        rows = _read_jsonl(rollout_path)
        audit = audit_logged_group_advantages(
            rows,
            group_size=group_size,
            expected_groups=expected_groups,
            correction=int(config["advantage"]["std_correction"]),
            epsilon=float(config["advantage"]["epsilon"]),
        )
        key = str(collection_index)
        advantage_audits[key] = audit
        collection_sources[key] = {
            "path": str(rollout_path),
            "sha256": _sha256_file(rollout_path),
            "rows": len(rows),
        }
        failures = [
            str(name)
            for name in source_failures.get(key, [])
            if str(name) != "group_advantages_zero_mean"
        ]
        if not audit["passed"]:
            failures.append("group_advantages_float32_eq8_replay_v2")
        if failures:
            corrected_failures[key] = sorted(set(failures))

    checks = dict(source_checks)
    checks["all_point_invariants_passed"] = not corrected_failures
    checks["float32_eq8_replay_passed_at_all_trained_points"] = all(
        bool(value["passed"]) for value in advantage_audits.values()
    )
    status = (
        "GRPO-MECHANISM-AUDIT-PASS"
        if all(bool(value) for value in checks.values())
        else "GRPO-MECHANISM-AUDIT-FAIL"
    )
    report = {
        "schema_version": 2,
        "status": status,
        "audit_revision": AUDIT_REVISION,
        "purpose": "corrected_float32_eq8_replay_without_training_rerun",
        "source_report": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
            "original_status": source.get("status"),
            "preserved_unmodified": True,
        },
        "correction_scope": {
            "replaced_invariant": "group_advantages_zero_mean",
            "reason": (
                "fixed 1e-5 ideal-zero threshold ignored conditioned float32 "
                "mean-subtraction residual"
            ),
            "new_invariants": [
                "logged_advantages_match_float32_eq8",
                "float32_centering_residual_within_conditioned_bound",
                "population_std_within_float32_bound",
            ],
            "training_or_checkpoint_state_modified": False,
        },
        "checks": checks,
        "invariant_failures_by_collection": corrected_failures,
        "advantage_audits_by_collection": advantage_audits,
        "collection_sources": collection_sources,
        "trends": source.get("trends", []),
    }
    output_path = (
        Path(output)
        if output is not None
        else run_dir / "mechanism_audits" / OUTPUT_NAME
    )
    if output_path.resolve() == source_path.resolve():
        raise ValueError("v2 re-audit must not overwrite the original report")
    _atomic_write_json(output_path, report)
    return report, output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-audit logged GRPO advantages with the exact float32 Eq. (8) path"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, output_path = reaudit_run(args.run_dir, output=args.output)
    print(f"GRPO mechanism re-audit: {report['status']}")
    print(f"Audit revision: {report['audit_revision']}")
    print(
        "Failed checks: "
        + str([name for name, passed in report["checks"].items() if not passed])
    )
    for collection_index, audit in report["advantage_audits_by_collection"].items():
        print(
            f"collection={collection_index} replay_error="
            f"{audit['max_abs_replay_error']:.3e} mean_residual="
            f"{audit['max_abs_group_advantage_mean']:.3e} bound="
            f"{audit['max_conditioned_centering_bound']:.3e} passed={audit['passed']}"
        )
    print(f"Report: {output_path}")
    if report["status"] != "GRPO-MECHANISM-AUDIT-PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
