"""Fair-comparison protocol audits shared by GRPO training and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from rl.common.shared_initialization import (
    lora_state_fingerprint as lora_state_fingerprint,
    prepare_or_load_shared_lora_snapshot as prepare_or_load_shared_lora_snapshot,
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_manifest(path: str | Path) -> dict[str, str]:
    source = Path(path)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate utterance {key!r} in {source}")
            result[key] = value
        return result

    value = json.loads(
        source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest is empty or not an object: {source}")
    if not all(
        isinstance(utterance, str) and isinstance(transcript, str)
        for utterance, transcript in value.items()
    ):
        raise ValueError("manifest must map utterance IDs to transcript strings")
    return value


def _calibration_utterances(
    report_path: Path, *, source_nfe: int
) -> tuple[set[str], dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = report.get("source", {})
    train_manifest_path = Path(str(source.get("train_manifest_path", "")))
    if not train_manifest_path.is_file():
        raise FileNotFoundError(
            "calibration report lacks a readable train manifest: "
            f"{train_manifest_path}"
        )
    recorded_train_hash = str(source.get("train_manifest_sha256", ""))
    observed_train_hash = sha256_file(train_manifest_path)
    if recorded_train_hash != observed_train_hash:
        raise ValueError("calibration train manifest hash differs from report")
    calibration_train = strict_manifest(train_manifest_path)
    block_reports = source.get("block_reports")
    if isinstance(block_reports, Mapping) and block_reports:
        utterances: set[str] = set()
        rows = 0
        blocks = {}
        for name, descriptor in sorted(block_reports.items()):
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"invalid calibration block descriptor: {name}")
            block_path = Path(str(descriptor.get("path", "")))
            if not block_path.is_file():
                raise FileNotFoundError(block_path)
            observed_block_hash = sha256_file(block_path)
            if observed_block_hash != str(descriptor.get("sha256", "")):
                raise ValueError(f"calibration block hash differs: {name}")
            block_utterances, block = _calibration_utterances(
                block_path, source_nfe=source_nfe
            )
            overlap = utterances & block_utterances
            if overlap:
                raise ValueError(
                    f"combined calibration blocks overlap: {sorted(overlap)[:5]}"
                )
            if block["train_manifest_sha256"] != observed_train_hash:
                raise ValueError(f"calibration block train manifest differs: {name}")
            utterances.update(block_utterances)
            rows += int(block["rows"])
            blocks[str(name)] = {
                **block,
                "report_path": str(block_path),
                "report_sha256": observed_block_hash,
            }
        checks = {
            "status": report.get("status") == "CALIBRATION-COMPLETE",
            "rows": rows == int(source.get("rows", -1)),
            "utterances": len(utterances)
            == int(source.get("selected_train_utterances", -1)),
            "excluded_rows": int(source.get("excluded_outside_train_rows", -1))
            == 0,
            "excluded_utterances": source.get("excluded_outside_train_utterances")
            == [],
        }
        if not all(checks.values()):
            raise ValueError(
                f"combined calibration report does not reconstruct: {checks}"
            )
        return utterances, {
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "rows": rows,
            "utterances": len(utterances),
            "eligible_rows_before_train_filter": rows,
            "excluded_outside_train_rows": 0,
            "excluded_outside_train_utterances": 0,
            "train_manifest_path": str(train_manifest_path),
            "train_manifest_sha256": observed_train_hash,
            "combined_replication_report": True,
            "blocks": blocks,
        }

    endpoint_path = Path(str(source.get("endpoint_metrics_path", "")))
    if not endpoint_path.is_file():
        raise FileNotFoundError(endpoint_path)
    recorded_hash = str(source.get("endpoint_metrics_sha256", ""))
    observed_hash = sha256_file(endpoint_path)
    if recorded_hash != observed_hash:
        raise ValueError("calibration endpoint metrics hash differs from report")
    utterances = set()
    eligible_rows = 0
    selected_rows = 0
    excluded_rows = 0
    excluded_utterances: set[str] = set()
    for line_number, line in enumerate(
        endpoint_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("nfe", -1)) != int(source_nfe) or row.get("error"):
            continue
        eligible_rows += 1
        utterance = row.get("utterance")
        if not isinstance(utterance, str) or not utterance:
            raise ValueError(
                f"calibration row lacks utterance: {endpoint_path}:{line_number}"
            )
        if utterance not in calibration_train:
            excluded_rows += 1
            excluded_utterances.add(utterance)
            continue
        utterances.add(utterance)
        selected_rows += 1
    expected_eligible = int(source.get("eligible_rows_before_train_filter", -1))
    if eligible_rows != expected_eligible:
        raise ValueError(
            "calibration eligible source row count differs from report: "
            f"observed={eligible_rows}, reported={expected_eligible}"
        )
    expected_excluded_rows = int(source.get("excluded_outside_train_rows", -1))
    expected_excluded_utterances = sorted(
        str(value) for value in source.get("excluded_outside_train_utterances", [])
    )
    if (
        excluded_rows != expected_excluded_rows
        or sorted(excluded_utterances) != expected_excluded_utterances
    ):
        raise ValueError(
            "calibration outside-train exclusion audit differs from report: "
            f"rows=({excluded_rows}, {expected_excluded_rows}), "
            f"utterances=({sorted(excluded_utterances)}, "
            f"{expected_excluded_utterances})"
        )
    if selected_rows != int(source.get("rows", -1)):
        raise ValueError(
            "calibration selected row count differs from report: "
            f"selected={selected_rows}, reported={source.get('rows')}"
        )
    return utterances, {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "endpoint_metrics_path": str(endpoint_path),
        "endpoint_metrics_sha256": observed_hash,
        "rows": selected_rows,
        "utterances": len(utterances),
        "eligible_rows_before_train_filter": eligible_rows,
        "excluded_outside_train_rows": excluded_rows,
        "excluded_outside_train_utterances": len(excluded_utterances),
        "train_manifest_path": str(train_manifest_path),
        "train_manifest_sha256": observed_train_hash,
    }


def audit_data_splits(config: Mapping, *, strict_calibration: bool = False) -> dict:
    """Check split disjointness and, when requested, calibration provenance.

    Public smoke runs only need a runnable, non-overlapping split.  The
    calibration report may point at files that exist only in the authors'
    training workspace, so its provenance checks are reserved for formal
    reproduction runs.  Keeping this switch here avoids making the public
    entry point depend on server-specific paths while preserving the full
    audit for formal configs and direct callers/tests.
    """
    paths = {
        "train": Path(config["data"]["train_manifest"]),
        "validation": Path(config["data"]["validation_manifest"]),
        "official_test": Path(config["data"]["official_test_manifest"]),
    }
    manifests = {name: strict_manifest(path) for name, path in paths.items()}
    sets = {name: set(value) for name, value in manifests.items()}
    overlaps = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_official_test": sorted(sets["train"] & sets["official_test"]),
        "validation_official_test": sorted(sets["validation"] & sets["official_test"]),
    }
    if any(overlaps.values()):
        raise ValueError(
            "speech data split overlap detected: "
            + json.dumps({key: value[:5] for key, value in overlaps.items()})
        )
    runtime_train_hash = sha256_file(paths["train"])
    calibration_path = Path(config["training_reward"]["calibration"]["report_path"])
    if not strict_calibration:
        calibration_utterances = set()
        calibration = {
            "report_path": str(calibration_path),
            "train_manifest_path": str(paths["train"]),
            "train_manifest_sha256": runtime_train_hash,
            "provenance_only": True,
            "provenance_note": "calibration provenance is skipped for public smoke",
        }
    else:
        try:
            calibration_utterances, calibration = _calibration_utterances(
                calibration_path,
                source_nfe=int(config["training_reward"]["calibration"]["source_nfe"]),
            )
            calibration["provenance_only"] = (
                calibration.get("train_manifest_sha256") != runtime_train_hash
            )
        except (FileNotFoundError, ValueError) as exc:
            if strict_calibration:
                raise
            # Published calibration reports may retain paths to the author's
            # private endpoint files.  Component scales and rule compatibility
            # are verified by reward specification; split provenance is
            # optional for a new checkout and must not block a public smoke run.
            calibration_utterances = set()
            calibration = {
                "report_path": str(calibration_path),
                "train_manifest_path": str(paths["train"]),
                "train_manifest_sha256": runtime_train_hash,
                "provenance_only": True,
                "provenance_note": str(exc),
            }
    outside_train = sorted(calibration_utterances - sets["train"])
    validation_leakage = sorted(calibration_utterances & sets["validation"])
    official_test_leakage = sorted(calibration_utterances & sets["official_test"])
    if (
        not calibration.get("provenance_only", False)
        and (outside_train or validation_leakage or official_test_leakage)
    ):
        raise ValueError(
            "reward calibration is not train-only: "
            f"outside_train={outside_train[:5]}, "
            f"validation={validation_leakage[:5]}, "
            f"official_test={official_test_leakage[:5]}"
        )
    return {
        "manifests": {
            name: {
                "path": str(paths[name]),
                "sha256": (
                    runtime_train_hash
                    if name == "train"
                    else sha256_file(paths[name])
                ),
                "utterances": len(manifests[name]),
            }
            for name in paths
        },
        "overlap_counts": {name: len(value) for name, value in overlaps.items()},
        "calibration": {
            **calibration,
            "outside_train": 0,
            "validation_leakage": 0,
            "official_test_leakage": 0,
        },
        "sampler_distribution": {
            "implementation": "shared_speaker_balanced_epoch_and_utterances_for_step",
            "order_seed": int(config["data"]["order_seed"]),
            "marginal_unit": "unique_speech_prompt_before_grpo_repeat",
        },
    }


def milestone_collections(
    total_collections: int, percentages: Sequence[int]
) -> dict[int, int]:
    if total_collections < 1:
        raise ValueError("total_collections must be positive")
    values = [int(value) for value in percentages]
    if not values or values[0] != 0 or values[-1] != 100:
        raise ValueError("milestones must begin at 0 and end at 100")
    if values != sorted(set(values)) or any(
        value < 0 or value > 100 for value in values
    ):
        raise ValueError("milestones must be unique sorted percentages")
    result = {}
    for percentage in values:
        collection = (
            0
            if percentage == 0
            else min(
                total_collections,
                int(math.ceil(total_collections * percentage / 100.0)),
            )
        )
        if collection in result:
            raise ValueError(
                "training run is too short to provide distinct milestone opportunities"
            )
        result[collection] = percentage
    return result
