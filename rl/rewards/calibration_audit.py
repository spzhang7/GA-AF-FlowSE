"""Audit two disjoint frozen composite-reward calibration blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


COMPONENTS = ("dnsmos", "speaker", "speechbertscore")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_vector(row: Mapping) -> np.ndarray:
    return np.asarray(
        [
            float(row["dnsmos_ovrl"]) / 4.0,
            float(row["eres2net_speaker_similarity"]),
            float(row["speechbertscore"]),
        ],
        dtype=np.float64,
    )


def _component_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(COMPONENTS)}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation requires equal nontrivial vectors")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator <= 1.0e-12:
        return 1.0 if np.allclose(left, right, atol=1.0e-12, rtol=0.0) else 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def _rewards(values: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[None, :] / stds[None, :], axis=1)


def _af_advantages(rewards: np.ndarray, *, clip: float) -> np.ndarray:
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(centered**2)))
    if scale <= 1.0e-12:
        raise ValueError("calibration sensitivity batch has zero reward scale")
    return np.clip(centered / (scale + 1.0e-6), -clip, clip)


def _cluster_bootstrap(
    clusters: list[np.ndarray], *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(clusters) < 2 or replicates < 100:
        raise ValueError("cluster bootstrap requires clusters and >=100 replicates")
    counts = np.asarray([cluster.shape[0] for cluster in clusters], dtype=np.float64)
    sums = np.stack([cluster.sum(axis=0) for cluster in clusters])
    squares = np.stack([(cluster**2).sum(axis=0) for cluster in clusters])
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, len(COMPONENTS)), dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        count = float(counts[sampled].sum())
        total = sums[sampled].sum(axis=0)
        total_squares = squares[sampled].sum(axis=0)
        variance = np.maximum(total_squares / count - (total / count) ** 2, 0.0)
        estimates[replicate] = np.sqrt(variance)
    low, high = np.percentile(estimates, [2.5, 97.5], axis=0)
    return low, high, estimates


def _group_conditions(rows_by_block: Mapping[str, list[dict]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for block, rows in rows_by_block.items():
        for row in rows:
            grouped[f"{block}|{row['utterance']}"] .append(row)
    result = {}
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
        result[key] = np.stack([_component_vector(row) for row in ordered])
    return result


def summarize_replication_audit(
    rows_by_block: Mapping[str, list[dict]],
    *,
    speaker_by_utterance: Mapping[str, str],
    candidates_per_condition: int,
    bootstrap_config: Mapping,
    replication_config: Mapping,
    sensitivity_config: Mapping,
    weights: Mapping[str, float],
) -> dict:
    if len(rows_by_block) != 2:
        raise ValueError("replication audit requires exactly two blocks")
    block_names = list(rows_by_block)
    condition_values = _group_conditions(rows_by_block)
    for key, values in condition_values.items():
        if values.shape != (candidates_per_condition, len(COMPONENTS)):
            raise ValueError(f"incomplete candidate group: {key}")

    utterances_by_block = {
        block: {str(row["utterance"]) for row in rows}
        for block, rows in rows_by_block.items()
    }
    overlap = utterances_by_block[block_names[0]] & utterances_by_block[block_names[1]]
    if overlap:
        raise ValueError(f"calibration blocks overlap: {sorted(overlap)[:5]}")
    missing_speakers = sorted(
        {
            utterance
            for utterances in utterances_by_block.values()
            for utterance in utterances
            if utterance not in speaker_by_utterance
        }
    )
    if missing_speakers:
        raise ValueError(f"calibration utterances lack speaker metadata: {missing_speakers[:5]}")

    block_arrays = {
        block: np.concatenate(
            [
                condition_values[f"{block}|{utterance}"]
                for utterance in sorted(utterances_by_block[block])
            ],
            axis=0,
        )
        for block in block_names
    }
    block_stds = {
        block: np.std(values, axis=0, ddof=0) for block, values in block_arrays.items()
    }
    combined = np.concatenate(list(block_arrays.values()), axis=0)
    combined_stds = np.std(combined, axis=0, ddof=0)
    relative_block_difference = np.abs(
        block_stds[block_names[0]] - block_stds[block_names[1]]
    ) / combined_stds

    condition_clusters = list(condition_values.values())
    speaker_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, values in condition_values.items():
        _, utterance = key.split("|", 1)
        speaker_groups[str(speaker_by_utterance[utterance])].append(values)
    speaker_clusters = [np.concatenate(groups, axis=0) for groups in speaker_groups.values()]

    bootstrap_replicates = int(bootstrap_config["replicates"])
    bootstrap_seed = int(bootstrap_config["seed"])
    condition_low, condition_high, _ = _cluster_bootstrap(
        condition_clusters,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    speaker_low, speaker_high, _ = _cluster_bootstrap(
        speaker_clusters,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
    )
    condition_width = (condition_high - condition_low) / combined_stds
    speaker_width = (speaker_high - speaker_low) / combined_stds

    weights_array = np.asarray([float(weights[name]) for name in COMPONENTS])
    left_stds = block_stds[block_names[0]]
    right_stds = block_stds[block_names[1]]
    reward_spearman = []
    top_bottom_agreement = []
    reward_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, values in condition_values.items():
        left = _rewards(values, left_stds, weights_array)
        right = _rewards(values, right_stds, weights_array)
        reward_pairs[key] = (left, right)
        reward_spearman.append(
            _correlation(_average_ranks(left), _average_ranks(right))
        )
        left_order = np.argsort(left)
        right_order = np.argsort(right)
        top_bottom_agreement.append(
            set(left_order[:2]) == set(right_order[:2])
            and set(left_order[-2:]) == set(right_order[-2:])
        )

    rng = np.random.default_rng(int(sensitivity_config["seed"]))
    condition_keys = sorted(condition_values)
    conditions_per_batch = int(sensitivity_config["conditions_per_af_batch"])
    if conditions_per_batch > len(condition_keys):
        raise ValueError("AF sensitivity batch is larger than calibration conditions")
    advantage_correlations = []
    advantage_sign_agreements = []
    advantage_mean_abs_differences = []
    for _ in range(int(sensitivity_config["af_batch_replicates"])):
        selected = rng.choice(
            len(condition_keys), size=conditions_per_batch, replace=False
        )
        left_rewards = np.stack([reward_pairs[condition_keys[index]][0] for index in selected])
        right_rewards = np.stack([reward_pairs[condition_keys[index]][1] for index in selected])
        left_advantage = _af_advantages(
            left_rewards, clip=float(sensitivity_config["advantage_clip"])
        )
        right_advantage = _af_advantages(
            right_rewards, clip=float(sensitivity_config["advantage_clip"])
        )
        advantage_correlations.append(
            _correlation(left_advantage.ravel(), right_advantage.ravel())
        )
        advantage_sign_agreements.append(
            float(np.mean(np.sign(left_advantage) == np.sign(right_advantage)))
        )
        advantage_mean_abs_differences.append(
            float(np.mean(np.abs(left_advantage - right_advantage)))
        )

    reward_spearman_array = np.asarray(reward_spearman)
    advantage_correlations_array = np.asarray(advantage_correlations)
    checks = {
        "blocks_are_disjoint": not overlap,
        "block_component_stds_equivalent": bool(
            np.all(
                relative_block_difference
                <= float(replication_config["max_block_relative_std_difference"])
            )
        ),
        "condition_bootstrap_precision": bool(
            np.all(
                condition_width
                <= float(bootstrap_config["max_condition_relative_ci_width"])
            )
        ),
        "speaker_bootstrap_precision": bool(
            np.all(
                speaker_width
                <= float(bootstrap_config["max_speaker_relative_ci_width"])
            )
        ),
        "reward_rank_sensitivity": bool(
            reward_spearman_array.mean()
            >= float(sensitivity_config["min_reward_spearman_mean"])
            and np.percentile(reward_spearman_array, 5)
            >= float(sensitivity_config["min_reward_spearman_p05"])
            and np.mean(top_bottom_agreement)
            >= float(sensitivity_config["min_top2_bottom2_exact_agreement"])
        ),
        "af_advantage_sensitivity": bool(
            advantage_correlations_array.mean()
            >= float(sensitivity_config["min_advantage_pearson_mean"])
            and np.percentile(advantage_correlations_array, 5)
            >= float(sensitivity_config["min_advantage_pearson_p05"])
            and np.mean(advantage_sign_agreements)
            >= float(sensitivity_config["min_advantage_sign_agreement"])
        ),
    }
    return {
        "audit_status": "AUDIT-PASS" if all(checks.values()) else "AUDIT-FAIL",
        "checks": checks,
        "conditions": len(condition_values),
        "speakers": len(speaker_groups),
        "rows": int(combined.shape[0]),
        "block_component_stds": {
            block: _component_dict(block_stds[block]) for block in block_names
        },
        "combined_component_stds": _component_dict(combined_stds),
        "block_relative_std_difference": _component_dict(relative_block_difference),
        "condition_bootstrap": {
            "clusters": len(condition_clusters),
            "replicates": bootstrap_replicates,
            "95_ci": {
                name: [float(condition_low[index]), float(condition_high[index])]
                for index, name in enumerate(COMPONENTS)
            },
            "relative_ci_width": _component_dict(condition_width),
        },
        "speaker_bootstrap": {
            "clusters": len(speaker_clusters),
            "replicates": bootstrap_replicates,
            "95_ci": {
                name: [float(speaker_low[index]), float(speaker_high[index])]
                for index, name in enumerate(COMPONENTS)
            },
            "relative_ci_width": _component_dict(speaker_width),
        },
        "reward_sensitivity": {
            "within_condition_spearman_mean": float(reward_spearman_array.mean()),
            "within_condition_spearman_p05": float(
                np.percentile(reward_spearman_array, 5)
            ),
            "within_condition_spearman_min": float(reward_spearman_array.min()),
            "top2_bottom2_exact_agreement": float(np.mean(top_bottom_agreement)),
        },
        "af_advantage_sensitivity": {
            "batch_replicates": len(advantage_correlations),
            "pearson_mean": float(advantage_correlations_array.mean()),
            "pearson_p05": float(np.percentile(advantage_correlations_array, 5)),
            "pearson_min": float(advantage_correlations_array.min()),
            "sign_agreement_mean": float(np.mean(advantage_sign_agreements)),
            "mean_abs_difference": float(np.mean(advantage_mean_abs_differences)),
        },
    }


def _validate_block(
    spec: Mapping, *, expected_conditions: int, candidates_per_condition: int
) -> tuple[dict, list[dict]]:
    report_path = Path(str(spec["calibration_report_path"]))
    metrics_path = Path(str(spec["enriched_metrics_path"]))
    report = _read_json(report_path)
    rows = _read_jsonl(metrics_path)
    expected_rows = expected_conditions * candidates_per_condition
    if report.get("status") != "CALIBRATION-COMPLETE" or len(rows) != expected_rows:
        raise ValueError(f"incomplete calibration block {spec['name']}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["utterance"])].append(row)
    if len(grouped) != expected_conditions:
        raise ValueError(f"wrong condition count in block {spec['name']}")
    for utterance, group in grouped.items():
        candidates = sorted(int(row["candidate_index"]) for row in group)
        if candidates != list(range(candidates_per_condition)):
            raise ValueError(f"incomplete K group for {spec['name']}:{utterance}")
    recomputed = np.std(
        np.stack([_component_vector(row) for row in rows]), axis=0, ddof=0
    )
    reported = np.asarray(
        [float(report["component_stds"][name]) for name in COMPONENTS]
    )
    if not np.allclose(recomputed, reported, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"reported stds do not reconstruct for block {spec['name']}")
    return report, rows


def run(config: Mapping) -> tuple[dict, Path]:
    input_config = config["input"]
    candidates = int(input_config["candidates_per_condition"])
    reports = {}
    source_rows_by_block = {}
    block_specs = {}
    default_expected_conditions = input_config.get("expected_conditions_per_block")
    for spec in input_config["blocks"]:
        name = str(spec["name"])
        if name in reports:
            raise ValueError(f"duplicate calibration block name: {name}")
        expected_conditions = spec.get(
            "expected_conditions", default_expected_conditions
        )
        if expected_conditions is None:
            raise ValueError(f"block {name} has no expected condition count")
        report, rows = _validate_block(
            spec,
            expected_conditions=int(expected_conditions),
            candidates_per_condition=candidates,
        )
        reports[name] = report
        source_rows_by_block[name] = rows
        block_specs[name] = spec
    if len(reports) < 2:
        raise ValueError("audit config must contain at least two calibration blocks")

    comparison_arms = input_config.get("comparison_arms")
    if comparison_arms is None:
        if len(source_rows_by_block) != 2:
            raise ValueError("more than two source blocks require comparison_arms")
        rows_by_block = source_rows_by_block
    else:
        rows_by_block = {}
        used_sources = []
        for arm in comparison_arms:
            arm_name = str(arm["name"])
            source_names = [str(name) for name in arm["blocks"]]
            if arm_name in rows_by_block or not source_names:
                raise ValueError("comparison arm names must be unique and nonempty")
            missing = [name for name in source_names if name not in source_rows_by_block]
            if missing:
                raise ValueError(f"comparison arm references missing blocks: {missing}")
            rows_by_block[arm_name] = [
                row for name in source_names for row in source_rows_by_block[name]
            ]
            used_sources.extend(source_names)
        if len(rows_by_block) != 2:
            raise ValueError("replication audit requires exactly two comparison arms")
        if sorted(used_sources) != sorted(source_rows_by_block):
            raise ValueError("comparison arms must partition every source block exactly once")

    first_report = next(iter(reports.values()))
    invariants = {
        "train_manifest_sha256": all(
            report["source"]["train_manifest_sha256"]
            == first_report["source"]["train_manifest_sha256"]
            for report in reports.values()
        ),
        "source_nfe": all(
            report["source"]["source_nfe"]
            == first_report["source"]["source_nfe"]
            == 10
            for report in reports.values()
        ),
        "std_ddof": all(
            report["source"]["std_ddof"]
            == first_report["source"]["std_ddof"]
            == 0
            for report in reports.values()
        ),
        "dnsmos_divisor": all(
            report["source"]["dnsmos_divisor"]
            == first_report["source"]["dnsmos_divisor"]
            == 4.0
            for report in reports.values()
        ),
        "evaluators": all(
            report["evaluators"] == first_report["evaluators"]
            for report in reports.values()
        ),
        "evaluator_fingerprint": all(
            report["evaluator_fingerprint_sha256"]
            == first_report["evaluator_fingerprint_sha256"]
            for report in reports.values()
        ),
    }
    if not all(invariants.values()):
        raise ValueError(f"calibration block invariants differ: {invariants}")

    manifest_path = Path(str(input_config["train_manifest_path"]))
    recipe_path = Path(str(input_config["train_recipe_path"]))
    if _sha256_file(manifest_path) != first_report["source"]["train_manifest_sha256"]:
        raise ValueError("audit train manifest does not match block reports")
    recipe_rows = _read_jsonl(recipe_path)
    speaker_by_utterance = {
        str(row["utterance"]): str(row["speaker"]) for row in recipe_rows
    }
    summary = summarize_replication_audit(
        rows_by_block,
        speaker_by_utterance=speaker_by_utterance,
        candidates_per_condition=candidates,
        bootstrap_config=config["bootstrap"],
        replication_config=config["replication"],
        sensitivity_config=config["sensitivity"],
        weights=config["reward"]["weights"],
    )
    audit_pass = summary["audit_status"] == "AUDIT-PASS"
    output_dir = Path(str(config["output_root"]))
    source_rows = int(summary["rows"])
    report = {
        "status": "CALIBRATION-COMPLETE" if audit_pass else "CALIBRATION-AUDIT-FAILED",
        "label": "libritts_disjoint_replication_frozen_composite_calibration",
        "source": {
            "source_nfe": 10,
            "std_ddof": 0,
            "dnsmos_divisor": 4.0,
            "rows": source_rows,
            "eligible_rows_before_train_filter": source_rows,
            "selected_train_rows": source_rows,
            "selected_train_utterances": int(summary["conditions"]),
            "excluded_outside_train_rows": 0,
            "excluded_outside_train_utterances": [],
            "train_manifest_path": str(manifest_path),
            "train_manifest_sha256": _sha256_file(manifest_path),
            "train_recipe_path": str(recipe_path),
            "train_recipe_sha256": _sha256_file(recipe_path),
            "block_reports": {
                name: {
                    "path": str(block_specs[name]["calibration_report_path"]),
                    "sha256": _sha256_file(
                        Path(str(block_specs[name]["calibration_report_path"]))
                    ),
                }
                for name in reports
            },
        },
        "component_stds": summary["combined_component_stds"],
        "weights": dict(config["reward"]["weights"]),
        "evaluators": first_report["evaluators"],
        "evaluator_fingerprint_sha256": first_report[
            "evaluator_fingerprint_sha256"
        ],
        "replication_audit": summary,
        "audit_config": json.loads(json.dumps(config, sort_keys=True)),
    }
    _write_json(output_dir / "calibration_report.json", report)
    return report, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit disjoint composite-calibration replications"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, output_dir = run(config)
    audit = report["replication_audit"]
    print("\nLibriTTS composite calibration replication audit")
    print("=" * 88)
    print(f"Status: {audit['audit_status']}")
    print(
        f"Conditions: {audit['conditions']}  Speakers: {audit['speakers']}  "
        f"Rows: {audit['rows']}"
    )
    for block, stds in audit["block_component_stds"].items():
        print(
            f"Block {block}: dnsmos={stds['dnsmos']:.8f} "
            f"speaker={stds['speaker']:.8f} "
            f"speechbertscore={stds['speechbertscore']:.8f}"
        )
    stds = audit["combined_component_stds"]
    print(
        f"Combined: dnsmos={stds['dnsmos']:.8f} speaker={stds['speaker']:.8f} "
        f"speechbertscore={stds['speechbertscore']:.8f}"
    )
    differences = audit["block_relative_std_difference"]
    print(
        "A/B relative std difference: "
        f"dnsmos={differences['dnsmos']:.2%} "
        f"speaker={differences['speaker']:.2%} "
        f"speechbertscore={differences['speechbertscore']:.2%}"
    )
    condition_width = audit["condition_bootstrap"]["relative_ci_width"]
    speaker_width = audit["speaker_bootstrap"]["relative_ci_width"]
    print(
        "Condition-bootstrap relative CI width: "
        f"{condition_width['dnsmos']:.2%} / {condition_width['speaker']:.2%} / "
        f"{condition_width['speechbertscore']:.2%}"
    )
    print(
        "Speaker-bootstrap relative CI width: "
        f"{speaker_width['dnsmos']:.2%} / {speaker_width['speaker']:.2%} / "
        f"{speaker_width['speechbertscore']:.2%}"
    )
    reward = audit["reward_sensitivity"]
    advantage = audit["af_advantage_sensitivity"]
    print(
        "Reward sensitivity: "
        f"Spearman mean={reward['within_condition_spearman_mean']:.6f}, "
        f"p05={reward['within_condition_spearman_p05']:.6f}, "
        f"top/bottom agreement={reward['top2_bottom2_exact_agreement']:.2%}"
    )
    print(
        "AF advantage sensitivity: "
        f"Pearson mean={advantage['pearson_mean']:.6f}, "
        f"p05={advantage['pearson_p05']:.6f}, "
        f"sign agreement={advantage['sign_agreement_mean']:.2%}"
    )
    print("Checks:")
    for name, passed in audit["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {output_dir / 'calibration_report.json'}")
    print("Training performed: NO")
    if audit["audit_status"] != "AUDIT-PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
