"""Re-score retained Gate-A candidates with the public FlowSE-GRPO composite."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from rl.rewards.evaluators import resolve_hf_model
from rl.rewards.composite import (
    FLOWSE_GRPO_COMPONENTS,
    FlowSEGRPOCompositeEvaluators,
    ModelScopeERes2NetEvaluator,
    SpeechBERTScoreEvaluator,
    composite_reward,
    evaluator_fingerprint_sha256,
    population_std,
)
from rl.common.protocol import sha256_file


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strict_manifest(path: Path) -> dict[str, str]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate utterance {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest is empty or not an object: {path}")
    if not all(
        isinstance(utterance, str) and isinstance(transcript, str)
        for utterance, transcript in value.items()
    ):
        raise ValueError("manifest must map utterance IDs to transcript strings")
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_jsonl(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(rows[key], sort_keys=True, ensure_ascii=False) + "\n"
            for key in sorted(rows)
        ),
        encoding="utf-8",
    )


def _row_key(row: dict) -> str:
    if row.get("cache_key"):
        return str(row["cache_key"])
    return "|".join(
        (
            str(row["utterance"]),
            str(row["nfe"]),
            str(row["latent_seed"]),
        )
    )


def select_train_only_rows(
    source_rows: list[dict], *, train_utterances: set[str], source_nfe: int
) -> tuple[list[dict], dict]:
    """Select complete source-NFE rows and exclude every non-train utterance."""

    eligible = [
        row
        for row in source_rows
        if int(row.get("nfe", -1)) == int(source_nfe) and not row.get("error")
    ]
    selected = []
    excluded = []
    seen_keys: set[str] = set()
    for row in eligible:
        utterance = row.get("utterance")
        if not isinstance(utterance, str) or not utterance:
            raise ValueError("calibration source row lacks a non-empty utterance ID")
        key = _row_key(row)
        if key in seen_keys:
            raise ValueError(f"duplicate calibration source row key: {key}")
        seen_keys.add(key)
        if utterance in train_utterances:
            selected.append(row)
        else:
            excluded.append(row)
    return selected, {
        "eligible_rows_before_train_filter": len(eligible),
        "excluded_outside_train_rows": len(excluded),
        "excluded_outside_train_utterances": sorted(
            {str(row["utterance"]) for row in excluded}
        ),
        "selected_train_rows": len(selected),
        "selected_train_utterances": len(
            {str(row["utterance"]) for row in selected}
        ),
    }


def calibration_cache_row_matches(
    cached_row: dict | None,
    *,
    audio_sha256: str,
    clean_sha256: str,
    evaluator_sha256: str,
) -> bool:
    """A cached score is reusable only for identical audio and evaluator code/models."""

    return bool(
        cached_row is not None
        and cached_row.get("audio_sha256") == audio_sha256
        and cached_row.get("clean_sha256") == clean_sha256
        and cached_row.get("evaluator_fingerprint_sha256") == evaluator_sha256
    )


def summarize_calibration(rows: list[dict], *, weights: dict[str, float]) -> dict:
    raw_values = {
        "dnsmos": [float(row["dnsmos_ovrl"]) / 4.0 for row in rows],
        "speaker": [float(row["eres2net_speaker_similarity"]) for row in rows],
        "speechbertscore": [float(row["speechbertscore"]) for row in rows],
    }
    stds = {name: population_std(values) for name, values in raw_values.items()}
    if any(value <= 1.0e-8 for value in stds.values()):
        raise ValueError(f"composite calibration has a collapsed component: {stds}")
    enriched = []
    for row in rows:
        reward = composite_reward(
            {
                "dnsmos": float(row["dnsmos_ovrl"]) / 4.0,
                "speaker": float(row["eres2net_speaker_similarity"]),
                "speechbertscore": float(row["speechbertscore"]),
            },
            component_stds=stds,
            weights=weights,
        )
        enriched.append({**row, **reward})

    by_utterance: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        by_utterance[str(row["utterance"])].append(row)
    ranges = [
        max(float(row["reward"]) for row in group)
        - min(float(row["reward"]) for row in group)
        for group in by_utterance.values()
    ]
    top_bottom_gaps = []
    for group in by_utterance.values():
        ordered = sorted(float(row["reward"]) for row in group)
        if len(ordered) >= 4:
            top_bottom_gaps.append(
                float(np.mean(ordered[-2:]) - np.mean(ordered[:2]))
            )
    matrix = np.asarray(
        [[row["raw_components"][name] for name in FLOWSE_GRPO_COMPONENTS] for row in enriched],
        dtype=np.float64,
    )
    correlation = np.corrcoef(matrix, rowvar=False)
    return {
        "component_stds": stds,
        "component_means": {
            name: float(np.mean(values)) for name, values in raw_values.items()
        },
        "weights": weights,
        "reward": {
            "mean": float(np.mean([row["reward"] for row in enriched])),
            "std": float(np.std([row["reward"] for row in enriched], ddof=0)),
            "per_condition_range_mean": float(np.mean(ranges)),
            "top2_bottom2_gap_mean": (
                float(np.mean(top_bottom_gaps)) if top_bottom_gaps else None
            ),
        },
        "component_correlation": {
            left: {
                right: float(correlation[i, j])
                for j, right in enumerate(FLOWSE_GRPO_COMPONENTS)
            }
            for i, left in enumerate(FLOWSE_GRPO_COMPONENTS)
        },
        "enriched_rows": enriched,
    }


def summarize_preflight(rows: list[dict], *, weights: dict[str, float]) -> dict:
    """Summarize evaluator connectivity without requiring estimable scales."""

    if not rows:
        raise ValueError("composite preflight produced no rows")
    raw_values = {
        "dnsmos": [float(row["dnsmos_ovrl"]) / 4.0 for row in rows],
        "speaker": [float(row["eres2net_speaker_similarity"]) for row in rows],
        "speechbertscore": [float(row["speechbertscore"]) for row in rows],
    }
    return {
        "component_stds": {
            name: population_std(values) for name, values in raw_values.items()
        },
        "component_means": {
            name: float(np.mean(values)) for name, values in raw_values.items()
        },
        "weights": weights,
        "reward": {
            "computed": False,
            "reason": "preflight does not estimate component scales",
        },
        "component_correlation": None,
        "enriched_rows": rows,
    }


def _component_vector(row: dict) -> np.ndarray:
    return np.asarray(
        [
            float(row["dnsmos_ovrl"]) / 4.0,
            float(row["eres2net_speaker_similarity"]),
            float(row["speechbertscore"]),
        ],
        dtype=np.float64,
    )


def summarize_calibration_stability(
    rows: list[dict],
    *,
    prefix_condition_counts: list[int],
    candidates_per_condition: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    max_last_prefix_relative_change: float,
    max_full_relative_ci_width: float,
) -> dict:
    """Audit frozen component scales on nested condition prefixes.

    The point estimates always use every endpoint in a prefix. Bootstrap
    resampling is performed at the condition level so the K candidates from
    one noisy input are never treated as independent observations.
    """

    components = tuple(FLOWSE_GRPO_COMPONENTS)
    prefix_counts = [int(value) for value in prefix_condition_counts]
    if (
        not prefix_counts
        or prefix_counts != sorted(set(prefix_counts))
        or prefix_counts[0] < 2
    ):
        raise ValueError("stability prefix condition counts must be unique and increasing")
    if candidates_per_condition < 2:
        raise ValueError("stability requires at least two candidates per condition")
    if bootstrap_replicates < 100:
        raise ValueError("stability requires at least 100 bootstrap replicates")
    if not 0.0 < max_last_prefix_relative_change < 1.0:
        raise ValueError("invalid last-prefix relative-change threshold")
    if not 0.0 < max_full_relative_ci_width < 1.0:
        raise ValueError("invalid full-prefix relative-CI-width threshold")

    grouped: dict[int, list[dict]] = defaultdict(list)
    utterance_for_index: dict[int, str] = {}
    for row in rows:
        if "condition_index" not in row or "candidate_index" not in row:
            raise ValueError(
                "stability audit requires condition_index and candidate_index"
            )
        condition_index = int(row["condition_index"])
        utterance = str(row["utterance"])
        previous = utterance_for_index.setdefault(condition_index, utterance)
        if previous != utterance:
            raise ValueError("one condition_index maps to multiple utterances")
        grouped[condition_index].append(row)

    ordered_indices = sorted(grouped)
    expected_indices = list(range(len(ordered_indices)))
    if ordered_indices != expected_indices:
        raise ValueError("stability condition indices are not contiguous from zero")
    if prefix_counts[-1] != len(ordered_indices):
        raise ValueError(
            "largest stability prefix must equal the complete condition count"
        )

    condition_arrays = []
    for condition_index in ordered_indices:
        group = grouped[condition_index]
        candidate_indices = sorted(int(row["candidate_index"]) for row in group)
        if candidate_indices != list(range(candidates_per_condition)):
            raise ValueError(
                f"condition {condition_index} does not contain one complete K group"
            )
        ordered = sorted(group, key=lambda row: int(row["candidate_index"]))
        condition_arrays.append(np.stack([_component_vector(row) for row in ordered]))
    matrix = np.stack(condition_arrays)

    rng = np.random.default_rng(int(bootstrap_seed))
    estimates = []
    previous_stds: np.ndarray | None = None
    full_stds: np.ndarray | None = None
    prefix_stds: list[np.ndarray] = []
    for count in prefix_counts:
        values = matrix[:count].reshape(-1, len(components))
        stds = np.std(values, axis=0, ddof=0)
        if np.any(stds <= 1.0e-8):
            raise ValueError(f"stability prefix {count} has a collapsed component")
        prefix_stds.append(stds)
    full_stds = prefix_stds[-1]

    for count, stds in zip(prefix_counts, prefix_stds, strict=True):
        prefix = matrix[:count]
        bootstrap = np.empty((bootstrap_replicates, len(components)), dtype=np.float64)
        for replicate in range(bootstrap_replicates):
            sampled = rng.integers(0, count, size=count)
            sampled_values = prefix[sampled].reshape(-1, len(components))
            bootstrap[replicate] = np.std(sampled_values, axis=0, ddof=0)
        low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        relative_change = (
            None if previous_stds is None else np.abs(stds - previous_stds) / previous_stds
        )
        estimates.append(
            {
                "conditions": count,
                "rows": count * candidates_per_condition,
                "component_stds": {
                    name: float(stds[index]) for index, name in enumerate(components)
                },
                "condition_bootstrap_95_ci": {
                    name: [float(low[index]), float(high[index])]
                    for index, name in enumerate(components)
                },
                "condition_bootstrap_relative_ci_width": {
                    name: float((high[index] - low[index]) / stds[index])
                    for index, name in enumerate(components)
                },
                "relative_change_from_previous_prefix": (
                    None
                    if relative_change is None
                    else {
                        name: float(relative_change[index])
                        for index, name in enumerate(components)
                    }
                ),
                "relative_change_from_full_prefix": {
                    name: float(abs(stds[index] - full_stds[index]) / full_stds[index])
                    for index, name in enumerate(components)
                },
            }
        )
        previous_stds = stds

    final = estimates[-1]
    last_change = final["relative_change_from_previous_prefix"]
    if last_change is None:
        raise AssertionError("stability audit requires at least two prefixes")
    previous = estimates[-2]
    previous_inside_final_ci = {
        name: (
            float(final["condition_bootstrap_95_ci"][name][0])
            <= float(previous["component_stds"][name])
            <= float(final["condition_bootstrap_95_ci"][name][1])
        )
        for name in components
    }
    checks = {
        "previous_prefix_inside_full_bootstrap_ci": all(
            previous_inside_final_ci.values()
        ),
        "full_prefix_relative_ci_width": all(
            float(final["condition_bootstrap_relative_ci_width"][name])
            <= max_full_relative_ci_width
            for name in components
        ),
    }
    return {
        "status": "STABILITY-PASS" if all(checks.values()) else "EXPAND-CALIBRATION",
        "bootstrap_unit": "condition",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": int(bootstrap_seed),
        "thresholds": {
            "max_full_relative_ci_width": max_full_relative_ci_width,
        },
        "checks": checks,
        "diagnostics": {
            "previous_prefix_inside_full_bootstrap_ci_by_component": (
                previous_inside_final_ci
            ),
            "last_prefix_relative_change_target": {
                "threshold": max_last_prefix_relative_change,
                "by_component": {
                    name: float(last_change[name]) <= max_last_prefix_relative_change
                    for name in components
                },
                "all_components_pass": all(
                    float(last_change[name]) <= max_last_prefix_relative_change
                    for name in components
                ),
                "gating": False,
            },
        },
        "prefix_estimates": estimates,
        "recommended_frozen_condition_count": (
            prefix_counts[-1] if all(checks.values()) else None
        ),
        "recommended_component_stds": (
            final["component_stds"] if all(checks.values()) else None
        ),
    }


def run(config: dict, *, limit: int | None = None) -> tuple[dict, Path]:
    source = Path(config["input"]["endpoint_metrics_path"])
    source_rows = _read_jsonl(source)
    nfe = int(config["input"]["source_nfe"])
    train_manifest_path = Path(config["input"]["train_manifest_path"])
    train_manifest = _strict_manifest(train_manifest_path)
    selected, ownership_audit = select_train_only_rows(
        source_rows,
        train_utterances=set(train_manifest),
        source_nfe=nfe,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if limit is None:
        expected_source = int(config["input"]["expected_source_rows"])
        expected = int(config["input"]["expected_rows"])
        expected_excluded_rows = int(
            config["input"]["expected_excluded_outside_train_rows"]
        )
        expected_excluded_utterances = int(
            config["input"]["expected_excluded_outside_train_utterances"]
        )
        checks = {
            "eligible_source_rows": ownership_audit[
                "eligible_rows_before_train_filter"
            ]
            == expected_source,
            "selected_train_rows": len(selected) == expected,
            "excluded_outside_train_rows": ownership_audit[
                "excluded_outside_train_rows"
            ]
            == expected_excluded_rows,
            "excluded_outside_train_utterances": len(
                ownership_audit["excluded_outside_train_utterances"]
            )
            == expected_excluded_utterances,
        }
        if not all(checks.values()):
            raise ValueError(
                "train-only calibration ownership audit failed: "
                f"checks={checks}, observed={ownership_audit}"
            )

    output_dir = Path(config["output_root"])
    if limit is not None:
        output_dir = output_dir / f"preflight_{limit}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "official_composite_metrics.jsonl"
    cached = {
        _row_key(row): row for row in _read_jsonl(metrics_path)
    } if metrics_path.is_file() else {}
    eligible_keys = {_row_key(row) for row in selected}

    evaluator_config = config["evaluators"]
    wavlm = resolve_hf_model(evaluator_config["speechbertscore"])
    evaluators = FlowSEGRPOCompositeEvaluators(
        ModelScopeERes2NetEvaluator(
            evaluator_config["speaker"], device=str(evaluator_config["device"])
        ),
        SpeechBERTScoreEvaluator(
            wavlm,
            device=str(evaluator_config["device"]),
            layer=int(evaluator_config["speechbertscore"]["layer"]),
            reference_cache_size=int(
                evaluator_config["speechbertscore"].get(
                    "reference_cache_size", 64
                )
            ),
        ),
    )
    evaluator_fingerprint = evaluators.fingerprint()
    evaluator_sha256 = evaluator_fingerprint_sha256(evaluator_fingerprint)
    clean_dir = Path(config["input"]["clean_dir"])
    # Preserve still-eligible cache entries while an expanded calibration pool
    # is traversed. Every row is revalidated below before the final report is
    # produced, but an interruption cannot discard unvisited valid scores.
    completed: dict[str, dict] = {
        key: row for key, row in cached.items() if key in eligible_keys
    }
    for source_row in tqdm(selected, desc="Official composite calibration"):
        key = _row_key(source_row)
        audio_path = Path(str(source_row["audio_path"]))
        clean_path = clean_dir / f"{source_row['utterance']}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not clean_path.is_file():
            raise FileNotFoundError(clean_path)
        cached_row = cached.get(key)
        audio_sha256 = sha256_file(audio_path)
        clean_sha256 = sha256_file(clean_path)
        if calibration_cache_row_matches(
            cached_row,
            audio_sha256=audio_sha256,
            clean_sha256=clean_sha256,
            evaluator_sha256=evaluator_sha256,
        ):
            completed[key] = {
                **cached_row,
                **(
                    {"condition_index": int(source_row["condition_index"])}
                    if "condition_index" in source_row
                    else {}
                ),
                **(
                    {"candidate_index": int(source_row["candidate_index"])}
                    if "candidate_index" in source_row
                    else {}
                ),
            }
            continue
        scores = evaluators.score(clean_path, audio_path)
        completed[key] = {
            "cache_key": key,
            "utterance": str(source_row["utterance"]),
            "nfe": nfe,
            "latent_seed": int(source_row["latent_seed"]),
            **(
                {"condition_index": int(source_row["condition_index"])}
                if "condition_index" in source_row
                else {}
            ),
            **(
                {"candidate_index": int(source_row["candidate_index"])}
                if "candidate_index" in source_row
                else {}
            ),
            "audio_path": str(audio_path),
            "audio_sha256": audio_sha256,
            "clean_path": str(clean_path),
            "clean_sha256": clean_sha256,
            "evaluator_fingerprint_sha256": evaluator_sha256,
            "dnsmos_ovrl": float(source_row["dnsmos_ovrl"]),
            **scores,
        }
        _rewrite_jsonl(metrics_path, completed)
    # Remove cached rows that are no longer eligible after a split change.
    _rewrite_jsonl(metrics_path, completed)

    weights = {
        name: float(config["reward"]["weights"][name])
        for name in FLOWSE_GRPO_COMPONENTS
    }
    if limit is None:
        summary = summarize_calibration(list(completed.values()), weights=weights)
    else:
        summary = summarize_preflight(list(completed.values()), weights=weights)
    stability = None
    if limit is None and config.get("stability") is not None:
        stability_config = config["stability"]
        stability = summarize_calibration_stability(
            summary["enriched_rows"],
            prefix_condition_counts=list(stability_config["prefix_condition_counts"]),
            candidates_per_condition=int(
                stability_config["candidates_per_condition"]
            ),
            bootstrap_replicates=int(stability_config["bootstrap_replicates"]),
            bootstrap_seed=int(stability_config["bootstrap_seed"]),
            max_last_prefix_relative_change=float(
                stability_config["max_last_prefix_relative_change"]
            ),
            max_full_relative_ci_width=float(
                stability_config["max_full_relative_ci_width"]
            ),
        )
    report = {
        "status": "PREFLIGHT" if limit is not None else "CALIBRATION-COMPLETE",
        "label": "paper_aligned_public_flowse_grpo_composite",
        "exact_author_evaluator_reproduction": False,
        "reason": (
            "FlowSE-GRPO specifies evaluator families but does not publish "
            "immutable evaluator checkpoint revisions"
        ),
        "source": {
            "endpoint_metrics_path": str(source),
            "endpoint_metrics_sha256": sha256_file(source),
            "train_manifest_path": str(train_manifest_path),
            "train_manifest_sha256": sha256_file(train_manifest_path),
            "train_manifest_utterances": len(train_manifest),
            "source_nfe": nfe,
            "rows": len(completed),
            "std_ddof": 0,
            "dnsmos_divisor": 4.0,
            **ownership_audit,
        },
        "evaluators": evaluator_fingerprint,
        "evaluator_fingerprint_sha256": evaluator_sha256,
        **{key: value for key, value in summary.items() if key != "enriched_rows"},
        **({"stability": stability} if stability is not None else {}),
    }
    _rewrite_jsonl(
        output_dir / "official_composite_enriched_metrics.jsonl",
        {_row_key(row): row for row in summary["enriched_rows"]},
    )
    _write_json(output_dir / "calibration_report.json", report)
    return report, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the public FlowSE-GRPO composite on retained Gate-A WAVs"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, output_dir = run(config, limit=args.limit)
    print("\nPaper-aligned public FlowSE-GRPO composite calibration")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Rows: {report['source']['rows']}")
    print(
        "Excluded outside-train rows: "
        f"{report['source']['excluded_outside_train_rows']} "
        "(" 
        f"{len(report['source']['excluded_outside_train_utterances'])} utterances)"
    )
    for name in FLOWSE_GRPO_COMPONENTS:
        print(
            f"{name:18s} mean={report['component_means'][name]:+.8f} "
            f"std={report['component_stds'][name]:.8f}"
        )
    geometry = report["reward"]
    if geometry.get("computed") is False:
        print("Composite geometry: not estimated during preflight")
    else:
        print(f"Composite std: {geometry['std']:.8f}")
        print(f"Mean per-condition range: {geometry['per_condition_range_mean']:.8f}")
        print(f"Mean top2-bottom2 gap: {geometry['top2_bottom2_gap_mean']}")
    if "stability" in report:
        stability = report["stability"]
        print("\nNested-prefix component-std stability")
        print("-" * 72)
        for estimate in stability["prefix_estimates"]:
            stds = estimate["component_stds"]
            print(
                f"conditions={estimate['conditions']:4d} rows={estimate['rows']:5d} "
                f"dnsmos={stds['dnsmos']:.8f} "
                f"speaker={stds['speaker']:.8f} "
                f"speechbertscore={stds['speechbertscore']:.8f}"
            )
        print(f"Stability decision: {stability['status']}")
    print(f"Report: {output_dir / 'calibration_report.json'}")
    print("Training performed: NO")


if __name__ == "__main__":
    main()
