"""Two-GPU matched-latent diagnostic for FlowSE-GRPO candidate diversity.

This entry point never trains or injects LoRA.  It freezes the released FlowSE
base policy, nests several Brownian continuations under each initial Gaussian
latent, and adds a paired deterministic ODE endpoint for every latent.  The
result separates across-latent variance from within-latent SDE variance before
the production GRPO sampling scope is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import yaml

from rl.common.conditioning import ConditioningProtocol
from rl.rewards.specification import (
    FLOWSE_GRPO_COMPOSITE,
    compute_training_reward,
    resolve_training_reward,
    verify_reward_calibration,
)

from .policy import policy_velocity
from .rollout import FlowSEWindowedSDESampler, WindowSpec
from .storage import atomic_write_json, atomic_write_jsonl


SCHEMA_VERSION = 1


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _canonical_hash(value: Mapping) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _fingerprint_differences(
    calibration: object, runtime: object, *, prefix: str = ""
) -> dict[str, dict[str, object]]:
    """Return exact leaf differences so stale calibration is actionable."""

    if isinstance(calibration, Mapping) and isinstance(runtime, Mapping):
        differences = {}
        for key in sorted(set(calibration).union(runtime)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in calibration:
                differences[path] = {"calibration": "<missing>", "runtime": runtime[key]}
            elif key not in runtime:
                differences[path] = {"calibration": calibration[key], "runtime": "<missing>"}
            else:
                differences.update(
                    _fingerprint_differences(
                        calibration[key], runtime[key], prefix=path
                    )
                )
        return differences
    if calibration == runtime:
        return {}
    return {prefix or "<root>": {"calibration": calibration, "runtime": runtime}}


def _expected_composite_evaluator_fingerprint(config: Mapping) -> dict:
    """Rebuild the runtime fingerprint without loading either neural evaluator."""

    from rl.rewards import composite as implementation
    from rl.rewards.evaluators import (
        package_version,
        resolve_hf_model,
        sha256_tree,
    )

    evaluator_config = config["composite_reward_evaluators"]
    speaker_config = evaluator_config["speaker"]
    local_model_dir = speaker_config.get("local_model_dir")
    local_snapshot = (
        Path(str(local_model_dir)).resolve() if local_model_dir is not None else None
    )
    if local_snapshot is not None and local_snapshot.is_dir():
        speaker_source = str(local_snapshot)
        speaker_source_type = "local_directory"
        speaker_source_sha256 = sha256_tree(local_snapshot)
    else:
        speaker_source = str(speaker_config["model_id"])
        speaker_source_type = "modelscope_model_id"
        speaker_source_sha256 = None
    speaker = {
        "backend": "modelscope_speaker_verification_eres2net",
        "implementation": "explicit_pair_embedding_cosine_v1",
        "model_id": str(speaker_config["model_id"]),
        "revision": str(speaker_config["revision"]),
        "model_source": speaker_source,
        "model_source_type": speaker_source_type,
        "modelscope_version": package_version("modelscope"),
        "numpy_version": package_version("numpy"),
    }
    if speaker_source_sha256 is not None:
        speaker["model_source_sha256"] = speaker_source_sha256

    speechbert_config = evaluator_config["speechbertscore"]
    resolved = resolve_hf_model(dict(speechbert_config))
    return {
        "schema_version": 1,
        "implementation_source_sha256": _sha256_file(implementation.__file__),
        "speaker": speaker,
        "speechbertscore": {
            "backend": "discrete_speech_metrics_speechbertscore_precision",
            "implementation": "wavlm_hidden_state_cosine_precision_v1",
            "model": resolved.fingerprint(),
            "layer": int(speechbert_config.get("layer", 14)),
            "sample_rate": 16000,
            "transformers_version": package_version("transformers"),
            "torch_version": package_version("torch"),
        },
    }


def _preflight_calibration_fingerprint(
    config: Mapping, reward_definition: Mapping
) -> dict:
    report_path = Path(reward_definition["calibration"]["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration = report.get("evaluators")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration report has no evaluator fingerprint")
    runtime = _expected_composite_evaluator_fingerprint(config)
    differences = _fingerprint_differences(calibration, runtime)
    if differences:
        raise ValueError(
            "reward calibration evaluator fingerprint is stale; regenerate it with "
            "the current environment and common evaluator. Differences: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    return runtime


def validate_sampling_diagnostic_config(config: Mapping) -> dict:
    """Validate the frozen diagnostic geometry without loading data or models."""

    required = {
        "run",
        "flowse_config",
        "dnsmos_official_dir",
        "output_root",
        "conditioning",
        "data",
        "diagnostic",
        "sampler",
        "resources",
        "artifacts",
        "normalization",
        "training_reward",
        "composite_reward_evaluators",
        "evaluation",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError("diagnostic config is missing: " + ", ".join(missing))
    if str(config["run"].get("mode")) != "sampling_diagnostic":
        raise ValueError("run.mode must be sampling_diagnostic")
    if str(config["run"].get("policy_state")) != "released_base_lora_disabled":
        raise ValueError("sampling diagnostic must use released_base_lora_disabled")
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    if conditioning.fingerprint() != {
        "mode": "wotext",
        "use_text": False,
        "drop_text": True,
    }:
        raise ValueError("sampling diagnostic must remain audio-only")

    data = config["data"]
    if Path(str(data["train_manifest"])).name != "voicebank_train_16k.json":
        raise ValueError("diagnostic must sample only the frozen VoiceBank train manifest")
    utterance_count = int(config["diagnostic"]["utterance_count"])
    initial_latents = int(config["diagnostic"]["initial_latents_per_utterance"])
    continuations = int(
        config["diagnostic"]["brownian_continuations_per_latent"]
    )
    if utterance_count < 1 or initial_latents < 2 or continuations < 2:
        raise ValueError("diagnostic requires utterances>=1, latents>=2, continuations>=2")
    if int(config["diagnostic"].get("ode_controls_per_latent", -1)) != 1:
        raise ValueError("exactly one paired ODE control per latent is required")

    sampler = config["sampler"]
    nfe = int(sampler["nfe"])
    window_size = int(sampler["window_size"])
    starts = [int(value) for value in sampler["window_starts"]]
    if nfe != 10:
        raise ValueError("the matched-latent diagnostic freezes candidate NFE=10")
    if window_size != 2:
        raise ValueError("the FlowSE-GRPO diagnostic requires a two-step SDE window")
    if starts != [1, 2, 3]:
        raise ValueError("window_starts must be the frozen paper range [1, 2, 3]")
    for start in starts:
        WindowSpec(nfe=nfe, start_step=start, window_size=window_size).validate()
    if _finite_float(sampler["diffusion"], name="sampler.diffusion") != 0.4:
        raise ValueError("the diagnostic requires FlowSE-GRPO diffusion=0.4")
    if _finite_float(sampler["cfg_strength"], name="sampler.cfg_strength") != 0.0:
        raise ValueError("the controlled diagnostic requires CFG=0")

    resources = config["resources"]
    world_size = int(resources["world_size"])
    device_ids = [int(value) for value in resources["device_ids"]]
    if world_size != 2 or len(device_ids) != 2 or len(set(device_ids)) != 2:
        raise ValueError("this diagnostic is frozen to two distinct GPUs")
    if min(device_ids) < 0:
        raise ValueError("CUDA device IDs must be non-negative")
    if utterance_count < world_size:
        raise ValueError("utterance_count must be at least the GPU world size")
    cpu_threads = int(resources["cpu_threads_per_worker"])
    interop_threads = int(resources["torch_interop_threads_per_worker"])
    if cpu_threads < 1 or interop_threads < 1:
        raise ValueError("per-worker CPU thread limits must be positive")
    if not isinstance(resources.get("enforce_cpu_affinity"), bool):
        raise ValueError("resources.enforce_cpu_affinity must be boolean")

    artifacts = config["artifacts"]
    if bool(artifacts.get("keep_all_audio", True)):
        raise ValueError("diagnostic must delete non-audit WAVs after scoring")
    audit_count = int(artifacts["audit_wavs_per_utterance"])
    if not 0 <= audit_count <= 2:
        raise ValueError("audit_wavs_per_utterance must be 0, 1, or 2")
    if str(config["normalization"]["output_subtype"]) != "PCM_16":
        raise ValueError("reward WAVs must use PCM_16")

    reward = resolve_training_reward(config, validate_artifacts=False)
    if reward["name"] != FLOWSE_GRPO_COMPOSITE:
        raise ValueError("diagnostic must use the frozen FlowSE-GRPO composite reward")
    repeat_audits = int(config["diagnostic"]["repeat_score_audits_per_utterance"])
    if repeat_audits not in {0, 1}:
        raise ValueError("repeat_score_audits_per_utterance must be 0 or 1")
    near_zero = _finite_float(
        config["diagnostic"]["near_zero_reward_std"],
        name="diagnostic.near_zero_reward_std",
    )
    if near_zero < 0.0:
        raise ValueError("near_zero_reward_std must be non-negative")
    if bool(config["evaluation"].get("fidelity", {}).get("enabled", False)):
        raise ValueError(
            "the train-split sampling diagnostic is transcript-free; use "
            "SpeechBERTScore/ERes2Net here and reserve ASR/WER for held-out evaluation"
        )

    sde_candidates = utterance_count * initial_latents * continuations
    ode_candidates = utterance_count * initial_latents
    return {
        "status": "SAMPLING-DIAGNOSTIC-CONFIG-VALID",
        "schema_version": SCHEMA_VERSION,
        "policy_state": "released_base_lora_disabled",
        "world_size": world_size,
        "device_ids": device_ids,
        "cpu_threads_per_worker": cpu_threads,
        "maximum_configured_cpu_threads": world_size * cpu_threads,
        "torch_interop_threads_per_worker": interop_threads,
        "enforce_cpu_affinity": bool(resources["enforce_cpu_affinity"]),
        "utterances": utterance_count,
        "initial_latents_per_utterance": initial_latents,
        "brownian_continuations_per_latent": continuations,
        "sde_candidates": sde_candidates,
        "ode_candidates": ode_candidates,
        "total_candidates": sde_candidates + ode_candidates,
        "nfe": nfe,
        "window_size": window_size,
        "window_starts": starts,
        "diffusion": 0.4,
        "cfg_strength": 0.0,
        "reward": reward,
    }


def partition_utterances(
    utterances: Sequence[Mapping], world_size: int
) -> list[list[dict]]:
    """Round-robin utterances while keeping every nested design on one GPU."""

    if world_size < 1 or len(utterances) < world_size:
        raise ValueError("invalid diagnostic partition geometry")
    partitions = [[] for _ in range(world_size)]
    seen = set()
    for index, item in enumerate(utterances):
        utterance = str(item["utterance"])
        if utterance in seen:
            raise ValueError(f"duplicate diagnostic utterance: {utterance}")
        seen.add(utterance)
        partitions[index % world_size].append(dict(item))
    if any(not shard for shard in partitions):
        raise AssertionError("each diagnostic GPU must receive at least one utterance")
    return partitions


def variance_decomposition(values: Sequence[Sequence[float]]) -> dict[str, float]:
    """Population-law decomposition for latent x Brownian nested samples."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError("variance decomposition requires a finite 2-D nested matrix")
    latent_means = array.mean(axis=1)
    across = float(np.var(latent_means, ddof=0))
    within = float(np.mean(np.var(array, axis=1, ddof=0)))
    total = float(np.var(array, ddof=0))
    if not math.isclose(total, across + within, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise AssertionError("law of total variance failed")
    if total == 0.0:
        across_fraction = 0.0
        within_fraction = 0.0
    else:
        across_fraction = across / total
        within_fraction = within / total
    return {
        "total_variance": total,
        "across_initial_latent_variance": across,
        "within_initial_latent_brownian_variance": within,
        "initial_latent_variance_fraction": across_fraction,
        "brownian_variance_fraction": within_fraction,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distribution requires finite non-empty values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _numeric_metrics(row: Mapping) -> dict[str, float]:
    result = {"reward": _finite_float(row["reward"], name="reward")}
    for prefix, values in (
        ("metrics", row.get("metrics", {})),
        ("raw_reward", row.get("reward_components", {}).get("raw_components", {})),
    ):
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            parsed = float(value)
            if math.isfinite(parsed):
                result[f"{prefix}.{name}"] = parsed
    return result


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _mean_pairwise_rms(values: Sequence[np.ndarray]) -> dict[str, float | int]:
    distances = []
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    for left_index, left in enumerate(arrays):
        for right in arrays[left_index + 1 :]:
            if left.shape != right.shape:
                raise ValueError("paired diagnostic arrays have inconsistent shapes")
            distances.append(float(np.sqrt(np.mean(np.square(left - right)))))
    if not distances:
        return {"pairs": 0, "mean": 0.0, "median": 0.0, "maximum": 0.0}
    array = np.asarray(distances, dtype=np.float64)
    return {
        "pairs": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def _distance_audit(
    ode_mels: Sequence[np.ndarray],
    ode_waves: Sequence[np.ndarray],
    sde_mels: Sequence[Sequence[np.ndarray]],
    sde_waves: Sequence[Sequence[np.ndarray]],
) -> dict:
    within_mels = []
    within_waves = []
    for mel_group, wave_group in zip(sde_mels, sde_waves, strict=True):
        within_mels.extend(
            float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))
            for index, left in enumerate(mel_group)
            for right in mel_group[index + 1 :]
        )
        within_waves.extend(
            float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))
            for index, left in enumerate(wave_group)
            for right in wave_group[index + 1 :]
        )
    latent_mean_mels = [np.mean(np.stack(values), axis=0) for values in sde_mels]
    latent_mean_waves = [np.mean(np.stack(values), axis=0) for values in sde_waves]
    ode_to_sde_mel = [
        float(np.sqrt(np.mean(np.square(np.asarray(ode) - sde))))
        for ode, sde in zip(ode_mels, latent_mean_mels, strict=True)
    ]
    ode_to_sde_wave = [
        float(np.sqrt(np.mean(np.square(np.asarray(ode) - sde))))
        for ode, sde in zip(ode_waves, latent_mean_waves, strict=True)
    ]
    return {
        "within_latent_brownian_terminal_mel_rms": _distribution(within_mels),
        "across_latent_mean_terminal_mel_rms": _mean_pairwise_rms(latent_mean_mels),
        "paired_ode_to_mean_sde_terminal_mel_rms": _distribution(ode_to_sde_mel),
        "within_latent_brownian_waveform_rms": _distribution(within_waves),
        "across_latent_mean_waveform_rms": _mean_pairwise_rms(latent_mean_waves),
        "paired_ode_to_mean_sde_waveform_rms": _distribution(ode_to_sde_wave),
    }


def analyze_sampling_rows(
    rows: Sequence[Mapping],
    utterance_summaries: Sequence[Mapping],
    config: Mapping,
) -> dict:
    """Audit exact geometry and aggregate nested variance components."""

    diagnostic = config["diagnostic"]
    utterance_count = int(diagnostic["utterance_count"])
    latent_count = int(diagnostic["initial_latents_per_utterance"])
    continuation_count = int(diagnostic["brownian_continuations_per_latent"])
    expected = utterance_count * latent_count * (continuation_count + 1)
    if len(rows) != expected:
        raise ValueError(f"candidate row count mismatch: got {len(rows)}, expected {expected}")
    ids = [str(row["candidate_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("diagnostic candidate IDs are not unique")

    by_utterance: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_utterance[str(row["utterance"])].append(row)
    if len(by_utterance) != utterance_count:
        raise ValueError("diagnostic utterance count is incomplete")

    common_metrics: set[str] | None = None
    decompositions: dict[str, list[dict[str, float]]] = defaultdict(list)
    ode_deltas: dict[str, list[float]] = defaultdict(list)
    shared_reward_stds = []
    across_latent_reward_stds = []
    repeat_reward_differences = []
    repeat_metric_differences: dict[str, list[float]] = defaultdict(list)
    duplicate_terminal = 0
    duplicate_wav = 0
    sde_row_count = 0
    reward_decomposition_by_utterance = {}

    for utterance, utterance_rows in sorted(by_utterance.items()):
        sde = [row for row in utterance_rows if row["method"] == "sde"]
        ode = [row for row in utterance_rows if row["method"] == "ode"]
        if len(sde) != latent_count * continuation_count or len(ode) != latent_count:
            raise ValueError(f"incomplete nested geometry for {utterance}")
        sde_cells = {
            (int(row["latent_index"]), int(row["brownian_index"])): row for row in sde
        }
        ode_cells = {int(row["latent_index"]): row for row in ode}
        if len(sde_cells) != latent_count * continuation_count or len(ode_cells) != latent_count:
            raise ValueError(f"duplicate nested cell for {utterance}")
        for latent_index in range(latent_count):
            seeds = {
                int(sde_cells[(latent_index, brownian)]["initial_latent_seed"])
                for brownian in range(continuation_count)
            }
            if len(seeds) != 1 or next(iter(seeds)) != int(
                ode_cells[latent_index]["initial_latent_seed"]
            ):
                raise ValueError(f"matched latent seed mismatch for {utterance}")
        latent_seeds = {int(row["initial_latent_seed"]) for row in ode}
        brownian_seeds = {int(row["brownian_seed"]) for row in sde}
        if len(latent_seeds) != latent_count or len(brownian_seeds) != len(sde):
            raise ValueError(f"seed uniqueness audit failed for {utterance}")

        numeric = [_numeric_metrics(row) for row in utterance_rows]
        utterance_common = set.intersection(*(set(item) for item in numeric))
        common_metrics = (
            utterance_common
            if common_metrics is None
            else common_metrics.intersection(utterance_common)
        )
        for metric in sorted(utterance_common):
            matrix = [
                [
                    _numeric_metrics(sde_cells[(latent, brownian)])[metric]
                    for brownian in range(continuation_count)
                ]
                for latent in range(latent_count)
            ]
            decomposition = variance_decomposition(matrix)
            decompositions[metric].append(decomposition)
            if metric == "reward":
                reward_decomposition_by_utterance[utterance] = decomposition
                array = np.asarray(matrix, dtype=np.float64)
                shared_reward_stds.extend(np.std(array, axis=1, ddof=0).tolist())
                across_latent_reward_stds.extend(np.std(array, axis=0, ddof=0).tolist())
            for latent_index in range(latent_count):
                sde_mean = float(np.mean(matrix[latent_index]))
                ode_value = _numeric_metrics(ode_cells[latent_index])[metric]
                ode_deltas[metric].append(sde_mean - ode_value)

        terminal_hashes = [str(row["terminal_mel_sha256"]) for row in sde]
        wav_hashes = [str(row["scored_wav_sha256"]) for row in sde]
        duplicate_terminal += len(terminal_hashes) - len(set(terminal_hashes))
        duplicate_wav += len(wav_hashes) - len(set(wav_hashes))
        sde_row_count += len(sde)
        for row in sde:
            repeat = row.get("repeat_score")
            if not isinstance(repeat, Mapping):
                continue
            repeat_reward_differences.append(abs(float(row["reward"]) - float(repeat["reward"])))
            original_metrics = _numeric_metrics(row)
            repeated_metrics = repeat.get("numeric_metrics", {})
            for metric in set(original_metrics).intersection(repeated_metrics):
                repeat_metric_differences[metric].append(
                    abs(original_metrics[metric] - float(repeated_metrics[metric]))
                )

    if not common_metrics or "reward" not in common_metrics:
        raise ValueError("diagnostic rows have no common scalar reward metrics")
    repeat_max = max(repeat_reward_differences, default=0.0)
    frozen_floor = float(diagnostic["near_zero_reward_std"])
    effective_floor = max(frozen_floor, 3.0 * repeat_max)
    shared_near = sum(value <= effective_floor for value in shared_reward_stds)
    across_near = sum(value <= effective_floor for value in across_latent_reward_stds)

    metric_report = {}
    for metric in sorted(common_metrics):
        parts = decompositions[metric]
        across_values = [item["across_initial_latent_variance"] for item in parts]
        within_values = [item["within_initial_latent_brownian_variance"] for item in parts]
        total_values = [item["total_variance"] for item in parts]
        across_sum = float(sum(across_values))
        within_sum = float(sum(within_values))
        total_sum = float(sum(total_values))
        metric_report[metric] = {
            "total_variance": _distribution(total_values),
            "across_initial_latent_variance": _distribution(across_values),
            "within_initial_latent_brownian_variance": _distribution(within_values),
            "ratio_of_aggregate_variance": {
                "initial_latent_fraction": across_sum / total_sum if total_sum else 0.0,
                "brownian_fraction": within_sum / total_sum if total_sum else 0.0,
            },
            "mean_sde_minus_paired_ode": _distribution(ode_deltas[metric]),
        }

    distance_fields: dict[str, list[float]] = defaultdict(list)
    for summary in utterance_summaries:
        for name, value in summary.get("distance_audit", {}).items():
            if isinstance(value, Mapping) and "mean" in value:
                distance_fields[name].append(float(value["mean"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "geometry_audit": {
            "utterances": len(by_utterance),
            "candidate_rows": len(rows),
            "sde_rows": sde_row_count,
            "ode_rows": len(rows) - sde_row_count,
            "complete": True,
        },
        "metric_variance_decomposition": metric_report,
        "reward_group_std": {
            "shared_initial_latent_brownian_groups": _distribution(shared_reward_stds),
            "across_initial_latent_surrogate_groups": _distribution(
                across_latent_reward_stds
            ),
            "frozen_near_zero_threshold": frozen_floor,
            "repeat_score_max_abs_reward_difference": repeat_max,
            "effective_near_zero_threshold": effective_floor,
            "shared_latent_exact_zero_rate": sum(
                value == 0.0 for value in shared_reward_stds
            )
            / len(shared_reward_stds),
            "shared_latent_near_zero_rate": shared_near / len(shared_reward_stds),
            "across_latent_exact_zero_rate": sum(
                value == 0.0 for value in across_latent_reward_stds
            )
            / len(across_latent_reward_stds),
            "across_latent_near_zero_rate": across_near
            / len(across_latent_reward_stds),
        },
        "repeat_score_audit": {
            "count": len(repeat_reward_differences),
            "reward_abs_difference": (
                _distribution(repeat_reward_differences)
                if repeat_reward_differences
                else None
            ),
            "metric_max_abs_difference": {
                name: max(values) for name, values in sorted(repeat_metric_differences.items())
            },
        },
        "duplicate_audit": {
            "sde_terminal_mel_duplicate_count": duplicate_terminal,
            "sde_terminal_mel_duplicate_rate": duplicate_terminal / sde_row_count,
            "sde_pcm16_wav_duplicate_count": duplicate_wav,
            "sde_pcm16_wav_duplicate_rate": duplicate_wav / sde_row_count,
        },
        "distance_audit_across_utterances": {
            name: _distribution(values) for name, values in sorted(distance_fields.items())
        },
        "reward_decomposition_by_utterance": reward_decomposition_by_utterance,
        "interpretation": {
            "status": "REVIEW_REQUIRED",
            "primary_fields": [
                "metric_variance_decomposition.reward.ratio_of_aggregate_variance",
                "reward_group_std.shared_latent_near_zero_rate",
                "reward_group_std.across_latent_near_zero_rate",
                "duplicate_audit",
                "ERes2Net speaker and SpeechBERTScore content dispersion",
            ],
            "note": (
                "Freeze the production initial-latent coupling only after reviewing "
                "reward signal, content/speaker stability, and ODE-vs-SDE deltas together."
            ),
        },
    }


def _retain_audio(
    *, method: str, latent_index: int, brownian_index: int | None, count: int
) -> bool:
    ordered = [("sde", 0, 0), ("ode", 0, None)]
    return (method, latent_index, brownian_index) in ordered[:count]


def _write_scored_wave(path: Path, endpoint, *, sample_rate: int, subtype: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, endpoint.normalized_waveform, sample_rate, subtype=subtype)
    return int(np.asarray(endpoint.normalized_waveform).size)


def _score_endpoint(
    *,
    endpoint,
    method: str,
    utterance: str,
    transcript: str,
    latent_index: int,
    brownian_index: int | None,
    initial_latent_seed: int,
    brownian_seed: int | None,
    window_start: int,
    worker_rank: int,
    clean_path: Path,
    audio_dir: Path,
    bundle,
    config: Mapping,
    dnsmos,
    fidelity,
    composite_evaluators,
    reward_definition: Mapping,
    repeat_score: bool,
) -> tuple[dict, int]:
    from rl.common.protocol import score_evaluation_file

    brownian_label = "none" if brownian_index is None else f"{brownian_index:02d}"
    candidate_id = (
        f"{utterance}:{method}:z{latent_index:02d}:b{brownian_label}:w{window_start}"
    )
    audio_path = audio_dir / (
        f"{utterance}__{method}__z{latent_index:02d}__b{brownian_label}.wav"
    )
    samples = _write_scored_wave(
        audio_path,
        endpoint,
        sample_rate=bundle.output_sample_rate,
        subtype=str(config["normalization"]["output_subtype"]),
    )
    scored_hash = _sha256_file(audio_path)
    metrics = score_evaluation_file(
        audio_path=audio_path,
        clean_path=clean_path,
        transcript=transcript,
        dnsmos=dnsmos,
        fidelity=fidelity,
        paired=bool(config["evaluation"]["paired_metrics"]),
        composite_evaluators=composite_evaluators,
        reward_definition=reward_definition,
    )
    reward = compute_training_reward(metrics, reward_definition)
    if _sha256_file(audio_path) != scored_hash:
        raise RuntimeError(f"scored diagnostic WAV changed: {audio_path}")
    repeat_payload = None
    if repeat_score:
        repeated_metrics = score_evaluation_file(
            audio_path=audio_path,
            clean_path=clean_path,
            transcript=transcript,
            dnsmos=dnsmos,
            fidelity=fidelity,
            paired=bool(config["evaluation"]["paired_metrics"]),
            composite_evaluators=composite_evaluators,
            reward_definition=reward_definition,
        )
        repeated_reward = compute_training_reward(repeated_metrics, reward_definition)
        if _sha256_file(audio_path) != scored_hash:
            raise RuntimeError(f"repeat-scored diagnostic WAV changed: {audio_path}")
        repeated_row = {
            "reward": float(repeated_reward["reward"]),
            "metrics": repeated_metrics,
            "reward_components": repeated_reward,
        }
        repeat_payload = {
            **repeated_row,
            "numeric_metrics": _numeric_metrics(repeated_row),
        }

    retain = _retain_audio(
        method=method,
        latent_index=latent_index,
        brownian_index=brownian_index,
        count=int(config["artifacts"]["audit_wavs_per_utterance"]),
    )
    if not retain:
        audio_path.unlink()
    row = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "worker_rank": int(worker_rank),
        "utterance": utterance,
        "method": method,
        "latent_index": int(latent_index),
        "brownian_index": brownian_index,
        "initial_latent_seed": int(initial_latent_seed),
        "brownian_seed": brownian_seed,
        "nfe": int(config["sampler"]["nfe"]),
        "window_start": int(window_start),
        "window_size": int(config["sampler"]["window_size"]),
        "diffusion": float(config["sampler"]["diffusion"]) if method == "sde" else 0.0,
        "reward": float(reward["reward"]),
        "reward_components": reward,
        "metrics": metrics,
        "terminal_mel_sha256": endpoint.terminal_mel_sha256,
        "normalized_float_waveform_sha256": endpoint.normalized_waveform_sha256,
        "scored_wav_sha256": scored_hash,
        "audio_samples": samples,
        "audit_audio_retained": retain,
        "audio_path": str(audio_path) if retain else None,
        "repeat_score": repeat_payload,
    }
    return row, samples


def _sample_utterance(
    *,
    task: Mapping,
    worker_rank: int,
    bundle,
    conditioning: ConditioningProtocol,
    config: Mapping,
    dnsmos,
    fidelity,
    composite_evaluators,
    reward_definition: Mapping,
    output_dir: Path,
) -> tuple[list[dict], dict]:
    from rl.common.flow_objective import waveform_to_mel

    utterance = str(task["utterance"])
    transcript = str(task["transcript"])
    window_start = int(task["window_start"])
    noisy_path = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
    clean_path = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
    if not noisy_path.is_file() or not clean_path.is_file():
        raise FileNotFoundError(f"missing diagnostic VoiceBank pair: {utterance}")
    condition = waveform_to_mel(bundle, str(noisy_path)).to(bundle.device)
    latent_count = int(config["diagnostic"]["initial_latents_per_utterance"])
    continuation_count = int(
        config["diagnostic"]["brownian_continuations_per_latent"]
    )
    latent_seeds = [
        stable_seed(
            int(config["sampler"]["latent_seed_base"]),
            "matched_latent",
            utterance,
            latent_index,
        )
        for latent_index in range(latent_count)
    ]
    initial = bundle._fixed_latents(tuple(condition.shape[1:]), latent_seeds, torch.float32)
    spec = WindowSpec(
        nfe=int(config["sampler"]["nfe"]),
        start_step=window_start,
        window_size=int(config["sampler"]["window_size"]),
    )
    mask = torch.ones(initial.shape[:2], device=bundle.device, dtype=torch.bool)

    def velocity_fn(state, time_value, frame_mask):
        return policy_velocity(
            bundle,
            state=state,
            condition_mel=condition,
            time=time_value,
            frame_mask=frame_mask,
            conditioning=conditioning,
            cfg_strength=float(config["sampler"]["cfg_strength"]),
        )

    rows = []
    audio_samples = 0
    audio_dir = output_dir / "audit_audio" / f"rank_{worker_rank}" / utterance
    ode_sampler = FlowSEWindowedSDESampler(diffusion=0.0, offload_records_to_cpu=True)
    ode_dummy_brownian = [
        stable_seed(
            int(config["sampler"]["brownian_seed_base"]),
            "ode_unused",
            utterance,
            latent_index,
        )
        for latent_index in range(latent_count)
    ]
    ode_rollout = ode_sampler.rollout_group(
        initial,
        frame_mask=mask,
        spec=spec,
        velocity_fn=velocity_fn,
        initial_latent_seeds=latent_seeds,
        brownian_seeds=ode_dummy_brownian,
    )
    ode_endpoints = bundle.decode_group(
        ode_rollout.terminal.to(bundle.device),
        latent_seeds,
        target_dbfs=float(config["normalization"]["target_dbfs"]),
        peak_ceiling=float(config["normalization"]["peak_ceiling"]),
    )
    ode_mels = []
    ode_waves = []
    for latent_index, endpoint in enumerate(ode_endpoints):
        row, samples = _score_endpoint(
            endpoint=endpoint,
            method="ode",
            utterance=utterance,
            transcript=transcript,
            latent_index=latent_index,
            brownian_index=None,
            initial_latent_seed=latent_seeds[latent_index],
            brownian_seed=None,
            window_start=window_start,
            worker_rank=worker_rank,
            clean_path=clean_path,
            audio_dir=audio_dir,
            bundle=bundle,
            config=config,
            dnsmos=dnsmos,
            fidelity=fidelity,
            composite_evaluators=composite_evaluators,
            reward_definition=reward_definition,
            repeat_score=False,
        )
        rows.append(row)
        audio_samples += samples
        ode_mels.append(endpoint.terminal_mel.numpy())
        ode_waves.append(np.asarray(endpoint.normalized_waveform, dtype=np.float32))

    sde_sampler = FlowSEWindowedSDESampler(
        diffusion=float(config["sampler"]["diffusion"]),
        offload_records_to_cpu=True,
    )
    sde_mels = []
    sde_waves = []
    for latent_index in range(latent_count):
        repeated_initial = initial[latent_index : latent_index + 1].expand(
            continuation_count, -1, -1
        ).clone()
        repeated_mask = mask[latent_index : latent_index + 1].expand(
            continuation_count, -1
        ).clone()
        brownian_seeds = [
            stable_seed(
                int(config["sampler"]["brownian_seed_base"]),
                "matched_brownian",
                utterance,
                latent_index,
                brownian_index,
            )
            for brownian_index in range(continuation_count)
        ]
        rollout = sde_sampler.rollout_group(
            repeated_initial,
            frame_mask=repeated_mask,
            spec=spec,
            velocity_fn=velocity_fn,
            initial_latent_seeds=[latent_seeds[latent_index]] * continuation_count,
            brownian_seeds=brownian_seeds,
            allow_repeated_initial_latent_seeds=True,
        )
        endpoints = bundle.decode_group(
            rollout.terminal.to(bundle.device),
            [latent_seeds[latent_index]] * continuation_count,
            target_dbfs=float(config["normalization"]["target_dbfs"]),
            peak_ceiling=float(config["normalization"]["peak_ceiling"]),
        )
        latent_mels = []
        latent_waves = []
        for brownian_index, endpoint in enumerate(endpoints):
            repeat_score = (
                int(config["diagnostic"]["repeat_score_audits_per_utterance"]) == 1
                and latent_index == 0
                and brownian_index == 0
            )
            row, samples = _score_endpoint(
                endpoint=endpoint,
                method="sde",
                utterance=utterance,
                transcript=transcript,
                latent_index=latent_index,
                brownian_index=brownian_index,
                initial_latent_seed=latent_seeds[latent_index],
                brownian_seed=brownian_seeds[brownian_index],
                window_start=window_start,
                worker_rank=worker_rank,
                clean_path=clean_path,
                audio_dir=audio_dir,
                bundle=bundle,
                config=config,
                dnsmos=dnsmos,
                fidelity=fidelity,
                composite_evaluators=composite_evaluators,
                reward_definition=reward_definition,
                repeat_score=repeat_score,
            )
            rows.append(row)
            audio_samples += samples
            latent_mels.append(endpoint.terminal_mel.numpy())
            latent_waves.append(np.asarray(endpoint.normalized_waveform, dtype=np.float32))
        sde_mels.append(latent_mels)
        sde_waves.append(latent_waves)

    return rows, {
        "schema_version": SCHEMA_VERSION,
        "worker_rank": worker_rank,
        "utterance": utterance,
        "window_start": window_start,
        "candidate_count": len(rows),
        "candidate_audio_seconds": float(audio_samples / bundle.output_sample_rate),
        "distance_audit": _distance_audit(ode_mels, ode_waves, sde_mels, sde_waves),
    }


def _diagnostic_worker_main(
    *,
    worker_rank: int,
    device_id: int,
    config: dict,
    tasks: list[dict],
    output_dir: str,
    result_queue,
) -> None:
    try:
        resources = config["resources"]
        cpu_threads = int(resources["cpu_threads_per_worker"])
        interop_threads = int(resources["torch_interop_threads_per_worker"])
        selected_cpu_affinity = None
        if bool(resources["enforce_cpu_affinity"]) and hasattr(
            os, "sched_getaffinity"
        ):
            available_cpus = sorted(os.sched_getaffinity(0))
            required_cpus = int(resources["world_size"]) * cpu_threads
            if len(available_cpus) < required_cpus:
                raise RuntimeError(
                    "the process CPU affinity is smaller than the frozen two-worker cap: "
                    f"available={len(available_cpus)}, required={required_cpus}"
                )
            begin = worker_rank * cpu_threads
            selected_cpu_affinity = available_cpus[begin : begin + cpu_threads]
            os.sched_setaffinity(0, selected_cpu_affinity)
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(interop_threads)

        from rl.common.flowse_interface import load_flowse_bundle
        from rl.rewards.composite import (
            load_composite_reward_evaluators,
        )
        from rl.common.protocol import load_fidelity
        from rl.rewards.metrics import DNSMOSScorer

        torch.cuda.set_device(device_id)
        seed = stable_seed(int(config["run"]["seed"]), "worker", worker_rank)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if bool(config["run"].get("deterministic", True)):
            torch.use_deterministic_algorithms(True, warn_only=True)

        worker_config = json.loads(json.dumps(config))
        worker_config["composite_reward_evaluators"]["device"] = f"cuda:{device_id}"
        if bool(worker_config["evaluation"].get("fidelity", {}).get("enabled", False)):
            worker_config["evaluation"]["fidelity"]["device"] = f"cuda:{device_id}"
        bundle = load_flowse_bundle(
            worker_config["flowse_config"],
            deterministic=bool(worker_config["run"].get("deterministic", True)),
        )
        bundle.model.eval()
        conditioning = ConditioningProtocol.from_config(worker_config["conditioning"])
        dnsmos = DNSMOSScorer(worker_config["dnsmos_official_dir"])
        composite, composite_fingerprint = load_composite_reward_evaluators(
            worker_config
        )
        fidelity, fidelity_fingerprint = load_fidelity(worker_config, lazy_asr=False)
        reward_definition = resolve_training_reward(worker_config)
        calibration = verify_reward_calibration(
            worker_config, evaluator_fingerprint=composite_fingerprint
        )
        started = time.perf_counter()
        rows = []
        summaries = []
        for task in tasks:
            utterance_rows, summary = _sample_utterance(
                task=task,
                worker_rank=worker_rank,
                bundle=bundle,
                conditioning=conditioning,
                config=worker_config,
                dnsmos=dnsmos,
                fidelity=fidelity,
                composite_evaluators=composite,
                reward_definition=reward_definition,
                output_dir=Path(output_dir),
            )
            rows.extend(utterance_rows)
            summaries.append(summary)
        elapsed = time.perf_counter() - started
        shard_dir = Path(output_dir) / "shards"
        row_path = shard_dir / f"candidate_rows_rank_{worker_rank}.jsonl"
        summary_path = shard_dir / f"utterance_summaries_rank_{worker_rank}.jsonl"
        atomic_write_jsonl(row_path, rows)
        atomic_write_jsonl(summary_path, summaries)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "worker_rank": worker_rank,
            "device_id": device_id,
            "device_name": torch.cuda.get_device_name(device_id),
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity": selected_cpu_affinity,
            "released_checkpoint_sha256": bundle.checkpoint_sha256,
            "composite_evaluator_fingerprint": composite_fingerprint,
            "fidelity_evaluator_fingerprint": fidelity_fingerprint,
            "reward_calibration": calibration,
            "utterance_count": len(tasks),
            "candidate_count": len(rows),
            "wall_seconds": float(elapsed),
            "row_path": str(row_path),
            "summary_path": str(summary_path),
        }
        metadata_path = shard_dir / f"worker_{worker_rank}.json"
        atomic_write_json(metadata_path, metadata)
        result_queue.put({"status": "ok", "worker_rank": worker_rank, **metadata})
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "worker_rank": worker_rank,
                "traceback": traceback.format_exc(),
            }
        )


def _select_tasks(config: Mapping) -> list[dict]:
    from rl.common.protocol import strict_manifest, utterances_for_step

    manifest = strict_manifest(config["data"]["train_manifest"])
    utterances = utterances_for_step(
        list(manifest),
        step=1,
        conditions_per_step=int(config["diagnostic"]["utterance_count"]),
        seed=int(config["data"]["selection_seed"]),
    )
    starts = [int(value) for value in config["sampler"]["window_starts"]]
    return [
        {
            "selection_index": index,
            "utterance": utterance,
            "transcript": manifest[utterance],
            "window_start": starts[index % len(starts)],
        }
        for index, utterance in enumerate(utterances)
    ]


def _collect_worker_results(processes: Sequence, result_queue, world_size: int) -> list[dict]:
    results = []
    while len(results) < world_size:
        try:
            result = result_queue.get(timeout=60.0)
        except queue.Empty:
            failed = [process.exitcode for process in processes if process.exitcode not in {None, 0}]
            if failed:
                raise RuntimeError(f"diagnostic worker exited without a result: {failed}")
            continue
        if result.get("status") == "error":
            raise RuntimeError(str(result["traceback"]))
        results.append(result)
    ranks = [int(result["worker_rank"]) for result in results]
    if len(set(ranks)) != world_size:
        raise RuntimeError("duplicate diagnostic worker result")
    return sorted(results, key=lambda item: int(item["worker_rank"]))


def run(config: dict) -> tuple[dict, Path]:
    validation = validate_sampling_diagnostic_config(config)
    # Resolve immutable reward artifacts in the coordinator before any worker
    # or multiprocessing semaphore is created.  Missing calibration must fail
    # as a clean preflight error, not as two independent worker failures.
    reward_definition = resolve_training_reward(config)
    _preflight_calibration_fingerprint(config, reward_definition)
    if not torch.cuda.is_available():
        raise RuntimeError("sampling diagnostic requires CUDA")
    device_ids = [int(value) for value in config["resources"]["device_ids"]]
    if max(device_ids) >= torch.cuda.device_count():
        raise RuntimeError(
            "configured two-GPU diagnostic topology is unavailable: "
            f"device_ids={device_ids}, visible={torch.cuda.device_count()}"
        )
    tasks = _select_tasks(config)
    partitions = partition_utterances(tasks, int(config["resources"]["world_size"]))
    config_hash = _canonical_hash(config)
    # Diagnostics share the same human-readable run directory as training;
    # the config fingerprint remains in the report metadata when needed.
    output_dir = Path(config["output_root"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"diagnostic output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "frozen_config.json", config)
    atomic_write_json(output_dir / "selected_utterances.json", tasks)

    cpu_threads = str(int(config["resources"]["cpu_threads_per_worker"]))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = cpu_threads
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    started = time.perf_counter()
    try:
        for rank, (device_id, shard) in enumerate(zip(device_ids, partitions, strict=True)):
            process = context.Process(
                target=_diagnostic_worker_main,
                kwargs={
                    "worker_rank": rank,
                    "device_id": device_id,
                    "config": config,
                    "tasks": shard,
                    "output_dir": str(output_dir),
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)
        worker_results = _collect_worker_results(
            processes, result_queue, int(config["resources"]["world_size"])
        )
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        result_queue.close()
        result_queue.join_thread()
    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(f"diagnostic worker exit code {process.exitcode}")

    checkpoint_hashes = {result["released_checkpoint_sha256"] for result in worker_results}
    evaluator_hashes = {
        json.dumps(result["composite_evaluator_fingerprint"], sort_keys=True)
        for result in worker_results
    }
    fidelity_hashes = {
        json.dumps(result["fidelity_evaluator_fingerprint"], sort_keys=True)
        for result in worker_results
    }
    if len(checkpoint_hashes) != 1 or len(evaluator_hashes) != 1 or len(fidelity_hashes) != 1:
        raise RuntimeError("two diagnostic workers did not use identical frozen states")

    rows = []
    summaries = []
    for result in worker_results:
        rows.extend(_read_jsonl(Path(result["row_path"])))
        summaries.extend(_read_jsonl(Path(result["summary_path"])))
    task_order = {task["utterance"]: int(task["selection_index"]) for task in tasks}
    method_order = {"ode": 0, "sde": 1}
    rows.sort(
        key=lambda row: (
            task_order[row["utterance"]],
            method_order[row["method"]],
            int(row["latent_index"]),
            -1 if row["brownian_index"] is None else int(row["brownian_index"]),
        )
    )
    summaries.sort(key=lambda row: task_order[row["utterance"]])
    atomic_write_jsonl(output_dir / "candidate_rows.jsonl", rows)
    atomic_write_jsonl(output_dir / "utterance_summaries.jsonl", summaries)
    analysis = analyze_sampling_rows(rows, summaries, config)
    wall_seconds = time.perf_counter() - started
    report = {
        "status": "SAMPLING-DIAGNOSTIC-COMPLETE",
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_hash,
        "validation": validation,
        "policy_state": "released_base_lora_disabled",
        "lora_injected": False,
        "released_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "workers": worker_results,
        "two_gpu_wall_seconds": float(wall_seconds),
        "analysis": analysis,
        "artifacts": {
            "frozen_config": str(output_dir / "frozen_config.json"),
            "selected_utterances": str(output_dir / "selected_utterances.json"),
            "candidate_rows": str(output_dir / "candidate_rows.jsonl"),
            "utterance_summaries": str(output_dir / "utterance_summaries.jsonl"),
        },
    }
    atomic_write_json(output_dir / "sampling_diagnostic_report.json", report)
    return report, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the two-GPU FlowSE-GRPO matched-latent sampling diagnostic"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validation = validate_sampling_diagnostic_config(config)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    report, output_dir = run(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'sampling_diagnostic_report.json'}")


if __name__ == "__main__":
    main()
