"""Iterative, audio-only AdvantageFlow trainer for the released FlowSE model.

The trainer deliberately lives outside upstream FlowSE.  It uses ODE candidates,
one complete-batch advantage scale, a real persistent AdamW optimizer, an EMA
rollout policy, and the LoRA-disabled released model as the frozen reference.
The training reward is config-selected; all DNSMOS components are reported.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import queue
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm.auto import tqdm

from .advantage_estimation import cluster_bootstrap_mean
from rl.common.conditioning import ConditioningProtocol
from rl.rewards.evaluators import FidelityEvaluators, resolve_hf_model
from rl.common.flowse_interface import load_flowse_bundle
from .reward_evaluators import (
    FlowSEGRPOCompositeEvaluators,
    load_flowse_grpo_composite_evaluators,
)
from rl.common.flow_objective import RolloutTrainingCondition, waveform_to_mel
from rl.common.lora import (
    AdapterState,
    inject_lora,
    load_lora,
    lora_parameters,
    named_lora_parameters,
    snapshot_lora,
)
from rl.rewards.metrics import DNSMOSScorer, paired_metrics
from .advantage_flow import (
    adapter_distance,
    apply_reward_constraints,
    compute_paper_global_advantages,
    ema_adapter_state,
    length_adaptive_microbatch_size,
    load_training_checkpoint,
    paper_advantageflow_loss,
    restore_rng_state,
    save_training_checkpoint,
    stable_seed,
)
from rl.gaaf.gradient_aligned_advantage_flow import (
    COMPONENTS as GAAF_COMPONENTS,
    calibration_due as gaaf_calibration_due,
    component_advantage_streams,
    convex_fuse_streams,
    marble_fuse_streams,
    marble_calibration_due,
    marble_simplex_weights,
    observed_gate_weights,
    projected_gaaf_calibration_due,
    projected_gaaf_observed_weights,
    replace_advantages,
    reward_induced_gradients,
    update_marble_state,
    update_projected_gaaf_state,
    validate_marble_state,
    update_gate_state,
    validate_gate_state,
    validate_projected_gaaf_state,
)
from .protocol import (
    build_training_protocol,
    sha256_file,
    training_nfe_spec,
)
from .protocol import sha256_json
from .checkpoint import (
    append_jsonl_batch_durable as _append_jsonl_batch,
    atomic_write_json as _write_json,
    commit_step_transaction,
    recover_step_transaction,
    step_commit_id,
    truncate_jsonl_after_step,
)
from rl.common.shared_initialization import prepare_or_load_shared_lora_snapshot
from rl.rewards.specification import (
    DNSMOS_OVRL_RAW,
    DNSMOS_SPEAKER,
    FLOWSE_GRPO_COMPOSITE,
    compute_training_reward,
    resolve_training_reward,
)


DNSMOS_KEYS = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808")
COMPOSITE_REPORT_KEYS = (
    "flowse_grpo_composite_reward",
    "eres2net_speaker_similarity",
    "speechbertscore",
)


def _uses_composite_evaluators(reward_definition: Mapping) -> bool:
    return reward_definition["name"] == FLOWSE_GRPO_COMPOSITE or isinstance(
        reward_definition.get("auxiliary_composite"), Mapping
    )


def _composite_definition(reward_definition: Mapping) -> Mapping | None:
    if reward_definition["name"] == FLOWSE_GRPO_COMPOSITE:
        return reward_definition
    auxiliary = reward_definition.get("auxiliary_composite")
    return auxiliary if isinstance(auxiliary, Mapping) else None


def _sample_training_nfe(
    config: Mapping,
    *,
    step: int,
    utterance: str,
    condition_index: int,
) -> int:
    """Select one deterministic training NFE for a logical candidate group."""

    rollout = config["rollout"]
    spec = training_nfe_spec(rollout)
    if not spec["mixed"]:
        return int(spec["configured"])
    seed = stable_seed(
        int(spec["seed_base"]),
        "training_nfe",
        int(step),
        str(utterance),
        int(condition_index),
    )
    return random.Random(seed).randint(int(spec["minimum"]), int(spec["maximum"]))


def _training_nfe_by_condition(
    config: Mapping,
    *,
    step: int,
    utterances: list[str],
) -> list[int]:
    """Return the deterministic training-NFE schedule for one AF step.

    ``balanced_per_step`` keeps the NFE composition of every logical batch
    fixed while shuffling which condition receives each solver budget. This
    gives solver-level diversity without changing AF's group-relative
    credit-assignment: all K candidates for a condition still use one NFE.
    The legacy ``random_per_group`` policy remains available for ablations.
    """

    rollout = config["rollout"]
    spec = training_nfe_spec(rollout)
    count = len(utterances)
    if count < 1:
        raise ValueError("at least one logical condition is required")
    if not spec["mixed"]:
        return [int(spec["configured"])] * count
    values = list(range(int(spec["minimum"]), int(spec["maximum"]) + 1))
    schedule = str(rollout.get("training_nfe_schedule", "random_per_group"))
    if schedule == "balanced_per_step":
        if count % len(values) != 0:
            raise ValueError(
                "balanced_per_step requires conditions_per_step to be divisible "
                f"by the number of NFE values ({len(values)}), got {count}"
            )
        repeats = count // len(values)
        result = [value for value in values for _ in range(repeats)]
        random.Random(
            stable_seed(int(spec["seed_base"]), "training_nfe_schedule", int(step))
        ).shuffle(result)
        return result
    return [
        _sample_training_nfe(
            config,
            step=step,
            utterance=utterance,
            condition_index=condition_index,
        )
        for condition_index, utterance in enumerate(utterances)
    ]


def _synchronize_cuda(device=None) -> None:
    if not torch.cuda.is_available():
        return
    if device is not None and not isinstance(device, int):
        if torch.device(device).type != "cuda":
            return
    torch.cuda.synchronize(device=device)


def _cuda_memory_reservation_enabled(config: Mapping) -> bool:
    reservation = config.get("cuda_memory_reservation")
    return isinstance(reservation, Mapping) and bool(reservation.get("enabled", False))


def _reserve_cuda_allocator_memory(
    config: Mapping,
    *,
    device,
    role: str,
) -> dict:
    """Claim the configured peak budget in this process's CUDA caching allocator."""

    reservation = config.get("cuda_memory_reservation")
    if not isinstance(reservation, Mapping) or not bool(
        reservation.get("enabled", False)
    ):
        return {"enabled": False, "role": role}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA memory reservation requires CUDA")

    if isinstance(device, int):
        device_index = int(device)
    else:
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError(f"CUDA memory reservation received non-CUDA device: {device}")
        device_index = (
            int(resolved_device.index)
            if resolved_device.index is not None
            else int(torch.cuda.current_device())
        )
    if int(torch.cuda.current_device()) != device_index:
        raise RuntimeError(
            "CUDA memory reservation must run in the owning process/device: "
            f"current={torch.cuda.current_device()} requested={device_index}"
        )

    gib = 1024**3
    target_key = (
        "coordinator_target_reserved_gib"
        if role == "coordinator"
        else "worker_target_reserved_gib"
    )
    target_bytes = int(float(reservation[target_key]) * gib)
    if target_bytes == 0:
        return {
            "enabled": False,
            "role": role,
            "reason": "role_target_disabled",
            "device_index": device_index,
        }
    minimum_free_bytes = int(
        float(reservation["minimum_driver_free_gib"]) * gib
    )
    chunk_bytes = max(1, int(float(reservation["allocation_chunk_gib"]) * gib))
    require_target = bool(reservation["require_target"])
    total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
    if target_bytes + minimum_free_bytes > total_bytes:
        raise ValueError(
            "CUDA reservation target plus driver safety margin exceeds device memory: "
            f"target={target_bytes / gib:.2f}GiB "
            f"minimum_free={minimum_free_bytes / gib:.2f}GiB "
            f"total={total_bytes / gib:.2f}GiB"
        )

    _synchronize_cuda(device_index)
    allocated_before = int(torch.cuda.memory_allocated(device_index))
    reserved_before = int(torch.cuda.memory_reserved(device_index))
    free_before, driver_total = torch.cuda.mem_get_info(device_index)
    free_before = int(free_before)
    driver_total = int(driver_total)
    new_driver_bytes_needed = max(0, target_bytes - reserved_before)
    maximum_new_driver_bytes = max(0, free_before - minimum_free_bytes)
    effective_target = target_bytes
    if new_driver_bytes_needed > maximum_new_driver_bytes:
        if require_target:
            raise RuntimeError(
                "CUDA peak-memory reservation failed before training because the "
                "selected GPU is already occupied: "
                f"role={role} device=cuda:{device_index} "
                f"target={target_bytes / gib:.2f}GiB "
                f"reserved_now={reserved_before / gib:.2f}GiB "
                f"driver_free={free_before / gib:.2f}GiB "
                f"required_driver_free_after={minimum_free_bytes / gib:.2f}GiB. "
                "Choose a freer physical GPU and restart."
            )
        effective_target = reserved_before + maximum_new_driver_bytes

    # Existing cached free blocks are included in memory_reserved(), so the
    # temporary live allocation must cover target - memory_allocated().  Holding
    # all chunks simultaneously first consumes cached blocks and then grows the
    # allocator to the requested target.  Deleting the chunks without calling
    # empty_cache() leaves that memory reserved and reusable by the real model.
    temporary_bytes = max(0, effective_target - allocated_before)
    buffers = []
    remaining = temporary_bytes
    try:
        while remaining > 0:
            size = min(chunk_bytes, remaining)
            buffers.append(
                torch.empty(size, dtype=torch.uint8, device=f"cuda:{device_index}")
            )
            remaining -= size
        _synchronize_cuda(device_index)
    except Exception as exc:
        buffers.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA peak-memory reservation allocation failed before training: "
            f"role={role} device=cuda:{device_index} "
            f"target={effective_target / gib:.2f}GiB"
        ) from exc
    buffers.clear()
    gc.collect()
    _synchronize_cuda(device_index)

    allocated_after = int(torch.cuda.memory_allocated(device_index))
    reserved_after = int(torch.cuda.memory_reserved(device_index))
    free_after, _ = torch.cuda.mem_get_info(device_index)
    free_after = int(free_after)
    if require_target and reserved_after < target_bytes:
        raise RuntimeError(
            "CUDA caching allocator did not retain the required peak budget: "
            f"role={role} device=cuda:{device_index} "
            f"target={target_bytes / gib:.2f}GiB "
            f"reserved={reserved_after / gib:.2f}GiB"
        )
    report = {
        "enabled": True,
        "role": role,
        "device_index": device_index,
        "target_reserved_bytes": target_bytes,
        "effective_target_reserved_bytes": effective_target,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "driver_free_before_bytes": free_before,
        "allocated_after_bytes": allocated_after,
        "reserved_after_bytes": reserved_after,
        "driver_free_after_bytes": free_after,
        "driver_total_bytes": driver_total,
        "require_target": require_target,
    }
    print(
        "CUDA peak-memory reservation: PASS "
        f"role={role} device=cuda:{device_index} "
        f"reserved={reserved_after / gib:.2f}GiB "
        f"driver_free={free_after / gib:.2f}GiB",
        flush=True,
    )
    return report


def _cuda_peak_memory_report(*, device, role: str) -> dict:
    if not torch.cuda.is_available():
        return {
            "role": role,
            "device_index": None,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "allocated_at_report_bytes": 0,
            "reserved_at_report_bytes": 0,
        }
    if isinstance(device, int):
        device_index = int(device)
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return {
                "role": role,
                "device_index": None,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "allocated_at_report_bytes": 0,
                "reserved_at_report_bytes": 0,
            }
        device_index = (
            int(resolved.index)
            if resolved.index is not None
            else int(torch.cuda.current_device())
        )
    return {
        "role": role,
        "device_index": device_index,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        "allocated_at_report_bytes": int(torch.cuda.memory_allocated(device_index)),
        "reserved_at_report_bytes": int(torch.cuda.memory_reserved(device_index)),
    }


def _compute_training_accounting(
    step_records: list[Mapping], *, output_dir: Path, world_size: int
) -> dict:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    observed_steps = [int(row.get("step", -1)) for row in step_records]
    if observed_steps != list(range(1, len(step_records) + 1)):
        raise ValueError("training accounting log is not contiguous from step 1")
    recorded_world_sizes = {
        int(row.get("allocated_rollout_world_size", world_size))
        for row in step_records
    }
    if recorded_world_sizes and recorded_world_sizes != {int(world_size)}:
        raise ValueError("rollout GPU topology changed across resume")
    accounting_records = []
    exact_step_timing = True
    for row in step_records:
        value = json.loads(json.dumps(row))
        timing_path = (
            output_dir
            / "accounting_steps"
            / f"step_{int(row['step']):06d}.json"
        )
        if timing_path.is_file():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            if int(timing.get("step", -1)) != int(row["step"]):
                raise ValueError("AF accounting-step artifact has the wrong step")
            value["timing_seconds"].update(timing["timing_seconds"])
            if timing.get("exact_accounting") is False:
                exact_step_timing = False
        else:
            exact_step_timing = False
        accounting_records.append(value)
    phase_names = (
        "rollout_and_reward",
        "loss_and_backward",
        "optimizer",
        "ema",
        "checkpoint_artifact_io",
    )
    phase_seconds = {
        phase: float(
            sum(
                float(row.get("timing_seconds", {}).get(phase, 0.0))
                for row in accounting_records
            )
        )
        for phase in phase_names
    }
    training_wall = float(
        sum(
            float(
                row.get("timing_seconds", {}).get(
                    "training_wall_excluding_validation",
                    row.get("timing_seconds", {}).get("through_update", 0.0),
                )
            )
            for row in accounting_records
        )
    )
    validation_seconds = 0.0
    validation_timed = 0
    validation_untimed = 0
    for path in sorted(output_dir.glob("evaluation_step_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if "timing_seconds" in report:
            validation_seconds += float(report["timing_seconds"])
            validation_timed += 1
        else:
            validation_untimed += 1
    active_rollout = float(
        sum(
            float(
                row.get("rollout_geometry", {})
                .get("active_gpu_seconds_by_phase", {})
                .get("rollout", 0.0)
            )
            for row in accounting_records
        )
    )
    active_reward = float(
        sum(
            float(
                row.get("rollout_geometry", {})
                .get("active_gpu_seconds_by_phase", {})
                .get("reward", 0.0)
            )
            for row in accounting_records
        )
    )
    active_gpu_seconds = {
        "rollout": active_rollout,
        "reward": active_reward,
        "loss_and_backward": phase_seconds["loss_and_backward"],
        "optimizer": phase_seconds["optimizer"],
        "ema": 0.0,
    }
    logical_conditions = int(
        sum(
            int(row.get("rollout_geometry", {}).get("logical_conditions", 0))
            for row in accounting_records
        )
    )
    logical_endpoints = int(
        sum(
            int(row.get("rollout_geometry", {}).get("logical_endpoints", 0))
            for row in accounting_records
        )
    )
    training_nfe_values = set()
    rollout_model_forwards = 0
    rollout_logical_velocity_examples = 0
    for row in accounting_records:
        geometry = row.get("rollout_geometry", {})
        per_condition = geometry.get("training_nfe_by_condition")
        if isinstance(per_condition, list) and per_condition:
            condition_nfe = [int(value) for value in per_condition]
            training_nfe_values.update(condition_nfe)
            rollout_model_forwards += sum(condition_nfe)
            candidates = int(geometry.get("candidates_per_condition", 0))
            rollout_logical_velocity_examples += sum(condition_nfe) * candidates
            continue
        scalar_nfe = geometry.get("training_nfe")
        if scalar_nfe is None:
            continue
        scalar_nfe = int(scalar_nfe)
        training_nfe_values.add(scalar_nfe)
        logical_conditions_in_step = int(geometry.get("logical_conditions", 0))
        logical_endpoints_in_step = int(geometry.get("logical_endpoints", 0))
        rollout_model_forwards += logical_conditions_in_step * scalar_nfe
        rollout_logical_velocity_examples += logical_endpoints_in_step * scalar_nfe
    workload = {
        "prompt_instances": logical_conditions,
        "unique_utterance_coverage": len(
            {
                str(value)
                for row in accounting_records
                for value in row.get("utterances", [])
            }
        ),
        "scored_trajectories": logical_endpoints,
        "eligible_trajectories": logical_endpoints,
        "used_trajectories": logical_endpoints,
        "reward_calls": logical_endpoints,
        "optimizer_updates": len(accounting_records),
        "gaaf_gradient_calibrations": sum(
            bool(row.get("gaaf", {}).get("calibration_performed", False))
            for row in accounting_records
        ),
        "gaaf_calibration_backward_calls": sum(
            int(row.get("gaaf", {}).get("calibration_backward_calls", 0))
            for row in accounting_records
        ),
        "projected_gaaf_gradient_calibrations": sum(
            bool(
                row.get("projected_gaaf", {}).get(
                    "calibration_performed", False
                )
            )
            for row in accounting_records
        ),
        "projected_gaaf_calibration_backward_calls": sum(
            int(
                row.get("projected_gaaf", {}).get(
                    "calibration_backward_calls", 0
                )
            )
            for row in accounting_records
        ),
        "marble_gradient_calibrations": sum(
            bool(row.get("marble", {}).get("calibration_performed", False))
            for row in accounting_records
        ),
        "marble_calibration_backward_calls": sum(
            int(row.get("marble", {}).get("calibration_backward_calls", 0))
            for row in accounting_records
        ),
        "training_nfe_values": sorted(training_nfe_values),
    }
    workload.update(
        {
            "rollout_logical_velocity_examples": int(
                rollout_logical_velocity_examples
            ),
            "rollout_model_forwards": int(rollout_model_forwards),
            "old_policy_loss_examples": logical_endpoints,
            "reference_policy_loss_examples": logical_endpoints,
            "current_policy_loss_examples": logical_endpoints,
            "backward_endpoint_examples": logical_endpoints,
            "gaaf_calibration_backward_endpoint_examples": sum(
                int(row.get("rollout_geometry", {}).get("logical_endpoints", 0))
                * int(row.get("gaaf", {}).get("calibration_backward_calls", 0))
                for row in accounting_records
            ),
            "projected_gaaf_calibration_backward_endpoint_examples": sum(
                int(row.get("rollout_geometry", {}).get("logical_endpoints", 0))
                * int(
                    row.get("projected_gaaf", {}).get(
                        "calibration_backward_calls", 0
                    )
                )
                for row in accounting_records
            ),
            "marble_calibration_backward_endpoint_examples": sum(
                int(row.get("rollout_geometry", {}).get("logical_endpoints", 0))
                * int(row.get("marble", {}).get("calibration_backward_calls", 0))
                for row in accounting_records
            ),
        }
    )
    return {
        "schema_version": 1,
        "accounting_source": (
            "committed_training_steps_jsonl_plus_exact_step_artifacts_across_resume"
        ),
        "physical_gpu_count": int(world_size),
        "phase_wall_seconds": phase_seconds,
        "active_gpu_seconds_by_phase": active_gpu_seconds,
        "active_phase_accounting_complete": all(
            "active_gpu_seconds_by_phase" in row.get("rollout_geometry", {})
            for row in accounting_records
        ),
        "training_wall_accounting_complete": all(
            "training_wall_excluding_validation" in row.get("timing_seconds", {})
            for row in accounting_records
        )
        and exact_step_timing,
        "allocated_training_wall_seconds": training_wall,
        "allocated_training_gpu_seconds": training_wall * world_size,
        "allocated_training_gpu_hours": training_wall * world_size / 3600.0,
        "validation_wall_seconds": validation_seconds,
        "validation_allocated_gpu_seconds": validation_seconds * world_size,
        "validation_allocated_gpu_hours": validation_seconds * world_size / 3600.0,
        "validation_reports_timed": validation_timed,
        "validation_reports_without_timing": validation_untimed,
        "completed_optimizer_steps": len(step_records),
        "workload_statistics": workload,
        "validation_excluded_from_training_budget": True,
    }
EVALUATION_KEYS = list(DNSMOS_KEYS) + [
    "pesq_wb",
    "stoi",
    "speaker_similarity",
    "wer",
    *COMPOSITE_REPORT_KEYS,
]


def _speaker(utterance: str) -> str:
    parts = utterance.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"cannot extract speaker from {utterance!r}")
    speaker = parts[0].lower()
    # VoiceBank IDs use pXXX_*, while LibriTTS IDs use numeric reader IDs.
    if speaker.startswith("p") and speaker[1:].isdigit():
        return speaker
    if speaker.isdigit():
        return speaker
    raise ValueError(f"cannot extract speaker from {utterance!r}")


def speaker_balanced_epoch(
    utterances: list[str], *, seed: int, epoch: int
) -> list[str]:
    """Round-robin a frozen per-speaker shuffle; never sort by utterance ID."""

    by_speaker: dict[str, list[str]] = {}
    for utterance in utterances:
        by_speaker.setdefault(_speaker(utterance), []).append(utterance)
    speakers = sorted(by_speaker)
    random.Random(stable_seed(seed, "speaker_order", epoch)).shuffle(speakers)
    for speaker, items in by_speaker.items():
        items.sort()
        random.Random(stable_seed(seed, "utterances", epoch, speaker)).shuffle(items)
    output = []
    position = 0
    while True:
        progressed = False
        for speaker in speakers:
            if position < len(by_speaker[speaker]):
                output.append(by_speaker[speaker][position])
                progressed = True
        if not progressed:
            break
        position += 1
    if len(output) != len(utterances) or len(set(output)) != len(output):
        raise AssertionError("speaker-balanced epoch is not a permutation")
    return output


def utterances_for_step(
    utterances: list[str],
    *,
    step: int,
    conditions_per_step: int,
    seed: int,
    stride_per_step: int | None = None,
) -> list[str]:
    if step < 1 or conditions_per_step < 1:
        raise ValueError("step and conditions_per_step must be positive")
    count = len(utterances)
    if count < conditions_per_step:
        raise ValueError("training manifest is smaller than one logical batch")
    stride = conditions_per_step if stride_per_step is None else int(stride_per_step)
    if stride < conditions_per_step:
        raise ValueError("stride_per_step must be at least conditions_per_step")
    start = (step - 1) * stride
    selected = []
    cursor = start
    while len(selected) < conditions_per_step:
        epoch = cursor // count
        position = cursor % count
        order = speaker_balanced_epoch(utterances, seed=seed, epoch=epoch)
        candidate = order[position]
        cursor += 1
        if candidate not in selected:
            selected.append(candidate)
    return selected


def _scheduler(optimizer, config: dict):
    total = int(
        config["optimizer"].get(
            "schedule_total_steps", config["run"]["optimizer_steps"]
        )
    )
    warmup = int(config["optimizer"].get("warmup_steps", 0))

    def multiplier(index: int) -> float:
        if warmup > 0 and index < warmup:
            return float(index + 1) / warmup
        progress = (index - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _load_fidelity(
    config: dict, *, lazy_asr: bool = False
) -> tuple[FidelityEvaluators | None, dict | None]:
    fidelity = config["evaluation"].get("fidelity", {"enabled": False})
    if not fidelity.get("enabled", False):
        return None, None
    locked_path = Path(fidelity["source_locked_config"])
    locked = yaml.safe_load(locked_path.read_text(encoding="utf-8"))
    evaluator_config = locked["evaluators"]
    speaker = resolve_hf_model(evaluator_config["speaker"])
    asr = resolve_hf_model(evaluator_config["asr"])
    evaluators = FidelityEvaluators.load(
        speaker,
        asr,
        device=str(fidelity.get("device", evaluator_config["device"])),
        lazy_asr=lazy_asr,
    )
    return evaluators, {
        "speaker": speaker.fingerprint(),
        "asr": asr.fingerprint(),
    }


def _reward_summary(matrix: np.ndarray) -> dict:
    return {
        "mean": float(matrix.mean()),
        "std": float(matrix.std(ddof=0)),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "per_condition_range_mean": float(np.ptp(matrix, axis=1).mean()),
        "top2_bottom2_gap_mean": float(
            np.mean(
                np.mean(np.sort(matrix, axis=1)[:, -2:], axis=1)
                - np.mean(np.sort(matrix, axis=1)[:, :2], axis=1)
            )
        ),
    }


def _write_wave(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str,
    *,
    compute_hash: bool = True,
) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype=subtype)
    return sha256_file(path) if compute_hash else None


def _generate_group_length_adaptive(
    bundle,
    noisy_path: Path,
    transcript: str,
    seeds: list[int],
    *,
    mel_frames: int,
    nfe: int,
    cfg_strength: float,
    conditioning: ConditioningProtocol,
    target_dbfs: float,
    peak_ceiling: float,
    adaptive_config: Mapping | None,
):
    """Generate the same logical K group in memory-bounded seed chunks."""

    microbatch_size = length_adaptive_microbatch_size(
        mel_frames,
        maximum_size=len(seeds),
        config=adaptive_config,
    )
    if microbatch_size == len(seeds):
        endpoints = bundle.generate_group(
            noisy_path,
            transcript,
            seeds,
            nfe=nfe,
            cfg_strength=cfg_strength,
            conditioning=conditioning,
            target_dbfs=target_dbfs,
            peak_ceiling=peak_ceiling,
        )
        return endpoints, microbatch_size

    condition_waveform = bundle.load_condition(noisy_path)
    endpoints = []
    for start in range(0, len(seeds), microbatch_size):
        chunk_seeds = seeds[start : start + microbatch_size]
        terminal = bundle.sample_fixed_latents(
            condition_waveform,
            transcript,
            chunk_seeds,
            nfe=nfe,
            cfg_strength=cfg_strength,
            conditioning=conditioning,
        )
        if int(terminal.shape[1]) != mel_frames:
            raise ValueError(
                "rollout/loss mel-frame mismatch: "
                f"expected={mel_frames}, got={int(terminal.shape[1])}"
            )
        endpoints.extend(
            bundle.decode_group(
                terminal,
                chunk_seeds,
                target_dbfs=target_dbfs,
                peak_ceiling=peak_ceiling,
            )
        )
        del terminal
    if [int(endpoint.latent_seed) for endpoint in endpoints] != [
        int(seed) for seed in seeds
    ]:
        raise AssertionError("length-adaptive rollout changed candidate order")
    return endpoints, microbatch_size


def _rollout_training_shard(
    *,
    bundle,
    rollout_state: Mapping[str, torch.Tensor],
    utterances: list[str],
    condition_indices: list[int],
    audit_audio_keys: set[tuple[int, int]],
    step: int,
    config: dict,
    conditioning: ConditioningProtocol,
    dnsmos: DNSMOSScorer,
    fidelity: FidelityEvaluators | None,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None,
    output_dir: Path,
) -> dict:
    rollout = config["rollout"]
    normalization = config["normalization"]
    candidates = int(rollout["candidates_per_condition"])
    load_lora(bundle.model.transformer, rollout_state)
    bundle.model.eval()
    step_audio = output_dir / "rollout_audio" / f"step_{step:06d}"
    keep_audio = bool(config["artifacts"]["keep_training_audio"])
    conditions_data = []
    rows = []
    reward_definition = resolve_training_reward(config)
    uses_speaker_reward = reward_definition["name"] == DNSMOS_SPEAKER
    uses_official_composite = _uses_composite_evaluators(reward_definition)
    if uses_speaker_reward and fidelity is None:
        raise ValueError("DNSMOS+speaker training reward requires speaker evaluator")
    if uses_official_composite and composite_evaluators is None:
        raise ValueError(
            "FlowSE-GRPO composite reward requires ERes2Net and SpeechBERTScore"
        )
    retained_audio_paths: set[Path] = set()
    phase_seconds = {"rollout": 0.0, "reward": 0.0, "audio_io": 0.0}
    microbatching = []
    training_nfe_by_condition = _training_nfe_by_condition(
        config, step=step, utterances=utterances
    )
    for condition_index in condition_indices:
        utterance = utterances[condition_index]
        noisy_path = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
        if not noisy_path.is_file():
            raise FileNotFoundError(noisy_path)
        clean_path = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
        if (
            uses_speaker_reward or uses_official_composite
        ) and not clean_path.is_file():
            raise FileNotFoundError(clean_path)
        _synchronize_cuda(bundle.device)
        rollout_started = time.perf_counter()
        condition_mel = waveform_to_mel(bundle, str(noisy_path)).detach().cpu()
        mel_frames = int(condition_mel.shape[1])
        training_nfe = int(training_nfe_by_condition[condition_index])
        seeds = [
            stable_seed(
                int(rollout["latent_seed_base"]),
                "rollout",
                step,
                utterance,
                candidate,
            )
            for candidate in range(candidates)
        ]
        # No manifest transcript is passed into the policy.  WER may use it only
        # in the post-generation evaluation function below.
        endpoints, candidate_microbatch_size = _generate_group_length_adaptive(
            bundle,
            noisy_path,
            "",
            seeds,
            mel_frames=mel_frames,
            nfe=training_nfe,
            cfg_strength=0.0,
            conditioning=conditioning,
            target_dbfs=float(normalization["target_dbfs"]),
            peak_ceiling=float(normalization["peak_ceiling"]),
            adaptive_config=config.get("length_adaptive_microbatching"),
        )
        microbatching.append(
            {
                "condition_index": condition_index,
                "utterance": utterance,
                "mel_frames": mel_frames,
                "candidate_microbatch_size": candidate_microbatch_size,
                "training_nfe": training_nfe,
            }
        )
        _synchronize_cuda(bundle.device)
        phase_seconds["rollout"] += time.perf_counter() - rollout_started
        terminals = []
        for candidate_index, endpoint in enumerate(endpoints):
            audit_audio_retained = (
                condition_index,
                candidate_index,
            ) in audit_audio_keys
            retain_audio = keep_audio or audit_audio_retained
            audio_path = step_audio / f"{utterance}__k{candidate_index}.wav"
            audio_io_started = time.perf_counter()
            scored_wav_sha256 = _write_wave(
                audio_path,
                endpoint.normalized_waveform,
                bundle.output_sample_rate,
                str(normalization["output_subtype"]),
                compute_hash=retain_audio,
            )
            phase_seconds["audio_io"] += time.perf_counter() - audio_io_started
            _synchronize_cuda(bundle.device)
            reward_started = time.perf_counter()
            metrics = dnsmos(audio_path)
            if uses_speaker_reward:
                metrics["speaker_similarity"] = float(
                    fidelity.speaker(clean_path, audio_path)
                )
            if uses_official_composite:
                metrics.update(composite_evaluators.score(clean_path, audio_path))
            reward = compute_training_reward(metrics, reward_definition)
            auxiliary_reward = None
            if reward_definition["name"] == DNSMOS_OVRL_RAW:
                auxiliary_reward = compute_training_reward(
                    metrics, reward_definition["auxiliary_composite"]
                )
            _synchronize_cuda(bundle.device)
            phase_seconds["reward"] += time.perf_counter() - reward_started
            if retain_audio:
                retained_audio_paths.add(audio_path)
            terminals.append(endpoint.terminal_mel)
            rows.append(
                {
                    "step": step,
                    "condition_index": condition_index,
                    "utterance": utterance,
                    "candidate_index": candidate_index,
                    "training_nfe": training_nfe,
                    "latent_seed": int(endpoint.latent_seed),
                    "mel_frames": mel_frames,
                    "candidate_microbatch_size": candidate_microbatch_size,
                    "reward": float(reward["reward"]),
                    "unconstrained_reward": float(reward["reward"]),
                    "reward_components": {
                        "raw": reward["raw_components"],
                        "normalized": reward["normalized_components"],
                        "weighted": reward["weighted_components"],
                    },
                    "auxiliary_composite_reward": (
                        float(auxiliary_reward["reward"])
                        if auxiliary_reward is not None
                        else None
                    ),
                    **metrics,
                    "terminal_mel_sha256": endpoint.terminal_mel_sha256,
                    "waveform_sha256": endpoint.normalized_waveform_sha256,
                    "prewrite_waveform_sha256": endpoint.normalized_waveform_sha256,
                    "scored_wav_sha256": scored_wav_sha256,
                    "scored_wav_subtype": str(normalization["output_subtype"]),
                    "audit_audio_retained": audit_audio_retained,
                    "audio_path": (
                        str(audio_path) if keep_audio or audit_audio_retained else None
                    ),
                }
            )
        conditions_data.append(
            (condition_index, utterance, condition_mel, torch.stack(terminals))
        )
    return {
        "condition_indices": list(condition_indices),
        "conditions_data": conditions_data,
        "rows": rows,
        "retained_audio_paths": [str(path) for path in retained_audio_paths],
        "phase_seconds": phase_seconds,
        "length_adaptive_microbatching": microbatching,
    }


def _audit_audio_keys(
    utterances: list[str], *, step: int, candidates: int, config: dict
) -> set[tuple[int, int]]:
    count = int(config["artifacts"].get("audit_audio_candidates_per_step", 0))
    if count < 0:
        raise ValueError("audit_audio_candidates_per_step must be non-negative")
    endpoint_keys = [
        (condition_index, candidate_index)
        for condition_index in range(len(utterances))
        for candidate_index in range(candidates)
    ]
    return set(
        sorted(
            endpoint_keys,
            key=lambda key: stable_seed(
                int(config["run"]["seed"]),
                "rollout_audit_audio",
                step,
                utterances[key[0]],
                key[1],
            ),
        )[: min(count, len(endpoint_keys))]
    )


def _finalize_rollout_training_batch(
    *,
    shards: list[dict],
    utterances: list[str],
    step: int,
    config: dict,
    output_dir: Path,
    audit_audio_keys: set[tuple[int, int]],
) -> tuple[list[RolloutTrainingCondition], list[dict], dict]:
    candidates = int(config["rollout"]["candidates_per_condition"])
    expected_indices = list(range(len(utterances)))
    observed_indices = sorted(
        index for shard in shards for index in shard["condition_indices"]
    )
    if observed_indices != expected_indices:
        raise ValueError(
            "rollout shards do not exactly cover the logical batch: "
            f"expected={expected_indices}, got={observed_indices}"
        )
    conditions_data = sorted(
        (item for shard in shards for item in shard["conditions_data"]),
        key=lambda item: item[0],
    )
    if [int(item[0]) for item in conditions_data] != expected_indices:
        raise ValueError(
            "rollout shards contain missing or duplicate condition tensors"
        )
    rows = sorted(
        (row for shard in shards for row in shard["rows"]),
        key=lambda row: (int(row["condition_index"]), int(row["candidate_index"])),
    )
    expected_endpoints = len(utterances) * candidates
    if len(rows) != expected_endpoints:
        raise ValueError(
            f"rollout endpoint count mismatch: expected={expected_endpoints}, got={len(rows)}"
        )
    expected_endpoint_keys = [
        (condition_index, candidate_index)
        for condition_index in range(len(utterances))
        for candidate_index in range(candidates)
    ]
    observed_endpoint_keys = [
        (int(row["condition_index"]), int(row["candidate_index"])) for row in rows
    ]
    if observed_endpoint_keys != expected_endpoint_keys:
        raise ValueError("rollout shards contain missing or duplicate endpoint keys")
    microbatching = sorted(
        (
            item
            for shard in shards
            for item in shard.get("length_adaptive_microbatching", [])
        ),
        key=lambda item: int(item["condition_index"]),
    )
    if [int(item["condition_index"]) for item in microbatching] != expected_indices:
        raise ValueError("rollout shards lack length-adaptive microbatch metadata")
    nfe_policy = training_nfe_spec(config["rollout"])
    expected_training_nfe_by_condition = _training_nfe_by_condition(
        config, step=step, utterances=utterances
    )
    training_nfe_by_condition = []
    for condition_index, utterance in enumerate(utterances):
        group_rows = [
            row for row in rows if int(row["condition_index"]) == condition_index
        ]
        observed_nfe = {
            int(row.get("training_nfe", nfe_policy["configured"]))
            for row in group_rows
        }
        if len(observed_nfe) != 1:
            raise ValueError(
                "all candidates in one AF condition must share the same training NFE"
            )
        actual_nfe = next(iter(observed_nfe))
        expected_nfe = int(expected_training_nfe_by_condition[condition_index])
        if actual_nfe != expected_nfe:
            raise ValueError(
                "rollout training NFE is not reproducible for "
                f"condition={condition_index}, utterance={utterance!r}: "
                f"expected={expected_nfe}, observed={actual_nfe}"
            )
        if actual_nfe < nfe_policy["minimum"] or actual_nfe > nfe_policy["maximum"]:
            raise ValueError("observed training NFE lies outside the configured range")
        training_nfe_by_condition.append(actual_nfe)
    training_nfe_values = sorted(set(training_nfe_by_condition))
    training_nfe_counts = {
        str(value): training_nfe_by_condition.count(value)
        for value in training_nfe_values
    }
    rewards = np.empty((len(utterances), candidates), dtype=np.float64)
    constraint_summary = apply_reward_constraints(rows, config.get("reward_constraints"))
    for row in rows:
        rewards[int(row["condition_index"]), int(row["candidate_index"])] = float(
            row["reward"]
        )
    result = compute_paper_global_advantages(
        rewards,
        clip=float(config["advantage"]["clip"]),
        minimum_scale=float(config["advantage"]["minimum_global_scale"]),
        mapping=str(config["advantage"].get("mapping", "linear_clipped")),
        temperature=float(config["advantage"].get("temperature", 1.0)),
    )
    reward_definition = resolve_training_reward(config)
    conditions = []
    row_index = 0
    for condition_index, utterance, condition_mel, terminals in conditions_data:
        advantages = torch.from_numpy(result.advantages[condition_index]).float()
        conditions.append(
            RolloutTrainingCondition(
                utterance=utterance,
                transcript="",
                condition_mel=condition_mel,
                terminal_mels=terminals,
                advantages=advantages,
            )
        )
        for candidate_index in range(candidates):
            rows[row_index]["advantage"] = float(
                result.advantages[condition_index, candidate_index]
            )
            rows[row_index]["centered_reward"] = float(
                result.centered_rewards[condition_index, candidate_index]
            )
            row_index += 1
    geometry = {
        **_reward_summary(rewards),
        "unconstrained_reward": _reward_summary(
            np.asarray(
                [float(row["unconstrained_reward"]) for row in rows],
                dtype=np.float64,
            ).reshape(len(utterances), candidates)
        ),
        "reward_constraints": constraint_summary,
        "reward_name": reward_definition["name"],
        "training_reward": reward_definition,
        "global_advantage_scale": result.pooled_scale,
        "advantage_clipped_fraction": result.clipped_fraction,
        "advantage_mapping": result.mapping,
        "advantage_temperature": result.temperature,
        "advantage_statistics": {
            "minimum": float(result.advantages.min()),
            "maximum": float(result.advantages.max()),
            "mean": float(result.advantages.mean()),
            "std": float(result.advantages.std(ddof=0)),
            "positive_fraction": float(np.mean(result.advantages > 0.0)),
            "negative_fraction": float(np.mean(result.advantages < 0.0)),
            "zero_fraction": float(np.mean(result.advantages == 0.0)),
            "maximum_abs_condition_mean": float(
                np.max(np.abs(result.advantages.mean(axis=1)))
            ),
        },
        "logical_conditions": len(utterances),
        "candidates_per_condition": candidates,
        # Keep the historical scalar field for fixed-NFE consumers.  Mixed runs
        # expose the actual per-condition schedule below and use null here.
        "training_nfe": (
            int(training_nfe_values[0]) if len(training_nfe_values) == 1 else None
        ),
        "training_nfe_values": training_nfe_values,
        "training_nfe_counts": training_nfe_counts,
        "training_nfe_by_condition": training_nfe_by_condition,
        "training_nfe_sampling": {
            "minimum": int(nfe_policy["minimum"]),
            "maximum": int(nfe_policy["maximum"]),
            "scope": str(nfe_policy["scope"]),
            "schedule": str(
                config["rollout"].get("training_nfe_schedule", "random_per_group")
            ),
            "seed_base": nfe_policy["seed_base"],
        },
        "logical_endpoints": int(rewards.size),
        "rollout_shards": len(shards),
        "audit_audio_retained": len(audit_audio_keys),
        "cuda_memory_by_rollout_rank": sorted(
            (
                dict(shard["cuda_memory"])
                for shard in shards
                if isinstance(shard.get("cuda_memory"), Mapping)
            ),
            key=lambda item: str(item["role"]),
        ),
        "length_adaptive_microbatching": {
            "policy": config.get("length_adaptive_microbatching"),
            "effective_size_by_condition": [
                int(item["candidate_microbatch_size"]) for item in microbatching
            ],
            "effective_size_histogram": {
                str(size): sum(
                    int(item["candidate_microbatch_size"]) == size
                    for item in microbatching
                )
                for size in sorted(
                    {int(item["candidate_microbatch_size"]) for item in microbatching}
                )
            },
            "adapted_conditions": sum(
                int(item["candidate_microbatch_size"]) < candidates
                for item in microbatching
            ),
            "maximum_mel_frames": max(int(item["mel_frames"]) for item in microbatching),
        },
        "active_gpu_seconds_by_phase": {
            phase: float(
                sum(
                    float(shard.get("phase_seconds", {}).get(phase, 0.0))
                    for shard in shards
                )
            )
            for phase in ("rollout", "reward")
        },
        "parallel_wall_seconds_by_phase": {
            phase: float(
                max(
                    float(shard.get("phase_seconds", {}).get(phase, 0.0))
                    for shard in shards
                )
            )
            for phase in ("rollout", "reward", "audio_io")
        },
        "dnsmos_components": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows], ddof=0)),
            }
            for key in DNSMOS_KEYS
        },
        "reward_components": {
            component: {
                space: {
                    "mean": float(
                        np.mean(
                            [row["reward_components"][space][component] for row in rows]
                        )
                    ),
                    "std": float(
                        np.std(
                            [
                                row["reward_components"][space][component]
                                for row in rows
                            ],
                            ddof=0,
                        )
                    ),
                }
                for space in ("raw", "normalized", "weighted")
            }
            for component in reward_definition["components"]
        },
    }
    auxiliary_values = [
        float(row["auxiliary_composite_reward"])
        for row in rows
        if row.get("auxiliary_composite_reward") is not None
    ]
    if auxiliary_values:
        geometry["auxiliary_composite_reward"] = _reward_summary(
            np.asarray(auxiliary_values, dtype=np.float64).reshape(
                len(utterances), candidates
            )
        )
    keep_audio = bool(config["artifacts"]["keep_training_audio"])
    step_audio = output_dir / "rollout_audio" / f"step_{step:06d}"
    retained_audio_paths = {
        Path(path) for shard in shards for path in shard["retained_audio_paths"]
    }
    if not keep_audio and step_audio.exists():
        for path in step_audio.glob("*.wav"):
            if path not in retained_audio_paths:
                path.unlink()
        if not retained_audio_paths:
            shutil.rmtree(step_audio)
    return conditions, rows, geometry


def _rollout_training_batch(
    *,
    bundle,
    rollout_state: Mapping[str, torch.Tensor],
    manifest: dict[str, str],
    utterances: list[str],
    step: int,
    config: dict,
    conditioning: ConditioningProtocol,
    dnsmos: DNSMOSScorer,
    fidelity: FidelityEvaluators | None,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None,
    output_dir: Path,
) -> tuple[list[RolloutTrainingCondition], list[dict], dict]:
    candidates = int(config["rollout"]["candidates_per_condition"])
    audit_audio_keys = _audit_audio_keys(
        utterances, step=step, candidates=candidates, config=config
    )
    if torch.cuda.is_available() and torch.device(bundle.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    shard = _rollout_training_shard(
        bundle=bundle,
        rollout_state=rollout_state,
        utterances=utterances,
        condition_indices=list(range(len(utterances))),
        audit_audio_keys=audit_audio_keys,
        step=step,
        config=config,
        conditioning=conditioning,
        dnsmos=dnsmos,
        fidelity=fidelity,
        composite_evaluators=composite_evaluators,
        output_dir=output_dir,
    )
    shard["cuda_memory"] = _cuda_peak_memory_report(
        device=bundle.device, role="rollout_rank_0"
    )
    return _finalize_rollout_training_batch(
        shards=[shard],
        utterances=utterances,
        step=step,
        config=config,
        output_dir=output_dir,
        audit_audio_keys=audit_audio_keys,
    )


def _partition_condition_indices(count: int, world_size: int) -> list[list[int]]:
    if count < world_size or world_size < 1:
        raise ValueError("logical conditions must be at least the rollout world size")
    if count % world_size != 0:
        raise ValueError("logical conditions must divide evenly across rollout workers")
    per_worker = count // world_size
    return [
        list(range(rank * per_worker, (rank + 1) * per_worker))
        for rank in range(world_size)
    ]


def _rollout_worker_main(
    *,
    worker_rank: int,
    device_index: int,
    config: dict,
    output_dir: str,
    shared_rollout_state: Mapping[str, torch.Tensor],
    task_queue,
    result_queue,
) -> None:
    """Persistent rollout/scoring worker for one non-coordinator GPU."""

    try:

        def report_initialization(stage: str) -> None:
            result_queue.put(
                {
                    "status": "initializing",
                    "worker_rank": worker_rank,
                    "device_index": device_index,
                    "stage": stage,
                }
            )

        torch.cuda.set_device(device_index)
        if torch.cuda.current_device() != device_index:
            raise RuntimeError(
                f"worker CUDA assignment failed: expected={device_index}, "
                f"got={torch.cuda.current_device()}"
            )
        report_initialization("cuda_selected")
        bundle = load_flowse_bundle(
            config["flowse_config"],
            deterministic=bool(config["run"]["deterministic"]),
            compute_artifact_hashes=(
                str(config["run"].get("startup_validation", "full"))
                != "lightweight"
            ),
        )
        report_initialization("flowse_loaded")
        torch.manual_seed(int(config["lora"]["initialization_seed"]))
        torch.cuda.manual_seed_all(int(config["lora"]["initialization_seed"]))
        inject_lora(
            bundle.model.transformer,
            target_patterns=config["lora"]["target_patterns"],
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
            expected_modules=int(config["lora"]["expected_modules"]),
        )
        report_initialization("lora_injected")
        conditioning = ConditioningProtocol.from_config(config["conditioning"])
        dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
        report_initialization("dnsmos_loaded")
        reward_definition = resolve_training_reward(config)
        fidelity = None
        composite_evaluators = None
        reward_evaluator_fingerprint = None
        if reward_definition["name"] == DNSMOS_SPEAKER:
            fidelity, _ = _load_fidelity(config, lazy_asr=True)
        elif _uses_composite_evaluators(reward_definition):
            report_initialization("composite_evaluators_loading")
            composite_evaluators, reward_evaluator_fingerprint = (
                load_flowse_grpo_composite_evaluators(
                    config, device_override=f"cuda:{device_index}"
                )
            )
            report_initialization("composite_evaluators_loaded")
        cuda_memory_reservation = _reserve_cuda_allocator_memory(
            config,
            device=device_index,
            role=f"rollout_worker_{worker_rank}",
        )
        report_initialization("cuda_memory_reserved")
        cuda_allocated = {
            index: int(torch.cuda.memory_allocated(index))
            for index in range(torch.cuda.device_count())
        }
        foreign_allocated = {
            index: allocated
            for index, allocated in cuda_allocated.items()
            if index != device_index and allocated > 16 * 1024 * 1024
        }
        if foreign_allocated:
            raise RuntimeError(
                "rollout worker allocated model tensors on a foreign CUDA device: "
                f"assigned=cuda:{device_index}, allocated={cuda_allocated}"
            )
        result_queue.put(
            {
                "status": "ready",
                "worker_rank": worker_rank,
                "device_index": device_index,
                "reward_evaluator_fingerprint": reward_evaluator_fingerprint,
                "cuda_memory_allocated_bytes_by_device": cuda_allocated,
                "cuda_memory_reservation": cuda_memory_reservation,
            }
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id = str(task["task_id"])
            try:
                _synchronize_cuda(device_index)
                torch.cuda.reset_peak_memory_stats(device_index)
                shard = _rollout_training_shard(
                    bundle=bundle,
                    rollout_state=shared_rollout_state,
                    utterances=task["utterances"],
                    condition_indices=task["condition_indices"],
                    audit_audio_keys=set(task["audit_audio_keys"]),
                    step=int(task["step"]),
                    config=config,
                    conditioning=conditioning,
                    dnsmos=dnsmos,
                    fidelity=fidelity,
                    composite_evaluators=composite_evaluators,
                    output_dir=Path(output_dir),
                )
                _synchronize_cuda(device_index)
                shard["cuda_memory"] = _cuda_peak_memory_report(
                    device=device_index,
                    role=f"rollout_rank_{worker_rank}",
                )
                result_queue.put(
                    {
                        "status": "ok",
                        "worker_rank": worker_rank,
                        "task_id": task_id,
                        "shard": shard,
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "status": "error",
                        "worker_rank": worker_rank,
                        "task_id": task_id,
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "worker_rank": worker_rank,
                "task_id": "initialization",
                "traceback": traceback.format_exc(),
            }
        )


class _ShardedRolloutPool:
    """Child workers plus the coordinator form two or four rollout shards."""

    def __init__(
        self,
        *,
        config: dict,
        output_dir: Path,
        rollout_state: Mapping[str, torch.Tensor],
        expected_reward_evaluator_fingerprint: Mapping | None,
    ) -> None:
        parallel = config["parallel_rollout"]
        self.world_size = int(parallel["world_size"])
        self.device_ids = [int(value) for value in parallel["device_ids"]]
        self.startup_timeout = float(parallel["worker_startup_timeout_seconds"])
        self.task_timeout = float(parallel["worker_task_timeout_seconds"])
        self.expected_reward_evaluator_fingerprint = (
            dict(expected_reward_evaluator_fingerprint)
            if expected_reward_evaluator_fingerprint is not None
            else None
        )
        if self.world_size not in {2, 4} or self.device_ids != list(
            range(self.world_size)
        ):
            raise ValueError(
                "sharded rollout requires world_size 2 or 4 and contiguous "
                "logical device IDs starting at 0"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"sharded rollout requires {self.world_size} visible GPUs, "
                f"got {torch.cuda.device_count()}"
            )
        if torch.cuda.current_device() != self.device_ids[0]:
            raise RuntimeError(
                "coordinator must run on the first parallel_rollout device"
            )
        self._context = torch.multiprocessing.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._task_queues = {}
        self._processes = {}
        self.worker_memory_reservations = {}
        self._closed = False
        self._shared_rollout_state = {}
        for name, value in rollout_state.items():
            shared = value.detach().cpu().clone().contiguous()
            shared.share_memory_()
            self._shared_rollout_state[name] = shared
        for worker_rank in range(1, self.world_size):
            task_queue = self._context.Queue(maxsize=1)
            process = self._context.Process(
                target=_rollout_worker_main,
                kwargs={
                    "worker_rank": worker_rank,
                    "device_index": self.device_ids[worker_rank],
                    "config": config,
                    "output_dir": str(output_dir),
                    "shared_rollout_state": self._shared_rollout_state,
                    "task_queue": task_queue,
                    "result_queue": self._result_queue,
                },
                daemon=True,
            )
            process.start()
            self._task_queues[worker_rank] = task_queue
            self._processes[worker_rank] = process
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def _get_result(self, timeout: float) -> dict:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty as exc:
            states = {
                rank: {"alive": process.is_alive(), "exitcode": process.exitcode}
                for rank, process in self._processes.items()
            }
            raise TimeoutError(f"rollout worker timeout; states={states}") from exc

    def _wait_until_ready(self) -> None:
        ready = set()
        stages = {rank: "spawned" for rank in self._processes}
        deadline = time.monotonic() + self.startup_timeout
        while len(ready) < self.world_size - 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                states = {
                    rank: {"alive": process.is_alive(), "exitcode": process.exitcode}
                    for rank, process in self._processes.items()
                }
                raise TimeoutError(
                    f"rollout worker startup timeout; states={states}; stages={stages}"
                )
            try:
                result = self._get_result(remaining)
            except TimeoutError as exc:
                raise TimeoutError(f"{exc}; startup_stages={stages}") from exc
            if result["status"] == "error":
                raise RuntimeError(
                    "sharded rollout worker initialization failed:\n"
                    + str(result["traceback"])
                )
            if result["status"] == "initializing":
                worker_rank = int(result["worker_rank"])
                stages[worker_rank] = str(result["stage"])
                print(
                    "parallel rollout worker initialization: "
                    f"rank={worker_rank} device=cuda:{int(result['device_index'])} "
                    f"stage={stages[worker_rank]}",
                    flush=True,
                )
                continue
            if result["status"] != "ready":
                raise RuntimeError(f"unexpected rollout worker message: {result}")
            worker_rank = int(result["worker_rank"])
            if (
                result.get("reward_evaluator_fingerprint")
                != self.expected_reward_evaluator_fingerprint
            ):
                raise RuntimeError(
                    f"rollout worker {worker_rank} evaluator fingerprint differs "
                    "from the coordinator"
                )
            ready.add(worker_rank)
            stages[worker_rank] = "ready"
            self.worker_memory_reservations[worker_rank] = result.get(
                "cuda_memory_reservation"
            )
            print(
                "parallel rollout worker ready: "
                f"rank={worker_rank} device=cuda:{int(result['device_index'])} "
                "allocated_bytes_by_device="
                f"{result['cuda_memory_allocated_bytes_by_device']}",
                flush=True,
            )

    def _update_shared_rollout_state(
        self, rollout_state: Mapping[str, torch.Tensor]
    ) -> None:
        if set(rollout_state) != set(self._shared_rollout_state):
            raise ValueError("EMA LoRA keys changed after rollout workers started")
        for name, target in self._shared_rollout_state.items():
            source = rollout_state[name].detach().cpu()
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"EMA LoRA tensor metadata changed for {name}")
            target.copy_(source)

    def rollout(
        self,
        *,
        bundle,
        rollout_state: Mapping[str, torch.Tensor],
        manifest: dict[str, str],
        utterances: list[str],
        step: int,
        config: dict,
        conditioning: ConditioningProtocol,
        dnsmos: DNSMOSScorer,
        fidelity: FidelityEvaluators | None,
        composite_evaluators: FlowSEGRPOCompositeEvaluators | None,
        output_dir: Path,
    ) -> tuple[list[RolloutTrainingCondition], list[dict], dict]:
        partitions = _partition_condition_indices(len(utterances), self.world_size)
        candidates = int(config["rollout"]["candidates_per_condition"])
        audit_audio_keys = _audit_audio_keys(
            utterances, step=step, candidates=candidates, config=config
        )
        self._update_shared_rollout_state(rollout_state)
        task_id = f"step_{step:06d}"
        for worker_rank in range(1, self.world_size):
            self._task_queues[worker_rank].put(
                {
                    "task_id": task_id,
                    "step": step,
                    "utterances": utterances,
                    "condition_indices": partitions[worker_rank],
                    "audit_audio_keys": sorted(audit_audio_keys),
                }
            )
        torch.cuda.reset_peak_memory_stats(self.device_ids[0])
        main_shard = _rollout_training_shard(
            bundle=bundle,
            rollout_state=rollout_state,
            utterances=utterances,
            condition_indices=partitions[0],
            audit_audio_keys=audit_audio_keys,
            step=step,
            config=config,
            conditioning=conditioning,
            dnsmos=dnsmos,
            fidelity=fidelity,
            composite_evaluators=composite_evaluators,
            output_dir=output_dir,
        )
        main_shard["cuda_memory"] = _cuda_peak_memory_report(
            device=self.device_ids[0], role="rollout_rank_0"
        )
        shards = [main_shard]
        completed = set()
        while len(completed) < self.world_size - 1:
            result = self._get_result(self.task_timeout)
            if result["status"] == "error":
                raise RuntimeError(
                    f"sharded rollout worker {result['worker_rank']} failed:\n"
                    + str(result["traceback"])
                )
            if result["status"] != "ok" or result["task_id"] != task_id:
                raise RuntimeError(f"unexpected rollout worker result: {result}")
            worker_rank = int(result["worker_rank"])
            if worker_rank in completed:
                raise RuntimeError(
                    f"duplicate result from rollout worker {worker_rank}"
                )
            completed.add(worker_rank)
            shards.append(result["shard"])
        return _finalize_rollout_training_batch(
            shards=shards,
            utterances=utterances,
            step=step,
            config=config,
            output_dir=output_dir,
            audit_audio_keys=audit_audio_keys,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task_queue in self._task_queues.values():
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        for process in self._processes.values():
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        for task_queue in self._task_queues.values():
            task_queue.close()
        self._result_queue.close()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass


def _gradient_and_update_statistics(
    before: Mapping[str, torch.Tensor],
    transformer: torch.nn.Module,
    gradient_norm: float,
) -> dict:
    after = snapshot_lora(transformer, device="cpu")
    a_update, b_update = 0.0, 0.0
    a_grad, b_grad = 0.0, 0.0
    for name, parameter in named_lora_parameters(transformer):
        delta = after[name].double() - before[name].double()
        if name.endswith("lora_A"):
            a_update += float(delta.square().sum().item())
            if parameter.grad is not None:
                a_grad += float(parameter.grad.detach().double().square().sum().item())
        else:
            b_update += float(delta.square().sum().item())
            if parameter.grad is not None:
                b_grad += float(parameter.grad.detach().double().square().sum().item())
    return {
        "gradient_norm_clipped_return": float(gradient_norm),
        "lora_A_gradient_norm": math.sqrt(a_grad),
        "lora_B_gradient_norm": math.sqrt(b_grad),
        "lora_A_update_norm": math.sqrt(a_update),
        "lora_B_update_norm": math.sqrt(b_update),
        "total_update_norm": adapter_distance(before, after),
    }


def _metric_summary(rows: list[dict], keys: list[str]) -> dict:
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        if values.size:
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "count": int(values.size),
            }
    return summary


def _assert_metric_summary_matches(
    observed: Mapping, expected: Mapping, *, label: str
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{label} metric key space differs from the cached rows")
    for metric, expected_statistics in expected.items():
        observed_statistics = observed[metric]
        if set(observed_statistics) != {"mean", "std", "count"}:
            raise ValueError(f"{label}/{metric} has an invalid summary schema")
        if int(observed_statistics["count"]) != int(expected_statistics["count"]):
            raise ValueError(f"{label}/{metric} count differs from the cached rows")
        for statistic in ("mean", "std"):
            if not math.isclose(
                float(observed_statistics[statistic]),
                float(expected_statistics[statistic]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{label}/{metric}/{statistic} differs from the cached rows"
                )


def _import_initial_evaluation_cache(
    *, manifest: dict[str, str], config: dict, output_dir: Path
) -> dict:
    """Validate and import an immutable step-0 report after an interrupted run."""

    source_report = Path(
        str(config["evaluation"]["initial_cache_report_path"])
    ).resolve()
    source_baseline = source_report.parent / "evaluation_noisy_baselines.json"
    report = json.loads(source_report.read_text(encoding="utf-8"))
    baseline_payload = json.loads(source_baseline.read_text(encoding="utf-8"))
    baseline_rows = baseline_payload.get("rows")
    enhanced_rows = report.get("rows")
    if not isinstance(baseline_rows, list) or not isinstance(enhanced_rows, list):
        raise ValueError("initial evaluation cache rows are missing")

    utterances = list(manifest)
    expected_count = len(utterances)
    if [str(row.get("utterance")) for row in baseline_rows] != utterances:
        raise ValueError("cached noisy baseline utterance order is incompatible")
    if [str(row.get("utterance")) for row in enhanced_rows] != utterances:
        raise ValueError("cached step-0 utterance order is incompatible")
    if int(report.get("step", -1)) != 0:
        raise ValueError("initial evaluation cache is not step 0")
    if str(report.get("policy")) != str(config["evaluation"]["policy"]):
        raise ValueError("initial evaluation cache policy is incompatible")
    if report.get("counts") != {"noisy": expected_count, "enhanced": expected_count}:
        raise ValueError("initial evaluation cache counts are incomplete")

    required_metrics = set(EVALUATION_KEYS)
    for label, rows in (("noisy", baseline_rows), ("enhanced", enhanced_rows)):
        for row in rows:
            missing = required_metrics - set(row)
            if missing:
                raise ValueError(
                    f"cached {label} row {row.get('utterance')} misses {sorted(missing)}"
                )
            if any(not math.isfinite(float(row[key])) for key in required_metrics):
                raise ValueError(
                    f"cached {label} row {row.get('utterance')} has a non-finite metric"
                )

    for utterance, row in zip(utterances, enhanced_rows):
        expected_seed = stable_seed(
            int(config["evaluation"]["latent_seed_base"]), "evaluation", utterance
        )
        if (
            int(row.get("step", -1)) != 0
            or int(row.get("latent_seed", -1)) != expected_seed
        ):
            raise ValueError(f"cached step-0 seed differs for {utterance}")
        audio_path = Path(str(row.get("audio_path", "")))
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        observed_hash = sha256_file(audio_path)
        if observed_hash != str(row.get("scored_wav_sha256")):
            raise ValueError(f"cached step-0 WAV hash differs for {utterance}")

    noisy_summary = _metric_summary(baseline_rows, EVALUATION_KEYS)
    enhanced_summary = _metric_summary(enhanced_rows, EVALUATION_KEYS)
    _assert_metric_summary_matches(
        report.get("noisy", {}), noisy_summary, label="noisy"
    )
    _assert_metric_summary_matches(
        report.get("enhanced", {}), enhanced_summary, label="enhanced"
    )
    expected_delta = {
        key: enhanced_summary[key]["mean"] - noisy_summary[key]["mean"]
        for key in enhanced_summary.keys() & noisy_summary.keys()
    }
    observed_delta = report.get("delta")
    if not isinstance(observed_delta, Mapping) or set(observed_delta) != set(
        expected_delta
    ):
        raise ValueError("initial evaluation cache delta key space is incompatible")
    for key, expected in expected_delta.items():
        if not math.isclose(
            float(observed_delta[key]), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"initial evaluation cache delta differs for {key}")

    # Composite reward is a deterministic derived scalar.  Recompute it from
    # the cached DNSMOS/ERes2Net/SpeechBERTScore raw metrics under the current
    # frozen calibration instead of regenerating and rescoring step-0 audio.
    reward_definition = resolve_training_reward(config)
    recalibrated_rows = 0
    composite_definition = _composite_definition(reward_definition)
    if composite_definition is not None:
        for rows in (baseline_rows, enhanced_rows):
            for row in rows:
                reward = compute_training_reward(row, composite_definition)
                row["flowse_grpo_composite_reward"] = float(reward["reward"])
                recalibrated_rows += 1
        noisy_summary = _metric_summary(baseline_rows, EVALUATION_KEYS)
        enhanced_summary = _metric_summary(enhanced_rows, EVALUATION_KEYS)
        report["noisy"] = noisy_summary
        report["enhanced"] = enhanced_summary
        report["delta"] = {
            key: enhanced_summary[key]["mean"] - noisy_summary[key]["mean"]
            for key in enhanced_summary.keys() & noisy_summary.keys()
        }

    descriptor = {
        "authorization": "immutable_step0_raw_metrics_with_current_reward_recompute",
        "source_report": str(source_report),
        "source_report_sha256": sha256_file(source_report),
        "source_noisy_baseline": str(source_baseline),
        "source_noisy_baseline_sha256": sha256_file(source_baseline),
        "validated_utterances": expected_count,
        "validated_wav_hashes": expected_count,
        "recomputed_composite_rows": recalibrated_rows,
        "current_training_reward": reward_definition,
    }
    imported_report = dict(report)
    imported_report["initial_evaluation_cache_import"] = descriptor
    _write_json(output_dir / "evaluation_noisy_baselines.json", baseline_payload)
    _write_json(output_dir / "evaluation_step_000000.json", imported_report)
    print(
        "Imported and validated immutable step-0 evaluation cache: "
        f"utterances={expected_count} report_sha256={descriptor['source_report_sha256']}",
        flush=True,
    )
    _print_evaluation_table(imported_report)
    return imported_report


def _score_evaluation_file(
    *,
    audio_path: Path,
    clean_path: Path,
    transcript: str,
    dnsmos: DNSMOSScorer,
    fidelity: FidelityEvaluators | None,
    paired: bool,
    reference_free: bool = False,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None = None,
    reward_definition: Mapping | None = None,
) -> dict:
    metrics = dnsmos(audio_path)
    if paired:
        pair = paired_metrics(clean_path, audio_path)
        metrics.update({"pesq_wb": pair["pesq_wb"], "stoi": pair["stoi"]})
    if fidelity is not None and not reference_free:
        metrics["speaker_similarity"] = fidelity.speaker(clean_path, audio_path)
    if fidelity is not None and transcript.strip():
        wer, hypothesis = fidelity.asr(transcript, audio_path)
        metrics["wer"] = wer
        metrics["asr_hypothesis"] = hypothesis
    if composite_evaluators is not None and not reference_free:
        metrics.update(composite_evaluators.score(clean_path, audio_path))
    if reward_definition is not None and not reference_free:
        composite_definition = _composite_definition(reward_definition)
    else:
        composite_definition = None
    if composite_definition is not None:
        if composite_evaluators is None:
            raise ValueError(
                "composite held-out reward requires the frozen composite evaluators"
            )
        reconstructed = compute_training_reward(metrics, composite_definition)
        metrics["flowse_grpo_composite_reward"] = float(reconstructed["reward"])
    return metrics


def _evaluation_baselines(
    *,
    manifest: dict[str, str],
    config: dict,
    dnsmos: DNSMOSScorer,
    fidelity: FidelityEvaluators | None,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None,
    output_dir: Path,
) -> tuple[list[dict], dict]:
    reward_definition = resolve_training_reward(config)
    cache = output_dir / "evaluation_noisy_baselines.json"
    if cache.is_file():
        rows = json.loads(cache.read_text(encoding="utf-8"))["rows"]
        required = (
            set(COMPOSITE_REPORT_KEYS)
            if _uses_composite_evaluators(reward_definition)
            else set()
        )
        cache_complete = all(required.issubset(row) for row in rows)
    else:
        rows = []
        cache_complete = False
    if not cache_complete:
        rows = []
        for utterance, transcript in tqdm(
            manifest.items(),
            total=len(manifest),
            desc="Validation noisy baseline",
            unit="utt",
            dynamic_ncols=True,
        ):
            noisy = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
            clean = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
            metrics = _score_evaluation_file(
                audio_path=noisy,
                clean_path=clean,
                transcript=transcript,
                dnsmos=dnsmos,
                fidelity=fidelity,
                composite_evaluators=composite_evaluators,
                paired=bool(config["evaluation"]["paired_metrics"]),
                reward_definition=reward_definition,
            )
            rows.append({"utterance": utterance, **metrics})
        _write_json(cache, {"rows": rows})
    return rows, _metric_summary(rows, EVALUATION_KEYS)


def _evaluate(
    *,
    bundle,
    state: Mapping[str, torch.Tensor],
    step: int,
    manifest: dict[str, str],
    config: dict,
    conditioning: ConditioningProtocol,
    dnsmos: DNSMOSScorer,
    fidelity: FidelityEvaluators | None,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None,
    output_dir: Path,
) -> dict:
    _synchronize_cuda(bundle.device)
    evaluation_started = time.perf_counter()
    reward_definition = resolve_training_reward(config)
    load_lora(bundle.model.transformer, state)
    baseline_rows, noisy_summary = _evaluation_baselines(
        manifest=manifest,
        config=config,
        dnsmos=dnsmos,
        fidelity=fidelity,
        composite_evaluators=composite_evaluators,
        output_dir=output_dir,
    )
    evaluation = config["evaluation"]
    normalization = config["normalization"]
    rows = []
    audio_dir = output_dir / "evaluation_audio" / f"step_{step:06d}"
    for utterance, transcript in tqdm(
        manifest.items(),
        total=len(manifest),
        desc=f"Validation {evaluation['policy']} step {step}",
        unit="utt",
        dynamic_ncols=True,
    ):
        noisy = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
        clean = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
        seed = stable_seed(int(evaluation["latent_seed_base"]), "evaluation", utterance)
        endpoint = bundle.generate_group(
            noisy,
            "",
            [seed],
            nfe=int(config["rollout"]["evaluation_nfe"]),
            cfg_strength=0.0,
            conditioning=conditioning,
            target_dbfs=float(normalization["target_dbfs"]),
            peak_ceiling=float(normalization["peak_ceiling"]),
        )[0]
        audio_path = audio_dir / f"{utterance}.wav"
        scored_wav_sha256 = _write_wave(
            audio_path,
            endpoint.normalized_waveform,
            bundle.output_sample_rate,
            str(normalization["output_subtype"]),
        )
        metrics = _score_evaluation_file(
            audio_path=audio_path,
            clean_path=clean,
            transcript=transcript,
            dnsmos=dnsmos,
            fidelity=fidelity,
            composite_evaluators=composite_evaluators,
            paired=bool(evaluation["paired_metrics"]),
            reward_definition=reward_definition,
        )
        rows.append(
            {
                "step": step,
                "utterance": utterance,
                "latent_seed": seed,
                "waveform_sha256": endpoint.normalized_waveform_sha256,
                "prewrite_waveform_sha256": endpoint.normalized_waveform_sha256,
                "scored_wav_sha256": scored_wav_sha256,
                "audio_path": str(audio_path),
                **metrics,
            }
        )
    enhanced_summary = _metric_summary(rows, EVALUATION_KEYS)
    delta = {
        key: enhanced_summary[key]["mean"] - noisy_summary[key]["mean"]
        for key in enhanced_summary.keys() & noisy_summary.keys()
    }
    _synchronize_cuda(bundle.device)
    report = {
        "step": step,
        "policy": str(evaluation["policy"]),
        "counts": {"noisy": len(baseline_rows), "enhanced": len(rows)},
        "noisy": noisy_summary,
        "enhanced": enhanced_summary,
        "delta": delta,
        "rows": rows,
        "timing_seconds": float(time.perf_counter() - evaluation_started),
    }
    _write_json(output_dir / f"evaluation_step_{step:06d}.json", report)
    _print_evaluation_table(report)
    if fidelity is not None:
        fidelity.release_asr(
            empty_cuda_cache=not _cuda_memory_reservation_enabled(config)
        )
    return report


def _print_evaluation_table(report: dict) -> None:
    labels = [
        ("dnsmos_sig", "DNSMOS SIG"),
        ("dnsmos_bak", "DNSMOS BAK"),
        ("dnsmos_ovrl", "DNSMOS OVRL"),
        ("dnsmos_p808", "DNSMOS P808"),
        ("speaker_similarity", "Speaker similarity"),
        ("eres2net_speaker_similarity", "ERes2Net speaker"),
        ("speechbertscore", "SpeechBERTScore"),
        ("flowse_grpo_composite_reward", "Composite reward"),
        ("stoi", "STOI"),
        ("pesq_wb", "PESQ-WB"),
        ("wer", "WER"),
    ]
    print("\n" + "=" * 68)
    print(f"Evaluation after optimizer step {report['step']}")
    print(f"{'Metric':<24}{'Noisy':>13}{'Enhanced':>13}{'Delta':>13}")
    print("-" * 68)
    for key, label in labels:
        if key not in report["enhanced"] or key not in report["noisy"]:
            continue
        noisy = report["noisy"][key]["mean"]
        enhanced = report["enhanced"][key]["mean"]
        print(f"{label:<24}{noisy:>13.5f}{enhanced:>13.5f}{enhanced - noisy:>+13.5f}")
    print("=" * 68)


def _speaker_cluster_bootstrap_mean(
    utterances: list[str],
    values: list[float],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> dict:
    grouped: dict[str, list[float]] = {}
    for utterance, value in zip(utterances, values, strict=True):
        speaker = utterance.split("_", 1)[0].lower()
        grouped.setdefault(speaker, []).append(float(value))
    speakers = sorted(grouped)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = generator.integers(0, len(speakers), size=len(speakers))
        draw = [value for item in selected for value in grouped[speakers[item]]]
        estimates[index] = np.mean(draw)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "speakers": len(speakers),
    }


def _paired_policy_gain(initial: dict, final: dict, config: dict) -> dict:
    """Paired final-minus-initial policy deltas on fixed conditions/latents."""

    initial_rows = {row["utterance"]: row for row in initial["rows"]}
    final_rows = {row["utterance"]: row for row in final["rows"]}
    if set(initial_rows) != set(final_rows):
        raise ValueError("initial/final evaluation utterance sets differ")
    decision = config.get("pilot_decision", {})
    seed = int(decision.get("bootstrap_seed", 41123))
    samples = int(decision.get("bootstrap_samples", 2000))
    confidence = float(decision.get("confidence", 0.95))
    metrics = {}
    for metric in list(DNSMOS_KEYS) + [
        "speaker_similarity",
        *COMPOSITE_REPORT_KEYS,
        "stoi",
        "pesq_wb",
        "wer",
    ]:
        if not all(
            metric in initial_rows[key] and metric in final_rows[key]
            for key in initial_rows
        ):
            continue
        utterances = sorted(initial_rows)
        deltas = [
            float(final_rows[key][metric] - initial_rows[key][metric])
            for key in utterances
        ]
        utterance_ci = cluster_bootstrap_mean(
            deltas, seed=seed, samples=samples, confidence=confidence
        )
        metrics[metric] = {
            **utterance_ci,
            "utterance_ci": utterance_ci,
            "speaker_ci": _speaker_cluster_bootstrap_mean(
                utterances,
                deltas,
                seed=stable_seed(seed, "speaker", metric),
                samples=samples,
                confidence=confidence,
            ),
            "positive_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
            "count": len(deltas),
        }
    return {
        "comparison": "final_policy_minus_initial_policy_common_conditions_and_latents",
        "initial_step": initial["step"],
        "final_step": final["step"],
        "metrics": metrics,
    }


def _pilot_decision(gain: dict, config: dict) -> dict:
    thresholds = config["pilot_decision"]
    metrics = gain["metrics"]
    criteria = {
        "dnsmos_ovrl_effect_size": metrics["dnsmos_ovrl"]["mean"]
        >= float(thresholds["dnsmos_ovrl_minimum_gain"]),
        "dnsmos_ovrl_ci_positive": (
            metrics["dnsmos_ovrl"]["ci_low"] > 0.0
            if bool(thresholds["require_ci_positive"])
            else True
        ),
    }
    safety = thresholds["safety"]
    for metric, tolerance in safety.items():
        if metric not in metrics:
            criteria[f"{metric}_available"] = False
            continue
        if metric == "wer":
            criteria[f"{metric}_safety"] = metrics[metric]["ci_high"] <= float(
                tolerance
            )
        else:
            criteria[f"{metric}_safety"] = metrics[metric]["ci_low"] >= float(tolerance)
    supported = all(criteria.values())
    return {
        "decision": "PILOT-SUPPORT" if supported else "PILOT-NO-SUPPORT",
        "criteria": criteria,
        "thresholds": thresholds,
        "interpretation": (
            "Preliminary iterative-mechanism evidence only; it does not authorize "
            "full training or final-test evaluation."
        ),
    }


def _specialization_pilot_decision(gain: dict, config: dict) -> dict:
    """Apply the preregistered gate to specialist-minus-composite control."""

    thresholds = config["pilot_decision"]
    metrics = gain["metrics"]
    ovrl = metrics["dnsmos_ovrl"]
    require_ci = bool(thresholds["require_ci_positive"])
    criteria = {
        "dnsmos_ovrl_effect_size": float(ovrl["mean"])
        >= float(thresholds["dnsmos_ovrl_minimum_gain"]),
        "dnsmos_ovrl_utterance_ci_positive": (
            not require_ci or float(ovrl["utterance_ci"]["ci_low"]) > 0.0
        ),
        "dnsmos_ovrl_speaker_ci_positive": (
            not require_ci or float(ovrl["speaker_ci"]["ci_low"]) > 0.0
        ),
    }
    for metric, tolerance in thresholds["safety"].items():
        if metric not in metrics:
            criteria[f"{metric}_available"] = False
            continue
        utterance = metrics[metric]["utterance_ci"]
        speaker = metrics[metric]["speaker_ci"]
        if metric == "wer":
            passed = max(
                float(utterance["ci_high"]), float(speaker["ci_high"])
            ) <= float(tolerance)
        else:
            passed = min(
                float(utterance["ci_low"]), float(speaker["ci_low"])
            ) >= float(tolerance)
        criteria[f"{metric}_safety"] = passed
    return {
        "decision": "PILOT-SUPPORT" if all(criteria.values()) else "PILOT-NO-SUPPORT",
        "criteria": criteria,
        "thresholds": thresholds,
        "comparison": "raw OVRL specialist minus historical composite continuation",
        "interpretation": (
            "Fixed 250-step validation pilot only; no official-test selection or "
            "automatic full-experiment authorization."
        ),
    }


def _print_policy_gain(initial: dict, final: dict, gain: dict) -> None:
    labels = [
        ("dnsmos_sig", "DNSMOS SIG"),
        ("dnsmos_bak", "DNSMOS BAK"),
        ("dnsmos_ovrl", "DNSMOS OVRL"),
        ("dnsmos_p808", "DNSMOS P808"),
        ("speaker_similarity", "Speaker similarity"),
        ("eres2net_speaker_similarity", "ERes2Net speaker"),
        ("speechbertscore", "SpeechBERTScore"),
        ("flowse_grpo_composite_reward", "Composite reward"),
        ("stoi", "STOI"),
        ("pesq_wb", "PESQ-WB"),
        ("wer", "WER"),
    ]
    print("\n" + "=" * 122)
    print("Fixed-condition/latent policy gain (final minus initial)")
    print(
        f"{'Metric':<23}{'Initial':>12}{'Final':>12}{'Delta':>12}"
        f"{'utterance 95% CI':>30}{'speaker 95% CI':>30}"
    )
    print("-" * 122)
    for key, label in labels:
        if key not in gain["metrics"]:
            continue
        initial_mean = initial["enhanced"][key]["mean"]
        final_mean = final["enhanced"][key]["mean"]
        row = gain["metrics"][key]
        utterance = row["utterance_ci"]
        speaker = row["speaker_ci"]
        utterance_interval = (
            f"[{utterance['ci_low']:+.5f}, {utterance['ci_high']:+.5f}]"
        )
        speaker_interval = f"[{speaker['ci_low']:+.5f}, {speaker['ci_high']:+.5f}]"
        print(
            f"{label:<23}{initial_mean:>12.5f}{final_mean:>12.5f}"
            f"{row['mean']:>+12.5f}{utterance_interval:>30}{speaker_interval:>30}"
        )
    print("=" * 122)


def _build_optimizer(transformer, config: dict):
    values = config["optimizer"]
    optimizer = torch.optim.AdamW(
        lora_parameters(transformer),
        lr=float(values["learning_rate"]),
        betas=tuple(float(value) for value in values["betas"]),
        eps=float(values["epsilon"]),
        weight_decay=float(values["weight_decay"]),
    )
    return optimizer, _scheduler(optimizer, config)


def _branch_compatibility_signature(config: Mapping) -> dict:
    data = config["data"]
    run = config["run"]
    return {
        "run": {
            key: run[key]
            for key in ("seed", "deterministic", "conditions_per_step")
        },
        "flowse_config": config["flowse_config"],
        "dnsmos_official_dir": config["dnsmos_official_dir"],
        "conditioning": config["conditioning"],
        "parallel_rollout": config.get("parallel_rollout"),
        "composite_reward_evaluators": config.get("composite_reward_evaluators"),
        "data": {
            key: data.get(key)
            for key in (
                "train_manifest",
                "evaluation_manifest",
                "noisy_dir",
                "clean_dir",
                "order_seed",
                "condition_stride_per_step",
            )
        },
        "rollout": config["rollout"],
        "advantage": config["advantage"],
        "lora": config["lora"],
        "loss": config["loss"],
        "length_adaptive_microbatching": config.get(
            "length_adaptive_microbatching"
        ),
        "optimizer": config["optimizer"],
        "ema": config["ema"],
        "normalization": config["normalization"],
    }


def _load_branch_checkpoint(
    config: Mapping,
    *,
    transformer,
    optimizer,
    scheduler,
) -> tuple[AdapterState, dict]:
    branch = config["branch"]
    source_dir = Path(str(branch["source_run_dir"])).resolve()
    source_step = int(branch["source_step"])
    source_protocol_path = source_dir / "protocol.json"
    source_checkpoint = source_dir / f"checkpoint_step_{source_step:06d}.pt"
    control_step = source_step + int(config["run"]["optimizer_steps"])
    control_checkpoint = source_dir / f"checkpoint_step_{control_step:06d}.pt"
    for path in (source_protocol_path, source_checkpoint, control_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if sha256_json(source_protocol) != source_dir.name:
        raise ValueError("branch source protocol hash does not match its run directory")
    source_config = source_protocol.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("branch source protocol lacks its frozen config")
    if _branch_compatibility_signature(source_config) != _branch_compatibility_signature(
        config
    ):
        raise ValueError("branch source and specialist changed a non-reward field")
    recorded_sources = dict(source_protocol.get("source_sha256") or {})
    current_sources = {
        name: sha256_file(name) if Path(name).is_file() else None
        for name in recorded_sources
    }
    source_mismatches = sorted(
        name
        for name in recorded_sources
        if recorded_sources[name] != current_sources[name]
    )
    allowed_source_mismatches = sorted(
        [
            "rl/rewards/specification.py",
            "rl/af/protocol.py",
            "rl/af/trainer.py",
        ]
    )
    # This is an algorithm branch from immutable checkpoint tensors, not a
    # confirmation audit of the historical Python environment.  Preserve all
    # observed mismatches for provenance without blocking the experiment.
    source_reward = resolve_training_reward(source_config)
    target_reward = resolve_training_reward(config)
    if source_reward["name"] != FLOWSE_GRPO_COMPOSITE:
        raise ValueError("sequential specialist must branch from composite AF")
    uses_fixed_fusion = isinstance(config.get("fixed_fusion"), Mapping)
    uses_gaaf = isinstance(config.get("gaaf"), Mapping)
    uses_projected_gaaf = isinstance(config.get("projected_gaaf"), Mapping)
    uses_marble = isinstance(config.get("marble"), Mapping)
    if sum(
        (uses_fixed_fusion, uses_gaaf, uses_projected_gaaf, uses_marble)
    ) > 1:
        raise ValueError(
            "configure only one of fixed_fusion, gaaf, projected_gaaf, or marble"
        )
    if uses_fixed_fusion or uses_gaaf or uses_projected_gaaf or uses_marble:
        if target_reward != source_reward:
            raise ValueError(
                "fixed-fusion/GA-AF/projected-GA-AF/MARBLE branch must retain the source "
                "composite reward"
            )
    elif target_reward.get("auxiliary_composite") != source_reward:
        raise ValueError(
            "specialist auxiliary composite differs from the source training reward"
        )

    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    criteria = {
        "schema": payload.get("schema_version") == 1,
        "source_protocol": str(payload.get("protocol_hash", "")) == source_dir.name,
        "source_step": int(payload.get("completed_step", -1)) == source_step,
        "current_lora": bool(payload.get("current_lora")),
        "rollout_lora": bool(payload.get("rollout_lora")),
        "optimizer": isinstance(payload.get("optimizer"), Mapping),
        "scheduler": isinstance(payload.get("scheduler"), Mapping),
        "rng": isinstance(payload.get("rng"), Mapping),
    }
    if not all(criteria.values()):
        raise ValueError(f"invalid branch source checkpoint: {criteria}")
    load_lora(transformer, payload["current_lora"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if int(scheduler.last_epoch) != source_step:
        raise ValueError(
            "branch source scheduler is not aligned with its optimizer step: "
            f"last_epoch={scheduler.last_epoch}, source_step={source_step}"
        )
    restore_rng_state(payload["rng"])
    rollout_state = {
        str(name): value.detach().cpu().clone()
        for name, value in payload["rollout_lora"].items()
    }
    if set(rollout_state) != set(snapshot_lora(transformer)):
        raise ValueError("branch source EMA LoRA keys differ from the active model")
    descriptor = {
        "mode": (
            "validation_selected_fixed_advantage_fusion"
            if uses_fixed_fusion
            else "ovrl_guarded_advantage_fusion"
            if uses_gaaf
            else "ovrl_primary_asymmetric_gradient_projection"
            if uses_projected_gaaf
            else "ovrl_preferred_marble"
            if uses_marble
            else "sequential_reward_specialization"
        ),
        "source_run_dir": str(source_dir),
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "source_mismatch_override": {
            "authorized": True,
            "historically_allowed": allowed_source_mismatches,
            "observed": source_mismatches,
            "scope": "post-hoc algorithm branch; checkpoint tensors remain immutable",
        },
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "composite_control_checkpoint": str(control_checkpoint),
        "composite_control_checkpoint_sha256": sha256_file(control_checkpoint),
        "composite_control_step": control_step,
        "source_step": source_step,
        "global_step_offset": int(branch["global_step_offset"]),
        "state_loaded": [
            "online_lora",
            "ema_rollout_lora",
            "optimizer",
            "scheduler",
            "rng",
        ],
        "source_training_reward": source_reward,
        "specialist_training_reward": target_reward,
    }
    return rollout_state, descriptor


def _load_composite_control_ema(branch_descriptor: Mapping) -> AdapterState:
    path = Path(str(branch_descriptor["composite_control_checkpoint"]))
    if sha256_file(path) != str(
        branch_descriptor["composite_control_checkpoint_sha256"]
    ):
        raise ValueError("composite control checkpoint changed after branch creation")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    criteria = {
        "schema": payload.get("schema_version") == 1,
        "protocol": str(payload.get("protocol_hash", ""))
        == Path(str(branch_descriptor["source_run_dir"])).name,
        "step": int(payload.get("completed_step", -1))
        == int(branch_descriptor["composite_control_step"]),
        "rollout_lora": bool(payload.get("rollout_lora")),
    }
    if not all(criteria.values()):
        raise ValueError(f"invalid historical composite control checkpoint: {criteria}")
    return {
        str(name): value.detach().cpu().clone()
        for name, value in payload["rollout_lora"].items()
    }


def run(config: dict, *, resume: Path | None = None) -> tuple[dict, Path]:
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    deterministic = bool(config["run"]["deterministic"])
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lightweight_startup = (
        str(config.get("run", {}).get("startup_validation", "full"))
        == "lightweight"
    )
    bundle = load_flowse_bundle(
        config["flowse_config"],
        deterministic=deterministic,
        compute_artifact_hashes=not lightweight_startup,
    )
    torch.manual_seed(int(config["lora"]["initialization_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["lora"]["initialization_seed"]))
    injection = inject_lora(
        bundle.model.transformer,
        target_patterns=config["lora"]["target_patterns"],
        rank=int(config["lora"]["rank"]),
        alpha=float(config["lora"]["alpha"]),
        dropout=float(config["lora"]["dropout"]),
        expected_modules=int(config["lora"]["expected_modules"]),
    )
    shared_initial_lora_snapshot = None
    if config["lora"].get("shared_initial_snapshot") is not None:
        shared_initial_lora_snapshot = prepare_or_load_shared_lora_snapshot(
            bundle.model.transformer,
            config=config,
            module_names=injection.module_names,
        )
    bundle.model.eval()
    fidelity, fidelity_fingerprint = _load_fidelity(config, lazy_asr=True)
    reward_definition = resolve_training_reward(config)
    composite_evaluators = None
    composite_fingerprint = None
    if _uses_composite_evaluators(reward_definition):
        composite_evaluators, composite_fingerprint = (
            load_flowse_grpo_composite_evaluators(config)
        )
    evaluator_fingerprint = {
        "fidelity": fidelity_fingerprint,
        "training_reward": composite_fingerprint,
    }
    protocol = build_training_protocol(
        config,
        bundle=bundle,
        evaluator_fingerprint=evaluator_fingerprint,
        shared_initial_lora_snapshot=shared_initial_lora_snapshot,
    )
    protocol_hash = protocol["protocol_hash"]
    output_dir = Path(config["output_root"]) / protocol_hash
    if resume is not None:
        resume_path = Path(resume)
        if resume_path.name != "checkpoint_latest.pt":
            raise ValueError(
                "transactional AdvantageFlow resume requires checkpoint_latest.pt"
            )
        resume_output_dir = resume_path.parent
        frozen_protocol_path = resume_output_dir / "protocol.json"
        if not frozen_protocol_path.is_file():
            raise FileNotFoundError(frozen_protocol_path)
        frozen_components = json.loads(
            frozen_protocol_path.read_text(encoding="utf-8")
        )
        frozen_protocol_hash = sha256_json(frozen_components)
        if frozen_protocol_hash != resume_output_dir.name:
            raise ValueError(
                "resume transaction directory does not match its frozen protocol"
            )
        # Runtime metadata and the explicitly enumerated maintenance sources
        # must not redirect a resume into a new transaction.  Model, loss,
        # LoRA, data, evaluator, and all other execution sources remain frozen.
        provenance_only = {"source_sha256", "runtime_environment"}
        current_components = protocol["components"]
        scientific_keys = sorted(
            (set(frozen_components) | set(current_components)) - provenance_only
        )
        scientific_mismatches = [
            key
            for key in scientific_keys
            if frozen_components.get(key) != current_components.get(key)
        ]
        if scientific_mismatches:
            raise ValueError(
                "resume scientific protocol differs from the checkpoint: "
                f"{scientific_mismatches}"
            )
        frozen_sources = dict(frozen_components.get("source_sha256") or {})
        current_sources = dict(current_components.get("source_sha256") or {})
        source_mismatches = sorted(
            name
            for name in set(frozen_sources) | set(current_sources)
            if frozen_sources.get(name) != current_sources.get(name)
        )
        # Source fingerprints are provenance only.  Training may be resumed
        # after local source maintenance without editing an allowlist.  The
        # checkpoint/run identity and the scientific config/state checks above
        # remain authoritative for transactional recovery.
        protocol_hash = frozen_protocol_hash
        output_dir = resume_output_dir
        protocol = {
            **protocol,
            "protocol_hash": protocol_hash,
            "components": frozen_components,
        }
        print(
            "Transactional resume: PASS "
            "(source-hash enforcement disabled; frozen run directory retained; "
            f"observed source changes={source_mismatches})",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        _write_json(output_dir / "protocol.json", protocol["components"])
    _write_json(
        Path(config["output_root"]) / "latest_protocol.json",
        {"protocol_hash": protocol_hash, "protocol_dir": str(output_dir)},
    )

    main_cuda_memory_reservation = _reserve_cuda_allocator_memory(
        config,
        device=bundle.device,
        role="coordinator",
    )

    optimizer, scheduler = _build_optimizer(bundle.model.transformer, config)
    rollout_state = snapshot_lora(bundle.model.transformer, device="cpu")
    total_steps = int(config["run"]["optimizer_steps"])
    checkpoint_interval = int(config["artifacts"]["checkpoint_interval"])
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    training_log = output_dir / "training_steps.jsonl"
    rollout_log = output_dir / "rollout_metrics.jsonl"
    completed_step = 0
    checkpoint_extra = {}
    branch_descriptor = None
    fixed_fusion_config = config.get("fixed_fusion")
    uses_fixed_fusion = isinstance(fixed_fusion_config, Mapping)
    gaaf_config = config.get("gaaf")
    uses_gaaf = isinstance(gaaf_config, Mapping)
    projected_gaaf_config = config.get("projected_gaaf")
    uses_projected_gaaf = isinstance(projected_gaaf_config, Mapping)
    marble_config = config.get("marble")
    uses_marble = isinstance(marble_config, Mapping)
    if sum(
        (uses_fixed_fusion, uses_gaaf, uses_projected_gaaf, uses_marble)
    ) > 1:
        raise ValueError(
            "configure only one of fixed_fusion, gaaf, projected_gaaf, or marble"
        )
    gaaf_gate_state = None
    projected_gaaf_state = None
    marble_state = None
    if resume is not None:
        if Path(resume).resolve() != checkpoint_path.resolve():
            raise ValueError(
                "transactional AdvantageFlow resume requires this run's "
                "checkpoint_latest.pt"
            )
        completed_step, rollout_state, checkpoint_extra = load_training_checkpoint(
            resume,
            transformer=bundle.model.transformer,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_protocol_hash=protocol_hash,
        )
        branch_descriptor = checkpoint_extra.get("branch_lineage")
        if "branch" in config and not isinstance(branch_descriptor, Mapping):
            raise ValueError("branch checkpoint is missing its frozen lineage")
        gaaf_gate_state = checkpoint_extra.get("gaaf_gate_state")
        projected_gaaf_state = checkpoint_extra.get("projected_gaaf_state")
        marble_state = checkpoint_extra.get("marble_state")
        if uses_gaaf:
            if completed_step > 0 and not isinstance(gaaf_gate_state, Mapping):
                raise ValueError("resumed GA-AF checkpoint lacks its gate state")
            if gaaf_gate_state is not None:
                validate_gate_state(gaaf_gate_state)
        if uses_projected_gaaf:
            if completed_step > 0 and not isinstance(
                projected_gaaf_state, Mapping
            ):
                raise ValueError(
                    "resumed projected GA-AF checkpoint lacks its state"
                )
            if projected_gaaf_state is not None:
                validate_projected_gaaf_state(projected_gaaf_state)
        if uses_marble:
            if completed_step > 0 and not isinstance(marble_state, Mapping):
                raise ValueError("resumed MARBLE checkpoint lacks its state")
            if marble_state is not None:
                validate_marble_state(marble_state)
    elif "branch" in config:
        rollout_state, branch_descriptor = _load_branch_checkpoint(
            config,
            transformer=bundle.model.transformer,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
    train_manifest = protocol["train_manifest"]
    evaluation_manifest = protocol["evaluation_manifest"]
    if resume is not None:
        step_records = recover_step_transaction(
            output_dir=output_dir,
            latest_checkpoint=checkpoint_path,
            completed_step=completed_step,
            checkpoint_extra=checkpoint_extra,
            protocol_hash=protocol_hash,
            checkpoint_interval=checkpoint_interval,
            total_steps=total_steps,
        )
        truncate_jsonl_after_step(rollout_log, completed_step)
    else:
        transaction_artifacts = [
            checkpoint_path,
            output_dir / "checkpoint_pending.pt",
            training_log,
            rollout_log,
        ]
        if any(path.exists() for path in transaction_artifacts) or any(
            (output_dir / "accounting_steps").glob("step_*.json")
        ) or any(output_dir.glob("checkpoint_step_*.pt")):
            raise FileExistsError(
                "existing AdvantageFlow transaction artifacts require --resume "
                f"{checkpoint_path}"
            )
        initial_commit_id = step_commit_id(protocol_hash, 0)
        save_training_checkpoint(
            checkpoint_path,
            transformer=bundle.model.transformer,
            rollout_state=rollout_state,
            optimizer=optimizer,
            scheduler=scheduler,
            step=0,
            protocol_hash=protocol_hash,
            extra={
                "branch_lineage": branch_descriptor,
                "gaaf_gate_state": gaaf_gate_state,
                "projected_gaaf_state": projected_gaaf_state,
                "marble_state": marble_state,
                "step_transaction": {
                    "schema_version": 1,
                    "step": 0,
                    "step_commit_id": initial_commit_id,
                    "commit_protocol": "initial_checkpoint",
                }
            },
        )
        step_records = []
    evaluation_reports = []

    interval = int(config["evaluation"]["interval_steps"])
    resume_evaluation_path = output_dir / f"evaluation_step_{completed_step:06d}.json"
    if (
        completed_step > 0
        and (completed_step % interval == 0 or completed_step == total_steps)
        and not resume_evaluation_path.is_file()
    ):
        current = snapshot_lora(bundle.model.transformer, device="cpu")
        evaluation_reports.append(
            _evaluate(
                bundle=bundle,
                state=rollout_state
                if config["evaluation"]["policy"] == "ema"
                else current,
                step=completed_step,
                manifest=evaluation_manifest,
                config=config,
                conditioning=conditioning,
                dnsmos=dnsmos,
                fidelity=fidelity,
                composite_evaluators=composite_evaluators,
                output_dir=output_dir,
            )
        )
        load_lora(bundle.model.transformer, current)

    parallel_pool = None
    if completed_step < total_steps and bool(
        config.get("parallel_rollout", {}).get("enabled", False)
    ):
        try:
            parallel_pool = _ShardedRolloutPool(
                config=config,
                output_dir=output_dir,
                rollout_state=rollout_state,
                expected_reward_evaluator_fingerprint=composite_fingerprint,
            )
        except Exception:
            # A brand-new run has not consumed data or updated parameters yet.
            # Remove only its initial step-0 checkpoint so a failed GPU claim can
            # be retried on freer cards without a meaningless --resume step.
            if (
                resume is None
                and completed_step == 0
                and checkpoint_path.is_file()
                and not training_log.exists()
                and not rollout_log.exists()
            ):
                checkpoint_path.unlink()
            raise

    if completed_step == 0 and bool(config["evaluation"]["run_initial"]):
        cache_path = config["evaluation"].get("initial_cache_report_path")
        if cache_path is not None:
            evaluation_reports.append(
                _import_initial_evaluation_cache(
                    manifest=evaluation_manifest,
                    config=config,
                    output_dir=output_dir,
                )
            )
        else:
            current = snapshot_lora(bundle.model.transformer, device="cpu")
            try:
                evaluation_reports.append(
                    _evaluate(
                        bundle=bundle,
                        state=(
                            rollout_state
                            if config["evaluation"]["policy"] == "ema"
                            else current
                        ),
                        step=0,
                        manifest=evaluation_manifest,
                        config=config,
                        conditioning=conditioning,
                        dnsmos=dnsmos,
                        fidelity=fidelity,
                        composite_evaluators=composite_evaluators,
                        output_dir=output_dir,
                    )
                )
            except Exception:
                if parallel_pool is not None:
                    parallel_pool.close()
                raise
            load_lora(bundle.model.transformer, current)

    for step in range(completed_step + 1, total_steps + 1):
        global_step = step + int(
            config.get("branch", {}).get("global_step_offset", 0)
        )
        step_started = time.perf_counter()
        current_before = snapshot_lora(bundle.model.transformer, device="cpu")
        utterances = utterances_for_step(
            list(train_manifest),
            step=global_step,
            conditions_per_step=int(config["run"]["conditions_per_step"]),
            seed=int(config["data"]["order_seed"]),
            stride_per_step=int(
                config["data"].get(
                    "condition_stride_per_step",
                    config["run"]["conditions_per_step"],
                )
            ),
        )
        rollout_function = (
            parallel_pool.rollout
            if parallel_pool is not None
            else _rollout_training_batch
        )
        _synchronize_cuda(bundle.device)
        rollout_started = time.perf_counter()
        conditions, rollout_rows, geometry = rollout_function(
            bundle=bundle,
            rollout_state=rollout_state,
            manifest=train_manifest,
            utterances=utterances,
            step=global_step,
            config=config,
            conditioning=conditioning,
            dnsmos=dnsmos,
            fidelity=fidelity,
            composite_evaluators=composite_evaluators,
            output_dir=output_dir,
        )
        for row in rollout_rows:
            row["global_optimizer_step"] = global_step
            row["step"] = step
        _synchronize_cuda(bundle.device)
        rollout_seconds = time.perf_counter() - rollout_started
        load_lora(bundle.model.transformer, current_before)
        optimizer.zero_grad(set_to_none=True)
        _synchronize_cuda(bundle.device)
        optimization_started = time.perf_counter()
        loss_backward_started = time.perf_counter()
        fixed_fusion_step = None
        gaaf_step = None
        projected_gaaf_step = None
        marble_step = None
        if uses_fixed_fusion or uses_gaaf or uses_projected_gaaf or uses_marble:
            component_streams, component_diagnostics = component_advantage_streams(
                rollout_rows,
                conditions=len(conditions),
                candidates=int(config["rollout"]["candidates_per_condition"]),
                advantage_config=config["advantage"],
            )
            if uses_fixed_fusion:
                fixed_weights = {
                    name: float(fixed_fusion_config["weights"][name])
                    for name in GAAF_COMPONENTS
                }
                fused_stream = convex_fuse_streams(component_streams, fixed_weights)
                total_fixed_weight = float(sum(fixed_weights.values()))
                fusion_state = {
                    "weights": fixed_weights,
                    "convex_weights": {
                        name: value / total_fixed_weight
                        for name, value in fixed_weights.items()
                    },
                    "weight_provenance": json.loads(
                        json.dumps(
                            fixed_fusion_config.get(
                                "validation_selection",
                                fixed_fusion_config.get("weight_source"),
                            )
                        )
                    ),
                }
                fusion_diagnostics = {
                    "method": "validation_selected_fixed_advantage_fusion",
                    "formula": "(A_D + b_S*A_S + b_C*A_C)/(1+b_S+b_C)",
                    "weights_constant_during_training": True,
                }
                calibration_due = False
                calibration_backward_calls = 0
                calibration_losses = None
            elif uses_gaaf:
                fusion_config = gaaf_config
                fusion_state = gaaf_gate_state
                calibration_function = gaaf_calibration_due
            elif uses_projected_gaaf:
                fusion_config = projected_gaaf_config
                fusion_state = projected_gaaf_state
                calibration_function = projected_gaaf_calibration_due
            else:
                fusion_config = marble_config
                fusion_state = marble_state
                calibration_function = marble_calibration_due
            if not uses_fixed_fusion:
                refresh_interval = int(fusion_config["gradient_refresh_interval"])
                calibration_due = calibration_function(
                    fusion_state,
                    local_step=step,
                    refresh_interval=refresh_interval,
                )
                calibration_backward_calls = 0
                calibration_losses = None
            if not uses_fixed_fusion and calibration_due:

                def component_loss_function(
                    stream, base_conditions=conditions
                ) -> torch.Tensor:
                    probe_conditions = replace_advantages(base_conditions, stream)
                    probe_loss, _ = paper_advantageflow_loss(
                        bundle,
                        probe_conditions,
                        current_state=current_before,
                        rollout_state=rollout_state,
                        conditioning=conditioning,
                        draw_seed_base=int(config["loss"]["draw_seed_base"]),
                        optimizer_step=global_step,
                        lambda_reference=float(config["loss"]["lambda_reference"]),
                        gamma_mode=str(config["loss"]["gamma_mode"]),
                        curvature_margin=float(config["loss"]["curvature_margin"]),
                        time_minimum=float(config["loss"]["time_minimum"]),
                        time_maximum=float(config["loss"]["time_maximum"]),
                        microbatch_size=int(config["loss"]["microbatch_size"]),
                        length_adaptive_microbatching=config.get(
                            "length_adaptive_microbatching"
                        ),
                        accumulate_gradients=True,
                    )
                    return probe_loss

                component_gradients, calibration_losses, calibration_backward_calls = (
                    reward_induced_gradients(
                        loss_function=component_loss_function,
                        streams=component_streams,
                        zero_grad=lambda: optimizer.zero_grad(set_to_none=True),
                        named_parameters=list(
                            named_lora_parameters(bundle.model.transformer)
                        ),
                    )
                )
                if uses_gaaf:
                    observed_weights, gradient_cosines, component_gradient_norms = (
                        observed_gate_weights(
                            component_gradients,
                            auxiliary_cap=float(gaaf_config["auxiliary_cap"]),
                        )
                    )
                    gaaf_gate_state = update_gate_state(
                        gaaf_gate_state,
                        observed_weights=observed_weights,
                        cosines=gradient_cosines,
                        gradient_norms=component_gradient_norms,
                        ema_decay=float(gaaf_config["coefficient_ema_decay"]),
                        local_step=step,
                        global_step=global_step,
                    )
                elif uses_projected_gaaf:
                    (
                        observed_weights,
                        gradient_cosines,
                        component_gradient_norms,
                        projection_diagnostics,
                    ) = projected_gaaf_observed_weights(
                        component_gradients,
                        auxiliary_target_norm_ratio=float(
                            projected_gaaf_config["auxiliary_target_norm_ratio"]
                        ),
                        auxiliary_coefficient_cap=float(
                            projected_gaaf_config["auxiliary_coefficient_cap"]
                        ),
                        projection_epsilon=float(
                            projected_gaaf_config["projection_epsilon"]
                        ),
                    )
                    projected_gaaf_state = update_projected_gaaf_state(
                        projected_gaaf_state,
                        observed_weights=observed_weights,
                        cosines=gradient_cosines,
                        gradient_norms=component_gradient_norms,
                        diagnostics=projection_diagnostics,
                        ema_decay=float(
                            projected_gaaf_config["coefficient_ema_decay"]
                        ),
                        local_step=step,
                        global_step=global_step,
                    )
                else:
                    (
                        observed_weights,
                        marble_alignments,
                        component_gradient_norms,
                        marble_qp_diagnostics,
                    ) = marble_simplex_weights(
                        component_gradients,
                        primary_preference=float(marble_config["ovrl_preference"]),
                        norm_epsilon=float(marble_config["norm_epsilon"]),
                    )
                    marble_state = update_marble_state(
                        marble_state,
                        observed_weights=observed_weights,
                        alignments=marble_alignments,
                        gradient_norms=component_gradient_norms,
                        diagnostics=marble_qp_diagnostics,
                        ema_decay=float(marble_config["coefficient_ema_decay"]),
                        local_step=step,
                        global_step=global_step,
                    )
            if uses_fixed_fusion:
                pass
            elif uses_gaaf:
                validate_gate_state(gaaf_gate_state)
                fused_stream = convex_fuse_streams(
                    component_streams, gaaf_gate_state["weights"]
                )
                fusion_state = gaaf_gate_state
                fusion_diagnostics = None
            elif uses_projected_gaaf:
                validate_projected_gaaf_state(projected_gaaf_state)
                fused_stream = convex_fuse_streams(
                    component_streams, projected_gaaf_state["weights"]
                )
                fusion_state = projected_gaaf_state
                fusion_diagnostics = {
                    "method": (
                        "convex_advantage_reconstruction_of_projected_gradient"
                    ),
                    "raw_stream_coefficients": dict(
                        projected_gaaf_state["weights"]
                    ),
                    "convex_stream_weights": dict(
                        projected_gaaf_state["convex_weights"]
                    ),
                }
            else:
                validate_marble_state(marble_state)
                fused_stream, fusion_diagnostics = marble_fuse_streams(
                    component_streams,
                    marble_state["convex_weights"],
                    marble_state["last_reward_induced_gradient_norms"],
                    clip=float(config["advantage"]["clip"]),
                    norm_epsilon=float(marble_config["norm_epsilon"]),
                )
                fusion_state = marble_state
            conditions = replace_advantages(conditions, fused_stream)
            rows_by_key = {
                (int(row["condition_index"]), int(row["candidate_index"])): row
                for row in rollout_rows
            }
            fused_values = []
            for condition_index, fused_advantages in enumerate(fused_stream):
                for candidate_index in range(len(fused_advantages)):
                    row = rows_by_key[(condition_index, candidate_index)]
                    row[
                        "fixed_fusion_component_advantages"
                        if uses_fixed_fusion
                        else "gaaf_component_advantages"
                        if uses_gaaf
                        else "projected_gaaf_component_advantages"
                        if uses_projected_gaaf
                        else "marble_component_advantages"
                    ] = {
                        name: float(
                            component_streams[name][condition_index][candidate_index]
                        )
                        for name in GAAF_COMPONENTS
                    }
                    row[
                        "fixed_fusion_fused_advantage"
                        if uses_fixed_fusion
                        else "gaaf_fused_advantage"
                        if uses_gaaf
                        else "projected_gaaf_fused_advantage"
                        if uses_projected_gaaf
                        else "marble_fused_advantage"
                    ] = float(
                        fused_advantages[candidate_index]
                    )
                    fused_values.append(float(fused_advantages[candidate_index]))
            fusion_name = (
                "fixed_fusion"
                if uses_fixed_fusion
                else "gaaf"
                if uses_gaaf
                else "projected_gaaf"
                if uses_projected_gaaf
                else "marble"
            )
            geometry[fusion_name] = {
                "component_advantages": component_diagnostics,
                "gate_state": json.loads(json.dumps(fusion_state))
                if uses_gaaf
                else None,
                "state": json.loads(json.dumps(fusion_state))
                if uses_fixed_fusion or uses_projected_gaaf or uses_marble
                else None,
                "fusion": fusion_diagnostics,
                "fused_advantage_statistics": {
                    "minimum": float(np.min(fused_values)),
                    "maximum": float(np.max(fused_values)),
                    "mean": float(np.mean(fused_values)),
                    "std": float(np.std(fused_values, ddof=0)),
                    "positive_fraction": float(np.mean(np.asarray(fused_values) > 0)),
                    "negative_fraction": float(np.mean(np.asarray(fused_values) < 0)),
                },
                "calibration_performed": bool(calibration_due),
            }
            optimizer.zero_grad(set_to_none=True)
            step_payload = {
                "calibration_performed": bool(calibration_due),
                "calibration_backward_calls": int(calibration_backward_calls),
                "calibration_losses": calibration_losses,
                "component_advantages": component_diagnostics,
                "fusion": fusion_diagnostics,
                "state": json.loads(json.dumps(fusion_state)),
            }
            if uses_fixed_fusion:
                fixed_fusion_step = step_payload
            elif uses_gaaf:
                gaaf_step = {**step_payload, "gate_state": step_payload["state"]}
            elif uses_projected_gaaf:
                projected_gaaf_step = step_payload
            else:
                marble_step = step_payload
        # Match GRPO's durability granularity: one batch append and one fsync
        # per logical step.  For GA-AF this occurs only after the actual fused
        # training advantages have been attached.  The latest checkpoint
        # remains commit authority and resume truncates any uncommitted tail.
        _append_jsonl_batch(rollout_log, rollout_rows)
        loss, loss_metrics = paper_advantageflow_loss(
            bundle,
            conditions,
            current_state=current_before,
            rollout_state=rollout_state,
            conditioning=conditioning,
            draw_seed_base=int(config["loss"]["draw_seed_base"]),
            optimizer_step=global_step,
            lambda_reference=float(config["loss"]["lambda_reference"]),
            gamma_mode=str(config["loss"]["gamma_mode"]),
            curvature_margin=float(config["loss"]["curvature_margin"]),
            time_minimum=float(config["loss"]["time_minimum"]),
            time_maximum=float(config["loss"]["time_maximum"]),
            microbatch_size=int(config["loss"]["microbatch_size"]),
            length_adaptive_microbatching=config.get(
                "length_adaptive_microbatching"
            ),
            accumulate_gradients=True,
        )
        _synchronize_cuda(bundle.device)
        loss_backward_seconds = time.perf_counter() - loss_backward_started
        optimizer_started = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            lora_parameters(bundle.model.transformer),
            max_norm=float(config["optimizer"]["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        _synchronize_cuda(bundle.device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        ema_started = time.perf_counter()
        current_after = snapshot_lora(bundle.model.transformer, device="cpu")
        update_metrics = _gradient_and_update_statistics(
            current_before, bundle.model.transformer, float(gradient_norm.item())
        )
        rollout_state = ema_adapter_state(
            rollout_state,
            current_after,
            float(config["ema"]["decay"]),
            device="cpu",
        )
        ema_seconds = time.perf_counter() - ema_started
        optimization_seconds = time.perf_counter() - optimization_started
        step_record = {
            "step": step,
            "global_optimizer_step": global_step,
            "protocol_hash": protocol_hash,
            "utterances": utterances,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "rollout_geometry": geometry,
            "loss": loss_metrics,
            "update": update_metrics,
            "current_rollout_distance": adapter_distance(current_after, rollout_state),
            "lora_A_has_updated": update_metrics["lora_A_update_norm"] > 0.0,
            "lora_B_has_updated": update_metrics["lora_B_update_norm"] > 0.0,
            "timing_seconds": {
                "rollout_and_reward": rollout_seconds,
                "loss_and_backward": loss_backward_seconds,
                "optimizer": optimizer_seconds,
                "ema": ema_seconds,
                "loss_optimizer_ema": optimization_seconds,
                "through_update": time.perf_counter() - step_started,
            },
            "allocated_rollout_world_size": int(
                config.get("parallel_rollout", {}).get("world_size", 1)
            ),
        }
        if fixed_fusion_step is not None:
            step_record["fixed_fusion"] = fixed_fusion_step
        if gaaf_step is not None:
            step_record["gaaf"] = gaaf_step
        if projected_gaaf_step is not None:
            step_record["projected_gaaf"] = projected_gaaf_step
        if marble_step is not None:
            step_record["marble"] = marble_step
        checkpoint_started = time.perf_counter()

        def checkpoint_writer(path: Path, extra: Mapping) -> None:
            save_training_checkpoint(
                path,
                transformer=bundle.model.transformer,
                rollout_state=rollout_state,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                protocol_hash=protocol_hash,
                extra={
                    **extra,
                    "branch_lineage": branch_descriptor,
                    "gaaf_gate_state": gaaf_gate_state,
                    "projected_gaaf_state": projected_gaaf_state,
                    "marble_state": marble_state,
                },
            )

        step_record = commit_step_transaction(
            output_dir=output_dir,
            latest_checkpoint=checkpoint_path,
            step_record=step_record,
            step=step,
            protocol_hash=protocol_hash,
            checkpoint_interval=checkpoint_interval,
            total_steps=total_steps,
            checkpoint_writer=checkpoint_writer,
            checkpoint_started=checkpoint_started,
            now=time.perf_counter,
            step_started=step_started,
        )
        step_records.append(step_record)
        speaker_text = (
            f" SPK={geometry['reward_components']['speaker']['raw']['mean']:.5f}"
            if "speaker" in geometry["reward_components"]
            else ""
        )
        gaaf_text = ""
        if fixed_fusion_step is not None:
            state = fixed_fusion_step["state"]
            gaaf_text = (
                " FixedFusion "
                f"bS={state['weights']['speaker']:.4f} "
                f"bC={state['weights']['speechbertscore']:.4f}"
            )
        if gaaf_step is not None:
            gate = gaaf_step["gate_state"]
            gaaf_text = (
                f" GA-AF_cal={int(gaaf_step['calibration_performed'])} "
                f"wE={gate['convex_weights']['speaker']:.4f} "
                f"wB={gate['convex_weights']['speechbertscore']:.4f}"
            )
        if projected_gaaf_step is not None:
            state = projected_gaaf_step["state"]
            cosines = state["last_gradient_cosines"]
            gaaf_text = (
                f" PGA-AF_cal={int(projected_gaaf_step['calibration_performed'])} "
                f"cosE={cosines['speaker']:+.3f} "
                f"cosB={cosines['speechbertscore']:+.3f} "
                f"wO={state['convex_weights']['dnsmos']:.4f} "
                f"wE={state['convex_weights']['speaker']:.4f} "
                f"wB={state['convex_weights']['speechbertscore']:.4f}"
            )
        if marble_step is not None:
            state = marble_step["state"]
            gaaf_text = (
                f" MARBLE_cal={int(marble_step['calibration_performed'])} "
                f"aO={state['convex_weights']['dnsmos']:.4f} "
                f"aE={state['convex_weights']['speaker']:.4f} "
                f"aB={state['convex_weights']['speechbertscore']:.4f}"
            )
        advantage_text = f"A={geometry['advantage_mapping']}"
        if geometry["advantage_temperature"] is not None:
            advantage_text += f"(tau={geometry['advantage_temperature']:g})"
        cuda_peak_text = ",".join(
            f"r{int(item['role'].rsplit('_', 1)[-1])}:"
            f"{float(item['peak_allocated_bytes']) / 1024**3:.2f}G"
            for item in geometry.get("cuda_memory_by_rollout_rank", [])
        )
        nfe_text = ""
        if len(geometry.get("training_nfe_values", [])) > 1:
            nfe_text = f"NFE={geometry['training_nfe_counts']}"
        constraint_text = ""
        constraints = geometry.get("reward_constraints", {})
        if constraints.get("enabled", False):
            constraint_text = (
                f" Cpen={float(constraints['total_penalty_mean']):.4f}"
                f" Cfrac={float(constraints['penalized_fraction']):.2%}"
            )
        print(
            f"step={step}/{total_steps} global_step={global_step} "
            f"SIG={geometry['dnsmos_components']['dnsmos_sig']['mean']:.5f} "
            f"BAK={geometry['dnsmos_components']['dnsmos_bak']['mean']:.5f} "
            f"OVRL={geometry['dnsmos_components']['dnsmos_ovrl']['mean']:.5f} "
            f"P808={geometry['dnsmos_components']['dnsmos_p808']['mean']:.5f} "
            f"{speaker_text} "
            f"{constraint_text} "
            f"{gaaf_text} "
            f"Z={geometry['global_advantage_scale']:.6g} {advantage_text} "
            f"loss={loss_metrics['loss']:.6g} "
            f"grad={update_metrics['gradient_norm_clipped_return']:.6g} "
            f"A_update={update_metrics['lora_A_update_norm']:.6g} "
            f"B_update={update_metrics['lora_B_update_norm']:.6g} "
            f"{nfe_text} "
            "mb="
            f"{geometry['length_adaptive_microbatching']['effective_size_histogram']}"
            f"/{int(loss_metrics['effective_microbatch_size_min'])}-"
            f"{int(loss_metrics['effective_microbatch_size_max'])} "
            f"rollout_cuda_peak=[{cuda_peak_text}] "
            f"rollout_s={rollout_seconds:.1f} train_s={optimization_seconds:.1f}"
        )

        if step % interval == 0 or step == total_steps:
            policy_state: AdapterState = (
                rollout_state
                if config["evaluation"]["policy"] == "ema"
                else current_after
            )
            evaluation_reports.append(
                _evaluate(
                    bundle=bundle,
                    state=policy_state,
                    step=step,
                    manifest=evaluation_manifest,
                    config=config,
                    conditioning=conditioning,
                    dnsmos=dnsmos,
                    fidelity=fidelity,
                    composite_evaluators=composite_evaluators,
                    output_dir=output_dir,
                )
            )
            load_lora(bundle.model.transformer, current_after)
        del conditions
        gc.collect()
        if torch.cuda.is_available() and not _cuda_memory_reservation_enabled(config):
            torch.cuda.empty_cache()

    if parallel_pool is not None:
        parallel_pool.close()

    # A smoke is an implementation/reproducibility check. A pilot reports a
    # signal but never authorizes full training automatically.
    final_path = output_dir / f"evaluation_step_{total_steps:06d}.json"
    final_evaluation = (
        evaluation_reports[-1]
        if evaluation_reports
        else (
            json.loads(final_path.read_text(encoding="utf-8"))
            if final_path.is_file()
            else None
        )
    )
    initial_path = output_dir / "evaluation_step_000000.json"
    initial_evaluation = (
        json.loads(initial_path.read_text(encoding="utf-8"))
        if initial_path.is_file()
        else None
    )
    composite_control_evaluation = None
    specialization_comparison = None
    specialization_pilot = None
    if branch_descriptor is not None and final_evaluation is not None:
        current = snapshot_lora(bundle.model.transformer, device="cpu")
        control_state = _load_composite_control_ema(branch_descriptor)
        control_output_dir = output_dir / "composite_control"
        control_output_dir.mkdir(parents=True, exist_ok=True)
        baseline_cache = output_dir / "evaluation_noisy_baselines.json"
        control_baseline_cache = control_output_dir / "evaluation_noisy_baselines.json"
        if not control_baseline_cache.is_file():
            shutil.copy2(baseline_cache, control_baseline_cache)
        composite_control_evaluation = _evaluate(
            bundle=bundle,
            state=control_state,
            step=total_steps,
            manifest=evaluation_manifest,
            config=config,
            conditioning=conditioning,
            dnsmos=dnsmos,
            fidelity=fidelity,
            composite_evaluators=composite_evaluators,
            output_dir=control_output_dir,
        )
        load_lora(bundle.model.transformer, current)
        specialization_comparison = _paired_policy_gain(
            composite_control_evaluation, final_evaluation, config
        )
        specialization_comparison["comparison"] = (
            "fixed_fusion_minus_composite_control_common_validation_and_latents"
            if uses_fixed_fusion
            else "gaaf_minus_composite_control_common_validation_and_latents"
            if uses_gaaf
            else (
                "projected_gaaf_minus_composite_control_common_validation_and_latents"
            )
            if uses_projected_gaaf
            else "raw_ovrl_specialist_minus_composite_control_common_validation_and_latents"
        )
        specialization_comparison["global_source_step"] = int(
            branch_descriptor["source_step"]
        )
        specialization_comparison["global_endpoint_step"] = int(
            branch_descriptor["composite_control_step"]
        )
        specialization_pilot = _specialization_pilot_decision(
            specialization_comparison, config
        )
        _write_json(
            output_dir
            / (
                "fixed_fusion_minus_composite_control.json"
                if uses_fixed_fusion
                else "gaaf_minus_composite_control.json"
                if uses_gaaf
                else "projected_gaaf_minus_composite_control.json"
                if uses_projected_gaaf
                else "specialist_minus_composite_control.json"
            ),
            {
                "status": specialization_pilot["decision"],
                "branch_lineage": branch_descriptor,
                "analysis": specialization_comparison,
                "pilot_decision": specialization_pilot,
            },
        )
        _print_policy_gain(
            composite_control_evaluation,
            final_evaluation,
            specialization_comparison,
        )
    gain = (
        _paired_policy_gain(initial_evaluation, final_evaluation, config)
        if initial_evaluation is not None and final_evaluation is not None
        else None
    )
    pilot = (
        _pilot_decision(gain, config)
        if config["run"]["mode"] == "pilot" and gain is not None
        else None
    )
    if gain is not None:
        _print_policy_gain(initial_evaluation, final_evaluation, gain)
    smoke_health = None
    if config["run"]["mode"] == "smoke":
        smoke_criteria = {
            "at_least_two_real_optimizer_steps": len(step_records) >= 2,
            "first_step_lora_B_gradient_nonzero": bool(
                step_records and step_records[0]["update"]["lora_B_gradient_norm"] > 0.0
            ),
            "second_step_lora_A_gradient_nonzero": bool(
                len(step_records) >= 2
                and step_records[1]["update"]["lora_A_gradient_norm"] > 0.0
            ),
            "ema_differs_from_current": bool(
                step_records and step_records[-1]["current_rollout_distance"] > 0.0
            ),
            "global_advantage_signal_each_step": bool(
                step_records
                and all(
                    row["rollout_geometry"]["global_advantage_scale"]
                    > float(config["advantage"]["minimum_global_scale"])
                    for row in step_records
                )
            ),
            "parallel_rollout_shards_each_step": bool(
                not config.get("parallel_rollout", {}).get("enabled", False)
                or (
                    step_records
                    and all(
                        int(row["rollout_geometry"].get("rollout_shards", -1))
                        == int(config["parallel_rollout"]["world_size"])
                        for row in step_records
                    )
                )
            ),
            "final_evaluation_complete": final_evaluation is not None,
        }
        smoke_health = {
            "passed": all(smoke_criteria.values()),
            "criteria": smoke_criteria,
        }
        status = "SMOKE-PASS" if smoke_health["passed"] else "SMOKE-FAIL"
    else:
        status = (
            specialization_pilot["decision"]
            if specialization_pilot is not None
            else (
                pilot["decision"]
                if pilot is not None
                else "PILOT-FINAL-EVALUATION-COMPLETE"
            )
        )
    world_size = int(config.get("parallel_rollout", {}).get("world_size", 1))
    compute_accounting = _compute_training_accounting(
        step_records, output_dir=output_dir, world_size=world_size
    )
    device_ids = [
        int(value)
        for value in config.get("parallel_rollout", {}).get("device_ids", [0])
    ]
    resources = {
        "physical_gpu_count": world_size,
        "device_ids": device_ids,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_devices": int(torch.cuda.device_count()),
        "cuda_memory_reservation": {
            "coordinator": main_cuda_memory_reservation,
            "workers": (
                dict(parallel_pool.worker_memory_reservations)
                if parallel_pool is not None
                else {}
            ),
        },
        "devices": (
            [
                {
                    "logical_id": device_id,
                    "name": torch.cuda.get_device_name(device_id),
                    "total_memory": int(
                        torch.cuda.get_device_properties(device_id).total_memory
                    ),
                }
                for device_id in device_ids
            ]
            if torch.cuda.is_available()
            else []
        ),
    }
    report = {
        "status": status,
        "authorization": "never_automatically_authorizes_full_training",
        "protocol_hash": protocol_hash,
        "completed_steps": total_steps,
        "conditioning": conditioning.fingerprint(),
        "training_reward": resolve_training_reward(config)["name"],
        "training_reward_definition": resolve_training_reward(config),
        "branch_lineage": branch_descriptor,
        "fixed_fusion": (
            {
                "config": dict(fixed_fusion_config),
                "weights_constant_during_training": True,
                "component_advantages_computed_independently": True,
            }
            if uses_fixed_fusion
            else None
        ),
        "gaaf": (
            {
                "config": dict(gaaf_config),
                "final_gate_state": gaaf_gate_state,
                "calibration_steps": [
                    int(row["step"])
                    for row in step_records
                    if row.get("gaaf", {}).get("calibration_performed", False)
                ],
            }
            if uses_gaaf
            else None
        ),
        "projected_gaaf": (
            {
                "config": dict(projected_gaaf_config),
                "final_state": projected_gaaf_state,
                "calibration_steps": [
                    int(row["step"])
                    for row in step_records
                    if row.get("projected_gaaf", {}).get(
                        "calibration_performed", False
                    )
                ],
            }
            if uses_projected_gaaf
            else None
        ),
        "marble": (
            {
                "config": dict(marble_config),
                "final_state": marble_state,
                "calibration_steps": [
                    int(row["step"])
                    for row in step_records
                    if row.get("marble", {}).get("calibration_performed", False)
                ],
            }
            if uses_marble
            else None
        ),
        "parallel_rollout": config.get(
            "parallel_rollout", {"enabled": False, "world_size": 1}
        ),
        "compute_accounting": compute_accounting,
        "resources": resources,
        "evaluator_fingerprint": evaluator_fingerprint,
        "policy_roles": {
            "current": "trainable_lora",
            "old": "ema_rollout_lora",
            "reference": "released_checkpoint_lora_disabled",
        },
        "lora": {
            "modules": len(injection.module_names),
            "trainable_parameters": injection.trainable_parameters,
            "shared_initial_snapshot": shared_initial_lora_snapshot,
        },
        "final_evaluation": final_evaluation,
        "paired_policy_gain": gain,
        "composite_control_evaluation": composite_control_evaluation,
        "specialist_minus_composite_control": specialization_comparison,
        "gaaf_minus_composite_control": (
            specialization_comparison if uses_gaaf else None
        ),
        "fixed_fusion_minus_composite_control": (
            specialization_comparison if uses_fixed_fusion else None
        ),
        "projected_gaaf_minus_composite_control": (
            specialization_comparison if uses_projected_gaaf else None
        ),
        "pilot_decision": pilot,
        "specialization_pilot_decision": specialization_pilot,
        "smoke_health": smoke_health,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": (
            None if lightweight_startup else sha256_file(checkpoint_path)
        ),
    }
    _write_json(output_dir / "training_report.json", report)
    return report, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train audio-only speech AdvantageFlow"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, output_dir = run(config, resume=args.resume)
    print("\nSpeech AdvantageFlow training")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Protocol: {report['protocol_hash']}")
    print(f"Completed optimizer steps: {report['completed_steps']}")
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"Report: {output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
