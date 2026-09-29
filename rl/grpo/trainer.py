"""Online-LoRA FlowSE-GRPO training entry point for the controlled A/B.

The production path reuses method-neutral released-FlowSE, LoRA, VoiceBank
ordering, and frozen-reward components. Checkpoints are GRPO-online-only and resume at
collection boundaries, which keeps old-policy semantics unambiguous.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import math
import os
import queue
import random
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import yaml

from rl.common.conditioning import ConditioningProtocol
from rl.common.fairness import (
    CONTROLLED_SETTING,
    LORA_TARGET_PATTERNS,
    validate_controlled_baseline,
    validate_shared_optimizer,
)
from rl.common.lora import (
    inject_lora,
    load_lora,
    lora_enabled,
    lora_parameters,
    snapshot_lora,
)
from rl.rewards.specification import (
    FLOWSE_GRPO_COMPOSITE,
    compute_training_reward,
    resolve_training_reward,
    verify_reward_calibration,
)
from rl.common.shared_initialization import (
    validate_shared_lora_snapshot_spec,
)

from .math import (
    compute_group_advantages,
    gaussian_transition_log_prob,
    reference_gaussian_kl,
    sde_transition_stats,
)
from .objective import GRPOObjectiveOutput, grpo_objective
from .policy import policy_velocity
from .rollout import (
    FlowSEWindowedSDESampler,
    TrajectoryRollout,
    sample_window_spec,
)
from .protocol import (
    audit_data_splits,
    lora_state_fingerprint,
    milestone_collections,
    prepare_or_load_shared_lora_snapshot,
)
from .storage import (
    append_jsonl_batch,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    reconcile_collection_logs,
)


@dataclass(frozen=True)
class GRPOUpdateExample:
    trajectory_id: str
    group_id: str
    condition_mel: torch.Tensor
    frame_mask: torch.Tensor
    trajectory: TrajectoryRollout
    advantage: float


DNS10S_DATA_DOMAIN = "libritts_dns10s_v1"
DNS10S_ARTIFACT_ROOT = (
    "artifacts/af/manifests/libritts_dns10s"
)
DNS10S_CALIBRATION_REPORT = (
    "artifacts/af/"
    "libritts_dns10s_calibration_audit_8192/"
    "calibration_report.json"
)


def _validate_dns10s_controlled_baseline(config: Mapping) -> dict:
    """Validate method-neutral fields for the LibriTTS/DNS10s comparison."""

    data = config["data"]
    evaluation = config["evaluation"]
    sampler = config["sampler"]
    reward = config["training_reward"]
    public_smoke = str(config["run"].get("mode")) == "smoke"
    validation_name = (
        "libritts_dns10s_smoke_validation.json"
        if str(config["run"]["mode"]) == "smoke"
        else "libritts_dns10s_validation.json"
    )
    expected_manifests = {
        "train_manifest": f"{DNS10S_ARTIFACT_ROOT}/libritts_dns10s_train_exposures.json",
        "validation_manifest": f"{DNS10S_ARTIFACT_ROOT}/{validation_name}",
        "official_test_manifest": f"{DNS10S_ARTIFACT_ROOT}/dns2020_official_test_all.json",
    }
    normalized_paths = {
        key: str(data[key]).replace("\\", "/") for key in expected_manifests
    }
    checks = {
        "data_domain": str(config["comparison"].get("data_domain", ""))
        == DNS10S_DATA_DOMAIN,
        "flowse_config_neutral": public_smoke
        or str(config["flowse_config"]).replace("\\", "/")
        == "configs/flowse/flowse_libritts_sft20k_wotext.yaml",
        "manifests_neutral": public_smoke or normalized_paths == expected_manifests,
        "order_seed": int(data["order_seed"]) == 260810,
        "audio_roots": public_smoke
        or (
            str(data["noisy_dir"]).replace("\\", "/")
            == "data/libritts_dns10s/audio/noisy"
            and str(data["clean_dir"]).replace("\\", "/")
            == "data/libritts_dns10s/audio/clean"
        ),
        "conditioning": dict(config["conditioning"])
        == {"mode": "wotext", "use_text": False, "drop_text": True},
        "cfg_zero": float(sampler["cfg_strength"]) == 0.0,
        "evaluation_nfe": int(evaluation["nfe"]) == 32,
        "evaluation_latent_seed": int(evaluation["latent_seed_base"]) == 2700100,
        "paired_metrics": bool(evaluation["paired_metrics"]),
        "fidelity_config": (
            dict(evaluation["fidelity"])
            == {
                "enabled": True,
                "source_locked_config": (
                    "artifacts/configs/"
                    "gate_a_wotext_v2.locked.yaml"
                ),
                "device": "cuda",
            }
            or (
                str(config["run"].get("mode")) == "smoke"
                and dict(evaluation["fidelity"]) == {"enabled": False}
            )
        ),
        "normalization": dict(config["normalization"])
        == {"target_dbfs": -25.0, "peak_ceiling": 0.99, "output_subtype": "PCM_16"},
        "lora_geometry": (
            int(config["lora"]["rank"]),
            float(config["lora"]["alpha"]),
            float(config["lora"]["dropout"]),
            int(config["lora"]["expected_modules"]),
            tuple(config["lora"]["target_patterns"]),
        )
        == (32, 64.0, 0.0, 132, LORA_TARGET_PATTERNS),
        "reward": str(reward["name"]) == FLOWSE_GRPO_COMPOSITE,
        "reward_weights": dict(reward["weights"])
        == {"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
        "reward_normalization": str(reward["component_normalization"])
        == "frozen_std",
        "reward_calibration": public_smoke
        or dict(reward["calibration"])
        == {
            "report_path": DNS10S_CALIBRATION_REPORT,
            "source_nfe": 10,
            "std_ddof": 0,
            "dnsmos_divisor": 4.0,
        },
        "dnsmos": public_smoke
        or str(config["dnsmos_official_dir"]) == "pretrainmodel/DNSMOS-official",
    }
    expected_evaluators = {
        "device": "cuda",
        "speaker": {
            "backend": "modelscope_speaker_verification",
            "model_id": "iic/speech_eres2net_sv_zh-cn_16k-common",
            "revision": "v1.0.5",
            "local_model_dir": "pretrainmodel/speech_eres2net_sv_zh-cn_16k-common",
        },
        "speechbertscore": {
            "repo_id": "microsoft/wavlm-large",
            "revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
            "local_model_dir": "pretrainmodel/wavlm-large",
            "local_files_only": True,
            "layer": 14,
            "reference_cache_size": 64,
        },
    }
    checks["composite_evaluators"] = public_smoke or (
        dict(config["composite_reward_evaluators"]) == expected_evaluators
    )
    snapshot = config["lora"].get("shared_initial_snapshot")
    checks["shared_snapshot_neutral"] = public_smoke or (
        snapshot is not None
        and (
            str(snapshot["path"]).replace("\\", "/").startswith(
                "artifacts/shared_initial_lora/"
            )
            or Path(str(snapshot["path"])).is_absolute()
        )
    )
    permitted_cross_method_artifacts = {
        *expected_manifests.values(),
        DNS10S_CALIBRATION_REPORT,
    }
    observed_cross_method_artifacts: set[str] = set()

    def collect_cross_method_artifacts(value) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect_cross_method_artifacts(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_cross_method_artifacts(child)
        elif isinstance(value, str):
            normalized = value.replace("\\", "/")
            if "artifacts/af/" in normalized:
                observed_cross_method_artifacts.add(normalized)

    collect_cross_method_artifacts(config)
    checks["cross_method_references_are_shared_artifacts_only"] = public_smoke or (
        observed_cross_method_artifacts == permitted_cross_method_artifacts
    )
    if not all(checks.values()):
        raise ValueError(
            "DNS10s controlled shared-baseline mismatch: "
            + json.dumps(checks, sort_keys=True)
        )
    return {"baseline": checks, "optimizer": validate_shared_optimizer(config["optimizer"])}


def _validate_shared_baseline(config: Mapping) -> dict:
    # Public smoke configs are runnable examples rather than formal paired
    # comparison claims.  Keep algorithm/optimizer validation, but do not
    # require the authors' exact artifact paths or provenance records.
    if str(config["run"].get("mode")) == "smoke":
        return {
            "baseline": {"public_smoke": True},
            "optimizer": validate_shared_optimizer(config["optimizer"]),
        }
    if str(config["comparison"].get("data_domain", "voicebank")) == DNS10S_DATA_DOMAIN:
        return _validate_dns10s_controlled_baseline(config)
    return validate_controlled_baseline(config, method="grpo")


@dataclass(frozen=True)
class CollectionResult:
    collection_index: int
    rewards: torch.Tensor
    examples: tuple[GRPOUpdateExample, ...]
    rollout_rows: tuple[dict, ...]
    group_rows: tuple[dict, ...]
    candidate_audio_seconds: float
    phase_seconds: dict[str, float]
    active_gpu_seconds_by_phase: dict[str, float]
    retained_audio_count: int


def _canonical_hash(value: Mapping) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_physical_gpu_ids(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if not tokens:
            raise ValueError("physical GPU IDs cannot be empty")
        try:
            device_ids = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("physical GPU IDs must be non-negative integers") from exc
    else:
        device_ids = [int(device_id) for device_id in value]
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError("physical GPU IDs must be non-negative integers")
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("physical GPU IDs must be unique")
    return device_ids


def resolve_physical_gpu_session(
    config: Mapping,
    *,
    resume: Path | None,
    requested_resume_physical_gpu_ids: str | Sequence[int] | None = None,
    observed_cuda_visible_devices: str | None = None,
) -> dict:
    """Validate a resume-only physical remap without changing the config hash."""
    resources = config["resources"]
    world_size = int(resources["rollout_world_size"])
    frozen_value = resources.get("expected_cuda_visible_devices")
    frozen = (
        None if frozen_value is None else _parse_physical_gpu_ids(frozen_value)
    )
    observed_text = (
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if observed_cuda_visible_devices is None
        else observed_cuda_visible_devices
    )
    if not observed_text.strip() and frozen is None:
        observed = [int(device_id) for device_id in resources["device_ids"]]
    else:
        observed = _parse_physical_gpu_ids(observed_text)
    if len(observed) != world_size:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the configured rollout "
            f"world size: expected={world_size}, observed={observed}"
        )

    requested = None
    if requested_resume_physical_gpu_ids is not None:
        if resume is None:
            raise ValueError("physical GPU remapping is permitted only with --resume")
        requested = _parse_physical_gpu_ids(requested_resume_physical_gpu_ids)
        if len(requested) != world_size:
            raise ValueError(
                "resume physical GPU mapping must preserve rollout_world_size: "
                f"expected={world_size}, requested={requested}"
            )
        if observed != requested:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES differs from --resume-physical-gpus: "
                f"observed={observed}, requested={requested}"
            )
    elif frozen is not None and observed != frozen:
        expected_text = ",".join(str(device_id) for device_id in frozen)
        observed_value = ",".join(str(device_id) for device_id in observed)
        raise RuntimeError(
            "formal GRPO physical GPU order differs from the frozen mapping: "
            f"expected CUDA_VISIBLE_DEVICES={expected_text}, observed={observed_value}"
        )

    remapped = frozen is not None and observed != frozen
    return {
        "schema_version": 1,
        "logical_device_ids": [
            int(device_id) for device_id in resources["device_ids"]
        ],
        "frozen_physical_gpu_ids": frozen,
        "session_physical_gpu_ids": observed,
        "resume_requested": resume is not None,
        "resume_checkpoint": None if resume is None else str(resume),
        "explicit_resume_remap_authorization": requested is not None,
        "physical_gpu_remapped": remapped,
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_json(path: Path, value) -> None:
    atomic_write_json(path, value)


def _append_jsonl(path: Path, value) -> None:
    append_jsonl_batch(path, [value])


def validate_grpo_config(config: Mapping) -> dict:
    """Fail closed on choices that would invalidate the controlled comparison."""

    conditioning = ConditioningProtocol.from_config(dict(config["conditioning"]))
    if conditioning.fingerprint() != {
        "mode": "wotext",
        "use_text": False,
        "drop_text": True,
    }:
        raise ValueError("GRPO training must remain audio-only")
    run = config["run"]
    mode = str(run["mode"])
    dns10s_domain = (
        str(config["comparison"].get("data_domain", "voicebank"))
        == DNS10S_DATA_DOMAIN
    )
    if mode not in {"smoke", "pilot", "train", "secondary"}:
        raise ValueError("run.mode must be smoke, pilot, train, or secondary")
    if str(config["comparison"]["setting"]) != CONTROLLED_SETTING:
        raise ValueError(f"primary config must use {CONTROLLED_SETTING}")
    if str(config["comparison"]["policy_kind"]) != "grpo_online":
        raise ValueError("controlled GRPO checkpoints must be online policy")
    shared_baseline_report = _validate_shared_baseline(config)
    if float(config["sampler"]["cfg_strength"]) != 0.0:
        raise ValueError("controlled GRPO requires CFG=0 in rollout/recompute")
    if float(config["evaluation"]["cfg_strength"]) != 0.0:
        raise ValueError("controlled GRPO requires CFG=0 in evaluation")
    if int(config["evaluation"]["nfe"]) != 32:
        raise ValueError("controlled validation/official-test requires ODE NFE=32")
    if float(config["sampler"]["diffusion"]) != 0.4:
        raise ValueError("controlled sampler requires a=0.4")
    if int(config["sampler"]["window_size"]) != 2:
        raise ValueError("controlled sampler requires a two-step SDE window")
    if str(config["sampler"]["direction"]) != "0_to_1":
        raise ValueError("FlowSE sampler direction must be 0_to_1")
    if str(config["sampler"]["nfe_window_sampling_scope"]) != "group":
        raise ValueError("NFE/window sampling scope is frozen to each logical group")
    if str(config["objective"]["logprob_reduction"]) != "mean_valid":
        raise ValueError("controlled log-prob reduction must be mean_valid")
    if str(config["objective"]["kl_reduction"]) != "mean_valid":
        raise ValueError("controlled KL reduction must be mean_valid")
    if int(config["advantage"]["std_correction"]) != 0:
        raise ValueError("controlled group advantage uses population std")
    if config["advantage"].get("clip") is not None:
        raise ValueError("controlled group advantages are not clipped")
    if float(config["objective"]["clip_epsilon"]) != 0.2:
        raise ValueError("controlled public reproduction choice uses PPO epsilon=0.2")
    if str(config["objective"]["ratio"]) != "plain":
        raise ValueError("controlled GRPO forbids RatioNorm and requires plain ratio")
    if bool(config["objective"]["loss_dt_scaling"]):
        raise ValueError("Eq. (9) policy loss must not be divided by dt")
    if bool(config["ema"]["enabled"]):
        raise ValueError("controlled GRPO primary cannot create or use EMA")

    lora = config["lora"]
    if (
        int(lora["rank"]) != 32
        or float(lora["alpha"]) != 64.0
        or float(lora["dropout"]) != 0.0
        or int(lora["expected_modules"]) != 132
    ):
        raise ValueError(
            "LoRA must match AdvantageFlow rank/alpha/dropout/module count"
        )
    if tuple(lora["target_patterns"]) != LORA_TARGET_PATTERNS:
        raise ValueError("LoRA target patterns must exactly match AdvantageFlow")
    # DNS10s AF and GRPO load the same existing LoRA initialization file.  Its
    # embedded tensor fingerprint is verified and the observed state hash is
    # recorded at runtime; a generated, hardware-bound YAML is not required.
    require_external_snapshot_hash = mode in {"train", "secondary"} and str(
        config["comparison"].get("data_domain", "voicebank")
    ) != DNS10S_DATA_DOMAIN
    validate_shared_lora_snapshot_spec(
        config, formal=require_external_snapshot_hash
    )
    if str(config["optimizer"]["type"]) != "AdamW":
        raise ValueError("GRPO uses persistent AdamW")
    if str(config["optimizer"]["schedule"]) != "linear_decay":
        raise ValueError("GRPO uses a linear-decay optimizer-step schedule")

    collection = config["collection"]
    group_size = int(collection["group_size"])
    prompts = int(collection["prompts_per_mini_batch"])
    repeats = int(collection["mini_batch_repeats"])
    updates = int(collection["optimizer_updates"])
    microbatch_size = int(collection["microbatch_size"])
    if "global_batch_size" in collection:
        raise ValueError(
            "collection.global_batch_size is ambiguous under gradient accumulation; "
            "use microbatch_size"
        )
    production_geometry_smoke = (
        mode == "smoke"
        and str(config["comparison"].get("smoke_geometry", ""))
        == "production_dns10s"
    )
    if mode == "smoke" and not production_geometry_smoke:
        if (
            group_size != 2
            or prompts * repeats != 2
            or updates != 2
            or microbatch_size != 2
        ):
            raise ValueError(
                "smoke geometry must be 2 conditions, G=2, 2 updates, microbatch 2"
            )
    elif production_geometry_smoke:
        if (group_size, prompts * repeats, updates, microbatch_size) != (10, 4, 4, 4):
            raise ValueError(
                "DNS10s peak-memory smoke must use 4 complete G10 groups, "
                "4 updates, and microbatch 4"
            )
    else:
        expected_microbatch = 4 if dns10s_domain else 12
        if (group_size, prompts, repeats, updates, microbatch_size) != (
        10,
        6,
        12,
        4,
        expected_microbatch,
        ):
            raise ValueError(
                "controlled geometry must be G10, 6x12 prompts, 4 updates, "
                f"microbatch {expected_microbatch}"
            )
    if (mode != "smoke" or production_geometry_smoke) and (
        int(config["sampler"]["nfe_minimum"]) != 10
        or int(config["sampler"]["nfe_maximum"]) != 10
        or int(config["sampler"]["start_minimum"]) != 1
        or int(config["sampler"]["start_maximum"]) != 3
    ):
        raise ValueError("controlled A/B sampler requires fixed NFE 10 and S_min [1,3]")
    if str(collection["batch_semantics"]) != "full_buffer_4update_gradaccum":
        raise ValueError(
            "primary batch semantics must consume the complete eligible buffer "
            "through four gradient-accumulated updates"
        )
    collections = int(run["collections"])
    expected_steps = collections * updates
    if int(config["optimizer"]["schedule_total_steps"]) != expected_steps:
        raise ValueError("schedule_total_steps must equal collections * updates")
    comparison = config.get("comparison", {})
    if "base_condition_selections" in comparison:
        expected_base_conditions = collections * prompts
        if int(comparison["base_condition_selections"]) != expected_base_conditions:
            raise ValueError(
                "comparison.base_condition_selections must equal "
                "collections * prompts_per_mini_batch"
            )
        if str(comparison.get("base_condition_selection_semantics", "")) != (
            f"{prompts}_per_collection_each_repeated_{repeats}_times"
        ):
            raise ValueError(
                "comparison.base_condition_selection_semantics differs from "
                "the frozen native GRPO collection geometry"
            )

    resources = config.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("resources must explicitly freeze the GPU topology")
    world_size = int(resources["rollout_world_size"])
    device_ids = [int(value) for value in resources["device_ids"]]
    if world_size not in {1, 2, 4} or len(device_ids) != world_size:
        raise ValueError("rollout_world_size must be 1, 2, or 4 and match device_ids")
    if len(device_ids) != len(set(device_ids)) or any(
        value < 0 for value in device_ids
    ):
        raise ValueError("resource device_ids must be unique non-negative integers")
    if int(resources["trainer_device_id"]) != device_ids[0]:
        raise ValueError("trainer_device_id must be the coordinator device_ids[0]")
    correctness_pilot_2gpu = (
        mode == "pilot"
        and str(config["comparison"].get("budget_role", ""))
        == "implementation_correctness_pilot_2gpu"
    )
    if correctness_pilot_2gpu and (
        world_size != 2 or collections != 75 or expected_steps != 300
    ):
        raise ValueError(
            "2-GPU correctness pilot requires 75 collections and 300 optimizer steps"
        )
    if (mode != "smoke" or production_geometry_smoke) and world_size != 4 and not correctness_pilot_2gpu:
        raise ValueError(
            "controlled pilot/train topology is frozen to four rollout GPUs"
        )
    if not bool(resources.get("accumulate_across_resume", False)):
        raise ValueError("GPU-hour accounting must accumulate across resume")
    if (
        mode != "smoke"
        and str(config["comparison"].get("data_domain", "voicebank"))
        == DNS10S_DATA_DOMAIN
    ):
        reservation = config.get("cuda_memory_reservation")
        if not isinstance(reservation, Mapping) or not bool(
            reservation.get("enabled", False)
        ):
            raise ValueError("formal DNS10s GRPO requires CUDA peak-memory reservation")
        reservation_values = {
            "coordinator_target_reserved_gib": float(
                reservation["coordinator_target_reserved_gib"]
            ),
            "worker_target_reserved_gib": float(
                reservation["worker_target_reserved_gib"]
            ),
            "minimum_driver_free_gib": float(
                reservation["minimum_driver_free_gib"]
            ),
            "allocation_chunk_gib": float(reservation["allocation_chunk_gib"]),
        }
        if (
            reservation_values["coordinator_target_reserved_gib"] <= 0.0
            or reservation_values["worker_target_reserved_gib"] <= 0.0
            or reservation_values["minimum_driver_free_gib"] < 0.0
            or reservation_values["allocation_chunk_gib"] <= 0.0
            or not bool(reservation.get("require_target", False))
        ):
            raise ValueError("invalid required CUDA peak-memory reservation")

    mechanism_audit = config.get("mechanism_audit")
    mechanism_checkpoints: list[int] = []
    if correctness_pilot_2gpu:
        if not isinstance(mechanism_audit, Mapping):
            raise ValueError("2-GPU correctness pilot requires mechanism_audit")
        mechanism_checkpoints = [
            int(value) for value in mechanism_audit.get("checkpoint_collections", [])
        ]
        expected_mechanism_checkpoints = [0, 1, 5, 10, 20, 40, 60, 75]
        if mechanism_checkpoints != expected_mechanism_checkpoints:
            raise ValueError(
                "2-GPU correctness pilot mechanism checkpoints must be "
                f"{expected_mechanism_checkpoints}"
            )
        if not bool(mechanism_audit.get("enabled", False)):
            raise ValueError("2-GPU correctness pilot mechanism audit must be enabled")
        if not bool(mechanism_audit.get("save_full_checkpoint", False)):
            raise ValueError("mechanism audit must save reloadable full checkpoints")
        if bool(mechanism_audit.get("run_validation", True)):
            raise ValueError("mechanism audit checkpoints must not query validation")
        if bool(mechanism_audit.get("selection_eligible", True)):
            raise ValueError(
                "mechanism audit checkpoints must never enter checkpoint selection"
            )
        if not bool(
            mechanism_audit.get(
                "analysis_artifact_io_excluded_from_training_budget", False
            )
        ):
            raise ValueError(
                "mechanism audit I/O must be explicitly excluded from the "
                "correctness-pilot training budget"
            )
    elif mechanism_audit is not None:
        raise ValueError(
            "mechanism_audit is reserved for the isolated 2-GPU correctness pilot"
        )

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must define the training-audio retention policy")
    audit_count = int(artifacts["audit_audio_candidates_per_collection"])
    if audit_count < 0 or audit_count > prompts * repeats * group_size:
        raise ValueError("invalid audit audio count")
    if mode != "smoke" and bool(artifacts["keep_training_audio"]):
        raise ValueError("pilot/train cannot retain every rollout WAV")
    reporting_checkpoints = reporting_checkpoint_collections(config)

    milestones = [int(value) for value in config["evaluation"]["selection_milestones"]]
    expected_milestones = (
        [0, 100]
        if mode == "smoke"
        else (
            [0, 25, 50, 75, 100]
            if mode == "pilot"
            else [0, 20, 40, 60, 80, 100]
        )
    )
    if milestones != expected_milestones:
        raise ValueError(
            f"{mode} selection milestones must be {expected_milestones}, got {milestones}"
        )
    milestone_basis = str(config["evaluation"].get("milestone_basis", ""))
    expected_basis = (
        "allocated_training_gpu_hours" if mode == "train" else "collection_fraction"
    )
    if milestone_basis != expected_basis:
        raise ValueError(f"{mode} milestone_basis must be {expected_basis}")
    if mode == "train":
        target_gpu_hours = float(
            config["comparison"].get("target_training_gpu_hours", 0.0)
        )
        if not math.isfinite(target_gpu_hours) or target_gpu_hours <= 0.0:
            raise ValueError("formal training requires target_training_gpu_hours")
        target_source = config["comparison"].get(
            "target_training_gpu_hours_source"
        )
        if not isinstance(target_source, Mapping) or not target_source.get("kind"):
            raise ValueError("formal target GPU-hours must record frozen provenance")
        tolerance = float(
            config["comparison"].get("gpu_time_relative_tolerance", 0.01)
        )
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= 0.01:
            raise ValueError("formal GPU-time tolerance must be in (0, 0.01]")
    else:
        milestone_collections(collections, milestones)
    if not bool(config["evaluation"].get("run_at_milestones", False)):
        raise ValueError("validation must run at every registered milestone")
    if int(config["evaluation"].get("latent_seed_base", -1)) < 0:
        raise ValueError("evaluation latent_seed_base must be frozen")
    if not bool(config["evaluation"].get("paired_metrics", False)):
        raise ValueError("controlled validation requires paired metrics")
    dns10s_domain = str(config["comparison"].get("data_domain", "voicebank")) == DNS10S_DATA_DOMAIN
    validation_name = (
        "libritts_dns10s_smoke_validation.json"
        if dns10s_domain and mode == "smoke"
        else "libritts_dns10s_validation.json"
    )
    expected_manifests = (
        {
            "train_manifest": "libritts_dns10s_train_exposures.json",
            "validation_manifest": validation_name,
            "official_test_manifest": "dns2020_official_test_all.json",
        }
        if dns10s_domain
        else {
            "train_manifest": "voicebank_train_16k.json",
            "validation_manifest": "voicebank_valid_16k.json",
            "official_test_manifest": "voicebank_official_test_16k.json",
        }
    )
    for key, expected_name in expected_manifests.items():
        if Path(config["data"][key]).name != expected_name:
            raise ValueError(f"data.{key} must use the shared {expected_name}")

    reward = config["training_reward"]
    if str(reward["name"]) != FLOWSE_GRPO_COMPOSITE:
        raise ValueError("fair A/B requires the frozen FlowSE-GRPO composite reward")
    if dict(reward["weights"]) != {
        "dnsmos": 0.6,
        "speaker": 1.0,
        "speechbertscore": 1.0,
    }:
        raise ValueError("composite reward weights must be 0.6/1.0/1.0")
    if str(reward["component_normalization"]) != "frozen_std":
        raise ValueError("composite reward must divide by frozen train-only stds")
    if int(reward["calibration"]["std_ddof"]) != 0:
        raise ValueError("reward calibration must use population std")
    if float(reward["calibration"]["dnsmos_divisor"]) != 4.0:
        raise ValueError("R_DNSMOS must be DNSMOS_OVRL/4")

    choices = config.get("implementation_choices")
    required_choices = {
        "cfg_strength",
        "ppo_clip_epsilon",
        "reference_kl_beta",
        "logprob_reduction",
        "kl_reduction",
        "advantage_std_correction",
        "nfe_window_sampling_scope",
        "batch_semantics",
        "optimizer",
        "lora_targets",
    }
    if not isinstance(choices, Mapping) or not required_choices.issubset(choices):
        raise ValueError("all unpublished implementation choices must be labeled")
    if any(
        str(choices[name]) != "public_reproduction_choice"
        for name in required_choices - {"batch_semantics"}
    ) or str(choices["batch_semantics"]) != "flow_grpo_full_buffer_reference":
        raise ValueError(
            "implementation choices must label full-buffer batching as the "
            "Flow-GRPO reference choice and all other choices as public reproduction"
        )
    return {
        "setting": CONTROLLED_SETTING,
        "mode": mode,
        "candidate_count_per_collection": prompts * repeats * group_size,
        "optimizer_updates_per_collection": updates,
        "training_microbatch_size": microbatch_size,
        "trajectory_consumption": "all_eligible_once_per_collection",
        "optimizer_steps": expected_steps,
        "policy_kind": "grpo_online",
        "ema_enabled": False,
        "rollout_world_size": world_size,
        "selection_milestones": milestones,
        "mechanism_audit_checkpoint_collections": mechanism_checkpoints,
        "reporting_checkpoint_collections": reporting_checkpoints,
        "shared_baseline_contract": shared_baseline_report,
    }


def reporting_checkpoint_collections(config: Mapping) -> list[int]:
    """Validate non-selection checkpoints retained for secondary budget reports."""

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return []
    specification = artifacts.get("reporting_checkpoints")
    if specification is None:
        return []
    if not isinstance(specification, Mapping):
        raise ValueError("artifacts.reporting_checkpoints must be a mapping")
    collections = [int(value) for value in specification.get("collections", [])]
    if not collections or collections != sorted(set(collections)):
        raise ValueError(
            "reporting checkpoint collections must be non-empty, unique, and sorted"
        )
    total_collections = int(config["run"]["collections"])
    if any(value < 1 or value > total_collections for value in collections):
        raise ValueError("reporting checkpoint collection is outside the run horizon")
    required_false = ("selection_eligible", "validation_queried")
    if any(bool(specification.get(name, True)) for name in required_false):
        raise ValueError(
            "reporting checkpoints cannot query validation or enter checkpoint selection"
        )
    if not bool(
        specification.get("artifact_io_excluded_from_training_budget", False)
    ):
        raise ValueError(
            "reporting checkpoint I/O must be excluded from the training budget"
        )
    if not str(specification.get("purpose", "")).strip():
        raise ValueError("reporting checkpoints require a frozen purpose")
    return collections


def _linear_scheduler(optimizer: torch.optim.Optimizer, config: Mapping):
    total = int(config["optimizer"]["schedule_total_steps"])
    warmup = int(config["optimizer"].get("warmup_steps", 0))

    def multiplier(index: int) -> float:
        if warmup > 0 and index < warmup:
            return float(index + 1) / warmup
        progress = (index - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_optimizer(transformer, config: Mapping):
    values = config["optimizer"]
    optimizer = torch.optim.AdamW(
        lora_parameters(transformer),
        lr=float(values["learning_rate"]),
        betas=tuple(float(item) for item in values["betas"]),
        eps=float(values["epsilon"]),
        weight_decay=float(values["weight_decay"]),
    )
    return optimizer, _linear_scheduler(optimizer, config)


def recompute_minibatch_objective(
    bundle,
    examples: Sequence[GRPOUpdateExample],
    *,
    conditioning: ConditioningProtocol,
    config: Mapping,
) -> GRPOObjectiveOutput:
    """Replay executed transitions under current and released-base policies."""

    if not examples:
        raise ValueError("GRPO update minibatch cannot be empty")
    expected_transitions = int(config["sampler"]["window_size"])
    current_rows = []
    old_rows = []
    kl_rows = []
    advantages = []
    for example in examples:
        if len(example.trajectory.transitions) != expected_transitions:
            raise ValueError("trajectory does not contain the frozen SDE window")
        condition = example.condition_mel.to(bundle.device)
        mask = example.frame_mask.to(bundle.device)
        current_values = []
        old_values = []
        kl_values = []
        for record in example.trajectory.transitions:
            state = record.state.to(bundle.device).float()
            next_state = record.next_state.to(bundle.device).float()
            time = torch.full(
                (state.shape[0],),
                float(record.time),
                device=bundle.device,
                dtype=torch.float32,
            )
            with lora_enabled(bundle.model.transformer, True):
                current_velocity = policy_velocity(
                    bundle,
                    state=state,
                    condition_mel=condition,
                    time=time,
                    frame_mask=mask,
                    conditioning=conditioning,
                    cfg_strength=float(config["sampler"]["cfg_strength"]),
                )
            current_stats = sde_transition_stats(
                state,
                current_velocity,
                time,
                float(record.dt),
                diffusion=float(config["sampler"]["diffusion"]),
            )
            current_log_prob = gaussian_transition_log_prob(
                next_state,
                current_stats.mean,
                current_stats.std,
                mask,
                reduction=str(config["objective"]["logprob_reduction"]),
            ).value
            with torch.no_grad(), lora_enabled(bundle.model.transformer, False):
                reference_velocity = policy_velocity(
                    bundle,
                    state=state,
                    condition_mel=condition,
                    time=time,
                    frame_mask=mask,
                    conditioning=conditioning,
                    cfg_strength=float(config["sampler"]["cfg_strength"]),
                )
                reference_mean = sde_transition_stats(
                    state,
                    reference_velocity,
                    time,
                    float(record.dt),
                    diffusion=float(config["sampler"]["diffusion"]),
                ).mean
            kl = reference_gaussian_kl(
                current_stats.mean,
                reference_mean,
                current_stats.std,
                mask,
                reduction=str(config["objective"]["kl_reduction"]),
            )
            current_values.append(current_log_prob.squeeze(0))
            old_values.append(record.old_log_prob.to(bundle.device).squeeze(0))
            kl_values.append(kl.squeeze(0))
        current_rows.append(torch.stack(current_values))
        old_rows.append(torch.stack(old_values))
        kl_rows.append(torch.stack(kl_values))
        advantages.append(float(example.advantage))
    return grpo_objective(
        torch.stack(current_rows),
        torch.stack(old_rows),
        torch.tensor(advantages, device=bundle.device, dtype=torch.float32),
        torch.stack(kl_rows),
        clip_epsilon=float(config["objective"]["clip_epsilon"]),
        beta=float(config["objective"]["kl_beta"]),
        log_ratio_clamp=float(config["objective"]["log_ratio_clamp"]),
    )


def _aggregate_minibatch_diagnostics(
    rows: Sequence[tuple[int, Mapping[str, float]]],
) -> dict[str, float]:
    """Combine microbatch diagnostics as if the macro-batch ran at once."""

    total = sum(int(count) for count, _ in rows)
    if total < 1 or any(int(count) < 1 for count, _ in rows):
        raise ValueError("diagnostic microbatch sizes must be positive")

    def weighted_mean(name: str) -> float:
        return float(
            sum(int(count) * float(values[name]) for count, values in rows) / total
        )

    result = {
        name: weighted_mean(name)
        for name in (
            "loss",
            "policy_loss",
            "reference_kl",
            "weighted_reference_kl",
            "approx_kl",
            "clip_fraction",
            "positive_clip_fraction",
            "negative_clip_fraction",
            "overflow_clamp_fraction",
        )
    }
    for prefix in ("ratio", "log_ratio"):
        mean = weighted_mean(f"{prefix}_mean")
        second_moment = sum(
            int(count)
            * (
                float(values[f"{prefix}_std"]) ** 2
                + float(values[f"{prefix}_mean"]) ** 2
            )
            for count, values in rows
        ) / total
        result[f"{prefix}_mean"] = float(mean)
        result[f"{prefix}_std"] = float(
            math.sqrt(max(0.0, second_moment - mean**2))
        )
    result["log_ratio_abs_max"] = max(
        float(values["log_ratio_abs_max"]) for _, values in rows
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("aggregated GRPO diagnostics are non-finite")
    return result


def optimizer_update_accumulated(
    bundle,
    microbatches: Sequence[Sequence[GRPOUpdateExample]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler,
    conditioning: ConditioningProtocol,
    config: Mapping,
) -> dict:
    """Consume one complete macro-batch with memory-bounded accumulation."""

    if not microbatches or any(not batch for batch in microbatches):
        raise ValueError("optimizer macro-batch cannot contain empty microbatches")
    total_examples = sum(len(batch) for batch in microbatches)
    optimizer.zero_grad(set_to_none=True)
    recompute_seconds = 0.0
    backward_seconds = 0.0
    diagnostic_rows = []
    trajectory_ids = []
    for examples in microbatches:
        _synchronize_cuda(bundle.device)
        recompute_started = time.perf_counter()
        output = recompute_minibatch_objective(
            bundle, examples, conditioning=conditioning, config=config
        )
        _synchronize_cuda(bundle.device)
        recompute_seconds += time.perf_counter() - recompute_started

        backward_started = time.perf_counter()
        weight = len(examples) / total_examples
        (output.loss * float(weight)).backward()
        _synchronize_cuda(bundle.device)
        backward_seconds += time.perf_counter() - backward_started
        diagnostic_rows.append((len(examples), dict(output.diagnostics)))
        trajectory_ids.extend(item.trajectory_id for item in examples)

    backward_started = time.perf_counter()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        lora_parameters(bundle.model.transformer),
        max_norm=float(config["optimizer"]["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    _synchronize_cuda(bundle.device)
    backward_seconds += time.perf_counter() - backward_started
    optimizer_started = time.perf_counter()
    optimizer.step()
    scheduler.step()
    _synchronize_cuda(bundle.device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    return {
        **_aggregate_minibatch_diagnostics(diagnostic_rows),
        "gradient_norm": float(gradient_norm.detach().item()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "trajectory_ids": trajectory_ids,
        "effective_batch_size": int(total_examples),
        "microbatch_count": len(microbatches),
        "microbatch_sizes": [len(batch) for batch in microbatches],
        "timing_seconds": {
            "recompute": recompute_seconds,
            "backward": backward_seconds,
            "optimizer": optimizer_seconds,
        },
}


def deterministic_full_buffer_update_batches(
    examples: Sequence[GRPOUpdateExample],
    *,
    seed: int,
    updates: int,
    microbatch_size: int,
) -> list[list[list[GRPOUpdateExample]]]:
    """Shuffle once, consume every eligible trajectory, and make four updates."""

    updates = int(updates)
    microbatch_size = int(microbatch_size)
    if updates < 1 or microbatch_size < 1 or len(examples) < updates:
        raise ValueError(
            "full-buffer batching requires positive updates/microbatch size and at "
            "least one eligible trajectory per optimizer update"
        )
    source_ids = [item.trajectory_id for item in examples]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("eligible trajectories contain duplicate IDs")
    indices = list(range(len(examples)))
    random.Random(int(seed)).shuffle(indices)
    base_size, extra = divmod(len(indices), updates)
    optimizer_batches = []
    cursor = 0
    for update_index in range(updates):
        macro_size = base_size + int(update_index < extra)
        macro_indices = indices[cursor : cursor + macro_size]
        cursor += macro_size
        optimizer_batches.append(
            [
                [examples[index] for index in macro_indices[start:stop]]
                for start in range(0, macro_size, microbatch_size)
                for stop in [min(start + microbatch_size, macro_size)]
            ]
        )
    used_ids = [
        item.trajectory_id
        for optimizer_batch in optimizer_batches
        for microbatch in optimizer_batch
        for item in microbatch
    ]
    if len(used_ids) != len(set(used_ids)) or set(used_ids) != set(source_ids):
        raise AssertionError("full-buffer batching did not consume each trajectory once")
    return optimizer_batches


def _write_scored_wave(path: Path, endpoint, *, sample_rate: int, subtype: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, endpoint.normalized_waveform, sample_rate, subtype=subtype)
    return int(np.asarray(endpoint.normalized_waveform).size)


def _synchronize_cuda(device=None) -> None:
    if not torch.cuda.is_available():
        return
    if device is not None and not isinstance(device, int):
        if torch.device(device).type != "cuda":
            return
    torch.cuda.synchronize(device=device)


def _reserve_cuda_allocator_memory(
    config: Mapping,
    *,
    device,
    role: str,
) -> dict:
    """Claim a peak budget in the owning process's CUDA caching allocator."""

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
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"CUDA reservation received non-CUDA device: {device}")
        device_index = (
            int(resolved.index)
            if resolved.index is not None
            else int(torch.cuda.current_device())
        )
    if int(torch.cuda.current_device()) != device_index:
        raise RuntimeError(
            "CUDA reservation must run in the owning process/device: "
            f"current={torch.cuda.current_device()} requested={device_index}"
        )

    gib = 1024**3
    target_key = (
        "coordinator_target_reserved_gib"
        if role == "coordinator"
        else "worker_target_reserved_gib"
    )
    target_bytes = int(float(reservation[target_key]) * gib)
    minimum_free_bytes = int(float(reservation["minimum_driver_free_gib"]) * gib)
    chunk_bytes = max(1, int(float(reservation["allocation_chunk_gib"]) * gib))
    require_target = bool(reservation["require_target"])
    total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
    if target_bytes + minimum_free_bytes > total_bytes:
        raise ValueError(
            "CUDA reservation target plus safety margin exceeds device memory: "
            f"target={target_bytes / gib:.2f}GiB "
            f"minimum_free={minimum_free_bytes / gib:.2f}GiB "
            f"total={total_bytes / gib:.2f}GiB"
        )

    _synchronize_cuda(device_index)
    allocated_before = int(torch.cuda.memory_allocated(device_index))
    reserved_before = int(torch.cuda.memory_reserved(device_index))
    free_before, driver_total = torch.cuda.mem_get_info(device_index)
    free_before = int(free_before)
    maximum_new_driver_bytes = max(0, free_before - minimum_free_bytes)
    required_new_driver_bytes = max(0, target_bytes - reserved_before)
    effective_target = target_bytes
    if required_new_driver_bytes > maximum_new_driver_bytes:
        if require_target:
            raise RuntimeError(
                "CUDA peak-memory reservation failed before training: "
                f"role={role} device=cuda:{device_index} "
                f"target={target_bytes / gib:.2f}GiB "
                f"reserved_now={reserved_before / gib:.2f}GiB "
                f"driver_free={free_before / gib:.2f}GiB. "
                "Choose a freer physical GPU and restart."
            )
        effective_target = reserved_before + maximum_new_driver_bytes

    buffers = []
    remaining = max(0, effective_target - allocated_before)
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

    reserved_after = int(torch.cuda.memory_reserved(device_index))
    allocated_after = int(torch.cuda.memory_allocated(device_index))
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
        "driver_total_bytes": int(driver_total),
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


def _audit_trajectory_ids(
    trajectory_ids: Sequence[str], *, count: int, seed: int
) -> set[str]:
    if count < 0 or count > len(trajectory_ids):
        raise ValueError("audit trajectory count is outside the collection geometry")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("audit trajectory IDs must be unique")
    return set(
        sorted(
            trajectory_ids,
            key=lambda value: stable_seed(seed, "audit_audio", value),
        )[:count]
    )


def rollout_collection(
    bundle,
    *,
    collection_index: int,
    utterances: list[str],
    config: Mapping,
    conditioning: ConditioningProtocol,
    dnsmos,
    composite_evaluators,
    reward_definition: Mapping,
    output_dir: Path,
    group_indices: Sequence[int] | None = None,
) -> CollectionResult:
    """Collect and score one frozen-old-policy logical collection."""

    # Imported lazily so ``--validate-only`` does not require the complete
    # FlowSE/torchaudio/reward runtime.
    from rl.common.flow_objective import waveform_to_mel

    collection = config["collection"]
    sampler_config = config["sampler"]
    group_size = int(collection["group_size"])
    repeats = int(collection["mini_batch_repeats"])
    sampler = FlowSEWindowedSDESampler(
        diffusion=float(sampler_config["diffusion"]),
        logprob_reduction=str(config["objective"]["logprob_reduction"]),
        offload_records_to_cpu=True,
    )
    reward_rows: list[list[float]] = []
    pending: list[
        tuple[str, str, torch.Tensor, torch.Tensor, TrajectoryRollout, dict]
    ] = []
    rollout_rows = []
    group_rows = []
    candidate_samples = 0
    phase_seconds = {"rollout": 0.0, "reward": 0.0, "audio_io": 0.0}
    audio_dir = output_dir / "rollout_audio" / f"collection_{collection_index:06d}"
    keep_training_audio = bool(config["artifacts"]["keep_training_audio"])
    audit_count = int(config["artifacts"]["audit_audio_candidates_per_collection"])
    total_groups = repeats * len(utterances)
    selected_group_indices = (
        list(range(total_groups))
        if group_indices is None
        else [int(value) for value in group_indices]
    )
    if selected_group_indices != sorted(set(selected_group_indices)) or any(
        value < 0 or value >= total_groups for value in selected_group_indices
    ):
        raise ValueError("group_indices must be unique sorted canonical group indices")
    all_trajectory_ids = [
        f"{collection_index}:{mini_batch_id}:{condition_slot}:{candidate}"
        for mini_batch_id in range(repeats)
        for condition_slot in range(len(utterances))
        for candidate in range(group_size)
    ]
    audit_ids = _audit_trajectory_ids(
        all_trajectory_ids,
        count=audit_count,
        seed=int(config["run"]["seed"]),
    )
    retained_audio_count = 0

    condition_cache = {}
    for group_index in selected_group_indices:
        mini_batch_id, condition_slot = divmod(group_index, len(utterances))
        utterance = utterances[condition_slot]
        _synchronize_cuda(bundle.device)
        rollout_started = time.perf_counter()
        group_id = f"{collection_index}:{mini_batch_id}:{condition_slot}"
        noisy_path = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
        clean_path = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
        if not noisy_path.is_file() or not clean_path.is_file():
            raise FileNotFoundError(f"missing VoiceBank pair for {utterance}")
        condition_mel = condition_cache.get(utterance)
        if condition_mel is None:
            condition_mel = waveform_to_mel(bundle, str(noisy_path)).detach().cpu()
            condition_cache[utterance] = condition_mel
        spec = sample_window_spec(
            stable_seed(
                int(sampler_config["window_seed_base"]),
                "window",
                collection_index,
                mini_batch_id,
                condition_slot,
            ),
            nfe_minimum=int(sampler_config["nfe_minimum"]),
            nfe_maximum=int(sampler_config["nfe_maximum"]),
            start_minimum=int(sampler_config["start_minimum"]),
            start_maximum=int(sampler_config["start_maximum"]),
            window_size=int(sampler_config["window_size"]),
        )
        latent_seeds = [
            stable_seed(
                int(sampler_config["latent_seed_base"]),
                "latent",
                collection_index,
                mini_batch_id,
                condition_slot,
                candidate,
            )
            for candidate in range(group_size)
        ]
        brownian_seeds = [
            stable_seed(
                int(sampler_config["brownian_seed_base"]),
                "brownian",
                collection_index,
                mini_batch_id,
                condition_slot,
                candidate,
            )
            for candidate in range(group_size)
        ]
        condition_device = condition_mel.to(bundle.device)
        initial = bundle._fixed_latents(
            tuple(condition_device.shape[1:]), latent_seeds, torch.float32
        )
        mask = torch.ones(initial.shape[:2], device=bundle.device, dtype=torch.bool)

        def velocity_fn(state, time, frame_mask):
            return policy_velocity(
                bundle,
                state=state,
                condition_mel=condition_device,
                time=time,
                frame_mask=frame_mask,
                conditioning=conditioning,
                cfg_strength=float(sampler_config["cfg_strength"]),
            )

        rollout = sampler.rollout_group(
            initial,
            frame_mask=mask,
            spec=spec,
            velocity_fn=velocity_fn,
            initial_latent_seeds=latent_seeds,
            brownian_seeds=brownian_seeds,
            retain_full_trajectory=False,
        )
        endpoints = bundle.decode_group(
            rollout.terminal.to(bundle.device),
            latent_seeds,
            target_dbfs=float(config["normalization"]["target_dbfs"]),
            peak_ceiling=float(config["normalization"]["peak_ceiling"]),
        )
        _synchronize_cuda(bundle.device)
        phase_seconds["rollout"] += time.perf_counter() - rollout_started
        group_rewards = []
        for candidate, (trajectory, endpoint) in enumerate(
            zip(rollout.trajectories, endpoints, strict=True)
        ):
            trajectory_id = f"{group_id}:{candidate}"
            audio_path = audio_dir / f"{group_id.replace(':', '_')}__g{candidate}.wav"
            io_started = time.perf_counter()
            candidate_samples += _write_scored_wave(
                audio_path,
                endpoint,
                sample_rate=bundle.output_sample_rate,
                subtype=str(config["normalization"]["output_subtype"]),
            )
            scored_wav_sha256 = sha256_file(audio_path)
            phase_seconds["audio_io"] += time.perf_counter() - io_started
            _synchronize_cuda(bundle.device)
            reward_started = time.perf_counter()
            metrics = dnsmos(audio_path)
            metrics.update(composite_evaluators.score(clean_path, audio_path))
            reward = compute_training_reward(metrics, reward_definition)
            if sha256_file(audio_path) != scored_wav_sha256:
                raise RuntimeError(f"scored rollout WAV changed: {audio_path}")
            _synchronize_cuda(bundle.device)
            phase_seconds["reward"] += time.perf_counter() - reward_started
            group_rewards.append(float(reward["reward"]))
            retain_audio = keep_training_audio or trajectory_id in audit_ids
            if retain_audio:
                retained_audio_count += 1
            else:
                io_started = time.perf_counter()
                audio_path.unlink()
                phase_seconds["audio_io"] += time.perf_counter() - io_started
            row = {
                "collection_index": collection_index,
                "group_id": group_id,
                "trajectory_id": trajectory_id,
                "utterance": utterance,
                "mini_batch_id": mini_batch_id,
                "condition_slot": condition_slot,
                "candidate_index": candidate,
                "nfe": spec.nfe,
                "window_start": spec.start_step,
                "window_size": spec.window_size,
                "initial_latent_seed": trajectory.initial_latent_seed,
                "brownian_seed": trajectory.brownian_seed,
                "reward": float(reward["reward"]),
                "reward_components": reward,
                "metrics": metrics,
                "terminal_mel_sha256": endpoint.terminal_mel_sha256,
                "scored_wav_sha256": scored_wav_sha256,
                "audit_audio_retained": retain_audio,
                "audio_path": str(audio_path) if retain_audio else None,
                "valid_dimensions": [
                    item.valid_dimensions for item in trajectory.transitions
                ],
                "old_log_prob_sum": [
                    float(item.old_log_prob_sum.item())
                    for item in trajectory.transitions
                ],
            }
            pending.append(
                (
                    trajectory_id,
                    group_id,
                    condition_mel,
                    torch.ones((1, condition_mel.shape[1]), dtype=torch.bool),
                    trajectory,
                    row,
                )
            )
            rollout_rows.append(row)
        reward_rows.append(group_rewards)
        group_rows.append(
            {
                "group_index": group_index,
                "group_id": group_id,
                "utterance": utterance,
                "nfe": spec.nfe,
                "window_start": spec.start_step,
                "candidate_count": group_size,
            }
        )

    rewards = torch.tensor(reward_rows, dtype=torch.float32)
    advantage_result = compute_group_advantages(
        rewards,
        correction=int(config["advantage"]["std_correction"]),
        epsilon=float(config["advantage"]["epsilon"]),
    )
    examples = []
    for flat_index, item in enumerate(pending):
        group_index, candidate = divmod(flat_index, group_size)
        item[-1]["advantage"] = float(
            advantage_result.advantages[group_index, candidate].item()
        )
        item[-1]["eligible"] = bool(
            advantage_result.eligible_candidates[group_index, candidate].item()
        )
        if item[-1]["eligible"]:
            examples.append(
                GRPOUpdateExample(
                    trajectory_id=item[0],
                    group_id=item[1],
                    condition_mel=item[2],
                    frame_mask=item[3],
                    trajectory=item[4],
                    advantage=float(item[-1]["advantage"]),
                )
            )
    for group_index, row in enumerate(group_rows):
        row["reward_mean"] = float(advantage_result.group_mean[group_index].item())
        row["reward_std"] = float(advantage_result.group_std[group_index].item())
        row["eligible"] = bool(advantage_result.valid_groups[group_index].item())
    return CollectionResult(
        collection_index=collection_index,
        rewards=rewards,
        examples=tuple(examples),
        rollout_rows=tuple(rollout_rows),
        group_rows=tuple(group_rows),
        candidate_audio_seconds=float(candidate_samples / bundle.output_sample_rate),
        phase_seconds=phase_seconds,
        active_gpu_seconds_by_phase={
            "rollout": float(phase_seconds["rollout"]),
            "reward": float(phase_seconds["reward"]),
        },
        retained_audio_count=retained_audio_count,
    )


def _partition_group_indices(total_groups: int, world_size: int) -> list[list[int]]:
    if total_groups < world_size or total_groups % world_size != 0:
        raise ValueError("logical GRPO groups must divide evenly across rollout GPUs")
    per_worker = total_groups // world_size
    return [
        list(range(rank * per_worker, (rank + 1) * per_worker))
        for rank in range(world_size)
    ]


def _merge_collection_shards(
    shards: Sequence[CollectionResult],
    *,
    collection_index: int,
    expected_groups: int,
    group_size: int,
) -> CollectionResult:
    group_rows = sorted(
        (row for shard in shards for row in shard.group_rows),
        key=lambda row: int(row["group_index"]),
    )
    if [int(row["group_index"]) for row in group_rows] != list(range(expected_groups)):
        raise ValueError("rollout shards do not exactly cover canonical GRPO groups")
    rollout_rows = sorted(
        (row for shard in shards for row in shard.rollout_rows),
        key=lambda row: (
            int(row["mini_batch_id"]),
            int(row["condition_slot"]),
            int(row["candidate_index"]),
        ),
    )
    expected_trajectories = expected_groups * group_size
    if len(rollout_rows) != expected_trajectories:
        raise ValueError("rollout shard trajectory count differs from frozen geometry")
    trajectory_ids = [str(row["trajectory_id"]) for row in rollout_rows]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("rollout shards produced duplicate trajectory IDs")
    examples = sorted(
        (example for shard in shards for example in shard.examples),
        key=lambda item: item.trajectory_id,
    )
    rewards = torch.tensor(
        [
            [float(row["reward"]) for row in rollout_rows[index : index + group_size]]
            for index in range(0, len(rollout_rows), group_size)
        ],
        dtype=torch.float32,
    )
    phases = {
        phase: max(float(shard.phase_seconds.get(phase, 0.0)) for shard in shards)
        for phase in ("rollout", "reward", "audio_io")
    }
    return CollectionResult(
        collection_index=collection_index,
        rewards=rewards,
        examples=tuple(examples),
        rollout_rows=tuple(rollout_rows),
        group_rows=tuple(group_rows),
        candidate_audio_seconds=float(
            sum(shard.candidate_audio_seconds for shard in shards)
        ),
        phase_seconds=phases,
        active_gpu_seconds_by_phase={
            phase: float(
                sum(
                    shard.active_gpu_seconds_by_phase.get(phase, 0.0)
                    for shard in shards
                )
            )
            for phase in ("rollout", "reward")
        },
        retained_audio_count=sum(shard.retained_audio_count for shard in shards),
    )


def _worker_rollout_shard_path(
    output_dir: str | Path, *, task_id: str, worker_rank: int
) -> Path:
    if not task_id or any(not (character.isalnum() or character == "_") for character in task_id):
        raise ValueError("worker rollout task ID contains unsafe characters")
    if int(worker_rank) < 1:
        raise ValueError("worker rollout shard rank must be positive")
    return (
        Path(output_dir)
        / ".rollout_worker_shards"
        / f"{task_id}_rank_{int(worker_rank):03d}.pt"
    )


def _cleanup_worker_rollout_shard(path: str | Path) -> None:
    destination = Path(path)
    if destination.is_file():
        destination.unlink()
    if destination.parent.is_dir():
        for temporary in destination.parent.glob(f".{destination.name}.tmp-*"):
            if temporary.is_file():
                temporary.unlink()


def _cleanup_worker_rollout_directory(output_dir: str | Path) -> None:
    directory = Path(output_dir) / ".rollout_worker_shards"
    if not directory.is_dir():
        return
    for candidate in directory.iterdir():
        name = candidate.name
        is_transport_artifact = (
            (name.startswith("collection_") or name.startswith(".collection_"))
            and "_rank_" in name
            and (name.endswith(".pt") or ".pt.tmp-" in name)
        )
        if candidate.is_file() and is_transport_artifact:
            candidate.unlink()
    try:
        directory.rmdir()
    except OSError:
        pass


def _write_worker_rollout_shard(
    path: str | Path,
    *,
    task_id: str,
    worker_rank: int,
    collection_index: int,
    state_sha256: str,
    shard: CollectionResult,
) -> dict:
    destination = Path(path)
    if int(shard.collection_index) != int(collection_index):
        raise ValueError("worker rollout shard collection index differs")
    started = time.perf_counter()
    try:
        atomic_torch_save(
            destination,
            {
                "schema_version": 1,
                "kind": "grpo_rollout_worker_shard",
                "task_id": str(task_id),
                "worker_rank": int(worker_rank),
                "collection_index": int(collection_index),
                "state_sha256": str(state_sha256),
                "shard": shard,
            },
        )
        return {
            "shard_path": str(destination),
            "shard_sha256": sha256_file(destination),
            "shard_bytes": int(destination.stat().st_size),
            "shard_io_seconds": float(time.perf_counter() - started),
        }
    except BaseException:
        _cleanup_worker_rollout_shard(destination)
        raise


def _load_worker_rollout_shard(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_task_id: str,
    expected_worker_rank: int,
    expected_collection_index: int,
    expected_state_sha256: str,
) -> CollectionResult:
    source = Path(path)
    if sha256_file(source) != str(expected_file_sha256):
        raise ValueError("worker rollout shard file SHA256 differs")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("worker rollout shard payload must be a mapping")
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("kind") == "grpo_rollout_worker_shard",
        "task": payload.get("task_id") == str(expected_task_id),
        "rank": int(payload.get("worker_rank", -1)) == int(expected_worker_rank),
        "collection": int(payload.get("collection_index", -1))
        == int(expected_collection_index),
        "old_policy": payload.get("state_sha256") == str(expected_state_sha256),
        "shard_type": isinstance(payload.get("shard"), CollectionResult),
    }
    if not all(checks.values()):
        raise ValueError(f"worker rollout shard compatibility failed: {checks}")
    shard = payload["shard"]
    if int(shard.collection_index) != int(expected_collection_index):
        raise ValueError("loaded worker rollout shard collection index differs")
    return shard


def _grpo_rollout_worker_main(
    *,
    worker_rank: int,
    device_id: int,
    config: dict,
    output_dir: str,
    shared_online_state: Mapping[str, torch.Tensor],
    task_queue,
    result_queue,
) -> None:
    try:
        from rl.common.flowse_interface import (
            load_flowse_bundle,
        )
        from rl.rewards.composite import (
            load_flowse_grpo_composite_evaluators,
        )
        from rl.rewards.metrics import DNSMOSScorer

        torch.cuda.set_device(device_id)
        cuda_memory_reservation = _reserve_cuda_allocator_memory(
            config,
            device=device_id,
            role=f"rollout_worker_{worker_rank}",
        )
        bundle = load_flowse_bundle(
            config["flowse_config"],
            deterministic=bool(config["run"]["deterministic"]),
            compute_artifact_hashes=False,
        )
        torch.manual_seed(int(config["lora"]["initialization_seed"]))
        torch.cuda.manual_seed_all(int(config["lora"]["initialization_seed"]))
        injection = inject_lora(
            bundle.model.transformer,
            target_patterns=config["lora"]["target_patterns"],
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
            expected_modules=int(config["lora"]["expected_modules"]),
        )
        conditioning = ConditioningProtocol.from_config(config["conditioning"])
        dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
        composite_evaluators, evaluator_fingerprint = (
            load_flowse_grpo_composite_evaluators(config)
        )
        reward_definition = resolve_training_reward(config)
        result_queue.put(
            {
                "status": "ready",
                "worker_rank": worker_rank,
                "device_id": device_id,
                "module_names": list(injection.module_names),
                "evaluator_fingerprint": evaluator_fingerprint,
                "cuda_memory_reservation": cuda_memory_reservation,
            }
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                load_lora(bundle.model.transformer, shared_online_state)
                state_hash = lora_state_fingerprint(
                    snapshot_lora(bundle.model.transformer, device="cpu")
                )["state_sha256"]
                shard = rollout_collection(
                    bundle,
                    collection_index=int(task["collection_index"]),
                    utterances=list(task["utterances"]),
                    config=config,
                    conditioning=conditioning,
                    dnsmos=dnsmos,
                    composite_evaluators=composite_evaluators,
                    reward_definition=reward_definition,
                    output_dir=Path(output_dir),
                    group_indices=list(task["group_indices"]),
                )
                shard_path = _worker_rollout_shard_path(
                    output_dir,
                    task_id=str(task["task_id"]),
                    worker_rank=worker_rank,
                )
                _cleanup_worker_rollout_shard(shard_path)
                shard_metadata = _write_worker_rollout_shard(
                    shard_path,
                    task_id=str(task["task_id"]),
                    worker_rank=worker_rank,
                    collection_index=int(task["collection_index"]),
                    state_sha256=state_hash,
                    shard=shard,
                )
                result_queue.put(
                    {
                        "status": "ok",
                        "task_id": str(task["task_id"]),
                        "worker_rank": worker_rank,
                        "state_sha256": state_hash,
                        **shard_metadata,
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "status": "error",
                        "task_id": str(task.get("task_id", "unknown")),
                        "worker_rank": worker_rank,
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "task_id": "initialization",
                "worker_rank": worker_rank,
                "traceback": traceback.format_exc(),
            }
        )


class _GRPORolloutPool:
    """Persistent coordinator-plus-workers pool preserving complete G=10 groups."""

    def __init__(
        self,
        *,
        config: dict,
        output_dir: Path,
        initial_state: Mapping[str, torch.Tensor],
        expected_module_names: Sequence[str],
        expected_evaluator_fingerprint: Mapping,
    ) -> None:
        resources = config["resources"]
        self.world_size = int(resources["rollout_world_size"])
        self.device_ids = [int(value) for value in resources["device_ids"]]
        self.timeout = float(resources.get("worker_task_timeout_seconds", 7200.0))
        self.startup_timeout = float(
            resources.get("worker_startup_timeout_seconds", 3600.0)
        )
        self._expected_module_names = list(expected_module_names)
        self._expected_evaluator_fingerprint = dict(expected_evaluator_fingerprint)
        self._output_dir = Path(output_dir)
        if self.world_size <= 1:
            raise ValueError("rollout pool requires more than one GPU")
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.world_size:
            raise RuntimeError("configured multi-GPU rollout topology is unavailable")
        if torch.cuda.current_device() != self.device_ids[0]:
            raise RuntimeError("GRPO coordinator must use resources.device_ids[0]")
        self._context = torch.multiprocessing.get_context("spawn")
        _cleanup_worker_rollout_directory(self._output_dir)
        self._result_queue = self._context.Queue()
        self._task_queues = {}
        self._processes = {}
        self._closed = False
        self.worker_memory_reservations = {}
        self._shared_state = {}
        for name, value in initial_state.items():
            shared = value.detach().cpu().clone().contiguous()
            shared.share_memory_()
            self._shared_state[name] = shared
        try:
            for rank in range(1, self.world_size):
                task_queue = self._context.Queue(maxsize=1)
                process = self._context.Process(
                    target=_grpo_rollout_worker_main,
                    kwargs={
                        "worker_rank": rank,
                        "device_id": self.device_ids[rank],
                        "config": config,
                        "output_dir": str(output_dir),
                        "shared_online_state": self._shared_state,
                        "task_queue": task_queue,
                        "result_queue": self._result_queue,
                    },
                    daemon=True,
                )
                process.start()
                self._task_queues[rank] = task_queue
                self._processes[rank] = process
            self._wait_ready()
        except BaseException:
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
            raise TimeoutError(f"GRPO rollout worker timeout: {states}") from exc

    def _wait_ready(self) -> None:
        ready = set()
        deadline = time.monotonic() + self.startup_timeout
        while len(ready) < self.world_size - 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("GRPO rollout worker initialization timed out")
            result = self._get_result(remaining)
            if result["status"] == "error":
                raise RuntimeError(result["traceback"])
            if result["status"] != "ready":
                raise RuntimeError(f"unexpected GRPO worker message: {result}")
            if result["module_names"] != self._expected_module_names:
                raise RuntimeError("GRPO worker LoRA module list differs")
            if result["evaluator_fingerprint"] != self._expected_evaluator_fingerprint:
                raise RuntimeError("GRPO worker evaluator fingerprint differs")
            worker_rank = int(result["worker_rank"])
            self.worker_memory_reservations[worker_rank] = dict(
                result["cuda_memory_reservation"]
            )
            ready.add(worker_rank)

    def _broadcast_state(self, state: Mapping[str, torch.Tensor]) -> str:
        if set(state) != set(self._shared_state):
            raise ValueError("online LoRA keys changed after workers started")
        for name, target in self._shared_state.items():
            source = state[name].detach().cpu()
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"online LoRA metadata changed for {name}")
            target.copy_(source)
        return lora_state_fingerprint(state)["state_sha256"]

    def rollout(
        self,
        *,
        bundle,
        online_state: Mapping[str, torch.Tensor],
        collection_index: int,
        utterances: list[str],
        config: dict,
        conditioning: ConditioningProtocol,
        dnsmos,
        composite_evaluators,
        reward_definition: Mapping,
        output_dir: Path,
    ) -> tuple[CollectionResult, dict[int, str]]:
        total_groups = int(config["collection"]["mini_batch_repeats"]) * len(utterances)
        partitions = _partition_group_indices(total_groups, self.world_size)
        expected_hash = self._broadcast_state(online_state)
        task_id = f"collection_{collection_index:06d}"
        for rank in range(1, self.world_size):
            self._task_queues[rank].put(
                {
                    "task_id": task_id,
                    "collection_index": collection_index,
                    "utterances": utterances,
                    "group_indices": partitions[rank],
                }
            )
        main_shard = rollout_collection(
            bundle,
            collection_index=collection_index,
            utterances=utterances,
            config=config,
            conditioning=conditioning,
            dnsmos=dnsmos,
            composite_evaluators=composite_evaluators,
            reward_definition=reward_definition,
            output_dir=output_dir,
            group_indices=partitions[0],
        )
        shards = [main_shard]
        worker_hashes = {0: expected_hash}
        worker_shard_io_seconds = []
        completed = set()
        while len(completed) < self.world_size - 1:
            result = self._get_result(self.timeout)
            if result["status"] == "error":
                raise RuntimeError(result["traceback"])
            if result["status"] != "ok" or result["task_id"] != task_id:
                raise RuntimeError(f"unexpected GRPO rollout result: {result}")
            rank = int(result["worker_rank"])
            if rank not in self._processes:
                raise RuntimeError(f"unexpected GRPO rollout worker rank {rank}")
            if rank in completed:
                raise RuntimeError(f"duplicate GRPO worker result from rank {rank}")
            if result["state_sha256"] != expected_hash:
                raise RuntimeError(f"GRPO worker {rank} used a different old policy")
            expected_shard_path = _worker_rollout_shard_path(
                output_dir, task_id=task_id, worker_rank=rank
            )
            observed_shard_path = Path(str(result.get("shard_path", "")))
            if observed_shard_path.resolve() != expected_shard_path.resolve():
                raise RuntimeError(
                    f"GRPO worker {rank} returned an unexpected shard path"
                )
            shard_bytes = int(result.get("shard_bytes", -1))
            shard_io_seconds = float(result.get("shard_io_seconds", float("nan")))
            if not math.isfinite(shard_io_seconds) or shard_io_seconds < 0.0:
                raise RuntimeError(
                    f"GRPO worker {rank} returned invalid shard I/O timing"
                )
            try:
                if shard_bytes <= 0 or shard_bytes != int(
                    observed_shard_path.stat().st_size
                ):
                    raise RuntimeError(
                        f"GRPO worker {rank} shard byte count differs"
                    )
                worker_shard = _load_worker_rollout_shard(
                    observed_shard_path,
                    expected_file_sha256=str(result.get("shard_sha256", "")),
                    expected_task_id=task_id,
                    expected_worker_rank=rank,
                    expected_collection_index=collection_index,
                    expected_state_sha256=expected_hash,
                )
            finally:
                _cleanup_worker_rollout_shard(expected_shard_path)
            completed.add(rank)
            worker_hashes[rank] = str(result["state_sha256"])
            worker_shard_io_seconds.append(shard_io_seconds)
            shards.append(worker_shard)
        merged = _merge_collection_shards(
            shards,
            collection_index=collection_index,
            expected_groups=total_groups,
            group_size=int(config["collection"]["group_size"]),
        )
        merged = replace(
            merged,
            phase_seconds={
                **merged.phase_seconds,
                "rollout_transport_io": max(worker_shard_io_seconds, default=0.0),
            },
        )
        return merged, worker_hashes

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
        _cleanup_worker_rollout_directory(self._output_dir)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass


def save_checkpoint(
    path: Path,
    *,
    bundle,
    optimizer,
    scheduler,
    collection_index: int,
    optimizer_step: int,
    config_hash: str,
    collection_commit_id: str,
    cumulative_accounting: Mapping,
    completed_milestones: Sequence[int],
) -> dict:
    online_lora_state = snapshot_lora(bundle.model.transformer, device="cpu")
    payload = {
        "schema_version": 1,
        "method": "flowse_grpo",
        "policy_kind": "grpo_online",
        "ema_enabled": False,
        "collection_boundary": True,
        "collection_index": int(collection_index),
        "optimizer_step": int(optimizer_step),
        "collection_commit_id": str(collection_commit_id),
        "config_sha256": config_hash,
        "released_checkpoint_sha256": bundle.checkpoint_sha256,
        "online_lora_state": online_lora_state,
        "online_lora_fingerprint": lora_state_fingerprint(online_lora_state),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "cumulative_accounting": json.loads(json.dumps(cumulative_accounting)),
        "completed_milestones": [int(value) for value in completed_milestones],
    }
    atomic_torch_save(path, payload)
    return payload


def load_checkpoint(
    path: Path,
    *,
    bundle,
    optimizer,
    scheduler,
    config_hash: str,
) -> tuple[int, int, dict, list[int], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "schema": payload.get("schema_version") == 1,
        "method": payload.get("method") == "flowse_grpo",
        "policy_kind": payload.get("policy_kind") == "grpo_online",
        "ema_disabled": payload.get("ema_enabled") is False,
        "collection_boundary": payload.get("collection_boundary") is True,
        "config": payload.get("config_sha256") == config_hash,
        "released_base": payload.get("released_checkpoint_sha256")
        == bundle.checkpoint_sha256,
        "online_state": isinstance(payload.get("online_lora_state"), Mapping),
    }
    if not all(checks.values()):
        raise ValueError(f"GRPO checkpoint compatibility failed: {checks}")
    observed_fingerprint = lora_state_fingerprint(payload["online_lora_state"])
    if observed_fingerprint != payload.get("online_lora_fingerprint"):
        raise ValueError("GRPO checkpoint online LoRA fingerprint is invalid")
    load_lora(bundle.model.transformer, payload["online_lora_state"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    accounting = payload.get("cumulative_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("checkpoint lacks cumulative GPU-time accounting")
    milestones = [int(value) for value in payload.get("completed_milestones", [])]
    return (
        int(payload["collection_index"]),
        int(payload["optimizer_step"]),
        dict(accounting),
        milestones,
        str(payload.get("collection_commit_id", "")),
    )


def _empty_accounting(config: Mapping) -> dict:
    world_size = int(config["resources"]["rollout_world_size"])
    return {
        "schema_version": 1,
        "rollout_world_size": world_size,
        "device_ids": [int(value) for value in config["resources"]["device_ids"]],
        "phase_seconds": {
            "rollout": 0.0,
            "reward": 0.0,
            "audio_io": 0.0,
            "rollout_transport_io": 0.0,
            "recompute": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
            "checkpoint_and_logging": 0.0,
        },
        "active_gpu_seconds_by_phase": {
            "rollout": 0.0,
            "reward": 0.0,
            "recompute": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
        },
        "allocated_training_wall_seconds": 0.0,
        "allocated_training_gpu_seconds": 0.0,
        "validation_wall_seconds": 0.0,
        "validation_gpu_seconds": 0.0,
        "completed_collections": 0,
        "completed_optimizer_steps": 0,
        "accounting_recovery_events": [],
        "unmeasured_checkpoint_commit_count": 0,
    }


def _record_training_accounting(
    accounting: dict,
    *,
    collection_wall_seconds: float,
    phase_seconds: Mapping[str, float],
    active_gpu_seconds_by_phase: Mapping[str, float] | None,
    optimizer_step: int,
) -> None:
    world_size = int(accounting["rollout_world_size"])
    for phase, seconds in phase_seconds.items():
        accounting["phase_seconds"][phase] = float(
            accounting["phase_seconds"].get(phase, 0.0) + float(seconds)
        )
    active_gpu_seconds_by_phase = active_gpu_seconds_by_phase or {}
    for phase in ("rollout", "reward", "recompute", "backward", "optimizer"):
        seconds = float(
            active_gpu_seconds_by_phase.get(phase, phase_seconds.get(phase, 0.0))
        )
        accounting["active_gpu_seconds_by_phase"][phase] = float(
            accounting["active_gpu_seconds_by_phase"].get(phase, 0.0) + seconds
        )
    accounting["allocated_training_wall_seconds"] = float(
        accounting["allocated_training_wall_seconds"] + collection_wall_seconds
    )
    accounting["allocated_training_gpu_seconds"] = float(
        accounting["allocated_training_gpu_seconds"]
        + collection_wall_seconds * world_size
    )
    accounting["completed_collections"] = int(accounting["completed_collections"] + 1)
    accounting["completed_optimizer_steps"] = int(optimizer_step)


def _accounting_report(accounting: Mapping) -> dict:
    result = json.loads(json.dumps(accounting))
    result["allocated_training_gpu_hours"] = float(
        result["allocated_training_gpu_seconds"] / 3600.0
    )
    result["validation_gpu_hours"] = float(result["validation_gpu_seconds"] / 3600.0)
    result["active_gpu_hours_by_phase"] = {
        phase: float(seconds / 3600.0)
        for phase, seconds in result["active_gpu_seconds_by_phase"].items()
    }
    return result


def _gpu_time_budget_match(config: Mapping, accounting: Mapping) -> dict | None:
    if str(config["run"]["mode"]) != "train":
        return None
    target = float(config["comparison"]["target_training_gpu_hours"])
    observed = float(accounting["allocated_training_gpu_seconds"]) / 3600.0
    relative_error = (observed - target) / target
    tolerance = float(
        config["comparison"].get("gpu_time_relative_tolerance", 0.01)
    )
    exact = int(accounting.get("unmeasured_checkpoint_commit_count", 0)) == 0
    return {
        "target_training_gpu_hours": target,
        "observed_training_gpu_hours": observed,
        "signed_relative_error": relative_error,
        "absolute_relative_error": abs(relative_error),
        "tolerance": tolerance,
        "accounting_exact": exact,
        "passed": exact and abs(relative_error) <= tolerance,
        "name": f"gpu_time_matched_within_{abs(relative_error) * 100:.4f}pct",
    }


def _aggregate_collection_statistics(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observed = [int(row["collection_index"]) for row in rows]
    if observed != list(range(1, len(rows) + 1)):
        raise ValueError("collection statistics log is not contiguous")
    scalar_fields = (
        "prompt_instances",
        "scored_trajectories",
        "eligible_trajectories",
        "used_trajectories",
        "unused_eligible_trajectories",
        "used_stochastic_transitions",
        "reward_calls",
        "candidate_audio_seconds",
        "retained_audit_audio",
        "zero_std_groups",
    )
    compute_fields = (
        "rollout_logical_velocity_examples",
        "rollout_old_policy_model_forwards",
        "current_recompute_model_forwards",
        "reference_recompute_model_forwards",
        "old_replay_audit_transition_examples",
        "cfg_unconditional_forwards",
        "backward_transition_examples",
        "backward_calls",
        "effective_mel_frame_tokens",
        "effective_mel_dimensions",
        "optimizer_updates",
    )
    return {
        "completed_collections": len(rows),
        **{
            field: float(sum(float(row.get(field, 0.0)) for row in rows))
            if field == "candidate_audio_seconds"
            else int(sum(int(row.get(field, 0)) for row in rows))
            for field in scalar_fields
        },
        "unique_utterance_coverage": len(
            {str(value) for row in rows for value in row.get("prompts", [])}
        ),
        "compute_accounting": {
            field: int(
                sum(int(row.get("compute_accounting", {}).get(field, 0)) for row in rows)
            )
            for field in compute_fields
        },
    }


def _numeric_distribution(values: Sequence[float]) -> dict:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    if not np.all(np.isfinite(array)):
        raise ValueError("mechanism audit received a non-finite value")
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


def _lora_delta_statistics(
    initial_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
) -> dict:
    if set(initial_state) != set(current_state):
        raise ValueError("mechanism audit LoRA state keys changed")
    initial_square = 0.0
    current_square = 0.0
    delta_square = 0.0
    maximum_delta = 0.0
    parameter_count = 0
    changed_tensors = 0
    for name in sorted(initial_state):
        initial = initial_state[name].detach().cpu().float()
        current = current_state[name].detach().cpu().float()
        if initial.shape != current.shape:
            raise ValueError(f"mechanism audit LoRA tensor shape changed: {name}")
        delta = current - initial
        initial_square += float(initial.double().square().sum().item())
        current_square += float(current.double().square().sum().item())
        delta_square += float(delta.double().square().sum().item())
        maximum_delta = max(
            maximum_delta,
            float(delta.abs().max().item()) if delta.numel() else 0.0,
        )
        parameter_count += int(delta.numel())
        changed_tensors += int(not torch.equal(initial, current))
    initial_l2 = math.sqrt(initial_square)
    current_l2 = math.sqrt(current_square)
    delta_l2 = math.sqrt(delta_square)
    return {
        "tensor_count": len(initial_state),
        "changed_tensor_count": changed_tensors,
        "parameter_count": parameter_count,
        "initial_l2": initial_l2,
        "current_l2": current_l2,
        "delta_l2": delta_l2,
        "delta_rms": math.sqrt(delta_square / max(parameter_count, 1)),
        "delta_max_abs": maximum_delta,
        "relative_delta_l2": delta_l2 / max(initial_l2, 1.0e-12),
    }


def _mechanism_collection_summary(
    *,
    collection_index: int,
    optimizer_step: int,
    collection_commit_id: str,
    checkpoint_path: Path,
    checkpoint_payload: Mapping,
    initial_lora_state: Mapping[str, torch.Tensor],
    config: Mapping,
    collection_row: Mapping | None,
    rollout_rows: Sequence[Mapping],
) -> dict:
    current_state = checkpoint_payload["online_lora_state"]
    state_delta = _lora_delta_statistics(initial_lora_state, current_state)
    expected_step = int(collection_index) * int(
        config["collection"]["optimizer_updates"]
    )
    invariants = {
        "checkpoint_collection_matches": int(checkpoint_payload["collection_index"])
        == int(collection_index),
        "checkpoint_optimizer_step_matches": int(
            checkpoint_payload["optimizer_step"]
        )
        == int(optimizer_step)
        == expected_step,
        "checkpoint_commit_matches": str(
            checkpoint_payload.get("collection_commit_id", "")
        )
        == str(collection_commit_id),
        "checkpoint_online_lora_fingerprint_valid": lora_state_fingerprint(
            current_state
        )
        == checkpoint_payload.get("online_lora_fingerprint"),
        "selection_eligible_is_false": not bool(
            config["mechanism_audit"]["selection_eligible"]
        ),
        "validation_queried_is_false": not bool(
            config["mechanism_audit"]["run_validation"]
        ),
        "analysis_artifact_io_exclusion_is_explicit": bool(
            config["mechanism_audit"][
                "analysis_artifact_io_excluded_from_training_budget"
            ]
        ),
    }
    summary = {
        "schema_version": 1,
        "status": "MECHANISM-AUDIT-POINT",
        "checkpoint_role": "correctness_analysis_only",
        "selection_eligible": False,
        "validation_queried": False,
        "official_test_eligible": False,
        "analysis_artifact_io_excluded_from_training_budget": True,
        "collection_index": int(collection_index),
        "optimizer_step": int(optimizer_step),
        "collection_commit_id": str(collection_commit_id),
        "checkpoint_path": str(checkpoint_path),
        "online_lora_fingerprint": checkpoint_payload["online_lora_fingerprint"],
        "lora_delta_from_shared_initialization": state_delta,
    }
    if collection_index == 0:
        invariants["initial_lora_is_unchanged"] = (
            state_delta["changed_tensor_count"] == 0
            and state_delta["delta_l2"] == 0.0
        )
        summary["collection_mechanism"] = None
        summary["invariants"] = {
            **invariants,
            "all_passed": all(invariants.values()),
        }
        return summary

    if collection_row is None or not rollout_rows:
        raise ValueError("nonzero mechanism audit point lacks collection artifacts")
    if int(collection_row["collection_index"]) != int(collection_index):
        raise ValueError("mechanism audit collection artifact index differs")
    group_size = int(config["collection"]["group_size"])
    expected_candidates = int(
        config["collection"]["prompts_per_mini_batch"]
        * config["collection"]["mini_batch_repeats"]
        * group_size
    )
    expected_groups = expected_candidates // group_size
    grouped: dict[str, list[Mapping]] = {}
    for row in rollout_rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    trajectory_ids = [str(row["trajectory_id"]) for row in rollout_rows]
    initial_latent_seeds = [int(row["initial_latent_seed"]) for row in rollout_rows]
    brownian_seeds = [int(row["brownian_seed"]) for row in rollout_rows]
    terminal_mel_hashes = [str(row["terminal_mel_sha256"]) for row in rollout_rows]
    scored_wav_hashes = [str(row["scored_wav_sha256"]) for row in rollout_rows]
    group_reward_stds = []
    group_advantage_mean_abs = []
    group_advantage_std_error = []
    valid_groups = 0
    malformed_groups = 0
    epsilon = float(config["advantage"]["epsilon"])
    for rows in grouped.values():
        if len(rows) != group_size:
            malformed_groups += 1
            continue
        rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
        reward_std = float(rewards.std(ddof=0))
        group_reward_stds.append(reward_std)
        eligible = [bool(row["eligible"]) for row in rows]
        if reward_std > epsilon:
            valid_groups += 1
            advantages = np.asarray(
                [float(row["advantage"]) for row in rows], dtype=np.float64
            )
            group_advantage_mean_abs.append(float(abs(advantages.mean())))
            group_advantage_std_error.append(
                float(abs(advantages.std(ddof=0) - 1.0))
            )
            if not all(eligible):
                malformed_groups += 1
        elif any(eligible):
            malformed_groups += 1

    from .mechanism_audit import audit_logged_group_advantages

    advantage_replay_audit = audit_logged_group_advantages(
        rollout_rows,
        group_size=group_size,
        expected_groups=expected_groups,
        correction=int(config["advantage"]["std_correction"]),
        epsilon=epsilon,
    )

    updates = list(collection_row["updates"])
    first_update = updates[0] if updates else {}
    later_updates = updates[1:]
    eligible_ids = [
        str(row["trajectory_id"]) for row in rollout_rows if bool(row["eligible"])
    ]
    used_ids = [str(value) for value in collection_row["used_trajectory_ids"]]
    update_trajectory_ids = [
        str(value) for update in updates for value in update["trajectory_ids"]
    ]
    worker_hashes = {
        str(key): str(value)
        for key, value in collection_row["rollout_worker_state_sha256"].items()
    }
    metric_names = (
        "loss",
        "policy_loss",
        "reference_kl",
        "weighted_reference_kl",
        "ratio_mean",
        "ratio_std",
        "log_ratio_abs_max",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "learning_rate",
    )
    update_finite = all(
        math.isfinite(float(update[name]))
        for update in updates
        for name in metric_names
    )
    invariants.update(
        {
            "collection_commit_matches_artifact": str(
                collection_row["collection_commit_id"]
            )
            == str(collection_commit_id),
            "complete_candidate_geometry": len(rollout_rows) == expected_candidates
            and len(grouped) == expected_groups
            and malformed_groups == 0
            and all(
                sorted(int(row["candidate_index"]) for row in rows)
                == list(range(group_size))
                for rows in grouped.values()
            ),
            "rollout_collection_and_commit_match": all(
                int(row["collection_index"]) == int(collection_index)
                and str(row["collection_commit_id"]) == str(collection_commit_id)
                for row in rollout_rows
            ),
            "trajectory_ids_unique": len(trajectory_ids) == len(set(trajectory_ids)),
            "independent_initial_latent_seeds": len(initial_latent_seeds)
            == len(set(initial_latent_seeds)),
            "independent_brownian_seeds": len(brownian_seeds)
            == len(set(brownian_seeds)),
            "terminal_mel_hashes_unique": len(terminal_mel_hashes)
            == len(set(terminal_mel_hashes)),
            "scored_wav_hashes_unique": len(scored_wav_hashes)
            == len(set(scored_wav_hashes)),
            "sampler_values_in_frozen_range": all(
                int(config["sampler"]["nfe_minimum"])
                <= int(row["nfe"])
                <= int(config["sampler"]["nfe_maximum"])
                and int(config["sampler"]["start_minimum"])
                <= int(row["window_start"])
                <= int(config["sampler"]["start_maximum"])
                for row in rollout_rows
            ),
            "complete_optimizer_geometry": len(updates)
            == int(config["collection"]["optimizer_updates"])
            and sum(int(update["effective_batch_size"]) for update in updates)
            == len(eligible_ids)
            and all(
                int(update["microbatch_count"])
                == len(update["microbatch_sizes"])
                and sum(int(value) for value in update["microbatch_sizes"])
                == int(update["effective_batch_size"])
                and all(
                    1 <= int(value) <= int(config["collection"]["microbatch_size"])
                    for value in update["microbatch_sizes"]
                )
                for update in updates
            ),
            "used_trajectories_unique": len(used_ids) == len(set(used_ids)),
            "used_trajectories_are_eligible": set(used_ids).issubset(eligible_ids),
            "optimizer_update_ids_match_collection": update_trajectory_ids
            == used_ids,
            "all_eligible_trajectories_consumed": len(used_ids)
            == len(eligible_ids)
            and set(used_ids) == set(eligible_ids),
            "optimizer_shuffle_seed_matches_protocol": int(
                collection_row["optimizer_batch_shuffle_seed"]
            )
            == stable_seed(
                int(config["run"]["seed"]),
                "update_batches",
                int(collection_index),
            ),
            "worker_policy_equals_frozen_old_policy": bool(worker_hashes)
            and all(
                value == str(collection_row["old_lora_state_sha256"])
                for value in worker_hashes.values()
            ),
            "first_update_replays_old_policy": bool(first_update)
            and float(first_update["log_ratio_abs_max"]) <= 1.0e-5,
            "group_advantages_match_float32_eq8": bool(
                advantage_replay_audit["checks"][
                    "logged_advantages_match_float32_eq8"
                ]
            )
            and bool(
                advantage_replay_audit["checks"][
                    "eligible_flags_match_float32_eq8"
                ]
            ),
            "group_advantages_zero_mean_within_float32_bound": bool(
                advantage_replay_audit["checks"][
                    "float32_centering_residual_within_conditioned_bound"
                ]
            ),
            "group_advantages_unit_population_std": bool(
                advantage_replay_audit["checks"][
                    "population_std_within_float32_bound"
                ]
            ),
            "update_metrics_finite": update_finite,
            "reference_kl_nonnegative": all(
                float(update["reference_kl"]) >= -1.0e-7 for update in updates
            ),
            "online_lora_changed_from_initialization": state_delta[
                "changed_tensor_count"
            ]
            > 0
            and state_delta["delta_l2"] > 0.0,
        }
    )
    raw_component_names = ("dnsmos", "speaker", "speechbertscore")
    raw_components = {
        name: _numeric_distribution(
            [
                float(row["reward_components"]["raw_components"][name])
                for row in rollout_rows
            ]
        )
        for name in raw_component_names
    }
    summary["collection_mechanism"] = {
        "reward": _numeric_distribution(
            [float(row["reward"]) for row in rollout_rows]
        ),
        "raw_reward_components": raw_components,
        "advantage": _numeric_distribution(
            [float(row["advantage"]) for row in rollout_rows if bool(row["eligible"])]
        ),
        "group_reward_std": _numeric_distribution(group_reward_stds),
        "valid_groups": valid_groups,
        "zero_std_groups": expected_groups - valid_groups,
        "max_abs_group_advantage_mean": max(group_advantage_mean_abs, default=None),
        "max_abs_group_advantage_std_error": max(
            group_advantage_std_error, default=None
        ),
        "float32_eq8_replay_audit": advantage_replay_audit,
        "nfe_counts": {
            str(value): sum(int(row["nfe"]) == value for row in rollout_rows)
            for value in range(
                int(config["sampler"]["nfe_minimum"]),
                int(config["sampler"]["nfe_maximum"]) + 1,
            )
        },
        "window_start_counts": {
            str(value): sum(int(row["window_start"]) == value for row in rollout_rows)
            for value in range(
                int(config["sampler"]["start_minimum"]),
                int(config["sampler"]["start_maximum"]) + 1,
            )
        },
        "unique_initial_latent_seeds": len(set(initial_latent_seeds)),
        "unique_brownian_seeds": len(set(brownian_seeds)),
        "duplicate_terminal_mel_hashes": len(terminal_mel_hashes)
        - len(set(terminal_mel_hashes)),
        "duplicate_scored_wav_hashes": len(scored_wav_hashes)
        - len(set(scored_wav_hashes)),
        "gradient_norm": _numeric_distribution(
            [float(update["gradient_norm"]) for update in updates]
        ),
        "reference_kl": _numeric_distribution(
            [float(update["reference_kl"]) for update in updates]
        ),
        "approx_kl": _numeric_distribution(
            [float(update["approx_kl"]) for update in updates]
        ),
        "clip_fraction": _numeric_distribution(
            [float(update["clip_fraction"]) for update in updates]
        ),
        "ratio_mean": _numeric_distribution(
            [float(update["ratio_mean"]) for update in updates]
        ),
        "later_update_log_ratio_abs_max": _numeric_distribution(
            [float(update["log_ratio_abs_max"]) for update in later_updates]
        ),
        "policy_loss": _numeric_distribution(
            [float(update["policy_loss"]) for update in updates]
        ),
        "learning_rate_after_collection": float(updates[-1]["learning_rate"]),
        "scored_trajectories": int(collection_row["scored_trajectories"]),
        "eligible_trajectories": int(collection_row["eligible_trajectories"]),
        "used_trajectories": int(collection_row["used_trajectories"]),
        "trajectory_utilization": float(
            int(collection_row["used_trajectories"])
            / max(int(collection_row["eligible_trajectories"]), 1)
        ),
        "optimizer_batch_shuffle_seed": int(
            collection_row["optimizer_batch_shuffle_seed"]
        ),
        "optimizer_effective_batch_sizes": [
            int(update["effective_batch_size"]) for update in updates
        ],
        "gradient_accumulation_microbatch_counts": [
            int(update["microbatch_count"]) for update in updates
        ],
    }
    summary["invariants"] = {
        **invariants,
        "all_passed": all(invariants.values()),
    }
    return summary


def _aggregate_mechanism_audits(
    summaries: Sequence[Mapping], *, expected_collections: Sequence[int]
) -> dict:
    ordered = sorted(summaries, key=lambda value: int(value["collection_index"]))
    observed = [int(value["collection_index"]) for value in ordered]
    failures = {
        str(value["collection_index"]): sorted(
            name
            for name, passed in value["invariants"].items()
            if name != "all_passed" and not bool(passed)
        )
        for value in ordered
        if not bool(value["invariants"]["all_passed"])
    }
    trained = [value for value in ordered if int(value["collection_index"]) > 0]
    gradients_observed = any(
        float(value["collection_mechanism"]["gradient_norm"]["maximum"]) > 0.0
        for value in trained
    )
    within_collection_policy_change = any(
        value["collection_mechanism"]["later_update_log_ratio_abs_max"]["maximum"]
        is not None
        and float(
            value["collection_mechanism"]["later_update_log_ratio_abs_max"][
                "maximum"
            ]
        )
        > 0.0
        for value in trained
    )
    final_delta = (
        float(ordered[-1]["lora_delta_from_shared_initialization"]["delta_l2"])
        if ordered
        else 0.0
    )
    checks = {
        "all_registered_points_present": observed
        == [int(value) for value in expected_collections],
        "all_point_invariants_passed": not failures,
        "nonzero_gradients_observed": gradients_observed,
        "later_updates_depart_from_frozen_old_policy": within_collection_policy_change,
        "online_lora_changed_from_shared_initialization": final_delta > 0.0,
        "no_validation_or_selection_reuse": all(
            not bool(value["selection_eligible"])
            and not bool(value["validation_queried"])
            for value in ordered
        ),
        "analysis_artifact_io_exclusion_explicit": all(
            bool(value["analysis_artifact_io_excluded_from_training_budget"])
            for value in ordered
        ),
    }
    trends = [
        {
            "collection_index": int(value["collection_index"]),
            "optimizer_step": int(value["optimizer_step"]),
            "lora_delta_l2": float(
                value["lora_delta_from_shared_initialization"]["delta_l2"]
            ),
            "reward_mean": (
                None
                if value["collection_mechanism"] is None
                else value["collection_mechanism"]["reward"]["mean"]
            ),
            "group_reward_std_mean": (
                None
                if value["collection_mechanism"] is None
                else value["collection_mechanism"]["group_reward_std"]["mean"]
            ),
            "gradient_norm_mean": (
                None
                if value["collection_mechanism"] is None
                else value["collection_mechanism"]["gradient_norm"]["mean"]
            ),
            "reference_kl_mean": (
                None
                if value["collection_mechanism"] is None
                else value["collection_mechanism"]["reference_kl"]["mean"]
            ),
            "clip_fraction_mean": (
                None
                if value["collection_mechanism"] is None
                else value["collection_mechanism"]["clip_fraction"]["mean"]
            ),
        }
        for value in ordered
    ]
    return {
        "status": (
            "GRPO-MECHANISM-AUDIT-PASS"
            if all(checks.values())
            else "GRPO-MECHANISM-AUDIT-FAIL"
        ),
        "schema_version": 1,
        "purpose": "correctness_and_learning_mechanism_not_checkpoint_selection",
        "checks": checks,
        "invariant_failures_by_collection": failures,
        "trend_is_descriptive_not_a_selection_rule": True,
        "analysis_artifact_accounting": {
            "excluded_from_training_budget": True,
            "materialization_wall_seconds": float(
                sum(
                    float(
                        value.get(
                            "materialization_wall_seconds_excluded_from_training_budget",
                            0.0,
                        )
                    )
                    for value in ordered
                )
            ),
        },
        "trends": trends,
    }


def _due_selection_milestones(
    config: Mapping,
    *,
    collection_index: int,
    cumulative_accounting: Mapping,
    completed_milestones: Sequence[int],
) -> list[int]:
    percentages = [
        int(value)
        for value in config["evaluation"]["selection_milestones"]
        if int(value) != 0
    ]
    completed = {int(value) for value in completed_milestones}
    basis = str(config["evaluation"]["milestone_basis"])
    if basis == "collection_fraction":
        mapping = milestone_collections(
            int(config["run"]["collections"]), [0, *percentages]
        )
        return [
            int(percentage)
            for boundary, percentage in mapping.items()
            if boundary <= int(collection_index) and int(percentage) not in completed
        ]
    if basis == "allocated_training_gpu_hours":
        target_seconds = (
            float(config["comparison"]["target_training_gpu_hours"]) * 3600.0
        )
        observed_seconds = float(
            cumulative_accounting["allocated_training_gpu_seconds"]
        )
        due = [
            percentage
            for percentage in percentages
            if percentage not in completed
            and observed_seconds >= target_seconds * percentage / 100.0
        ]
        if len(due) > 1:
            raise RuntimeError(
                "one collection crossed multiple registered GPU-hour milestones; "
                "the collection geometry is too coarse for a fair first-crossing audit: "
                f"due={due}, observed_gpu_seconds={observed_seconds:.6f}"
            )
        return due
    raise ValueError(f"unsupported milestone basis: {basis}")


def _selection_milestone_metadata(
    config: Mapping,
    *,
    percentage: int,
    collection_index: int,
    cumulative_accounting: Mapping,
) -> dict:
    basis = str(config["evaluation"]["milestone_basis"])
    if basis == "collection_fraction":
        inverse = {
            int(value): int(boundary)
            for boundary, value in milestone_collections(
                int(config["run"]["collections"]),
                config["evaluation"]["selection_milestones"],
            ).items()
        }
        threshold_collection = inverse[int(percentage)]
        return {
            "basis": basis,
            "percentage": int(percentage),
            "threshold_collection": threshold_collection,
            "first_crossing_collection": int(collection_index),
            "collection_overshoot": int(collection_index - threshold_collection),
        }
    if basis == "allocated_training_gpu_hours":
        target_seconds = (
            float(config["comparison"]["target_training_gpu_hours"]) * 3600.0
        )
        threshold_seconds = target_seconds * int(percentage) / 100.0
        observed_seconds = float(
            cumulative_accounting["allocated_training_gpu_seconds"]
        )
        return {
            "basis": basis,
            "percentage": int(percentage),
            "target_training_gpu_hours": float(
                config["comparison"]["target_training_gpu_hours"]
            ),
            "threshold_allocated_training_gpu_seconds": threshold_seconds,
            "observed_allocated_training_gpu_seconds": observed_seconds,
            "observed_allocated_training_gpu_hours": observed_seconds / 3600.0,
            "overshoot_gpu_seconds": observed_seconds - threshold_seconds,
            "first_crossing_collection": int(collection_index),
            "accounting_is_exact": int(
                cumulative_accounting.get("unmeasured_checkpoint_commit_count", 0)
            )
            == 0,
        }
    raise ValueError(f"unsupported milestone basis: {basis}")


def _reconcile_accounting_sidecar(
    path: Path,
    *,
    config_hash: str,
    completed_collection: int,
    collection_commit_id: str,
    checkpoint_accounting: Mapping,
) -> dict:
    """Recover the narrow checkpoint-committed/sidecar-stale crash window."""

    accounting = json.loads(json.dumps(checkpoint_accounting))
    accounting.setdefault("accounting_recovery_events", [])
    accounting.setdefault("unmeasured_checkpoint_commit_count", 0)
    if not path.is_file():
        if completed_collection > 0:
            accounting["accounting_recovery_events"].append(
                {
                    "kind": "checkpoint_committed_before_accounting_sidecar",
                    "collection_index": completed_collection,
                    "collection_commit_id": collection_commit_id,
                    "recovery": "checkpoint_embedded_lower_bound_accounting",
                }
            )
            accounting["unmeasured_checkpoint_commit_count"] = int(
                accounting["unmeasured_checkpoint_commit_count"] + 1
            )
            atomic_write_json(
                path,
                {
                    "schema_version": 1,
                    "config_sha256": config_hash,
                    "collection_index": completed_collection,
                    "collection_commit_id": collection_commit_id,
                    "cumulative_accounting": accounting,
                    "recovered_from_missing_sidecar": True,
                },
            )
        return accounting
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    base_checks = {
        "schema": sidecar.get("schema_version") == 1,
        "config": sidecar.get("config_sha256") == config_hash,
        "accounting": isinstance(sidecar.get("cumulative_accounting"), Mapping),
    }
    if not all(base_checks.values()):
        raise ValueError(f"accounting sidecar is invalid: {base_checks}")
    sidecar_collection = int(sidecar.get("collection_index", -1))
    sidecar_commit = str(sidecar.get("collection_commit_id", ""))
    if (
        sidecar_collection == completed_collection
        and sidecar_commit == collection_commit_id
    ):
        exact = json.loads(json.dumps(sidecar["cumulative_accounting"]))
        exact["validation_wall_seconds"] = accounting["validation_wall_seconds"]
        exact["validation_gpu_seconds"] = accounting["validation_gpu_seconds"]
        exact.setdefault("accounting_recovery_events", [])
        exact.setdefault("unmeasured_checkpoint_commit_count", 0)
        return exact
    if completed_collection > 0 and sidecar_collection == completed_collection - 1:
        event = {
            "kind": "checkpoint_committed_before_accounting_sidecar",
            "collection_index": completed_collection,
            "collection_commit_id": collection_commit_id,
            "recovery": "checkpoint_embedded_lower_bound_accounting",
        }
        accounting["accounting_recovery_events"].append(event)
        accounting["unmeasured_checkpoint_commit_count"] = int(
            accounting["unmeasured_checkpoint_commit_count"] + 1
        )
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "collection_index": completed_collection,
                "collection_commit_id": collection_commit_id,
                "cumulative_accounting": accounting,
                "recovered_from_stale_sidecar": True,
            },
        )
        return accounting
    raise ValueError(
        "accounting sidecar cannot be reconciled with checkpoint: "
        f"checkpoint=({completed_collection}, {collection_commit_id}), "
        f"sidecar=({sidecar_collection}, {sidecar_commit})"
    )


def run(
    config: dict,
    *,
    resume: Path | None = None,
    resume_physical_gpu_ids: str | Sequence[int] | None = None,
) -> tuple[dict, Path]:
    summary = validate_grpo_config(config)
    public_smoke = str(config["run"].get("mode")) == "smoke"
    split_audit = audit_data_splits(config, strict_calibration=not public_smoke)
    physical_gpu_session = resolve_physical_gpu_session(
        config,
        resume=resume,
        requested_resume_physical_gpu_ids=resume_physical_gpu_ids,
    )

    # Heavy runtime imports stay behind config validation.  This keeps protocol
    # audit usable on CPU/login nodes without the FlowSE audio stack.
    from rl.common.flowse_interface import (
        load_flowse_bundle,
    )
    from rl.rewards.composite import (
        load_flowse_grpo_composite_evaluators,
    )
    from rl.common.protocol import (
        load_fidelity,
        strict_manifest,
        utterances_for_step,
    )
    from rl.rewards.metrics import DNSMOSScorer
    from .evaluation import (
        evaluate_validation_state,
        format_validation_comparison_table,
        load_grpo_online_checkpoint,
        reuse_released_base_for_zero_lora_validation,
        select_milestone_checkpoints,
    )

    config_hash = _canonical_hash(config)
    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(config["run"].get("deterministic", True)):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if not torch.cuda.is_available():
        raise RuntimeError("FlowSE-GRPO training requires CUDA")
    device_ids = [int(value) for value in config["resources"]["device_ids"]]
    if max(device_ids) >= torch.cuda.device_count():
        raise RuntimeError(
            "configured GRPO device ID is not visible: "
            f"device_ids={device_ids}, visible={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(int(config["resources"]["trainer_device_id"]))
    main_cuda_memory_reservation = _reserve_cuda_allocator_memory(
        config,
        device=int(config["resources"]["trainer_device_id"]),
        role="coordinator",
    )

    bundle = load_flowse_bundle(
        config["flowse_config"],
        deterministic=bool(config["run"].get("deterministic", True)),
    )
    released_hash_before = bundle.checkpoint_sha256
    torch.manual_seed(int(config["lora"]["initialization_seed"]))
    injection = inject_lora(
        bundle.model.transformer,
        target_patterns=config["lora"]["target_patterns"],
        rank=int(config["lora"]["rank"]),
        alpha=float(config["lora"]["alpha"]),
        dropout=float(config["lora"]["dropout"]),
        expected_modules=int(config["lora"]["expected_modules"]),
    )
    shared_snapshot = prepare_or_load_shared_lora_snapshot(
        bundle.model.transformer,
        config=config,
        module_names=injection.module_names,
    )
    initial_lora_state = snapshot_lora(bundle.model.transformer, device="cpu")
    bundle.model.eval()
    optimizer, scheduler = build_optimizer(bundle.model.transformer, config)
    conditioning = ConditioningProtocol.from_config(config["conditioning"])

    output_dir = Path(config["output_root"]) / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        existing_run_artifacts = [
            path
            for path in (
                output_dir / "checkpoint_latest.pt",
                output_dir / "collections.jsonl",
                output_dir / "rollout_rows.jsonl",
                output_dir / "milestones",
                output_dir / "mechanism_audits",
                output_dir / "reporting_checkpoints",
            )
            if path.exists()
        ]
        if existing_run_artifacts:
            raise FileExistsError(
                "GRPO run directory already contains resumable artifacts; pass --resume: "
                + ", ".join(str(path) for path in existing_run_artifacts)
            )
    _write_json(output_dir / "frozen_config.json", config)
    resource_fingerprint = {
        "rollout_world_size": int(config["resources"]["rollout_world_size"]),
        "device_ids": [int(value) for value in config["resources"]["device_ids"]],
        "trainer_device_id": int(config["resources"]["trainer_device_id"]),
        "expected_cuda_visible_devices": config["resources"].get(
            "expected_cuda_visible_devices"
        ),
        "observed_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_session": physical_gpu_session,
        "resource_session_log": str(output_dir / "resource_sessions.jsonl"),
        "cuda_memory_reservation": {
            "coordinator": main_cuda_memory_reservation,
            "workers": {},
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_devices": int(torch.cuda.device_count()),
        "devices": (
            [
                {
                    "logical_id": device_id,
                    "name": torch.cuda.get_device_name(device_id),
                    "total_memory": int(
                        torch.cuda.get_device_properties(device_id).total_memory
                    ),
                    "major": int(torch.cuda.get_device_properties(device_id).major),
                    "minor": int(torch.cuda.get_device_properties(device_id).minor),
                }
                for device_id in config["resources"]["device_ids"]
            ]
            if torch.cuda.is_available()
            else []
        ),
    }
    _write_json(
        output_dir / "implementation.json",
        {
            **summary,
            "config_sha256": config_hash,
            "released_checkpoint_sha256": released_hash_before,
            "lora_module_names": list(injection.module_names),
            "lora_trainable_parameters": injection.trainable_parameters,
            "shared_initial_lora_snapshot": shared_snapshot,
            "data_split_audit": split_audit,
            "resources": resource_fingerprint,
        },
    )

    # A public smoke run uses the same reward implementation and frozen
    # component scales, but does not require the formal calibration report's
    # private manifest/evaluator provenance.  Formal runs retain the complete
    # artifact verification path.
    reward_definition = resolve_training_reward(
        config, validate_artifacts=not public_smoke
    )
    if reward_definition["name"] != FLOWSE_GRPO_COMPOSITE:
        raise AssertionError("validated config changed reward implementation")
    composite_evaluators, evaluator_fingerprint = load_flowse_grpo_composite_evaluators(
        config
    )
    if not public_smoke:
        verify_reward_calibration(
            config,
            evaluator_fingerprint=evaluator_fingerprint,
            strict_provenance=True,
        )
    rollout_pool = None
    if int(config["resources"]["rollout_world_size"]) > 1:
        rollout_pool = _GRPORolloutPool(
            config=config,
            output_dir=output_dir,
            initial_state=snapshot_lora(bundle.model.transformer, device="cpu"),
            expected_module_names=injection.module_names,
            expected_evaluator_fingerprint=evaluator_fingerprint,
        )
        resource_fingerprint["cuda_memory_reservation"]["workers"] = dict(
            rollout_pool.worker_memory_reservations
        )
    dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
    manifest = strict_manifest(config["data"]["train_manifest"])
    validation_manifest = strict_manifest(config["data"]["validation_manifest"])
    print(
        "validation startup: loading fidelity evaluators "
        "(speaker/ASR; Wav2Vec2 loader warnings may appear)",
        flush=True,
    )
    fidelity, fidelity_fingerprint = load_fidelity(config)
    print(
        "validation startup: evaluators ready; "
        f"validation_utterances={len(validation_manifest)} nfe="
        f"{int(config['evaluation']['nfe'])}",
        flush=True,
    )
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    accounting_sidecar_path = output_dir / "accounting_latest.json"
    collection_log = output_dir / "collections.jsonl"
    rollout_log = output_dir / "rollout_rows.jsonl"
    completed_collection = 0
    optimizer_step = 0
    cumulative_accounting = _empty_accounting(config)
    completed_milestones: list[int] = []
    last_commit_id = "initial_uncommitted"
    if resume is not None:
        (
            completed_collection,
            optimizer_step,
            cumulative_accounting,
            completed_milestones,
            last_commit_id,
        ) = load_checkpoint(
            resume,
            bundle=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            config_hash=config_hash,
        )
    reconcile_collection_logs(
        collection_log=collection_log,
        rollout_log=rollout_log,
        completed_collection=completed_collection,
        expected_last_commit_id=(last_commit_id if completed_collection > 0 else None),
        expected_rollout_rows_per_collection=int(
            config["collection"]["prompts_per_mini_batch"]
            * config["collection"]["mini_batch_repeats"]
            * config["collection"]["group_size"]
        ),
    )
    if resume is not None:
        cumulative_accounting = _reconcile_accounting_sidecar(
            accounting_sidecar_path,
            config_hash=config_hash,
            completed_collection=completed_collection,
            collection_commit_id=last_commit_id,
            checkpoint_accounting=cumulative_accounting,
        )
    resource_session_record = {
        **physical_gpu_session,
        "session_id": f"{time.time_ns()}_{os.getpid()}",
        "config_sha256": config_hash,
        "checkpoint_collection": int(completed_collection),
        "checkpoint_optimizer_step": int(optimizer_step),
        "devices": resource_fingerprint["devices"],
    }
    append_jsonl_batch(
        output_dir / "resource_sessions.jsonl", [resource_session_record]
    )
    session_started = time.perf_counter()
    session_training_wall_start = float(
        cumulative_accounting["allocated_training_wall_seconds"]
    )
    session_training_gpu_start = float(
        cumulative_accounting["allocated_training_gpu_seconds"]
    )
    session_validation_wall_start = float(
        cumulative_accounting["validation_wall_seconds"]
    )
    session_validation_gpu_start = float(
        cumulative_accounting["validation_gpu_seconds"]
    )

    checkpoint_paths: dict[int, str] = {}
    milestone_reports: dict[int, dict] = {}
    milestone_metadata: dict[int, dict] = {}
    milestone_dir = output_dir / "milestones"
    mechanism_config = config.get("mechanism_audit")
    mechanism_collections = (
        [
            int(value)
            for value in mechanism_config.get("checkpoint_collections", [])
        ]
        if isinstance(mechanism_config, Mapping)
        else []
    )
    mechanism_dir = output_dir / "mechanism_audits"
    mechanism_checkpoint_paths: dict[int, str] = {}
    mechanism_summaries: dict[int, dict] = {}
    reporting_collections = reporting_checkpoint_collections(config)
    reporting_dir = output_dir / "reporting_checkpoints"
    reporting_checkpoint_paths: dict[int, str] = {}

    def save_current_checkpoint(
        *, destination: Path, collection_index: int, commit_id: str
    ) -> dict:
        return save_checkpoint(
            destination,
            bundle=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            collection_index=collection_index,
            optimizer_step=optimizer_step,
            config_hash=config_hash,
            collection_commit_id=commit_id,
            cumulative_accounting=cumulative_accounting,
            completed_milestones=completed_milestones,
        )

    def materialize_mechanism_audit(
        *, collection_index: int, commit_id: str
    ) -> None:
        materialization_started = time.perf_counter()
        expected_step = int(collection_index) * int(
            config["collection"]["optimizer_updates"]
        )
        checkpoint = mechanism_dir / (
            f"checkpoint_collection_{int(collection_index):06d}_"
            f"step_{expected_step:06d}.pt"
        )
        summary_path = mechanism_dir / (
            f"mechanism_collection_{int(collection_index):06d}_"
            f"step_{expected_step:06d}.json"
        )
        if checkpoint.is_file():
            payload = load_grpo_online_checkpoint(
                checkpoint, expected_config_hash=config_hash
            )
        else:
            if int(optimizer_step) != expected_step:
                raise ValueError(
                    "cannot reconstruct a missing historical mechanism checkpoint: "
                    f"collection={collection_index}, current_step={optimizer_step}, "
                    f"expected_step={expected_step}"
                )
            payload = save_current_checkpoint(
                destination=checkpoint,
                collection_index=int(collection_index),
                commit_id=str(commit_id),
            )
        if int(payload["collection_index"]) != int(collection_index):
            raise ValueError("mechanism checkpoint collection index differs")
        if int(payload["optimizer_step"]) != expected_step:
            raise ValueError("mechanism checkpoint optimizer step differs")
        if str(payload.get("collection_commit_id", "")) != str(commit_id):
            raise ValueError("mechanism checkpoint commit ID differs")
        if summary_path.is_file():
            audit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checks = {
                "collection": int(audit_summary.get("collection_index", -1))
                == int(collection_index),
                "step": int(audit_summary.get("optimizer_step", -1))
                == expected_step,
                "commit": str(audit_summary.get("collection_commit_id", ""))
                == str(commit_id),
                "checkpoint": str(audit_summary.get("checkpoint_path", ""))
                == str(checkpoint),
                "fingerprint": audit_summary.get("online_lora_fingerprint")
                == payload.get("online_lora_fingerprint"),
                "selection_excluded": audit_summary.get("selection_eligible")
                is False,
                "validation_excluded": audit_summary.get("validation_queried")
                is False,
                "analysis_io_excluded": audit_summary.get(
                    "analysis_artifact_io_excluded_from_training_budget"
                )
                is True,
            }
            if not all(checks.values()):
                raise ValueError(
                    f"mechanism audit sidecar differs from checkpoint: {checks}"
                )
        else:
            if int(collection_index) == 0:
                collection_row = None
                rollout_rows: list[dict] = []
            else:
                collection_path = output_dir / "collection_artifacts" / (
                    f"collection_{int(collection_index):06d}.json"
                )
                rollout_path = output_dir / "collection_artifacts" / (
                    f"rollout_collection_{int(collection_index):06d}.jsonl"
                )
                if not collection_path.is_file() or not rollout_path.is_file():
                    raise FileNotFoundError(
                        "mechanism audit lacks committed collection artifacts"
                    )
                collection_row = json.loads(
                    collection_path.read_text(encoding="utf-8")
                )
                rollout_rows = [
                    json.loads(line)
                    for line in rollout_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            audit_summary = _mechanism_collection_summary(
                collection_index=int(collection_index),
                optimizer_step=expected_step,
                collection_commit_id=str(commit_id),
                checkpoint_path=checkpoint,
                checkpoint_payload=payload,
                initial_lora_state=initial_lora_state,
                config=config,
                collection_row=collection_row,
                rollout_rows=rollout_rows,
            )
            audit_summary[
                "materialization_wall_seconds_excluded_from_training_budget"
            ] = float(time.perf_counter() - materialization_started)
            atomic_write_json(summary_path, audit_summary)
        mechanism_checkpoint_paths[int(collection_index)] = str(checkpoint)
        mechanism_summaries[int(collection_index)] = audit_summary

    def materialize_reporting_checkpoint(
        *, collection_index: int, commit_id: str
    ) -> None:
        expected_step = int(collection_index) * int(
            config["collection"]["optimizer_updates"]
        )
        checkpoint = reporting_dir / (
            f"checkpoint_collection_{int(collection_index):06d}_"
            f"step_{expected_step:06d}.pt"
        )
        metadata_path = reporting_dir / (
            f"reporting_collection_{int(collection_index):06d}_"
            f"step_{expected_step:06d}.json"
        )
        if checkpoint.is_file():
            payload = load_grpo_online_checkpoint(
                checkpoint, expected_config_hash=config_hash
            )
        else:
            if int(optimizer_step) != expected_step:
                raise ValueError(
                    "cannot reconstruct a missing historical reporting checkpoint: "
                    f"collection={collection_index}, current_step={optimizer_step}, "
                    f"expected_step={expected_step}"
                )
            payload = save_current_checkpoint(
                destination=checkpoint,
                collection_index=int(collection_index),
                commit_id=str(commit_id),
            )
        checks = {
            "collection": int(payload["collection_index"]) == int(collection_index),
            "optimizer_step": int(payload["optimizer_step"]) == expected_step,
            "commit": str(payload.get("collection_commit_id", ""))
            == str(commit_id),
        }
        if not all(checks.values()):
            raise ValueError(
                f"reporting checkpoint differs from its budget boundary: {checks}"
            )
        specification = config["artifacts"]["reporting_checkpoints"]
        metadata = {
            "schema_version": 1,
            "kind": "grpo_nonselection_reporting_checkpoint",
            "collection_index": int(collection_index),
            "optimizer_step": expected_step,
            "collection_commit_id": str(commit_id),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "online_lora_fingerprint": payload["online_lora_fingerprint"],
            "scored_trajectories": int(collection_index)
            * int(summary["candidate_count_per_collection"]),
            "purpose": str(specification["purpose"]),
            "selection_eligible": False,
            "validation_queried": False,
            "artifact_io_excluded_from_training_budget": True,
        }
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != metadata:
                raise ValueError(
                    "reporting checkpoint metadata differs from the checkpoint"
                )
        else:
            atomic_write_json(metadata_path, metadata)
        reporting_checkpoint_paths[int(collection_index)] = str(checkpoint)

    def materialize_milestone(
        *, percentage: int, collection_index: int, commit_id: str
    ) -> None:
        metadata = _selection_milestone_metadata(
            config,
            percentage=percentage,
            collection_index=collection_index,
            cumulative_accounting=cumulative_accounting,
        )
        milestone_path = (
            milestone_dir / f"checkpoint_milestone_{int(percentage):03d}pct.pt"
        )
        if milestone_path.is_file():
            payload = load_grpo_online_checkpoint(
                milestone_path, expected_config_hash=config_hash
            )
        else:
            payload = save_current_checkpoint(
                destination=milestone_path,
                collection_index=collection_index,
                commit_id=commit_id,
            )
        if int(payload["collection_index"]) != int(collection_index):
            raise ValueError(
                f"milestone {percentage}% is not the first crossing collection: "
                f"checkpoint={payload['collection_index']}, expected={collection_index}"
            )
        if str(payload.get("collection_commit_id", "")) != str(commit_id):
            raise ValueError(f"milestone {percentage}% commit ID differs from checkpoint")
        checkpoint_paths[int(percentage)] = str(milestone_path)
        if int(percentage) == 0:
            if payload["online_lora_fingerprint"] != lora_state_fingerprint(
                initial_lora_state
            ):
                raise RuntimeError("0% milestone differs from shared initial LoRA")
            validation_report, validation_seconds = (
                reuse_released_base_for_zero_lora_validation(
                    base_report=base_report,
                    lora_state=payload["online_lora_state"],
                    state_id="grpo_online_000pct",
                    percentage=0,
                    collection_index=0,
                    checkpoint_path=str(milestone_path),
                    config=config,
                    output_dir=output_dir,
                    milestone_metadata=metadata,
                )
            )
        else:
            validation_report, validation_seconds = evaluate_validation_state(
                bundle=bundle,
                lora_state=payload["online_lora_state"],
                state_id=f"grpo_online_{int(percentage):03d}pct",
                policy_kind="grpo_online",
                percentage=int(percentage),
                collection_index=int(collection_index),
                checkpoint_path=str(milestone_path),
                manifest=validation_manifest,
                config=config,
                conditioning=conditioning,
                dnsmos=dnsmos,
                fidelity=fidelity,
                composite_evaluators=composite_evaluators,
                reward_definition=reward_definition,
                output_dir=output_dir,
                milestone_metadata=metadata,
            )
        print(
            "\n"
            + format_validation_comparison_table(
                base_report=base_report,
                state_report=validation_report,
                optimizer_step=int(payload["optimizer_step"]),
                collection_index=int(collection_index),
                percentage=int(percentage),
            ),
            flush=True,
        )
        milestone_reports[int(percentage)] = validation_report
        milestone_metadata[int(percentage)] = metadata
        cumulative_accounting["validation_wall_seconds"] = float(
            cumulative_accounting["validation_wall_seconds"] + validation_seconds
        )
        cumulative_accounting["validation_gpu_seconds"] = float(
            cumulative_accounting["validation_gpu_seconds"] + validation_seconds
        )
        atomic_write_json(
            milestone_dir / f"milestone_{int(percentage):03d}pct.json", metadata
        )
        if int(percentage) not in completed_milestones:
            completed_milestones.append(int(percentage))
            completed_milestones.sort()
        save_current_checkpoint(
            destination=checkpoint_path,
            collection_index=collection_index,
            commit_id=commit_id,
        )

    sft20k_base = str(config["flowse_config"]).replace("\\", "/").endswith(
        "flowse_libritts_sft20k_wotext.yaml"
    )
    base_report, base_validation_seconds = evaluate_validation_state(
        bundle=bundle,
        lora_state=None,
        state_id=(
            "sft20k_base_cfg0_nfe32" if sft20k_base else "released_base_cfg0_nfe32"
        ),
        policy_kind="sft20k_base" if sft20k_base else "released_base",
        percentage=None,
        collection_index=0,
        checkpoint_path=None,
        manifest=validation_manifest,
        config=config,
        conditioning=conditioning,
        dnsmos=dnsmos,
        fidelity=fidelity,
        composite_evaluators=composite_evaluators,
        reward_definition=reward_definition,
        output_dir=output_dir,
    )
    cumulative_accounting["validation_wall_seconds"] = float(
        cumulative_accounting["validation_wall_seconds"] + base_validation_seconds
    )
    cumulative_accounting["validation_gpu_seconds"] = float(
        cumulative_accounting["validation_gpu_seconds"] + base_validation_seconds
    )

    if 0 not in completed_milestones:
        materialize_milestone(percentage=0, collection_index=0, commit_id="initial")
    if 0 in mechanism_collections:
        materialize_mechanism_audit(collection_index=0, commit_id="initial")

    for percentage in completed_milestones:
        milestone_path = (
            milestone_dir / f"checkpoint_milestone_{int(percentage):03d}pct.pt"
        )
        report_path = (
            output_dir / "validation" / f"grpo_online_{int(percentage):03d}pct.json"
        )
        if not milestone_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(
                f"completed milestone {percentage}% lacks checkpoint/evaluation"
            )
        checkpoint_paths[int(percentage)] = str(milestone_path)
        milestone_reports[int(percentage)] = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        metadata = milestone_reports[int(percentage)].get("milestone_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"milestone {percentage}% lacks budget metadata")
        milestone_metadata[int(percentage)] = dict(metadata)

    # Resume may restore a checkpoint that crossed a budget threshold just
    # before the immutable milestone copy was committed.  Materialize only the
    # exactly restored collection; an older missed boundary is unrecoverable.
    due_after_resume = _due_selection_milestones(
        config,
        collection_index=completed_collection,
        cumulative_accounting=cumulative_accounting,
        completed_milestones=completed_milestones,
    )
    for percentage in due_after_resume:
        materialize_milestone(
            percentage=int(percentage),
            collection_index=completed_collection,
            commit_id=last_commit_id,
        )

    for reporting_collection in reporting_collections:
        if reporting_collection > completed_collection:
            continue
        checkpoint = reporting_dir / (
            f"checkpoint_collection_{int(reporting_collection):06d}_"
            f"step_{int(reporting_collection) * int(config['collection']['optimizer_updates']):06d}.pt"
        )
        if checkpoint.is_file():
            payload = load_grpo_online_checkpoint(
                checkpoint, expected_config_hash=config_hash
            )
            materialize_reporting_checkpoint(
                collection_index=int(reporting_collection),
                commit_id=str(payload.get("collection_commit_id", "")),
            )
        elif int(reporting_collection) == int(completed_collection):
            materialize_reporting_checkpoint(
                collection_index=int(reporting_collection),
                commit_id=last_commit_id,
            )
        else:
            raise FileNotFoundError(
                "completed reporting checkpoint is missing and cannot be "
                f"reconstructed: collection={reporting_collection}"
            )

    for audit_collection in mechanism_collections:
        if audit_collection == 0 or audit_collection > completed_collection:
            continue
        audit_step = int(audit_collection) * int(
            config["collection"]["optimizer_updates"]
        )
        audit_checkpoint = mechanism_dir / (
            f"checkpoint_collection_{int(audit_collection):06d}_"
            f"step_{audit_step:06d}.pt"
        )
        if audit_checkpoint.is_file():
            audit_payload = load_grpo_online_checkpoint(
                audit_checkpoint, expected_config_hash=config_hash
            )
            audit_commit_id = str(audit_payload.get("collection_commit_id", ""))
        elif audit_collection == completed_collection:
            audit_commit_id = last_commit_id
        else:
            raise FileNotFoundError(
                "completed run prefix lacks immutable mechanism checkpoint: "
                f"collection={audit_collection}"
            )
        materialize_mechanism_audit(
            collection_index=int(audit_collection), commit_id=audit_commit_id
        )

    target_already_reached = (
        str(config["evaluation"]["milestone_basis"])
        == "allocated_training_gpu_hours"
        and 100 in completed_milestones
    )
    if target_already_reached and rollout_pool is not None:
        rollout_pool.close()
        rollout_pool = None
    collection_indices = (
        range(0)
        if target_already_reached
        else range(
            completed_collection + 1, int(config["run"]["collections"]) + 1
        )
    )
    for collection_index in collection_indices:
        collection_started = time.perf_counter()
        prompts = utterances_for_step(
            list(manifest),
            step=collection_index,
            conditions_per_step=int(config["collection"]["prompts_per_mini_batch"]),
            seed=int(config["data"]["order_seed"]),
            stride_per_step=int(config["collection"]["prompts_per_mini_batch"]),
        )
        old_state = snapshot_lora(bundle.model.transformer, device="cpu")
        old_state_sha256 = lora_state_fingerprint(old_state)["state_sha256"]
        collection_commit_id = hashlib.sha256(
            (
                f"{config_hash}|{collection_index}|{optimizer_step}|{old_state_sha256}"
            ).encode("utf-8")
        ).hexdigest()
        if rollout_pool is None:
            result = rollout_collection(
                bundle,
                collection_index=collection_index,
                utterances=prompts,
                config=config,
                conditioning=conditioning,
                dnsmos=dnsmos,
                composite_evaluators=composite_evaluators,
                reward_definition=reward_definition,
                output_dir=output_dir,
            )
            worker_state_hashes = {0: old_state_sha256}
        else:
            result, worker_state_hashes = rollout_pool.rollout(
                bundle=bundle,
                online_state=old_state,
                collection_index=collection_index,
                utterances=prompts,
                config=config,
                conditioning=conditioning,
                dnsmos=dnsmos,
                composite_evaluators=composite_evaluators,
                reward_definition=reward_definition,
                output_dir=output_dir,
            )
        # No policy mutation is permitted during collection/reward scoring.
        current_after_rollout = snapshot_lora(bundle.model.transformer, device="cpu")
        if any(
            not torch.equal(old_state[name], current_after_rollout[name])
            for name in old_state
        ):
            raise RuntimeError("online policy changed during frozen collection")
        all_ids = [str(row["trajectory_id"]) for row in result.rollout_rows]
        eligible_ids = [
            str(row["trajectory_id"])
            for row in result.rollout_rows
            if bool(row["eligible"])
        ]
        example_ids = [str(item.trajectory_id) for item in result.examples]
        preupdate_geometry = {
            "rollout_ids_unique": len(all_ids) == len(set(all_ids)),
            "eligible_ids_unique": len(eligible_ids) == len(set(eligible_ids)),
            "example_ids_unique": len(example_ids) == len(set(example_ids)),
            "examples_match_all_eligible": len(example_ids) == len(eligible_ids)
            and set(example_ids) == set(eligible_ids),
        }
        if not all(preupdate_geometry.values()):
            raise RuntimeError(
                "GRPO full-buffer pre-update geometry failed: "
                f"{preupdate_geometry}"
            )
        update_shuffle_seed = stable_seed(seed, "update_batches", collection_index)
        optimizer_batches = deterministic_full_buffer_update_batches(
            result.examples,
            seed=update_shuffle_seed,
            updates=int(config["collection"]["optimizer_updates"]),
            microbatch_size=int(config["collection"]["microbatch_size"]),
        )
        update_rows = []
        update_phase_seconds = {"recompute": 0.0, "backward": 0.0, "optimizer": 0.0}
        for microbatches in optimizer_batches:
            optimizer_step += 1
            metrics = optimizer_update_accumulated(
                bundle,
                microbatches,
                optimizer=optimizer,
                scheduler=scheduler,
                conditioning=conditioning,
                config=config,
            )
            if not update_rows and metrics["log_ratio_abs_max"] > 1.0e-5:
                raise RuntimeError(
                    "first update did not replay the frozen old policy: "
                    f"max |log ratio|={metrics['log_ratio_abs_max']}"
                )
            metrics["optimizer_step"] = optimizer_step
            update_rows.append(metrics)
            for phase in update_phase_seconds:
                update_phase_seconds[phase] += float(metrics["timing_seconds"][phase])
        used_ids = [value for row in update_rows for value in row["trajectory_ids"]]
        example_by_id = {item.trajectory_id: item for item in result.examples}
        used_examples = [example_by_id[value] for value in used_ids]
        used_transitions = sum(
            len(item.trajectory.transitions) for item in used_examples
        )
        used_valid_dimensions = sum(
            record.valid_dimensions
            for item in used_examples
            for record in item.trajectory.transitions
        )
        used_mel_frame_tokens = sum(
            int(item.frame_mask.sum().item()) * len(item.trajectory.transitions)
            for item in used_examples
        )
        rollout_logical_velocity_examples = sum(
            int(row["nfe"]) for row in result.rollout_rows
        )
        rollout_model_forwards = sum(int(row["nfe"]) for row in result.group_rows)
        collection_row = {
            "collection_index": collection_index,
            "collection_commit_id": collection_commit_id,
            "old_lora_state_sha256": old_state_sha256,
            "rollout_worker_state_sha256": {
                str(rank): value for rank, value in sorted(worker_state_hashes.items())
            },
            "prompts": prompts,
            "prompt_instances": int(
                config["collection"]["prompts_per_mini_batch"]
                * config["collection"]["mini_batch_repeats"]
            ),
            "unique_utterances": len(set(prompts)),
            "scored_trajectories": len(all_ids),
            "eligible_trajectories": len(eligible_ids),
            "used_trajectories": len(used_ids),
            "unused_eligible_trajectories": len(eligible_ids) - len(used_ids),
            "trajectory_utilization": float(
                len(used_ids) / max(len(eligible_ids), 1)
            ),
            "batch_semantics": str(config["collection"]["batch_semantics"]),
            "optimizer_batch_shuffle_seed": int(update_shuffle_seed),
            "training_microbatch_size": int(
                config["collection"]["microbatch_size"]
            ),
            "optimizer_effective_batch_sizes": [
                int(row["effective_batch_size"]) for row in update_rows
            ],
            "gradient_accumulation_microbatch_counts": [
                int(row["microbatch_count"]) for row in update_rows
            ],
            "used_stochastic_transitions": used_transitions,
            "reward_calls": len(all_ids),
            "candidate_audio_seconds": result.candidate_audio_seconds,
            "retained_audit_audio": result.retained_audio_count,
            "zero_std_groups": sum(not row["eligible"] for row in result.group_rows),
            "used_trajectory_ids": used_ids,
            "eligible_trajectory_ids": eligible_ids,
            "unused_trajectory_ids": sorted(set(all_ids) - set(used_ids)),
            "compute_accounting": {
                "rollout_logical_velocity_examples": rollout_logical_velocity_examples,
                "rollout_old_policy_model_forwards": rollout_model_forwards,
                "current_recompute_model_forwards": used_transitions,
                "reference_recompute_model_forwards": used_transitions,
                "old_replay_audit_transition_examples": sum(
                    len(microbatch) for microbatch in optimizer_batches[0]
                )
                * int(config["sampler"]["window_size"]),
                "cfg_unconditional_forwards": 0,
                "backward_transition_examples": used_transitions,
                "backward_calls": sum(
                    len(microbatches) for microbatches in optimizer_batches
                ),
                "effective_mel_frame_tokens": used_mel_frame_tokens,
                "effective_mel_dimensions": used_valid_dimensions,
                "optimizer_updates": len(optimizer_batches),
            },
            "updates": update_rows,
            "phase_seconds": {
                **result.phase_seconds,
                **update_phase_seconds,
            },
            "active_gpu_seconds_by_phase": {
                **result.active_gpu_seconds_by_phase,
                **update_phase_seconds,
            },
        }
        transaction_started = time.perf_counter()
        committed_rollout_rows = [
            {**row, "collection_commit_id": collection_commit_id}
            for row in result.rollout_rows
        ]
        per_collection_dir = output_dir / "collection_artifacts"
        atomic_write_jsonl(
            per_collection_dir / f"rollout_collection_{collection_index:06d}.jsonl",
            committed_rollout_rows,
        )
        append_jsonl_batch(rollout_log, committed_rollout_rows)
        transaction_seconds = time.perf_counter() - transaction_started
        collection_row["phase_seconds"]["checkpoint_and_logging"] = transaction_seconds
        training_wall_seconds = time.perf_counter() - collection_started
        collection_row["training_wall_seconds_excluding_validation"] = (
            training_wall_seconds
        )
        atomic_write_json(
            per_collection_dir / f"collection_{collection_index:06d}.json",
            collection_row,
        )
        append_jsonl_batch(collection_log, [collection_row])
        _record_training_accounting(
            cumulative_accounting,
            collection_wall_seconds=training_wall_seconds,
            phase_seconds=collection_row["phase_seconds"],
            active_gpu_seconds_by_phase=collection_row["active_gpu_seconds_by_phase"],
            optimizer_step=optimizer_step,
        )
        last_commit_id = collection_commit_id
        save_current_checkpoint(
            destination=checkpoint_path,
            collection_index=collection_index,
            commit_id=collection_commit_id,
        )
        checkpoint_completed = time.perf_counter()
        committed_transaction_seconds = checkpoint_completed - transaction_started
        committed_training_wall_seconds = checkpoint_completed - collection_started
        transaction_delta = committed_transaction_seconds - transaction_seconds
        training_wall_delta = committed_training_wall_seconds - training_wall_seconds
        collection_row["phase_seconds"]["checkpoint_and_logging"] = float(
            committed_transaction_seconds
        )
        collection_row["training_wall_seconds_excluding_validation"] = float(
            committed_training_wall_seconds
        )
        cumulative_accounting["phase_seconds"]["checkpoint_and_logging"] = float(
            cumulative_accounting["phase_seconds"]["checkpoint_and_logging"]
            + transaction_delta
        )
        cumulative_accounting["allocated_training_wall_seconds"] = float(
            cumulative_accounting["allocated_training_wall_seconds"]
            + training_wall_delta
        )
        cumulative_accounting["allocated_training_gpu_seconds"] = float(
            cumulative_accounting["allocated_training_gpu_seconds"]
            + training_wall_delta * int(cumulative_accounting["rollout_world_size"])
        )
        atomic_write_json(
            per_collection_dir / f"collection_{collection_index:06d}.json",
            collection_row,
        )
        atomic_write_json(
            accounting_sidecar_path,
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "collection_index": collection_index,
                "collection_commit_id": collection_commit_id,
                "cumulative_accounting": cumulative_accounting,
            },
        )
        training_wall_seconds = committed_training_wall_seconds

        if collection_index in mechanism_collections:
            materialize_mechanism_audit(
                collection_index=collection_index,
                commit_id=collection_commit_id,
            )
        if collection_index in reporting_collections:
            materialize_reporting_checkpoint(
                collection_index=collection_index,
                commit_id=collection_commit_id,
            )

        due_milestones = _due_selection_milestones(
            config,
            collection_index=collection_index,
            cumulative_accounting=cumulative_accounting,
            completed_milestones=completed_milestones,
        )
        for percentage in due_milestones:
            materialize_milestone(
                percentage=int(percentage),
                collection_index=collection_index,
                commit_id=collection_commit_id,
            )
        print(
            f"collection={collection_index}/{config['run']['collections']} "
            f"scored={len(all_ids)} eligible={len(eligible_ids)} used={len(used_ids)} "
            f"updates={optimizer_step} seconds={training_wall_seconds:.1f}"
        )
        if (
            str(config["evaluation"]["milestone_basis"])
            == "allocated_training_gpu_hours"
            and 100 in completed_milestones
        ):
            break

    if rollout_pool is not None:
        rollout_pool.close()
    released_hash_after = sha256_file(bundle.checkpoint_path)
    if released_hash_after != released_hash_before:
        raise RuntimeError("released FlowSE checkpoint changed during GRPO training")
    missing_milestones = sorted(
        set(int(value) for value in config["evaluation"]["selection_milestones"])
        - set(completed_milestones)
    )
    if missing_milestones:
        observed_hours = float(
            cumulative_accounting["allocated_training_gpu_seconds"] / 3600.0
        )
        raise RuntimeError(
            "training ended before all registered milestones were reached: "
            f"missing={missing_milestones}, observed_gpu_hours={observed_hours:.6f}"
        )
    missing_reporting = sorted(
        set(reporting_collections) - set(reporting_checkpoint_paths)
    )
    if missing_reporting:
        raise RuntimeError(
            "training ended before all reporting checkpoints were committed: "
            f"{missing_reporting}"
        )
    for percentage in config["evaluation"]["selection_milestones"]:
        percentage = int(percentage)
        if percentage not in checkpoint_paths:
            checkpoint_paths[percentage] = str(
                milestone_dir / f"checkpoint_milestone_{percentage:03d}pct.pt"
            )
        if percentage not in milestone_reports:
            milestone_reports[percentage] = json.loads(
                (
                    output_dir / "validation" / f"grpo_online_{percentage:03d}pct.json"
                ).read_text(encoding="utf-8")
            )
    budget_match = _gpu_time_budget_match(config, cumulative_accounting)
    collection_statistics = _aggregate_collection_statistics(collection_log)
    selection = select_milestone_checkpoints(
        base_report=base_report,
        milestone_reports=[
            milestone_reports[int(percentage)]
            for percentage in config["evaluation"]["selection_milestones"]
        ],
        checkpoint_paths=checkpoint_paths,
        config=config,
        output_dir=output_dir,
    )
    selection["training_config_sha256"] = config_hash
    selection["shared_initial_lora_snapshot"] = shared_snapshot
    selection["gpu_time_budget_match"] = budget_match
    selection["resources"] = resource_fingerprint
    selection["training_compute_accounting"] = _accounting_report(
        cumulative_accounting
    )
    selection["workload_statistics"] = collection_statistics
    atomic_write_json(output_dir / "checkpoint_selection.json", selection)
    mechanism_report = None
    if mechanism_collections:
        missing_mechanism_points = sorted(
            set(mechanism_collections) - set(mechanism_summaries)
        )
        if missing_mechanism_points:
            raise RuntimeError(
                "training ended before all mechanism checkpoints were committed: "
                f"{missing_mechanism_points}"
            )
        mechanism_report = _aggregate_mechanism_audits(
            [mechanism_summaries[value] for value in mechanism_collections],
            expected_collections=mechanism_collections,
        )
        atomic_write_json(
            mechanism_dir / "mechanism_audit_report.json", mechanism_report
        )
    if mechanism_report is not None and mechanism_report["status"].endswith("FAIL"):
        training_status = "GRPO-TRAINING-COMPLETE-MECHANISM-AUDIT-FAILED"
    elif budget_match is not None and not bool(budget_match["passed"]):
        training_status = "GRPO-TRAINING-COMPLETE-BUDGET-AUDIT-FAILED"
    else:
        training_status = "GRPO-TRAINING-COMPLETE"
    report = {
        "status": training_status,
        **summary,
        "config_sha256": config_hash,
        "completed_collections": int(cumulative_accounting["completed_collections"]),
        "completed_optimizer_steps": optimizer_step,
        "released_checkpoint_sha256_before": released_hash_before,
        "released_checkpoint_sha256_after": released_hash_after,
        "checkpoint": str(checkpoint_path),
        "current_session_end_to_end_wall_seconds": time.perf_counter()
        - session_started,
        "current_session_training_wall_seconds": float(
            cumulative_accounting["allocated_training_wall_seconds"]
            - session_training_wall_start
        ),
        "current_session_training_gpu_hours": float(
            (
                cumulative_accounting["allocated_training_gpu_seconds"]
                - session_training_gpu_start
            )
            / 3600.0
        ),
        "current_session_validation_wall_seconds": float(
            cumulative_accounting["validation_wall_seconds"]
            - session_validation_wall_start
        ),
        "current_session_validation_gpu_hours": float(
            (
                cumulative_accounting["validation_gpu_seconds"]
                - session_validation_gpu_start
            )
            / 3600.0
        ),
        "cumulative_accounting": _accounting_report(cumulative_accounting),
        "collection_statistics": collection_statistics,
        "gpu_time_budget_match": budget_match,
        "resources": resource_fingerprint,
        "data_split_audit": split_audit,
        "shared_initial_lora_snapshot": shared_snapshot,
        "milestone_checkpoints": checkpoint_paths,
        "milestone_budget_metadata": milestone_metadata,
        "checkpoint_selection": selection,
        "reporting_checkpoint_paths": reporting_checkpoint_paths,
        "mechanism_audit_checkpoint_paths": mechanism_checkpoint_paths,
        "mechanism_audit": mechanism_report,
        "evaluation_status": "milestone_validation_and_selection_complete",
    }
    _write_json(output_dir / "training_report.json", report)
    return report, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train online FlowSE-GRPO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-physical-gpus",
        help=(
            "comma-separated physical GPU order for an explicit resume-only "
            "hardware remap; must equal CUDA_VISIBLE_DEVICES"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validation = validate_grpo_config(config)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    report, output_dir = run(
        config,
        resume=args.resume,
        resume_physical_gpu_ids=args.resume_physical_gpus,
    )
    from .run_summary import (
        build_acceptance_summary,
        format_acceptance_summary,
    )

    acceptance = build_acceptance_summary(output_dir)
    _write_json(output_dir / "acceptance_summary.json", acceptance)
    print(format_acceptance_summary(acceptance))
    print(f"Report: {output_dir / 'training_report.json'}")
    print(f"Acceptance: {output_dir / 'acceptance_summary.json'}")
    if report["status"] != "GRPO-TRAINING-COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
