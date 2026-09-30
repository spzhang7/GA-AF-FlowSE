"""Validation and immutable fingerprints for speech AdvantageFlow training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Mapping

import yaml

from rl.common.fairness import validate_controlled_baseline

from rl.common.conditioning import ConditioningProtocol
from .advantage_flow import validate_reward_constraints
from rl.common.dataset_audit import (
    CERTIFICATE_CONFIG_KEY,
    validate_dataset_audit_certificate,
)
from rl.rewards.specification import (
    DNSMOS_OVRL_RAW,
    DNSMOS_SPEAKER,
    FLOWSE_GRPO_COMPOSITE,
    resolve_training_reward,
    verify_reward_calibration,
)
from rl.common.dataset_validation import validate_paired_audio_dataset
from rl.common.shared_initialization import validate_shared_lora_snapshot_spec


PAPER_BASELINE = {
    "conditions_per_step": 32,
    "candidates_per_condition": 4,
    "training_nfe": 10,
    "lora_rank": 32,
    "lora_alpha": 64.0,
    "lambda_reference": 0.001,
}

CONTROLLED_L16_K8_BASELINE = {
    **PAPER_BASELINE,
    "conditions_per_step": 16,
    "candidates_per_condition": 8,
}


def training_nfe_spec(rollout: Mapping) -> dict:
    """Resolve the training-rollout NFE policy.

    Historical AF configurations provide a single ``training_nfe``.  Mixed-NFE
    experiments additionally provide an inclusive minimum/maximum range and a
    deterministic group-level seed.  Keeping the scalar field as the nominal
    value preserves compatibility with the existing reward-calibration and
    protocol schemas while the resolved range records the actual rollout
    geometry.
    """

    if not isinstance(rollout, Mapping):
        raise ValueError("rollout must be a mapping")
    if "training_nfe" not in rollout:
        raise ValueError("rollout.training_nfe is required")
    configured = int(rollout["training_nfe"])
    minimum = int(rollout.get("training_nfe_minimum", configured))
    maximum = int(rollout.get("training_nfe_maximum", configured))
    if configured < 1 or minimum < 1 or maximum < 1:
        raise ValueError("training NFE values must be positive")
    if minimum > maximum:
        raise ValueError("training_nfe_minimum cannot exceed training_nfe_maximum")
    if not minimum <= configured <= maximum:
        raise ValueError(
            "rollout.training_nfe must lie within the configured training NFE range"
        )
    mixed = minimum != maximum
    scope = str(rollout.get("training_nfe_sampling_scope", "fixed"))
    schedule = str(rollout.get("training_nfe_schedule", "random_per_group"))
    if schedule not in {"random_per_group", "balanced_per_step"}:
        raise ValueError(
            "training_nfe_schedule must be random_per_group or balanced_per_step"
        )
    if mixed and scope != "group":
        raise ValueError(
            "mixed training NFE must use group sampling so all K candidates "
            "share one integration grid"
        )
    if not mixed and scope not in {"fixed", "group"}:
        raise ValueError("fixed training NFE scope must be fixed or group")
    seed_base = rollout.get("training_nfe_seed_base")
    if mixed and seed_base is None:
        raise ValueError("mixed training NFE requires training_nfe_seed_base")
    if schedule == "balanced_per_step" and not mixed:
        raise ValueError("balanced_per_step requires a mixed training NFE range")
    if seed_base is not None:
        seed_base = int(seed_base)
    return {
        "configured": configured,
        "minimum": minimum,
        "maximum": maximum,
        "mixed": mixed,
        "scope": scope,
        "seed_base": seed_base,
    }
LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING = {
    "mode": "quadratic_mel_frame_budget",
    "reference_mel_frames": 960,
    "reference_batch_size": 8,
    "minimum_size": 1,
}

RUNTIME_PACKAGES = (
    "huggingface-hub",
    "librosa",
    "modelscope",
    "numpy",
    "onnxruntime",
    "pesq",
    "PyYAML",
    "scipy",
    "soundfile",
    "torch",
    "torchaudio",
    "torchdiffeq",
    "transformers",
    "vocos",
    "x-transformers",
)

EXPERIMENT_SOURCE_NAMES = (
    "rl/common/conditioning.py",
    "rl/rewards/evaluators.py",
    "rl/common/fairness.py",
    "rl/common/flow_matching.py",
    "rl/common/flowse_interface.py",
    "rl/rewards/composite.py",
    "rl/common/flow_objective.py",
    "rl/common/lora.py",
    "rl/common/normalization.py",
    "rl/common/protocol.py",
    "rl/rewards/metrics.py",
    "rl/common/shared_initialization.py",
    "rl/rewards/specification.py",
    "rl/af/advantage_flow.py",
    "rl/gaaf/gradient_aligned_advantage_flow.py",
    "rl/af/protocol.py",
    "rl/af/checkpoint.py",
    "rl/af/trainer.py",
    "rl/common/dataset_audit.py",
    "rl/common/dataset_validation.py",
    "rl/af/advantage_estimation.py",
    "rl/af/reward_evaluators.py",
    "tools/evaluate_dnsmos.py",
    "flowse/infer.py",
)

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _initial_evaluation_compatibility_signature(components: dict) -> dict:
    """Fields that must match before a completed step-0 evaluation is reused."""

    config = components["config"]
    evaluation = config["evaluation"]
    return {
        "conditioning": config["conditioning"],
        "rollout": {
            key: config["rollout"][key]
            for key in ("solver", "evaluation_nfe", "cfg_strength")
        },
        "normalization": config["normalization"],
        "evaluation": {
            key: evaluation[key]
            for key in (
                "policy",
                "latent_seed_base",
                "paired_metrics",
                "fidelity",
            )
        },
        "training_reward": config["training_reward"],
        "composite_reward_evaluators": config.get("composite_reward_evaluators"),
        "evaluation_manifest": components["evaluation_manifest"],
        "evaluation_audio_sha256": {
            key: components["audio_aggregate_sha256"][key]
            for key in ("evaluation_noisy", "evaluation_clean")
        },
        "checkpoint": components["checkpoint"],
        "flowse_inputs_sha256": components["flowse_inputs_sha256"],
        "vocoder": components["vocoder"],
        "dnsmos_sha256": components["dnsmos_sha256"],
        "locked_evaluator_config_sha256": components["locked_evaluator_config_sha256"],
        "evaluator_fingerprint": components["evaluator_fingerprint"],
    }


def initial_evaluation_cache_descriptor(
    config: dict, *, current_components: dict
) -> dict | None:
    """Fingerprint and authorize an immutable, compatible step-0 cache."""

    configured = config["evaluation"].get("initial_cache_report_path")
    if configured is None:
        return None
    report_path = Path(str(configured)).resolve()
    baseline_path = report_path.parent / "evaluation_noisy_baselines.json"
    protocol_path = report_path.parent / "protocol.json"
    for path in (report_path, baseline_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if report_path.name != "evaluation_step_000000.json":
        raise ValueError(
            "initial evaluation cache must be named evaluation_step_000000.json"
        )
    source_components = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_protocol_hash = sha256_json(source_components)
    if report_path.parent.name != source_protocol_hash:
        raise ValueError(
            "initial evaluation cache directory does not match its protocol hash"
        )
    source_signature = _initial_evaluation_compatibility_signature(source_components)
    current_signature = _initial_evaluation_compatibility_signature(current_components)
    if source_signature != current_signature:
        differing = sorted(
            key
            for key in source_signature.keys() | current_signature.keys()
            if source_signature.get(key) != current_signature.get(key)
        )
        raise ValueError(
            "initial evaluation cache is incompatible with the current protocol: "
            f"{differing}"
        )
    source_reward_calibration = source_components.get("reward_calibration")
    current_reward_calibration = current_components.get("reward_calibration")
    return {
        "authorization": "immutable_step0_raw_metrics_with_current_reward_recompute",
        "source_protocol_hash": source_protocol_hash,
        "source_protocol_path": str(protocol_path),
        "source_protocol_sha256": sha256_file(protocol_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "noisy_baseline_path": str(baseline_path),
        "noisy_baseline_sha256": sha256_file(baseline_path),
        "reward_calibration_changed": (
            source_reward_calibration != current_reward_calibration
        ),
        "source_reward_calibration": source_reward_calibration,
        "current_reward_calibration": current_reward_calibration,
    }


def training_source_names(root: str | Path = ".") -> list[str]:
    """Return every repository source file that can affect training execution."""

    root = Path(root).resolve()
    names = set(EXPERIMENT_SOURCE_NAMES)
    for directory in ("flowse/model", "flowse/loader"):
        source_root = root / directory
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        names.update(
            path.relative_to(root).as_posix()
            for path in source_root.rglob("*.py")
            if path.is_file()
        )
    return sorted(names)


def source_fingerprint(root: str | Path = ".") -> dict[str, str]:
    root = Path(root).resolve()
    output = {}
    for name in training_source_names(root):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = sha256_file(path)
    return output


def runtime_environment_fingerprint() -> dict:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "missing"
    torch_runtime = {}
    try:
        import torch

        torch_runtime = {
            "torch_version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
        }
    except ImportError:
        torch_runtime = {
            "torch_version": "missing",
            "cuda_build": None,
            "cudnn_version": None,
        }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": torch_runtime,
        "determinism_environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def verify_execution_dependencies(
    frozen_components: dict, root: str | Path = "."
) -> dict:
    """Report source/runtime drift without blocking training or evaluation."""

    root = Path(root).resolve()
    recorded_sources = dict(frozen_components.get("source_sha256") or {})
    current_sources = {}
    for name in sorted(recorded_sources):
        path = root / name
        current_sources[name] = sha256_file(path) if path.is_file() else None
    required_sources = set(training_source_names(root))
    missing_coverage = sorted(required_sources - set(recorded_sources))
    recorded_environment = frozen_components.get("runtime_environment")
    current_environment = runtime_environment_fingerprint()
    criteria = {
        "protocol_schema_v2_or_newer": int(frozen_components.get("schema_version", 0))
        >= 2,
        "source_hashes_match": bool(recorded_sources)
        and current_sources == recorded_sources,
        "source_coverage_complete": not missing_coverage,
        "runtime_environment_recorded": isinstance(recorded_environment, dict),
        "runtime_environment_matches": recorded_environment == current_environment,
    }
    return {
        # Source and runtime hashes are retained only as optional provenance.
        # They deliberately do not gate resume or evaluation.
        "passed": True,
        "enforcement_disabled": True,
        "diagnostic_match": all(criteria.values()),
        "criteria": criteria,
        "missing_source_coverage": missing_coverage,
        "recorded_source_sha256": recorded_sources,
        "current_source_sha256": current_sources,
        "recorded_runtime_environment": recorded_environment,
        "current_runtime_environment": current_environment,
    }


def strict_manifest(path: str | Path) -> dict[str, str]:
    path = Path(path)

    def reject_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate utterance {key!r} in {path}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest is empty or not an object: {path}")
    if not all(
        isinstance(key, str) and isinstance(text, str) for key, text in value.items()
    ):
        raise ValueError(
            f"manifest must map utterance IDs to transcript strings: {path}"
        )
    return value


def validate_training_config(
    config: dict, *, validate_artifacts: bool = True
) -> None:
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    if conditioning.fingerprint() != {
        "mode": "wotext",
        "use_text": False,
        "drop_text": True,
    }:
        raise ValueError("speech AdvantageFlow is permanently audio-only")
    run = config["run"]
    mode = str(run["mode"])
    if mode not in {"smoke", "pilot"}:
        raise ValueError(
            "run.mode must be smoke or pilot; full training is not authorized"
        )
    if bool(run.get("authorizes_full_training", False)):
        raise ValueError("smoke/pilot config cannot authorize full training")
    if bool(run.get("require_cublas_workspace_config", False)):
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in {":4096:8", ":16:8"}:
            raise ValueError(
                "deterministic CUDA training requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    rollout = config["rollout"]
    if str(rollout["solver"]) != "euler" or float(rollout["cfg_strength"]) != 0.0:
        raise ValueError("speech AdvantageFlow requires Euler ODE and cfg_strength=0")
    if int(rollout["candidates_per_condition"]) < 2:
        raise ValueError("at least two ODE candidates per condition are required")
    training_nfe = training_nfe_spec(rollout)
    if int(rollout["evaluation_nfe"]) < 1:
        raise ValueError("NFE values must be positive")
    if int(run["conditions_per_step"]) < 1 or int(run["optimizer_steps"]) < 2:
        raise ValueError(
            "smoke/pilot needs at least one condition and two optimizer steps"
        )
    parallel = config.get("parallel_rollout", {"enabled": False})
    if not isinstance(parallel, dict):
        raise ValueError("parallel_rollout must be a mapping")
    if bool(parallel.get("enabled", False)):
        required_parallel = {
            "enabled",
            "world_size",
            "device_ids",
            "worker_startup_timeout_seconds",
            "worker_task_timeout_seconds",
        }
        if set(parallel) != required_parallel:
            raise ValueError(
                "enabled parallel_rollout must contain exactly "
                f"{sorted(required_parallel)}"
            )
        world_size = int(parallel["world_size"])
        device_ids = [int(value) for value in parallel["device_ids"]]
        if world_size not in {2, 4} or device_ids != list(range(world_size)):
            raise ValueError(
                "sharded rollout requires world_size 2 or 4 and contiguous "
                "logical device_ids starting at 0"
            )
        if int(run["conditions_per_step"]) % world_size != 0:
            raise ValueError(
                "conditions_per_step must divide evenly across rollout GPUs"
            )
        if float(parallel["worker_startup_timeout_seconds"]) <= 0.0:
            raise ValueError("rollout worker startup timeout must be positive")
        if float(parallel["worker_task_timeout_seconds"]) <= 0.0:
            raise ValueError("rollout worker task timeout must be positive")
    reservation = config.get("cuda_memory_reservation")
    if reservation is not None:
        required_reservation = {
            "enabled",
            "coordinator_target_reserved_gib",
            "worker_target_reserved_gib",
            "minimum_driver_free_gib",
            "allocation_chunk_gib",
            "require_target",
        }
        if not isinstance(reservation, dict) or set(reservation) != required_reservation:
            raise ValueError(
                "cuda_memory_reservation must contain exactly "
                f"{sorted(required_reservation)}"
            )
        if float(reservation["coordinator_target_reserved_gib"]) <= 0.0:
            raise ValueError("CUDA coordinator target must be positive")
        if float(reservation["worker_target_reserved_gib"]) < 0.0:
            raise ValueError("CUDA worker target must be non-negative")
        if float(reservation["minimum_driver_free_gib"]) < 0.0:
            raise ValueError("CUDA minimum_driver_free_gib must be non-negative")
        if float(reservation["allocation_chunk_gib"]) <= 0.0:
            raise ValueError("CUDA allocation_chunk_gib must be positive")
    lora = config["lora"]
    if lora.get("shared_initial_snapshot") is not None:
        validate_shared_lora_snapshot_spec(config)
    if float(lora["dropout"]) != 0.0:
        raise ValueError("LoRA dropout must be zero for rollout/training parity")
    loss = config["loss"]
    if loss["gamma_mode"] not in {"constant_1p1", "one_minus_advantage"}:
        raise ValueError("unsupported AdvantageFlow gamma mode")
    if bool(loss.get("paired_fidelity_anchor", False)):
        raise ValueError(
            "paper-aligned baseline must not include the custom fidelity anchor"
        )
    adaptive_microbatching = config.get("length_adaptive_microbatching")
    if adaptive_microbatching is not None:
        if not isinstance(adaptive_microbatching, dict) or set(
            adaptive_microbatching
        ) != set(LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING):
            raise ValueError(
                "length_adaptive_microbatching must contain exactly mode, "
                "reference_mel_frames, reference_batch_size, and minimum_size"
            )
        if str(adaptive_microbatching["mode"]) != "quadratic_mel_frame_budget":
            raise ValueError("unsupported length-adaptive microbatch mode")
        if int(adaptive_microbatching["reference_mel_frames"]) < 1:
            raise ValueError("adaptive reference_mel_frames must be positive")
        if int(adaptive_microbatching["reference_batch_size"]) < 1:
            raise ValueError("adaptive reference_batch_size must be positive")
        minimum_size = int(adaptive_microbatching["minimum_size"])
        maximum_size = min(
            int(loss["microbatch_size"]),
            int(rollout["candidates_per_condition"]),
        )
        if minimum_size < 1 or minimum_size > maximum_size:
            raise ValueError("adaptive minimum_size exceeds the logical K/microbatch")
    advantage = config["advantage"]
    if str(advantage["normalization"]) != "global_complete_LxK":
        raise ValueError("advantage normalization must be global_complete_LxK")
    advantage_mapping = str(advantage.get("mapping", "linear_clipped"))
    if advantage_mapping not in {"linear_clipped", "exponential_awr"}:
        raise ValueError(
            "advantage mapping must be linear_clipped or exponential_awr"
        )
    if float(advantage["clip"]) <= 0.0:
        raise ValueError("advantage clip must be positive")
    if float(advantage["minimum_global_scale"]) < 0.0:
        raise ValueError("advantage minimum_global_scale must be non-negative")
    advantage_temperature = float(advantage.get("temperature", 1.0))
    if not math.isfinite(advantage_temperature) or advantage_temperature <= 0.0:
        raise ValueError("advantage temperature must be finite and positive")
    optimizer = config["optimizer"]
    if str(optimizer["type"]) != "AdamW":
        raise ValueError("speech AdvantageFlow requires real persistent AdamW")
    if str(optimizer["schedule"]) != "linear_decay":
        raise ValueError("only the frozen linear_decay schedule is supported")
    schedule_total_steps = int(
        optimizer.get("schedule_total_steps", run["optimizer_steps"])
    )
    if schedule_total_steps < int(run["optimizer_steps"]):
        raise ValueError("schedule_total_steps cannot be shorter than optimizer_steps")
    condition_stride = int(
        config["data"].get("condition_stride_per_step", run["conditions_per_step"])
    )
    if condition_stride < int(run["conditions_per_step"]):
        raise ValueError(
            "condition_stride_per_step must be at least conditions_per_step"
        )
    preflight = config["data"].get("preflight")
    if preflight is not None:
        if not isinstance(preflight, dict):
            raise ValueError("data.preflight must be a mapping")
        lightweight_startup = str(run.get("startup_validation", "full")) == "lightweight"
        lightweight_required = {
            "expected_train_utterances",
            "expected_evaluation_utterances",
            "require_single_coverage",
        }
        full_required = lightweight_required | {
            "expected_sample_rate",
            "expected_channels",
            "full_decode",
        }
        required = lightweight_required if lightweight_startup else full_required
        allowed = (
            lightweight_required
            | {"allow_partial_coverage"}
            if lightweight_startup
            else full_required
            | {
                "allow_unlisted_audio",
                "allow_partial_coverage",
                CERTIFICATE_CONFIG_KEY,
            }
        )
        if not required.issubset(preflight) or not set(preflight).issubset(allowed):
            raise ValueError(
                "data.preflight must contain all required fields and only supported "
                f"optional fields: required={sorted(required)}, allowed={sorted(allowed)}"
            )
        if not lightweight_startup and not bool(preflight["full_decode"]):
            raise ValueError("data preflight must fully decode every WAV")
        expected_steps = (
            int(preflight["expected_train_utterances"])
            + int(run["conditions_per_step"])
            - 1
        ) // int(run["conditions_per_step"])
        require_single_coverage = bool(preflight["require_single_coverage"])
        allow_partial_coverage = bool(preflight.get("allow_partial_coverage", False))
        if require_single_coverage and allow_partial_coverage:
            raise ValueError(
                "data preflight cannot require single coverage and allow partial "
                "coverage simultaneously"
            )
        if require_single_coverage and int(run["optimizer_steps"]) != expected_steps:
            raise ValueError(
                "optimizer_steps must equal ceil(expected_train_utterances / "
                "conditions_per_step)"
            )
        if (
            not require_single_coverage
            and not allow_partial_coverage
            and int(run["optimizer_steps"]) < expected_steps
        ):
            raise ValueError(
                "multi-epoch optimizer_steps must cover the complete training manifest"
            )
    if not 0.0 <= float(config["ema"]["decay"]) < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")
    if int(config["artifacts"].get("audit_audio_candidates_per_step", 0)) < 0:
        raise ValueError("audit_audio_candidates_per_step must be non-negative")
    reward = resolve_training_reward(config, validate_artifacts=validate_artifacts)
    validate_reward_constraints(config.get("reward_constraints"))
    reward_constraints = config.get("reward_constraints")
    if isinstance(reward_constraints, dict) and bool(
        reward_constraints.get("enabled", False)
    ):
        if reward["name"] != FLOWSE_GRPO_COMPOSITE:
            raise ValueError(
                "enabled reward_constraints require the three-component composite reward"
            )
        component_metrics = {
            "dnsmos_ovrl",
            "eres2net_speaker_similarity",
            "speechbertscore",
        }
        configured_metrics = {
            str(spec["metric"])
            for spec in reward_constraints["constraints"].values()
        }
        if not configured_metrics.issubset(component_metrics):
            raise ValueError(
                "enabled reward_constraints may only use metrics emitted by the "
                "three-component composite reward"
            )
    fixed_fusion = config.get("fixed_fusion")
    gaaf = config.get("gaaf")
    projected_gaaf = config.get("projected_gaaf")
    marble = config.get("marble")
    if sum(
        value is not None
        for value in (fixed_fusion, gaaf, projected_gaaf, marble)
    ) > 1:
        raise ValueError(
            "configure only one of fixed_fusion, gaaf, projected_gaaf, or marble"
        )
    if fixed_fusion is not None:
        required_fixed_fusion = {"primary", "auxiliaries", "weights"}
        if not isinstance(fixed_fusion, dict) or not required_fixed_fusion.issubset(
            fixed_fusion
        ) or set(fixed_fusion) - required_fixed_fusion not in (
            {"validation_selection"},
            {"weight_source"},
        ):
            raise ValueError(
                "fixed_fusion must contain primary, auxiliaries, weights, and "
                "exactly one of validation_selection or weight_source"
            )
        if str(fixed_fusion["primary"]) != "dnsmos_ovrl":
            raise ValueError("fixed fusion primary reward must be DNSMOS OVRL")
        if list(fixed_fusion["auxiliaries"]) != [
            "eres2net_speaker_similarity",
            "speechbertscore",
        ]:
            raise ValueError(
                "fixed fusion auxiliaries must be ERes2Net and SpeechBERTScore"
            )
        weights = fixed_fusion["weights"]
        required_weights = {"dnsmos", "speaker", "speechbertscore"}
        if not isinstance(weights, dict) or set(weights) != required_weights:
            raise ValueError("fixed fusion weights must cover exactly D, S, and C")
        parsed_weights = {name: float(weights[name]) for name in required_weights}
        if parsed_weights["dnsmos"] != 1.0:
            raise ValueError("fixed fusion must use the identifiable anchor b_D=1")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in parsed_weights.values()
        ):
            raise ValueError("fixed fusion weights must be finite and non-negative")
        weight_source = fixed_fusion.get("weight_source")
        if weight_source is not None:
            required_source = {
                "method",
                "source_steps",
                "raw_means",
                "rounded_weights",
                "rounding_digits",
                "test_set_accessed",
            }
            if not isinstance(weight_source, dict) or set(weight_source) != required_source:
                raise ValueError(
                    "fixed fusion weight_source must fully record the GA-AF mean"
                )
            if str(weight_source["method"]) != "ga_af_training_step_raw_weight_mean":
                raise ValueError("unsupported fixed fusion weight source")
            steps = weight_source["source_steps"]
            if steps != {"first": 1, "last": 5000, "count": 5000}:
                raise ValueError("GA-AF mean must cover all 5000 training steps")
            raw_means = weight_source["raw_means"]
            rounded = weight_source["rounded_weights"]
            if not isinstance(raw_means, dict) or set(raw_means) != {"b_S", "b_C"}:
                raise ValueError("GA-AF raw means must contain b_S and b_C")
            if not isinstance(rounded, dict) or set(rounded) != {"b_S", "b_C"}:
                raise ValueError("rounded fixed weights must contain b_S and b_C")
            digits = int(weight_source["rounding_digits"])
            if digits != 4:
                raise ValueError("GA-AF mean weights must be rounded to four decimals")
            for name in ("b_S", "b_C"):
                raw_value = float(raw_means[name])
                rounded_value = float(rounded[name])
                if not math.isfinite(raw_value) or raw_value < 0.0:
                    raise ValueError("GA-AF raw mean weights must be non-negative")
                if rounded_value != round(raw_value, digits):
                    raise ValueError("recorded Fixed-fusion rounding is inconsistent")
            if parsed_weights["speaker"] != float(rounded["b_S"]) or parsed_weights[
                "speechbertscore"
            ] != float(rounded["b_C"]):
                raise ValueError("runtime weights must equal the rounded GA-AF means")
            if weight_source["test_set_accessed"] is not False:
                raise ValueError("Fixed-fusion weight construction must not use test data")
            selection = None
        else:
            selection = fixed_fusion["validation_selection"]
        if selection is None:
            if reward["name"] != FLOWSE_GRPO_COMPOSITE:
                raise ValueError(
                    "fixed fusion requires the frozen three-component reward"
                )
            selection_status = None
        else:
            selection_status = str(selection.get("status"))
        if selection is None:
            pass
        else:
            common_selection = {"status", "split", "manifest", "search_space"}
            if not isinstance(selection, dict) or not common_selection.issubset(selection):
                raise ValueError(
                    "fixed fusion validation_selection must contain status, split, "
                    "manifest, and search_space"
                )
            if str(selection["split"]) != "validation":
                raise ValueError("fixed fusion weights must be selected on validation")
            if str(selection["manifest"]) != str(config["data"]["evaluation_manifest"]):
                raise ValueError(
                    "fixed fusion selection manifest must equal data.evaluation_manifest"
                )
            search_space = selection["search_space"]
            if not isinstance(search_space, dict) or set(search_space) != {"b_S", "b_C"}:
                raise ValueError("fixed fusion search_space must contain b_S and b_C")
            grids = {}
            for name in ("b_S", "b_C"):
                values = search_space[name]
                if not isinstance(values, list) or not values:
                    raise ValueError(f"fixed fusion {name} search grid must be non-empty")
                grids[name] = [float(value) for value in values]
                if any(
                    not math.isfinite(value) or value < 0.0 for value in grids[name]
                ) or len(set(grids[name])) != len(grids[name]):
                    raise ValueError(
                        f"fixed fusion {name} search grid must contain unique finite "
                        "non-negative values"
                    )
        if selection_status is None:
            pass
        elif selection_status == "candidate":
            if set(selection) != common_selection | {"candidate"}:
                raise ValueError(
                    "fixed fusion candidate selection metadata has unsupported fields"
                )
            candidate = selection["candidate"]
            if not isinstance(candidate, dict) or set(candidate) != {"b_S", "b_C"}:
                raise ValueError("fixed fusion candidate must contain b_S and b_C")
            candidate_b_s = float(candidate["b_S"])
            candidate_b_c = float(candidate["b_C"])
            if candidate_b_s not in grids["b_S"] or candidate_b_c not in grids["b_C"]:
                raise ValueError("fixed fusion candidate must lie on the search grid")
            if parsed_weights["speaker"] != candidate_b_s or parsed_weights[
                "speechbertscore"
            ] != candidate_b_c:
                raise ValueError("fixed fusion runtime and candidate weights differ")
            if int(run["optimizer_steps"]) >= 5000:
                raise ValueError(
                    "an unselected fixed-fusion candidate cannot be used for the "
                    "formal 5k run"
                )
        elif selection_status == "selected":
            required_selected = common_selection | {
                "selected",
                "report_path",
                "report_sha256",
            }
            if set(selection) != required_selected:
                raise ValueError(
                    "selected fixed fusion metadata must add selected, report_path, "
                    "and report_sha256"
                )
            selected = selection["selected"]
            if not isinstance(selected, dict) or set(selected) != {"b_S", "b_C"}:
                raise ValueError("fixed fusion selected weights must contain b_S and b_C")
            selected_b_s = float(selected["b_S"])
            selected_b_c = float(selected["b_C"])
            if selected_b_s not in grids["b_S"] or selected_b_c not in grids["b_C"]:
                raise ValueError(
                    "fixed fusion selected weights must lie on the search grid"
                )
            if parsed_weights["speaker"] != selected_b_s or parsed_weights[
                "speechbertscore"
            ] != selected_b_c:
                raise ValueError(
                    "fixed fusion runtime weights must equal validation-selected b_S/b_C"
                )
            report_sha256 = str(selection["report_sha256"])
            if len(report_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in report_sha256
            ):
                raise ValueError(
                    "fixed fusion selection report_sha256 must be lowercase hex"
                )
            if validate_artifacts:
                report_path = Path(str(selection["report_path"]))
                if not report_path.is_file():
                    raise FileNotFoundError(report_path)
                if sha256_file(report_path) != report_sha256:
                    raise ValueError("fixed fusion selection report hash mismatch")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("split") != "validation":
                    raise ValueError(
                        "fixed fusion selection report is not validation-only"
                    )
                if str(report.get("validation_manifest")) != str(
                    selection["manifest"]
                ):
                    raise ValueError(
                        "fixed fusion config/report validation manifests differ"
                    )
                if report.get("test_set_accessed") is not False:
                    raise ValueError(
                        "fixed fusion selection report must deny test-set access"
                    )
                report_selected = report.get("selected", {})
                if float(report_selected.get("b_S", math.nan)) != selected_b_s or float(
                    report_selected.get("b_C", math.nan)
                ) != selected_b_c:
                    raise ValueError("fixed fusion config/report selected weights differ")
                if report.get("search_space") != search_space:
                    raise ValueError("fixed fusion config/report search spaces differ")
        else:
            raise ValueError(
                "fixed fusion validation_selection.status must be candidate or selected"
            )
        if reward["name"] != FLOWSE_GRPO_COMPOSITE:
            raise ValueError("fixed fusion requires the frozen three-component reward")
    if gaaf is not None:
        if not isinstance(gaaf, dict) or set(gaaf) != {
            "primary",
            "auxiliaries",
            "gradient_refresh_interval",
            "coefficient_ema_decay",
            "auxiliary_cap",
        }:
            raise ValueError("gaaf must contain exactly the frozen gate fields")
        if str(gaaf["primary"]) != "dnsmos_ovrl":
            raise ValueError("GA-AF primary reward must be DNSMOS OVRL")
        if list(gaaf["auxiliaries"]) != [
            "eres2net_speaker_similarity",
            "speechbertscore",
        ]:
            raise ValueError("GA-AF auxiliaries must be ERes2Net and SpeechBERTScore")
        if int(gaaf["gradient_refresh_interval"]) != 10:
            raise ValueError("GA-AF pilot requires gradient_refresh_interval=10")
        if float(gaaf["coefficient_ema_decay"]) != 0.7:
            raise ValueError("GA-AF pilot requires coefficient_ema_decay=0.7")
        if float(gaaf["auxiliary_cap"]) != 0.5:
            raise ValueError("GA-AF pilot requires auxiliary_cap=0.5")
        if reward["name"] != FLOWSE_GRPO_COMPOSITE:
            raise ValueError("GA-AF requires the frozen three-component composite")
    if projected_gaaf is not None:
        required_projected_gaaf = {
            "primary",
            "auxiliaries",
            "gradient_refresh_interval",
            "coefficient_ema_decay",
            "auxiliary_target_norm_ratio",
            "auxiliary_coefficient_cap",
            "projection_epsilon",
        }
        if (
            not isinstance(projected_gaaf, dict)
            or set(projected_gaaf) != required_projected_gaaf
        ):
            raise ValueError(
                "projected_gaaf must contain exactly primary, auxiliaries, "
                "gradient_refresh_interval, coefficient_ema_decay, "
                "auxiliary_target_norm_ratio, auxiliary_coefficient_cap, and "
                "projection_epsilon"
            )
        if str(projected_gaaf["primary"]) != "dnsmos_ovrl":
            raise ValueError("projected GA-AF primary reward must be DNSMOS OVRL")
        if list(projected_gaaf["auxiliaries"]) != [
            "eres2net_speaker_similarity",
            "speechbertscore",
        ]:
            raise ValueError(
                "projected GA-AF auxiliaries must be ERes2Net and SpeechBERTScore"
            )
        if int(projected_gaaf["gradient_refresh_interval"]) != 10:
            raise ValueError(
                "projected GA-AF requires gradient_refresh_interval=10"
            )
        if float(projected_gaaf["coefficient_ema_decay"]) != 0.7:
            raise ValueError("projected GA-AF requires coefficient_ema_decay=0.7")
        if float(projected_gaaf["auxiliary_target_norm_ratio"]) != 0.15:
            raise ValueError(
                "projected GA-AF requires auxiliary_target_norm_ratio=0.15"
            )
        if float(projected_gaaf["auxiliary_coefficient_cap"]) != 0.5:
            raise ValueError(
                "projected GA-AF requires auxiliary_coefficient_cap=0.5"
            )
        if (
            not math.isfinite(float(projected_gaaf["projection_epsilon"]))
            or float(projected_gaaf["projection_epsilon"]) != 1.0e-12
        ):
            raise ValueError("projected GA-AF requires projection_epsilon=1e-12")
        if reward["name"] != FLOWSE_GRPO_COMPOSITE:
            raise ValueError(
                "projected GA-AF requires the frozen three-component composite"
            )
    if marble is not None:
        required_marble = {
            "primary",
            "auxiliaries",
            "gradient_refresh_interval",
            "coefficient_ema_decay",
            "ovrl_preference",
            "norm_epsilon",
        }
        if not isinstance(marble, dict) or set(marble) != required_marble:
            raise ValueError(
                "marble must contain exactly primary, auxiliaries, "
                "gradient_refresh_interval, coefficient_ema_decay, "
                "ovrl_preference, and norm_epsilon"
            )
        if str(marble["primary"]) != "dnsmos_ovrl":
            raise ValueError("MARBLE primary reward must be DNSMOS OVRL")
        if list(marble["auxiliaries"]) != [
            "eres2net_speaker_similarity",
            "speechbertscore",
        ]:
            raise ValueError(
                "MARBLE auxiliaries must be ERes2Net and SpeechBERTScore"
            )
        if int(marble["gradient_refresh_interval"]) < 1:
            raise ValueError("MARBLE gradient_refresh_interval must be positive")
        if not 0.0 <= float(marble["coefficient_ema_decay"]) < 1.0:
            raise ValueError("MARBLE coefficient_ema_decay must lie in [0,1)")
        if not 0.0 <= float(marble["ovrl_preference"]) <= 1.0:
            raise ValueError("MARBLE ovrl_preference must lie in [0,1]")
        if not math.isfinite(float(marble["norm_epsilon"])) or float(
            marble["norm_epsilon"]
        ) <= 0.0:
            raise ValueError("MARBLE norm_epsilon must be finite and positive")
        if reward["name"] != FLOWSE_GRPO_COMPOSITE:
            raise ValueError("MARBLE requires the frozen three-component composite")
    if reward["name"] == DNSMOS_SPEAKER and not bool(
        config["evaluation"].get("fidelity", {}).get("enabled", False)
    ):
        raise ValueError("DNSMOS+speaker training reward requires fidelity evaluators")
    if reward["name"] == FLOWSE_GRPO_COMPOSITE:
        composite_to_validate = reward
    elif reward["name"] == DNSMOS_OVRL_RAW:
        composite_to_validate = reward.get("auxiliary_composite")
    else:
        composite_to_validate = None
    if composite_to_validate is not None:
        evaluators = config.get("composite_reward_evaluators")
        if not isinstance(evaluators, dict):
            raise ValueError(
                "FlowSE-GRPO composite reward requires composite_reward_evaluators"
            )
        if set(evaluators) != {"device", "speaker", "speechbertscore"}:
            raise ValueError(
                "composite_reward_evaluators must contain device, speaker, and speechbertscore"
            )
        speaker = evaluators["speaker"]
        speechbert = evaluators["speechbertscore"]
        if str(speaker.get("backend")) != "modelscope_speaker_verification":
            raise ValueError("composite speaker backend must be ModelScope ERes2Net")
        if str(speaker.get("model_id")) != "iic/speech_eres2net_sv_zh-cn_16k-common":
            raise ValueError("unexpected public ERes2Net model")
        if str(speaker.get("revision")) != "v1.0.5":
            raise ValueError("public ERes2Net revision must be v1.0.5")
        if str(speechbert.get("repo_id")) != "microsoft/wavlm-large":
            raise ValueError("SpeechBERTScore must use microsoft/wavlm-large")
        if int(speechbert.get("layer", -1)) != 14:
            raise ValueError("SpeechBERTScore must use WavLM-large layer 14")
    if reward["name"] == DNSMOS_OVRL_RAW:
        if "auxiliary_composite" not in reward:
            raise ValueError(
                "sequential raw-OVRL specialist must retain the auxiliary composite "
                "evaluators for an isolated, equal-query experiment"
            )
        evaluators = config.get("composite_reward_evaluators")
        if not isinstance(evaluators, dict):
            raise ValueError(
                "raw-OVRL specialist requires composite_reward_evaluators"
            )

    if mode == "pilot":
        observed = {
            "conditions_per_step": int(run["conditions_per_step"]),
            "candidates_per_condition": int(rollout["candidates_per_condition"]),
            "training_nfe": int(rollout["training_nfe"]),
            "lora_rank": int(lora["rank"]),
            "lora_alpha": float(lora["alpha"]),
            "lambda_reference": float(loss["lambda_reference"]),
        }
        experiment_kind = str(run.get("experiment_kind", "paper_baseline"))
        if experiment_kind == "controlled_l16_k8_baseline":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "controlled AF baseline requires L=16, K=8 and frozen "
                    f"non-L/K fields: expected={CONTROLLED_L16_K8_BASELINE}, "
                    f"got={observed}"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "controlled L16/K8 baseline requires 5000 optimizer/schedule steps"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "controlled L16/K8 baseline requires the validated "
                    "gradient-equivalent microbatch_size=8"
                )
            validate_controlled_baseline(config, method="advantageflow")
        elif experiment_kind in {
            "libritts_dns_l16_k8_baseline",
            "libritts_dns_l16_k8_sft20k_base",
            "libritts_dns_l16_k8_sft20k_mixed_nfe_7_to_10",
        }:
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "LibriTTS/DNS AF baseline requires L=16, K=8 and frozen "
                    f"non-L/K fields: expected={CONTROLLED_L16_K8_BASELINE}, "
                    f"got={observed}"
                )
            if experiment_kind == "libritts_dns_l16_k8_sft20k_mixed_nfe_7_to_10":
                if training_nfe != {
                    "configured": 10,
                    "minimum": 7,
                    "maximum": 10,
                    "mixed": True,
                    "scope": "group",
                    "seed_base": int(rollout["training_nfe_seed_base"]),
                }:
                    raise ValueError(
                        "mixed-NFE AF requires training NFE range 7..10, "
                        "group sampling, and a deterministic seed base"
                    )
                if str(rollout.get("training_nfe_schedule", "random_per_group")) != (
                    "balanced_per_step"
                ):
                    raise ValueError(
                        "the L16/K8 mixed-NFE protocol requires a balanced_per_step "
                        "schedule (with a random permutation per step)"
                    )
            elif training_nfe["mixed"]:
                raise ValueError(
                    "the fixed LibriTTS/DNS AF baseline cannot enable mixed training NFE"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "LibriTTS/DNS AF baseline requires 5000 optimizer/schedule steps"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "LibriTTS/DNS AF baseline requires validated microbatch_size=8"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "LibriTTS/DNS AF baseline requires the frozen length-adaptive "
                    "K execution policy"
                )
            if not bool(preflight and preflight.get("require_single_coverage", False)):
                raise ValueError(
                    "LibriTTS/DNS AF baseline requires one frozen exposure coverage"
                )
            if int(preflight["expected_train_utterances"]) != 80_000:
                raise ValueError(
                    "LibriTTS/DNS AF baseline requires exactly 80,000 exposures"
                )
        elif experiment_kind.startswith("fixed_fusion_validation_screen_"):
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "Fixed-fusion validation screen must retain the frozen L16/K8 "
                    "baseline"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "Fixed-fusion validation screen requires the frozen "
                    "length-adaptive K execution policy"
                )
            if int(run["optimizer_steps"]) != 250 or schedule_total_steps != 5000:
                raise ValueError(
                    "Fixed-fusion validation screen requires 250 updates on the "
                    "formal 5k LR schedule"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "Fixed-fusion validation screen requires microbatch_size=8"
                )
            if not isinstance(fixed_fusion, dict) or str(
                fixed_fusion["validation_selection"]["status"]
            ) != "candidate":
                raise ValueError(
                    "Fixed-fusion validation screen requires candidate metadata"
                )
            if not bool(preflight and preflight.get("allow_partial_coverage", False)):
                raise ValueError(
                    "Fixed-fusion validation screen must allow partial train coverage"
                )
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 250
                or int(evaluation["primary_checkpoint_step"]) != 250
                or int(config["artifacts"]["checkpoint_interval"]) != 250
                or int(rollout["evaluation_nfe"]) != 32
            ):
                raise ValueError(
                    "Fixed-fusion screen must evaluate once at step 250 with NFE=32"
                )
        elif experiment_kind == "fixed_fusion_l16_k8_sft20k_0_to_5000":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "formal Fixed fusion must retain the frozen L16/K8 baseline"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "formal Fixed fusion requires the frozen length-adaptive K policy"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError("formal Fixed fusion requires exactly 5k updates")
            if int(loss["microbatch_size"]) != 8:
                raise ValueError("formal Fixed fusion requires microbatch_size=8")
            fixed_source_is_selected = isinstance(fixed_fusion, dict) and (
                str(fixed_fusion.get("validation_selection", {}).get("status"))
                == "selected"
                or str(fixed_fusion.get("weight_source", {}).get("method"))
                == "ga_af_training_step_raw_weight_mean"
            )
            if not fixed_source_is_selected:
                raise ValueError(
                    "formal Fixed fusion requires frozen, provenance-recorded weights"
                )
            if not bool(preflight and preflight.get("require_single_coverage", False)):
                raise ValueError(
                    "formal Fixed fusion requires one frozen exposure coverage"
                )
            if int(preflight["expected_train_utterances"]) != 80_000:
                raise ValueError("formal Fixed fusion requires 80,000 exposures")
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 1000
                or int(evaluation["primary_checkpoint_step"]) != 5000
                or int(config["artifacts"]["checkpoint_interval"]) != 250
                or int(rollout["evaluation_nfe"]) != 32
            ):
                raise ValueError(
                    "formal Fixed fusion must checkpoint every 250 updates and "
                    "evaluate every 1000 updates through step 5000 at NFE=32"
                )
        elif experiment_kind == "libritts_dns_gaaf_l16_k8_sft20k_0_to_5000":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires L=16, K=8 and frozen "
                    f"non-L/K fields: expected={CONTROLLED_L16_K8_BASELINE}, "
                    f"got={observed}"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires the frozen length-adaptive "
                    "K execution policy"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires 5000 optimizer/schedule steps"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires validated microbatch_size=8"
                )
            if not isinstance(gaaf, dict):
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires an gaaf definition"
                )
            if not bool(preflight and preflight.get("require_single_coverage", False)):
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires one frozen exposure coverage"
                )
            if int(preflight["expected_train_utterances"]) != 80_000:
                raise ValueError(
                    "step-0 LibriTTS GA-AF requires exactly 80,000 exposures"
                )
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 1000
                or int(evaluation["primary_checkpoint_step"]) != 5000
                or int(config["artifacts"]["checkpoint_interval"]) != 250
            ):
                raise ValueError(
                    "step-0 LibriTTS GA-AF must checkpoint every 250 steps and "
                    "evaluate at 1000-step intervals through endpoint 5000"
                )
        elif experiment_kind == (
            "libritts_dns_projected_gaaf_l16_k8_sft20k_0_to_5000"
        ):
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires L=16, K=8 and "
                    "frozen non-L/K fields: "
                    f"expected={CONTROLLED_L16_K8_BASELINE}, got={observed}"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires the frozen "
                    "length-adaptive K execution policy"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires 5000 "
                    "optimizer/schedule steps"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires validated "
                    "microbatch_size=8"
                )
            if not isinstance(projected_gaaf, dict):
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires a "
                    "projected_gaaf definition"
                )
            if not bool(preflight and preflight.get("require_single_coverage", False)):
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires one frozen "
                    "exposure coverage"
                )
            if int(preflight["expected_train_utterances"]) != 80_000:
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF requires exactly 80,000 "
                    "exposures"
                )
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 1000
                or int(evaluation["primary_checkpoint_step"]) != 5000
                or int(config["artifacts"]["checkpoint_interval"]) != 250
            ):
                raise ValueError(
                    "step-0 LibriTTS projected GA-AF must checkpoint every 250 "
                    "steps and evaluate at 1000-step intervals through "
                    "endpoint 5000"
                )
        elif experiment_kind == (
            "libritts_dns_ovrl_preferred_marble_l16_k8_sft20k_0_to_5000"
        ):
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "step-0 LibriTTS MARBLE requires L=16, K=8 and frozen "
                    f"non-L/K fields: expected={CONTROLLED_L16_K8_BASELINE}, "
                    f"got={observed}"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "step-0 LibriTTS MARBLE requires the frozen length-adaptive "
                    "K execution policy"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "step-0 LibriTTS MARBLE requires 5000 optimizer/schedule steps"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "step-0 LibriTTS MARBLE requires validated microbatch_size=8"
                )
            if not isinstance(marble, dict):
                raise ValueError("step-0 LibriTTS MARBLE requires a marble definition")
            if not bool(preflight and preflight.get("require_single_coverage", False)):
                raise ValueError(
                    "step-0 LibriTTS MARBLE requires one frozen exposure coverage"
                )
            if int(preflight["expected_train_utterances"]) != 80_000:
                raise ValueError(
                    "step-0 LibriTTS MARBLE requires exactly 80,000 exposures"
                )
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 1000
                or int(evaluation["primary_checkpoint_step"]) != 5000
                or int(config["artifacts"]["checkpoint_interval"]) != 250
            ):
                raise ValueError(
                    "step-0 LibriTTS MARBLE must checkpoint every 250 steps and "
                    "evaluate at 1000-step intervals through endpoint 5000"
                )
        elif experiment_kind == "paper_baseline":
            if observed != PAPER_BASELINE:
                raise ValueError(
                    "pilot paper-baseline fields changed: "
                    f"expected={PAPER_BASELINE}, got={observed}"
                )
        elif experiment_kind == "long_horizon_paper_baseline":
            if observed != PAPER_BASELINE:
                raise ValueError(
                    "long-horizon paper baseline fields changed: "
                    f"expected={PAPER_BASELINE}, got={observed}"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "long-horizon paper baseline requires 5000 optimizer/schedule steps"
                )
        elif experiment_kind == "long_horizon_3000step_paper_baseline":
            if observed != PAPER_BASELINE:
                raise ValueError(
                    "3000-step paper baseline fields changed: "
                    f"expected={PAPER_BASELINE}, got={observed}"
                )
            if int(run["optimizer_steps"]) != 3000 or schedule_total_steps != 3000:
                raise ValueError(
                    "3000-step paper baseline requires 3000 optimizer/schedule steps"
                )
        elif experiment_kind == "training_nfe32_ablation":
            expected = {**PAPER_BASELINE, "training_nfe": 32}
            if observed != expected:
                raise ValueError(
                    "NFE=32 ablation changed a non-NFE paper baseline field: "
                    f"expected={expected}, got={observed}"
                )
            if int(run["optimizer_steps"]) != 5000 or schedule_total_steps != 5000:
                raise ValueError(
                    "NFE=32 long-horizon ablation requires 5000 optimizer/schedule steps"
                )
        elif experiment_kind == "lk_budget_ablation":
            invariant_names = (
                "training_nfe",
                "lora_rank",
                "lora_alpha",
                "lambda_reference",
            )
            if any(observed[name] != PAPER_BASELINE[name] for name in invariant_names):
                raise ValueError("L/K ablation changed a non-L/K paper baseline field")
            observed_budget = (
                observed["conditions_per_step"] * observed["candidates_per_condition"]
            )
            baseline_budget = (
                PAPER_BASELINE["conditions_per_step"]
                * PAPER_BASELINE["candidates_per_condition"]
            )
            if observed_budget != baseline_budget:
                raise ValueError(
                    "L/K ablation must preserve the 128-endpoint step budget"
                )
            if condition_stride != PAPER_BASELINE["conditions_per_step"]:
                raise ValueError(
                    "L/K ablation conditions must be nested in the L=32 baseline"
                )
            if schedule_total_steps != 100:
                raise ValueError(
                    "L/K ablation must retain the A-run 100-step LR schedule"
                )
        elif experiment_kind == "endpoint_budget_256_ablation":
            expected = {
                **PAPER_BASELINE,
                "candidates_per_condition": 8,
            }
            if observed != expected:
                raise ValueError(
                    "256-endpoint ablation must use L32/K8 and retain all "
                    "non-budget paper-baseline fields"
                )
            if int(run["optimizer_steps"]) != 100 or schedule_total_steps != 100:
                raise ValueError(
                    "256-endpoint ablation requires 100 optimizer/schedule steps"
                )
            if condition_stride != PAPER_BASELINE["conditions_per_step"]:
                raise ValueError(
                    "256-endpoint ablation must retain the L=32 condition stride"
                )
            observed_budget = (
                observed["conditions_per_step"] * observed["candidates_per_condition"]
            )
            baseline_budget = (
                PAPER_BASELINE["conditions_per_step"]
                * PAPER_BASELINE["candidates_per_condition"]
            )
            if observed_budget != 2 * baseline_budget:
                raise ValueError(
                    "256-endpoint ablation must double the 128-endpoint step budget"
                )
        elif experiment_kind == "k_candidate_ablation":
            invariant_names = (
                "conditions_per_step",
                "training_nfe",
                "lora_rank",
                "lora_alpha",
                "lambda_reference",
            )
            if any(observed[name] != PAPER_BASELINE[name] for name in invariant_names):
                raise ValueError("K ablation changed a non-K paper baseline field")
            if observed["candidates_per_condition"] != 8:
                raise ValueError(
                    "K ablation requires exactly 8 candidates per condition"
                )
            if int(run["optimizer_steps"]) != 20:
                raise ValueError(
                    "K=8 short ablation requires exactly 20 optimizer steps"
                )
            if condition_stride != PAPER_BASELINE["conditions_per_step"]:
                raise ValueError("K ablation must retain the L=32 condition stride")
            if schedule_total_steps != 100:
                raise ValueError(
                    "20-step K ablation must retain the A-run 100-step LR schedule"
                )
        elif experiment_kind == "k10_candidate_ablation":
            invariant_names = (
                "conditions_per_step",
                "training_nfe",
                "lora_rank",
                "lora_alpha",
                "lambda_reference",
            )
            if any(observed[name] != PAPER_BASELINE[name] for name in invariant_names):
                raise ValueError("K=10 ablation changed a non-K paper baseline field")
            if observed["candidates_per_condition"] != 10:
                raise ValueError(
                    "K=10 ablation requires exactly 10 candidates per condition"
                )
            if int(run["optimizer_steps"]) != 20:
                raise ValueError(
                    "K=10 short ablation requires exactly 20 optimizer steps"
                )
            if condition_stride != PAPER_BASELINE["conditions_per_step"]:
                raise ValueError("K=10 ablation must retain the L=32 condition stride")
            if schedule_total_steps != 100:
                raise ValueError(
                    "20-step K=10 ablation must retain the A-run 100-step LR schedule"
                )
        elif experiment_kind == "candidate_budget_ablation":
            invariant_names = (
                "conditions_per_step",
                "training_nfe",
                "lora_rank",
                "lora_alpha",
                "lambda_reference",
            )
            if any(observed[name] != PAPER_BASELINE[name] for name in invariant_names):
                raise ValueError(
                    "candidate-budget ablation changed a non-K paper baseline field"
                )
            if observed["candidates_per_condition"] != 8:
                raise ValueError(
                    "candidate-budget ablation requires exactly 8 candidates per condition"
                )
            if int(run["optimizer_steps"]) != 50 or schedule_total_steps != 50:
                raise ValueError(
                    "candidate-budget ablation requires a 50-step run and LR horizon"
                )
            if condition_stride != PAPER_BASELINE["conditions_per_step"]:
                raise ValueError(
                    "candidate-budget ablation must retain the L=32 condition stride"
                )
            observed_budget = (
                observed["conditions_per_step"]
                * observed["candidates_per_condition"]
                * int(run["optimizer_steps"])
            )
            baseline_budget = (
                PAPER_BASELINE["conditions_per_step"]
                * PAPER_BASELINE["candidates_per_condition"]
                * 100
            )
            if observed_budget != baseline_budget:
                raise ValueError(
                    "candidate-budget ablation must match the A-run 12800 endpoints"
                )
        elif experiment_kind == "learning_rate_screen_l16_k8_300step":
            expected = {
                **PAPER_BASELINE,
                "conditions_per_step": 16,
                "candidates_per_condition": 8,
            }
            if observed != expected:
                raise ValueError(
                    "learning-rate screen must use L16/K8 and retain all "
                    "non-L/K paper-baseline fields"
                )
            if int(run["optimizer_steps"]) != 300 or schedule_total_steps != 5000:
                raise ValueError(
                    "learning-rate screen requires 300 optimizer steps on the "
                    "planned 5000-step LR horizon"
                )
            if not bool(preflight and preflight.get("allow_partial_coverage", False)):
                raise ValueError(
                    "300-step learning-rate screen must explicitly allow partial "
                    "training-manifest coverage"
                )
            if float(optimizer["learning_rate"]) not in {1.0e-4, 2.0e-4}:
                raise ValueError("learning-rate screen supports only 1e-4 or 2e-4")
        elif experiment_kind == "advantage_mapping_screen_l16_k8_300step":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "advantage-mapping screen must use the frozen L16/K8 baseline"
                )
            if int(run["optimizer_steps"]) != 300 or schedule_total_steps != 5000:
                raise ValueError(
                    "advantage-mapping screen requires 300 optimizer steps on the "
                    "planned 5000-step LR horizon"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "advantage-mapping screen requires validated microbatch_size=8"
                )
            if not bool(preflight and preflight.get("allow_partial_coverage", False)):
                raise ValueError(
                    "300-step advantage-mapping screen must explicitly allow "
                    "partial training-manifest coverage"
                )
            if float(optimizer["learning_rate"]) != 2.0e-4:
                raise ValueError("advantage-mapping screen requires learning_rate=2e-4")
            if advantage_mapping == "linear_clipped":
                if float(advantage["clip"]) != 1.0:
                    raise ValueError(
                        "linear advantage screen requires the frozen clip=1"
                    )
            elif advantage_temperature not in {0.5, 1.0, 2.0}:
                raise ValueError("AWR screen temperature must be 0.5, 1.0, or 2.0")
            validate_controlled_baseline(config, method="advantageflow")
        elif experiment_kind == "sequential_ovrl_specialist_l16_k8_250step":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "sequential OVRL specialist must retain the frozen L16/K8 baseline"
                )
            branch = config.get("branch")
            if not isinstance(branch, dict) or set(branch) != {
                "source_run_dir",
                "source_step",
                "global_step_offset",
            }:
                raise ValueError(
                    "sequential OVRL specialist requires an exact branch source"
                )
            if int(branch["source_step"]) != 4000 or int(
                branch["global_step_offset"]
            ) != 4000:
                raise ValueError("sequential OVRL specialist must branch at step 4000")
            if int(run["optimizer_steps"]) != 250 or schedule_total_steps != 5000:
                raise ValueError(
                    "sequential OVRL specialist requires 250 local steps on the "
                    "original 5000-step LR horizon"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "sequential OVRL specialist requires validated microbatch_size=8"
                )
            if reward["name"] != DNSMOS_OVRL_RAW:
                raise ValueError("sequential specialist must optimize raw DNSMOS OVRL")
            if not bool(preflight and preflight.get("allow_partial_coverage", False)):
                raise ValueError(
                    "250-step sequential specialist must allow partial manifest coverage"
                )
        elif experiment_kind == "gaaf_l16_k8_250step":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError("GA-AF pilot must retain the frozen L16/K8 baseline")
            branch = config.get("branch")
            if not isinstance(branch, dict) or set(branch) != {
                "source_run_dir",
                "source_step",
                "global_step_offset",
            }:
                raise ValueError("GA-AF pilot requires an exact step-4000 branch")
            if int(branch["source_step"]) != 4000 or int(
                branch["global_step_offset"]
            ) != 4000:
                raise ValueError("GA-AF pilot must branch at global step 4000")
            if int(run["optimizer_steps"]) != 250 or schedule_total_steps != 5000:
                raise ValueError(
                    "GA-AF pilot requires 250 local steps on the original LR horizon"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError("GA-AF pilot requires validated microbatch_size=8")
            if not isinstance(gaaf, dict):
                raise ValueError("GA-AF experiment_kind requires an gaaf definition")
            if not bool(preflight and preflight.get("allow_partial_coverage", False)):
                raise ValueError("250-step GA-AF pilot must allow partial coverage")
        elif experiment_kind == "gaaf_l16_k8_1000step":
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "formal GA-AF branch must retain the frozen L16/K8 baseline"
                )
            branch = config.get("branch")
            if not isinstance(branch, dict) or set(branch) != {
                "source_run_dir",
                "source_step",
                "global_step_offset",
            }:
                raise ValueError("formal GA-AF branch requires an exact step-4000 source")
            if int(branch["source_step"]) != 4000 or int(
                branch["global_step_offset"]
            ) != 4000:
                raise ValueError("formal GA-AF branch must start at global step 4000")
            if int(run["optimizer_steps"]) != 1000 or schedule_total_steps != 5000:
                raise ValueError(
                    "formal GA-AF branch requires 1000 local steps on the original "
                    "5000-step LR horizon"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "formal GA-AF branch requires validated microbatch_size=8"
                )
            if not isinstance(gaaf, dict):
                raise ValueError("formal GA-AF experiment requires an gaaf definition")
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 250
                or int(evaluation["primary_checkpoint_step"]) != 1000
                or int(config["artifacts"]["checkpoint_interval"]) != 250
            ):
                raise ValueError(
                    "formal GA-AF branch must evaluate and checkpoint every 250 local "
                    "steps, with local step 1000 as the endpoint"
                )
            if bool(preflight and preflight.get("allow_partial_coverage", False)):
                raise ValueError(
                    "formal 1000-step GA-AF branch must not allow partial data coverage"
                )
        elif experiment_kind in {
            "libritts_dns_gaaf_l16_k8_sft20k_3000_to_5000",
            "libritts_dns_gaaf_l16_k8_sft20k_4000_to_5000",
        }:
            branch_design = {
                "libritts_dns_gaaf_l16_k8_sft20k_3000_to_5000": (3000, 2000),
                "libritts_dns_gaaf_l16_k8_sft20k_4000_to_5000": (4000, 1000),
            }
            required_source_step, required_local_steps = branch_design[
                experiment_kind
            ]
            if observed != CONTROLLED_L16_K8_BASELINE:
                raise ValueError(
                    "LibriTTS GA-AF branch must retain the frozen L16/K8 baseline"
                )
            if adaptive_microbatching != LIBRITTS_LENGTH_ADAPTIVE_MICROBATCHING:
                raise ValueError(
                    "LibriTTS GA-AF requires the frozen length-adaptive K execution"
                )
            branch = config.get("branch")
            if not isinstance(branch, dict) or set(branch) != {
                "source_run_dir",
                "source_step",
                "global_step_offset",
            }:
                raise ValueError(
                    "LibriTTS GA-AF requires an exact frozen AF branch source"
                )
            if int(branch["source_step"]) != required_source_step or int(
                branch["global_step_offset"]
            ) != required_source_step:
                raise ValueError(
                    "LibriTTS GA-AF branch source/global offset must equal "
                    f"{required_source_step}"
                )
            if (
                int(run["optimizer_steps"]) != required_local_steps
                or schedule_total_steps != 5000
            ):
                raise ValueError(
                    f"LibriTTS GA-AF requires {required_local_steps} local updates "
                    "on the original 5000-step LR horizon"
                )
            if int(loss["microbatch_size"]) != 8:
                raise ValueError(
                    "LibriTTS GA-AF requires validated microbatch_size=8"
                )
            if not isinstance(gaaf, dict):
                raise ValueError("LibriTTS GA-AF requires an gaaf definition")
            evaluation = config["evaluation"]
            if (
                int(evaluation["interval_steps"]) != 250
                or int(evaluation["primary_checkpoint_step"])
                != required_local_steps
                or int(config["artifacts"]["checkpoint_interval"]) != 250
            ):
                raise ValueError(
                    "LibriTTS GA-AF must evaluate and checkpoint every 250 local "
                    f"steps, with local step {required_local_steps} as the endpoint"
                )
            if not bool(
                preflight and preflight.get("allow_partial_coverage", False)
            ):
                raise ValueError(
                    "LibriTTS GA-AF must declare local partial-manifest coverage"
                )
            expected_utterances = int(preflight["expected_train_utterances"])
            global_end_step = int(branch["source_step"]) + int(
                run["optimizer_steps"]
            )
            if (
                global_end_step * int(run["conditions_per_step"])
                != expected_utterances
            ):
                raise ValueError(
                    "LibriTTS GA-AF parent plus branch must exactly cover the frozen "
                    "training exposure schedule"
                )
        else:
            raise ValueError(f"unsupported pilot experiment_kind: {experiment_kind}")
        decision = config.get("pilot_decision")
        if not isinstance(decision, dict):
            raise ValueError("pilot must preregister pilot_decision thresholds")
        if float(decision["dnsmos_ovrl_minimum_gain"]) <= 0:
            raise ValueError("pilot DNSMOS OVRL minimum gain must be positive")


def build_training_protocol(
    config: dict,
    *,
    bundle,
    evaluator_fingerprint: dict | None = None,
    shared_initial_lora_snapshot: dict | None = None,
) -> dict:
    """Build the immutable scientific protocol for a training run."""

    # Public smoke runs validate the runnable algorithm and data schema, but
    # do not require formal calibration/source-artifact provenance.  Pilot and
    # formal configs keep the complete artifact checks.
    validate_training_config(
        config, validate_artifacts=str(config["run"].get("mode")) != "smoke"
    )
    root = Path(".").resolve()
    train_manifest_path = Path(config["data"]["train_manifest"])
    eval_manifest_path = Path(config["data"]["evaluation_manifest"])
    train_manifest = strict_manifest(train_manifest_path)
    evaluation_manifest = strict_manifest(eval_manifest_path)
    overlap = set(train_manifest) & set(evaluation_manifest)
    if overlap:
        raise ValueError(f"train/evaluation manifests overlap: {sorted(overlap)[:3]}")

    lightweight_startup = (
        str(config.get("run", {}).get("startup_validation", "full"))
        == "lightweight"
    )
    source_hashes = {} if lightweight_startup else source_fingerprint(root)

    dnsmos_root = Path(config["dnsmos_official_dir"])
    dnsmos_files = {
        "script": dnsmos_root / "dnsmos_local.py",
        "p808": dnsmos_root / "DNSMOS/model_v8.onnx",
        "primary": dnsmos_root / "DNSMOS/sig_bak_ovr.onnx",
    }
    for path in dnsmos_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    def aggregate_audio_hash(manifest: dict[str, str], directory: Path) -> str:
        digest = hashlib.sha256()
        for utterance in sorted(manifest):
            audio = directory / f"{utterance}.wav"
            if not audio.is_file():
                raise FileNotFoundError(audio)
            digest.update(utterance.encode("utf-8"))
            digest.update(bytes.fromhex(sha256_file(audio)))
        return digest.hexdigest()

    noisy_dir = Path(config["data"]["noisy_dir"])
    clean_dir = Path(config["data"]["clean_dir"])
    preflight_config = config["data"].get("preflight")
    dataset_preflight = None
    dataset_audit_certificate = None
    audio_hashes = None
    if preflight_config is not None and lightweight_startup:
        expected_train = int(preflight_config["expected_train_utterances"])
        expected_evaluation = int(
            preflight_config["expected_evaluation_utterances"]
        )
        if len(train_manifest) != expected_train:
            raise ValueError(
                f"train manifest count differs: {len(train_manifest)} != {expected_train}"
            )
        if len(evaluation_manifest) != expected_evaluation:
            raise ValueError(
                "evaluation manifest count differs: "
                f"{len(evaluation_manifest)} != {expected_evaluation}"
            )
        for directory in (noisy_dir, clean_dir):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
        expected_exposures = int(config["run"]["conditions_per_step"]) * int(
            config["run"]["optimizer_steps"]
        )
        if (
            bool(preflight_config.get("require_single_coverage", False))
            and len(train_manifest) != expected_exposures
        ):
            raise ValueError(
                "single-coverage schedule differs from the train manifest: "
                f"{expected_exposures} != {len(train_manifest)}"
            )
        dataset_preflight = {
            "passed": True,
            "mode": "lightweight_manifest_only",
            "observed": {
                "train_utterances": len(train_manifest),
                "evaluation_utterances": len(evaluation_manifest),
            },
            "coverage": {
                "conditions_per_step": int(config["run"]["conditions_per_step"]),
                "optimizer_steps": int(config["run"]["optimizer_steps"]),
            },
        }
        print(
            "Lightweight dataset startup check: PASS "
            "(WAV scan and audit/hash certificate skipped)",
            flush=True,
        )
    elif preflight_config is not None:
        certificate_path = preflight_config.get(CERTIFICATE_CONFIG_KEY)
        if certificate_path is not None:
            frozen_audit = validate_dataset_audit_certificate(
                certificate_path=certificate_path,
                train_manifest_path=train_manifest_path,
                evaluation_manifest_path=eval_manifest_path,
                train_utterances=len(train_manifest),
                evaluation_utterances=len(evaluation_manifest),
                noisy_dir=noisy_dir,
                clean_dir=clean_dir,
                preflight_config=preflight_config,
                conditions_per_step=int(config["run"]["conditions_per_step"]),
                optimizer_steps=int(config["run"]["optimizer_steps"]),
            )
            dataset_preflight = frozen_audit["dataset_preflight"]
            audio_hashes = frozen_audit["audio_aggregate_sha256"]
            dataset_audit_certificate = frozen_audit["certificate"]
            print(
                "Frozen dataset audit certificate: PASS "
                "(full WAV decode/hash skipped)",
                flush=True,
            )
        else:
            dataset_preflight = validate_paired_audio_dataset(
                train_manifest=train_manifest,
                evaluation_manifest=evaluation_manifest,
                noisy_dir=noisy_dir,
                clean_dir=clean_dir,
                expected_train_utterances=int(
                    preflight_config["expected_train_utterances"]
                ),
                expected_evaluation_utterances=int(
                    preflight_config["expected_evaluation_utterances"]
                ),
                expected_sample_rate=int(preflight_config["expected_sample_rate"]),
                expected_channels=int(preflight_config["expected_channels"]),
                conditions_per_step=int(config["run"]["conditions_per_step"]),
                optimizer_steps=int(config["run"]["optimizer_steps"]),
                full_decode=bool(preflight_config["full_decode"]),
                require_single_coverage=bool(
                    preflight_config["require_single_coverage"]
                ),
                allow_partial_coverage=bool(
                    preflight_config.get("allow_partial_coverage", False)
                ),
                allow_unlisted_audio=bool(
                    preflight_config.get("allow_unlisted_audio", False)
                ),
            )
    if audio_hashes is None and not lightweight_startup:
        audio_hashes = {
            "training_noisy": aggregate_audio_hash(train_manifest, noisy_dir),
            "training_clean": aggregate_audio_hash(train_manifest, clean_dir),
            "evaluation_noisy": aggregate_audio_hash(evaluation_manifest, noisy_dir),
            "evaluation_clean": aggregate_audio_hash(evaluation_manifest, clean_dir),
        }
    if audio_hashes is None:
        audio_hashes = {
            "training_noisy": None,
            "training_clean": None,
            "evaluation_noisy": None,
            "evaluation_clean": None,
        }
    locked_evaluator_path = (
        config["evaluation"].get("fidelity", {}).get("source_locked_config")
    )
    evaluator_config_hash = None
    if config["evaluation"].get("fidelity", {}).get("enabled", False):
        if not locked_evaluator_path or not Path(locked_evaluator_path).is_file():
            raise FileNotFoundError(locked_evaluator_path)
        evaluator_config_hash = sha256_file(locked_evaluator_path)

    reward_evaluator_fingerprint = None
    if isinstance(evaluator_fingerprint, dict):
        candidate = evaluator_fingerprint.get("training_reward")
        if isinstance(candidate, dict):
            reward_evaluator_fingerprint = candidate
    reward_calibration_config = config
    resolved_reward = resolve_training_reward(config)
    if resolved_reward["name"] == DNSMOS_OVRL_RAW:
        auxiliary = config.get("training_reward", {}).get("auxiliary_composite")
        if not isinstance(auxiliary, dict):
            raise ValueError("raw OVRL protocol lacks auxiliary composite calibration")
        reward_calibration_config = {**config, "training_reward": auxiliary}

    # The public FlowSE YAML stores the tokenizer path explicitly.  Resolve it
    # from that configuration instead of assuming the old upstream working
    # directory (``Emilia_ZH_EN_pinyin/vocab.txt`` at repository root).
    flowse_config_path = Path(config["flowse_config"])
    flowse_spec = yaml.safe_load(flowse_config_path.read_text(encoding="utf-8"))
    configured_tokenizer = Path(str(flowse_spec["model"]["tokenizer_path"]))
    tokenizer_candidates = [
        configured_tokenizer,
        flowse_config_path.parent / configured_tokenizer,
        root / "flowse" / configured_tokenizer,
    ]
    tokenizer_path = next(
        (candidate for candidate in tokenizer_candidates if candidate.is_file()),
        None,
    )
    if tokenizer_path is None:
        raise FileNotFoundError(
            "FlowSE tokenizer vocabulary not found; checked: "
            + ", ".join(str(candidate) for candidate in tokenizer_candidates)
        )

    components = {
        "schema_version": 2,
        "method": "speech_advantageflow_iterative_audio_only",
        "config": config,
        "train_manifest": {
            "path": str(train_manifest_path),
            "sha256": (
                None if lightweight_startup else sha256_file(train_manifest_path)
            ),
            "utterances": len(train_manifest),
        },
        "evaluation_manifest": {
            "path": str(eval_manifest_path),
            "sha256": (
                None if lightweight_startup else sha256_file(eval_manifest_path)
            ),
            "utterances": len(evaluation_manifest),
        },
        "checkpoint": {
            "path": str(bundle.checkpoint_path),
            "sha256": bundle.checkpoint_sha256,
        },
        "flowse_inputs_sha256": {
            "flowse_config": (
                None if lightweight_startup else sha256_file(config["flowse_config"])
            ),
            "tokenizer_vocabulary": (
                None
                if lightweight_startup
                else sha256_file(tokenizer_path)
            ),
            "vocoder_config": (
                None
                if lightweight_startup
                else sha256_file(Path(bundle.vocoder_model_path).parent / "config.yaml")
            ),
        },
        "vocoder": {
            "path": str(bundle.vocoder_model_path),
            "sha256": bundle.vocoder_sha256,
        },
        "dnsmos_sha256": {
            name: (None if lightweight_startup else sha256_file(path))
            for name, path in dnsmos_files.items()
        },
        "audio_aggregate_sha256": audio_hashes,
        "dataset_preflight": dataset_preflight,
        "dataset_audit_certificate": dataset_audit_certificate,
        "locked_evaluator_config_sha256": evaluator_config_hash,
        "evaluator_fingerprint": evaluator_fingerprint,
        "shared_initial_lora_snapshot": shared_initial_lora_snapshot,
        "reward_calibration": (
            verify_reward_calibration(
                reward_calibration_config,
                evaluator_fingerprint=reward_evaluator_fingerprint,
                lightweight=lightweight_startup,
                strict_provenance=validate_artifacts,
            )
            if validate_artifacts
            else {
                "mode": "public_smoke",
                "verification": "skipped_formal_provenance",
            }
        ),
        "source_sha256": source_hashes,
        "runtime_environment": (
            None if lightweight_startup else runtime_environment_fingerprint()
        ),
        "adaptation_choices": {
            "optimizer_learning_rate": config["optimizer"]["learning_rate"],
            "ema_decay": config["ema"]["decay"],
            "note": (
                "Explicit speech-pilot choices; not claimed as values reported "
                "by the AdvantageFlow image experiments."
            ),
        },
    }
    branch_source = None
    if "branch" in config:
        branch = config["branch"]
        source_dir = Path(str(branch["source_run_dir"])).resolve()
        source_step = int(branch["source_step"])
        endpoint_step = source_step + int(config["run"]["optimizer_steps"])
        source_protocol_path = source_dir / "protocol.json"
        source_checkpoint = source_dir / f"checkpoint_step_{source_step:06d}.pt"
        control_checkpoint = source_dir / f"checkpoint_step_{endpoint_step:06d}.pt"
        for path in (source_protocol_path, source_checkpoint, control_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_components = json.loads(
            source_protocol_path.read_text(encoding="utf-8")
        )
        if sha256_json(source_components) != source_dir.name:
            raise ValueError("branch source protocol hash differs from its directory")
        branch_source = {
            "source_run_dir": str(source_dir),
            "source_protocol": str(source_protocol_path),
            "source_protocol_sha256": sha256_file(source_protocol_path),
            "source_step": source_step,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "composite_control_step": endpoint_step,
            "composite_control_checkpoint": str(control_checkpoint),
            "composite_control_checkpoint_sha256": sha256_file(control_checkpoint),
        }
    components["branch_source"] = branch_source
    components["initial_evaluation_cache"] = initial_evaluation_cache_descriptor(
        config, current_components=components
    )
    protocol_hash = sha256_json(components)
    return {
        "protocol_hash": protocol_hash,
        "components": components,
        "train_manifest": train_manifest,
        "evaluation_manifest": evaluation_manifest,
    }


