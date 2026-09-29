"""Frozen shared-baseline contract for the controlled AF/GRPO comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


CONTROLLED_SETTING = "controlled_cfg0_eval_nfe32"
NEUTRAL_ROOT = "configs/flowse"
SHARED_SNAPSHOT_ROOT = "artifacts/shared_initial_lora"
LORA_TARGET_PATTERNS = (
    r"^transformer_blocks\.[0-9]+\.attn\.(to_q|to_k|to_v|to_out\.0)$",
    r"^transformer_blocks\.[0-9]+\.ff\.ff\.(0\.0|2)$",
)
SHARED_OPTIMIZER = {
    "type": "AdamW",
    "learning_rate": 2.0e-4,
    "betas": [0.9, 0.999],
    "epsilon": 1.0e-8,
    "weight_decay": 0.01,
    "gradient_clip_norm": 5.0,
    "schedule": "linear_decay",
    "warmup_steps": 0,
}


def _algorithm_references(value, needle: str) -> list[str]:
    found: list[str] = []

    def visit(item, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str) and needle.lower() in item.replace("\\", "/").lower():
            found.append(path)

    visit(value, "")
    return found


def assert_algorithm_directory_isolation(config: Mapping, *, method: str) -> None:
    """Reject configs that point into the other algorithm's implementation tree."""

    if method == "grpo":
        needle = "rl/af"
    elif method == "advantageflow":
        needle = "grpo/"
    else:
        raise ValueError("method must be grpo or advantageflow")
    found = _algorithm_references(config, needle)
    if found:
        raise ValueError(
            f"{method} config references the other algorithm directory at {found}"
        )


def validate_shared_optimizer(optimizer: Mapping) -> dict[str, bool]:
    checks = {
        key: (
            list(optimizer.get(key, [])) == expected
            if isinstance(expected, list)
            else optimizer.get(key) == expected
        )
        for key, expected in SHARED_OPTIMIZER.items()
    }
    if not all(checks.values()):
        raise ValueError(f"shared optimizer contract mismatch: {checks}")
    return checks


def validate_controlled_baseline(config: Mapping, *, method: str) -> dict:
    """Validate fields that are external to either RL algorithm."""

    assert_algorithm_directory_isolation(config, method=method)
    data = config["data"]
    evaluation = config["evaluation"]
    rollout = config["sampler"] if method == "grpo" else config["rollout"]
    validation_key = "validation_manifest" if method == "grpo" else "evaluation_manifest"
    checks = {
        "flowse_config_neutral": str(config["flowse_config"]).replace("\\", "/")
        == "configs/flowse/flowse_voicebank_wotext.yaml",
        "train_manifest_neutral": str(data["train_manifest"]).replace("\\", "/")
        == "artifacts/manifests/voicebank_train_16k.json",
        "validation_manifest_neutral": str(data[validation_key]).replace("\\", "/")
        == "artifacts/manifests/voicebank_valid_16k.json",
        "order_seed": int(data["order_seed"]) == 51121,
        "conditioning": dict(config["conditioning"])
        == {"mode": "wotext", "use_text": False, "drop_text": True},
        "cfg_zero": float(rollout["cfg_strength"]) == 0.0,
        "evaluation_nfe": int(
            evaluation["nfe"] if method == "grpo" else rollout["evaluation_nfe"]
        )
        == 32,
        "evaluation_latent_seed": int(evaluation["latent_seed_base"]) == 2700100,
        "paired_metrics": bool(evaluation["paired_metrics"]),
        "fidelity_config": dict(evaluation["fidelity"])
        == {
            "enabled": True,
            "source_locked_config": "artifacts/configs/gate_a_wotext_v2.locked.yaml",
            "device": "cuda",
        },
        "normalization": dict(config["normalization"])
        == {
            "target_dbfs": -25.0,
            "peak_ceiling": 0.99,
            "output_subtype": "PCM_16",
        },
        "lora_geometry": (
            int(config["lora"]["rank"]),
            float(config["lora"]["alpha"]),
            float(config["lora"]["dropout"]),
            int(config["lora"]["expected_modules"]),
            tuple(config["lora"]["target_patterns"]),
        )
        == (32, 64.0, 0.0, 132, LORA_TARGET_PATTERNS),
        "reward": dict(config["training_reward"])["name"]
        == "flowse_grpo_public_composite",
        "reward_weights": dict(config["training_reward"]["weights"])
        == {"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
        "reward_normalization": str(
            config["training_reward"]["component_normalization"]
        )
        == "frozen_std",
        "reward_calibration": dict(config["training_reward"]["calibration"])
        == {
            "report_path": "artifacts/flowse_grpo_calibrate_composite/calibration_report.json",
            "source_nfe": 10,
            "std_ddof": 0,
            "dnsmos_divisor": 4.0,
        },
        "dnsmos": str(config["dnsmos_official_dir"])
        == "pretrainmodel/DNSMOS-official",
        "composite_evaluators": dict(config["composite_reward_evaluators"])
        == {
            "device": "cuda",
            "speaker": {
                "backend": "modelscope_speaker_verification",
                "model_id": "iic/speech_eres2net_sv_zh-cn_16k-common",
                "revision": "v1.0.5",
                "local_model_dir": (
                    "pretrainmodel/speech_eres2net_sv_zh-cn_16k-common"
                ),
            },
            "speechbertscore": {
                "repo_id": "microsoft/wavlm-large",
                "revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
                "local_model_dir": "pretrainmodel/wavlm-large",
                "local_files_only": True,
                "layer": 14,
                "reference_cache_size": 64,
            },
        },
    }
    snapshot = config["lora"].get("shared_initial_snapshot")
    if snapshot is not None:
        normalized = str(snapshot["path"]).replace("\\", "/")
        checks["shared_snapshot_neutral"] = normalized.startswith(
            f"{SHARED_SNAPSHOT_ROOT}/"
        ) or Path(normalized).is_absolute()
    if method == "grpo":
        checks["official_test_manifest_neutral"] = str(
            data["official_test_manifest"]
        ).replace("\\", "/") == "artifacts/manifests/voicebank_official_test_16k.json"
    else:
        checks["advantageflow_l16_k8"] = (
            int(config["run"]["conditions_per_step"]),
            int(config["rollout"]["candidates_per_condition"]),
        ) == (16, 8)
    if not all(checks.values()):
        raise ValueError(
            "controlled shared-baseline mismatch: "
            + json.dumps(checks, sort_keys=True)
        )
    optimizer_checks = validate_shared_optimizer(config["optimizer"])
    return {"baseline": checks, "optimizer": optimizer_checks}


def validate_paired_run_configs(grpo: Mapping, advantageflow: Mapping) -> dict:
    """Neutral freeze-time check for per-seed values that must match exactly."""

    grpo_report = validate_controlled_baseline(grpo, method="grpo")
    af_report = validate_controlled_baseline(advantageflow, method="advantageflow")
    grpo_snapshot = grpo["lora"]["shared_initial_snapshot"]
    af_snapshot = advantageflow["lora"]["shared_initial_snapshot"]
    checks = {
        "training_seed": int(grpo["run"]["seed"])
        == int(advantageflow["run"]["seed"]),
        "lora_initialization_seed": int(grpo["lora"]["initialization_seed"])
        == int(advantageflow["lora"]["initialization_seed"]),
        "shared_snapshot_path": str(grpo_snapshot["path"])
        == str(af_snapshot["path"]),
        "shared_snapshot_hash": grpo_snapshot.get("expected_state_sha256")
        == af_snapshot.get("expected_state_sha256"),
        "noisy_audio_root": str(grpo["data"]["noisy_dir"])
        == str(advantageflow["data"]["noisy_dir"]),
        "clean_audio_root": str(grpo["data"]["clean_dir"])
        == str(advantageflow["data"]["clean_dir"]),
        "gpu_world_size": int(grpo["resources"]["rollout_world_size"])
        == int(advantageflow["parallel_rollout"]["world_size"]),
        "gpu_device_ids": list(grpo["resources"]["device_ids"])
        == list(advantageflow["parallel_rollout"]["device_ids"]),
    }
    if not all(checks.values()):
        raise ValueError(f"paired run contract mismatch: {checks}")
    return {"grpo": grpo_report, "advantageflow": af_report, "paired": checks}

