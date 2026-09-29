"""Prepare and freeze the LibriTTS/DNS10s Flow-GRPO experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from rl.common.shared_initialization import (
    load_shared_lora_snapshot_payload,
)


def _read_manifest(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value or not all(
        isinstance(key, str) and isinstance(text, str) for key, text in value.items()
    ):
        raise ValueError(f"invalid manifest: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    from rl.grpo.protocol import audit_data_splits
    from rl.grpo.trainer import validate_grpo_config

    manifest_root = args.manifest_root.resolve()
    train_path = manifest_root / "libritts_dns10s_train_exposures.json"
    validation_path = manifest_root / "libritts_dns10s_validation.json"
    smoke_validation_path = manifest_root / "libritts_dns10s_smoke_validation.json"
    dns_paths = [
        manifest_root / "dns2020_no_reverb.json",
        manifest_root / "dns2020_with_reverb.json",
        manifest_root / "dns2020_real_recordings.json",
    ]
    train = _read_manifest(train_path)
    validation = _read_manifest(validation_path)
    smoke_validation = _read_manifest(smoke_validation_path)
    dns_manifests = [_read_manifest(path) for path in dns_paths]
    combined_test = {}
    for manifest in dns_manifests:
        overlap = set(combined_test) & set(manifest)
        if overlap:
            raise ValueError(f"DNS2020 manifests overlap: {sorted(overlap)[:5]}")
        combined_test.update(manifest)
    checks = {
        "train_80000": len(train) == 80000,
        "validation_512": len(validation) == 512,
        "smoke_validation_nonempty_subset": bool(smoke_validation)
        and set(smoke_validation).issubset(validation),
        "dns_split_counts": [len(value) for value in dns_manifests] == [150, 150, 300],
        "dns_combined_600": len(combined_test) == 600,
        "train_validation_disjoint": not (set(train) & set(validation)),
        "train_dns_disjoint": not (set(train) & set(combined_test)),
        "validation_dns_disjoint": not (set(validation) & set(combined_test)),
    }
    if not all(checks.values()):
        raise ValueError(f"DNS10s GRPO manifest preparation failed: {checks}")
    combined_path = manifest_root / "dns2020_official_test_all.json"
    if combined_path.exists():
        if _read_manifest(combined_path) != combined_test:
            raise ValueError(f"frozen combined DNS2020 manifest differs: {combined_path}")
    else:
        _atomic_json(combined_path, combined_test)

    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    expected_stds = {
        "dnsmos": 0.07032371,
        "speaker": 0.13865593,
        "speechbertscore": 0.09550272,
    }
    calibration_checks = {
        "complete": calibration.get("status") == "CALIBRATION-COMPLETE",
        "audit_pass": calibration.get("replication_audit", {}).get("audit_status")
        == "AUDIT-PASS",
        "conditions": int(calibration.get("replication_audit", {}).get("conditions", -1))
        == 8192,
        "rows": int(calibration.get("source", {}).get("rows", -1)) == 65536,
        "train_manifest": calibration.get("source", {}).get("train_manifest_sha256")
        == _sha256_file(train_path),
        "stds": all(
            abs(float(calibration.get("component_stds", {}).get(name, -1.0)) - expected)
            <= 5.0e-9
            for name, expected in expected_stds.items()
        ),
    }
    if not all(calibration_checks.values()):
        raise ValueError(f"DNS10s calibration contract failed: {calibration_checks}")

    payload = load_shared_lora_snapshot_payload(args.shared_lora_snapshot)
    state_sha256 = str(payload["state_sha256"])
    snapshot_checks = {
        "training_seed": int(payload.get("training_seed", -1)) == 260521,
        "initialization_seed": int(payload.get("initialization_seed", -1)) == 32001,
        "rank": int(payload.get("rank", -1)) == 32,
        "alpha": float(payload.get("alpha", -1.0)) == 64.0,
        "recorded_state_hash": len(state_sha256) == 64,
    }
    if not all(snapshot_checks.values()):
        raise ValueError(f"shared LoRA snapshot contract failed: {snapshot_checks}")

    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    config["lora"]["shared_initial_snapshot"]["expected_state_sha256"] = state_sha256
    config["resources"].pop("expected_cuda_visible_devices", None)
    smoke_config = yaml.safe_load(args.smoke_template.read_text(encoding="utf-8"))
    smoke_config["lora"]["shared_initial_snapshot"][
        "expected_state_sha256"
    ] = state_sha256
    smoke_config["resources"].pop("expected_cuda_visible_devices", None)
    validation = {
        "formal": validate_grpo_config(config),
        "smoke": validate_grpo_config(smoke_config),
        "formal_splits": audit_data_splits(config),
        "smoke_splits": audit_data_splits(smoke_config),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen formal config: {args.output}")
    if args.smoke_output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen smoke config: {args.smoke_output}"
        )
    _atomic_yaml(args.output, config)
    _atomic_yaml(args.smoke_output, smoke_config)
    return {
        "status": "DNS10S-GRPO-CONFIG-READY",
        "checks": checks,
        "calibration_checks": calibration_checks,
        "snapshot_checks": snapshot_checks,
        "manifest_sha256": {
            "train": _sha256_file(train_path),
            "validation": _sha256_file(validation_path),
            "official_test": _sha256_file(combined_path),
        },
        "shared_lora_state_sha256": state_sha256,
        "physical_gpu_policy": "runtime_cuda_visible_devices_exactly_four",
        "formal_config": str(args.output),
        "smoke_config": str(args.smoke_output),
        "validation": validation,
    }


def main() -> None:
    root = Path(
        "artifacts/af/manifests/libritts_dns10s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=root)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=Path(
            "artifacts/af/"
            "libritts_dns10s_calibration_audit_8192/"
            "calibration_report.json"
        ),
    )
    parser.add_argument(
        "--shared-lora-snapshot",
        type=Path,
        default=Path("artifacts/shared_initial_lora/seed_260521.pt"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            "configs/grpo/"
            "grpo_libritts_dns10s_4gpu_5000update.template.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "configs/grpo/"
            "grpo_libritts_dns10s_sft20k_4gpu_5000update.yaml"
        ),
    )
    parser.add_argument(
        "--smoke-template",
        type=Path,
        default=Path(
            "configs/grpo/"
            "grpo_libritts_dns10s_4gpu_production_geometry_smoke.yaml"
        ),
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=Path(
            "configs/grpo/"
            "grpo_libritts_dns10s_sft20k_4gpu_"
            "production_geometry_smoke.frozen.yaml"
        ),
    )
    args = parser.parse_args()
    report = run(args)
    print("\nLibriTTS/DNS10s Flow-GRPO formal preparation")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Formal config: {report['formal_config']}")
    print(f"Smoke config: {report['smoke_config']}")
    print(f"Shared LoRA state SHA256: {report['shared_lora_state_sha256']}")
    print("Physical GPUs: selected at runtime through CUDA_VISIBLE_DEVICES")


if __name__ == "__main__":
    main()
