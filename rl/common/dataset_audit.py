"""Freeze and verify a one-time full-decode paired-audio dataset audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


CERTIFICATE_SCHEMA_VERSION = 1
CERTIFICATE_KIND = "speech_paired_audio_dataset_full_decode_audit"
CERTIFICATE_CONFIG_KEY = "frozen_audit_certificate"
EXPECTED_AUDIO_HASH_KEYS = {
    "training_noisy",
    "training_clean",
    "evaluation_noisy",
    "evaluation_clean",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: str | Path) -> dict:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write_json(path: str | Path, value: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _without_certificate_path(preflight: Mapping) -> dict:
    return {
        str(key): value
        for key, value in preflight.items()
        if str(key) != CERTIFICATE_CONFIG_KEY
    }


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_completed_protocol(latest_protocol_path: str | Path) -> tuple[dict, dict, Path]:
    latest_protocol_path = Path(latest_protocol_path)
    pointer = _read_json_object(latest_protocol_path)
    if set(pointer) != {"protocol_hash", "protocol_dir"}:
        raise ValueError("latest protocol pointer has unsupported fields")
    protocol_path = Path(str(pointer["protocol_dir"])) / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    components = _read_json_object(protocol_path)
    observed_hash = _sha256_json(components)
    if observed_hash != pointer["protocol_hash"]:
        raise ValueError("completed protocol components do not match protocol_hash")
    return pointer, components, protocol_path


def freeze_dataset_audit_certificate(
    *, latest_protocol_path: str | Path, output_path: str | Path
) -> dict:
    """Extract a reusable certificate from a completed full-decode protocol build."""

    pointer, components, protocol_path = _load_completed_protocol(latest_protocol_path)
    config = components.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("source protocol does not contain a data configuration")
    data = config["data"]
    preflight = data.get("preflight")
    report = components.get("dataset_preflight")
    audio_hashes = components.get("audio_aggregate_sha256")
    source_hashes = components.get("source_sha256")
    recorded_auditor_hash = (
        source_hashes.get(
            "rl/common/dataset_validation.py"
        )
        if isinstance(source_hashes, dict)
        else None
    )
    auditor_path = Path(__file__).with_name("dataset_validation.py")
    criteria = {
        "protocol_schema_v2_or_newer": int(components.get("schema_version", 0)) >= 2,
        "preflight_config_present": isinstance(preflight, dict),
        "preflight_passed": isinstance(report, dict) and report.get("passed") is True,
        "full_decode_completed": isinstance(report, dict)
        and report.get("requirements", {}).get("full_decode") is True,
        "audio_hashes_complete": isinstance(audio_hashes, dict)
        and set(audio_hashes) == EXPECTED_AUDIO_HASH_KEYS
        and all(_valid_sha256(value) for value in audio_hashes.values()),
        "auditor_source_recorded": _valid_sha256(recorded_auditor_hash),
        "auditor_source_unchanged": recorded_auditor_hash
        == _sha256_file(auditor_path),
    }
    if not all(criteria.values()):
        raise ValueError(f"source dataset audit is incomplete: {criteria}")

    train_manifest = components["train_manifest"]
    evaluation_manifest = components["evaluation_manifest"]
    for name, descriptor in (
        ("train", train_manifest),
        ("evaluation", evaluation_manifest),
    ):
        if not isinstance(descriptor, dict):
            raise ValueError(f"source {name} manifest descriptor is missing")
        manifest_path = Path(str(descriptor["path"]))
        if _sha256_file(manifest_path) != descriptor["sha256"]:
            raise ValueError(f"source {name} manifest changed after protocol creation")

    certificate = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "kind": CERTIFICATE_KIND,
        "source_protocol_hash": str(pointer["protocol_hash"]),
        "source_protocol_path": str(protocol_path.resolve()),
        "source_protocol_file_sha256": _sha256_file(protocol_path),
        "auditor_source_sha256": _sha256_file(auditor_path),
        "train_manifest": {
            **train_manifest,
            "path": str(Path(str(train_manifest["path"])).resolve()),
        },
        "evaluation_manifest": {
            **evaluation_manifest,
            "path": str(Path(str(evaluation_manifest["path"])).resolve()),
        },
        "noisy_dir": str(Path(str(data["noisy_dir"])).resolve()),
        "clean_dir": str(Path(str(data["clean_dir"])).resolve()),
        "preflight_config": _without_certificate_path(preflight),
        "dataset_preflight": report,
        "audio_aggregate_sha256": audio_hashes,
    }
    _atomic_write_json(output_path, certificate)
    return {
        "path": str(Path(output_path)),
        "sha256": _sha256_file(output_path),
        "source_protocol_hash": certificate["source_protocol_hash"],
        "dataset_preflight": report,
        "audio_aggregate_sha256": audio_hashes,
    }


def validate_dataset_audit_certificate(
    *,
    certificate_path: str | Path,
    train_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    train_utterances: int,
    evaluation_utterances: int,
    noisy_dir: str | Path,
    clean_dir: str | Path,
    preflight_config: Mapping,
    conditions_per_step: int,
    optimizer_steps: int,
) -> dict:
    """Verify a frozen audit without reopening any WAV files."""

    certificate_path = Path(certificate_path)
    certificate = _read_json_object(certificate_path)
    source_protocol_path = Path(str(certificate.get("source_protocol_path", "")))
    source_protocol = (
        _read_json_object(source_protocol_path)
        if source_protocol_path.is_file()
        else None
    )
    report = certificate.get("dataset_preflight")
    audio_hashes = certificate.get("audio_aggregate_sha256")
    current_preflight = _without_certificate_path(preflight_config)
    auditor_path = Path(__file__).with_name("dataset_validation.py")
    train_descriptor = certificate.get("train_manifest", {})
    evaluation_descriptor = certificate.get("evaluation_manifest", {})
    coverage = report.get("coverage", {}) if isinstance(report, dict) else {}
    requirements = report.get("requirements", {}) if isinstance(report, dict) else {}
    observed = report.get("observed", {}) if isinstance(report, dict) else {}

    train_manifest_path = Path(train_manifest_path)
    evaluation_manifest_path = Path(evaluation_manifest_path)
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    criteria = {
        "schema": certificate.get("schema_version") == CERTIFICATE_SCHEMA_VERSION,
        "kind": certificate.get("kind") == CERTIFICATE_KIND,
        "source_protocol_hash": _valid_sha256(
            certificate.get("source_protocol_hash")
        ),
        "source_protocol_file_hash": _valid_sha256(
            certificate.get("source_protocol_file_sha256")
        ),
        "source_protocol_file_unchanged": source_protocol is not None
        and certificate.get("source_protocol_file_sha256")
        == _sha256_file(source_protocol_path),
        "source_protocol_components_unchanged": source_protocol is not None
        and certificate.get("source_protocol_hash")
        == _sha256_json(source_protocol),
        "auditor_source_unchanged": certificate.get("auditor_source_sha256")
        == _sha256_file(auditor_path),
        "train_manifest_path": train_descriptor.get("path")
        == str(train_manifest_path.resolve()),
        "train_manifest_hash": train_descriptor.get("sha256")
        == _sha256_file(train_manifest_path),
        "train_manifest_count": int(train_descriptor.get("utterances", -1))
        == int(train_utterances),
        "evaluation_manifest_path": evaluation_descriptor.get("path")
        == str(evaluation_manifest_path.resolve()),
        "evaluation_manifest_hash": evaluation_descriptor.get("sha256")
        == _sha256_file(evaluation_manifest_path),
        "evaluation_manifest_count": int(
            evaluation_descriptor.get("utterances", -1)
        )
        == int(evaluation_utterances),
        "noisy_directory": certificate.get("noisy_dir") == str(noisy_dir.resolve())
        and noisy_dir.is_dir(),
        "clean_directory": certificate.get("clean_dir") == str(clean_dir.resolve())
        and clean_dir.is_dir(),
        "preflight_config": certificate.get("preflight_config")
        == current_preflight,
        "preflight_passed": isinstance(report, dict) and report.get("passed") is True,
        "full_decode_completed": requirements.get("full_decode") is True,
        "observed_train_count": int(observed.get("train_utterances", -1))
        == int(train_utterances),
        "observed_evaluation_count": int(
            observed.get("evaluation_utterances", -1)
        )
        == int(evaluation_utterances),
        "conditions_per_step": int(coverage.get("conditions_per_step", -1))
        == int(conditions_per_step),
        "optimizer_steps": int(coverage.get("optimizer_steps", -1))
        == int(optimizer_steps),
        "audio_hashes_complete": isinstance(audio_hashes, dict)
        and set(audio_hashes) == EXPECTED_AUDIO_HASH_KEYS
        and all(_valid_sha256(value) for value in audio_hashes.values()),
    }
    if not all(criteria.values()):
        raise ValueError(f"frozen dataset audit verification failed: {criteria}")
    return {
        "dataset_preflight": report,
        "audio_aggregate_sha256": dict(audio_hashes),
        "certificate": {
            "path": str(certificate_path),
            "sha256": _sha256_file(certificate_path),
            "source_protocol_hash": certificate["source_protocol_hash"],
            "authorization": (
                "one_time_full_decode_and_content_hash; "
                "subsequent_runs_require_immutable_audio_tree"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a completed full-decode dataset audit certificate"
    )
    parser.add_argument("--latest-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_dataset_audit_certificate(
        latest_protocol_path=args.latest_protocol,
        output_path=args.output,
    )
    print("Frozen paired-audio dataset audit")
    print("=" * 72)
    print(f"Certificate: {result['path']}")
    print(f"SHA256: {result['sha256']}")
    print(f"Source protocol: {result['source_protocol_hash']}")
    print("Future training/resume full WAV decode: SKIPPED")


if __name__ == "__main__":
    main()

