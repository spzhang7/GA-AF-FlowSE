import hashlib
import json
from pathlib import Path

import pytest

from rl.common.dataset_audit import (
    freeze_dataset_audit_certificate,
    validate_dataset_audit_certificate,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _completed_protocol(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    train_manifest = tmp_path / "train.json"
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(train_manifest, {"train_a": "", "train_b": ""})
    _write_json(evaluation_manifest, {"valid_a": "reference"})
    noisy_dir = tmp_path / "noisy"
    clean_dir = tmp_path / "clean"
    noisy_dir.mkdir()
    clean_dir.mkdir()
    preflight = {
        "expected_train_utterances": 2,
        "expected_evaluation_utterances": 1,
        "expected_sample_rate": 16000,
        "expected_channels": 1,
        "full_decode": True,
        "require_single_coverage": True,
        "allow_unlisted_audio": True,
    }
    dataset_report = {
        "passed": True,
        "requirements": {
            **preflight,
            "allow_partial_coverage": False,
            "exact_directory_membership": False,
            "clean_noisy_frame_parity": True,
        },
        "observed": {
            "train_utterances": 2,
            "evaluation_utterances": 1,
            "paired_utterances": 3,
            "audio_files": 6,
        },
        "coverage": {
            "conditions_per_step": 2,
            "optimizer_steps": 1,
            "scheduled_conditions": 2,
        },
    }
    preflight_source = (
        Path("rl/common/dataset_validation.py")
    )
    components = {
        "schema_version": 2,
        "config": {
            "data": {
                "train_manifest": str(train_manifest),
                "evaluation_manifest": str(evaluation_manifest),
                "noisy_dir": str(noisy_dir),
                "clean_dir": str(clean_dir),
                "preflight": preflight,
            }
        },
        "train_manifest": {
            "path": str(train_manifest),
            "sha256": _sha256_file(train_manifest),
            "utterances": 2,
        },
        "evaluation_manifest": {
            "path": str(evaluation_manifest),
            "sha256": _sha256_file(evaluation_manifest),
            "utterances": 1,
        },
        "dataset_preflight": dataset_report,
        "audio_aggregate_sha256": {
            "training_noisy": "1" * 64,
            "training_clean": "2" * 64,
            "evaluation_noisy": "3" * 64,
            "evaluation_clean": "4" * 64,
        },
        "source_sha256": {
            "rl/common/dataset_validation.py": _sha256_file(preflight_source)
        },
    }
    protocol_hash = _sha256_json(components)
    protocol_dir = tmp_path / protocol_hash
    protocol_path = protocol_dir / "protocol.json"
    _write_json(protocol_path, components)
    latest_protocol = tmp_path / "latest_protocol.json"
    _write_json(
        latest_protocol,
        {"protocol_hash": protocol_hash, "protocol_dir": str(protocol_dir)},
    )
    return latest_protocol, train_manifest, evaluation_manifest, preflight


def test_frozen_audit_skips_wav_scan_but_binds_manifests_and_config(tmp_path):
    latest, train_manifest, evaluation_manifest, preflight = _completed_protocol(
        tmp_path
    )
    certificate_path = tmp_path / "dataset_audit_certificate.json"
    frozen = freeze_dataset_audit_certificate(
        latest_protocol_path=latest,
        output_path=certificate_path,
    )
    assert len(frozen["sha256"]) == 64

    # No WAV exists. Successful verification therefore proves the resume path does
    # not reopen or decode the already certified audio tree.
    current_preflight = {
        **preflight,
        "frozen_audit_certificate": str(certificate_path),
    }
    verified = validate_dataset_audit_certificate(
        certificate_path=certificate_path,
        train_manifest_path=train_manifest,
        evaluation_manifest_path=evaluation_manifest,
        train_utterances=2,
        evaluation_utterances=1,
        noisy_dir=tmp_path / "noisy",
        clean_dir=tmp_path / "clean",
        preflight_config=current_preflight,
        conditions_per_step=2,
        optimizer_steps=1,
    )
    assert verified["dataset_preflight"]["passed"] is True
    assert set(verified["audio_aggregate_sha256"]) == {
        "training_noisy",
        "training_clean",
        "evaluation_noisy",
        "evaluation_clean",
    }


def test_frozen_audit_rejects_manifest_changes(tmp_path):
    latest, train_manifest, evaluation_manifest, preflight = _completed_protocol(
        tmp_path
    )
    certificate_path = tmp_path / "dataset_audit_certificate.json"
    freeze_dataset_audit_certificate(
        latest_protocol_path=latest,
        output_path=certificate_path,
    )
    _write_json(train_manifest, {"train_a": "changed", "train_b": ""})
    with pytest.raises(ValueError, match="frozen dataset audit verification failed"):
        validate_dataset_audit_certificate(
            certificate_path=certificate_path,
            train_manifest_path=train_manifest,
            evaluation_manifest_path=evaluation_manifest,
            train_utterances=2,
            evaluation_utterances=1,
            noisy_dir=tmp_path / "noisy",
            clean_dir=tmp_path / "clean",
            preflight_config={
                **preflight,
                "frozen_audit_certificate": str(certificate_path),
            },
            conditions_per_step=2,
            optimizer_steps=1,
        )

