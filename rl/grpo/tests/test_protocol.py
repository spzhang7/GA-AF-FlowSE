import json
import pytest
import torch
from torch import nn

from rl.common.lora import inject_lora, snapshot_lora
from rl.grpo.protocol import (
    audit_data_splits,
    lora_state_fingerprint,
    milestone_collections,
    prepare_or_load_shared_lora_snapshot,
    sha256_file,
)


def _manifest(path, values):
    path.write_text(json.dumps({value: value for value in values}), encoding="utf-8")


def _split_config(
    tmp_path,
    *,
    train,
    validation,
    test,
    calibration,
    calibration_source_train=None,
):
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    test_path = tmp_path / "test.json"
    calibration_train_path = tmp_path / "calibration_train.json"
    endpoint_path = tmp_path / "endpoint.jsonl"
    report_path = tmp_path / "calibration.json"
    _manifest(train_path, train)
    _manifest(validation_path, validation)
    _manifest(test_path, test)
    source_train = (
        list(train)
        if calibration_source_train is None
        else list(calibration_source_train)
    )
    _manifest(calibration_train_path, source_train)
    endpoint_path.write_text(
        "".join(
            json.dumps({"utterance": utterance, "nfe": 10}) + "\n"
            for utterance in calibration
        ),
        encoding="utf-8",
    )
    excluded = sorted(set(calibration) - set(source_train))
    selected = [value for value in calibration if value in set(source_train)]
    report_path.write_text(
        json.dumps(
            {
                "source": {
                    "endpoint_metrics_path": str(endpoint_path),
                    "endpoint_metrics_sha256": sha256_file(endpoint_path),
                    "train_manifest_path": str(calibration_train_path),
                    "train_manifest_sha256": sha256_file(calibration_train_path),
                    "eligible_rows_before_train_filter": len(calibration),
                    "excluded_outside_train_rows": len(calibration) - len(selected),
                    "excluded_outside_train_utterances": excluded,
                    "rows": len(selected),
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "data": {
            "train_manifest": str(train_path),
            "validation_manifest": str(validation_path),
            "official_test_manifest": str(test_path),
            "order_seed": 17,
        },
        "training_reward": {
            "calibration": {"report_path": str(report_path), "source_nfe": 10}
        },
    }


def test_split_audit_accepts_train_only_calibration(tmp_path):
    config = _split_config(
        tmp_path,
        train=["p1_001", "p1_002"],
        validation=["p2_001"],
        test=["p3_001"],
        calibration=["p1_001", "p1_002"],
    )
    report = audit_data_splits(config)
    assert report["calibration"]["outside_train"] == 0


def test_split_audit_rejects_manifest_overlap(tmp_path):
    config = _split_config(
        tmp_path,
        train=["p1_001", "overlap"],
        validation=["overlap"],
        test=["p2_001"],
        calibration=["p1_001"],
    )
    with pytest.raises(ValueError, match="split overlap"):
        audit_data_splits(config)


def test_split_audit_records_calibration_train_manifest_mismatch_as_provenance(tmp_path):
    config = _split_config(
        tmp_path,
        train=["p1_001"],
        validation=["p2_001"],
        test=["p3_001"],
        calibration=["not_train"],
        calibration_source_train=["not_train"],
    )
    report = audit_data_splits(config)
    assert report["calibration"]["provenance_only"] is True
    assert report["calibration"]["outside_train"] == 0


def _adapted_model():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(3, 4))
    report = inject_lora(
        model,
        target_patterns=[r"0"],
        rank=2,
        alpha=4.0,
        expected_modules=1,
    )
    return model, report


def test_shared_lora_snapshot_round_trip_and_tensor_hashes(tmp_path):
    model, report = _adapted_model()
    config = {
        "run": {"seed": 19},
        "lora": {
            "initialization_seed": 4,
            "rank": 2,
            "alpha": 4.0,
            "shared_initial_snapshot": {
                "path": str(tmp_path / "shared.pt"),
                "create_if_missing": True,
                "expected_state_sha256": None,
            },
        },
    }
    expected = lora_state_fingerprint(snapshot_lora(model, device="cpu"))
    created = prepare_or_load_shared_lora_snapshot(
        model, config=config, module_names=report.module_names
    )
    assert created["state_sha256"] == expected["state_sha256"]
    assert created["tensor_sha256"] == expected["tensor_sha256"]

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)
    loaded = prepare_or_load_shared_lora_snapshot(
        model, config=config, module_names=report.module_names
    )
    assert lora_state_fingerprint(snapshot_lora(model)) == {
        key: loaded[key] for key in ("state_sha256", "tensor_sha256", "tensor_count")
    }


def test_shared_lora_snapshot_detects_tensor_tampering(tmp_path):
    model, report = _adapted_model()
    config = {
        "run": {"seed": 19},
        "lora": {
            "initialization_seed": 4,
            "rank": 2,
            "alpha": 4.0,
            "shared_initial_snapshot": {
                "path": str(tmp_path / "shared.pt"),
                "create_if_missing": True,
                "expected_state_sha256": None,
            },
        },
    }
    prepare_or_load_shared_lora_snapshot(
        model, config=config, module_names=report.module_names
    )
    payload = torch.load(config["lora"]["shared_initial_snapshot"]["path"], weights_only=False)
    first = next(iter(payload["state"]))
    payload["state"][first].add_(1.0)
    torch.save(payload, config["lora"]["shared_initial_snapshot"]["path"])
    with pytest.raises(ValueError, match="state hash"):
        prepare_or_load_shared_lora_snapshot(
            model, config=config, module_names=report.module_names
        )


@pytest.mark.parametrize(
    ("collections", "percentages", "expected"),
    [
        (50, [0, 25, 50, 75, 100], {0: 0, 13: 25, 25: 50, 38: 75, 50: 100}),
        (
            1250,
            [0, 20, 40, 60, 80, 100],
            {0: 0, 250: 20, 500: 40, 750: 60, 1000: 80, 1250: 100},
        ),
    ],
)
def test_registered_milestone_collection_boundaries(collections, percentages, expected):
    assert milestone_collections(collections, percentages) == expected
