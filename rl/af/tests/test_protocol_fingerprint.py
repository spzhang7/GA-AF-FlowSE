import json

import pytest

from rl.af.protocol import (
    initial_evaluation_cache_descriptor,
    runtime_environment_fingerprint,
    sha256_json,
    source_fingerprint,
    training_source_names,
    verify_execution_dependencies,
)


def _cache_components(config):
    return {
        "config": config,
        "evaluation_manifest": {"sha256": "manifest", "utterances": 2},
        "audio_aggregate_sha256": {
            "evaluation_noisy": "noisy",
            "evaluation_clean": "clean",
        },
        "checkpoint": {"sha256": "checkpoint"},
        "flowse_inputs_sha256": {"flowse_config": "flowse"},
        "vocoder": {"sha256": "vocoder"},
        "dnsmos_sha256": {"primary": "dnsmos"},
        "locked_evaluator_config_sha256": "locked",
        "evaluator_fingerprint": {"training_reward": "evaluator"},
        "reward_calibration": {"sha256": "calibration"},
    }


def _cache_config():
    return {
        "conditioning": {"mode": "wotext"},
        "rollout": {"solver": "euler", "evaluation_nfe": 32, "cfg_strength": 0.0},
        "normalization": {"target_dbfs": -25.0},
        "evaluation": {
            "policy": "ema",
            "latent_seed_base": 17,
            "paired_metrics": True,
            "fidelity": {"enabled": True},
        },
        "training_reward": {"name": "composite"},
        "composite_reward_evaluators": {"device": "cuda"},
    }


def test_initial_evaluation_cache_is_hashed_and_semantically_compatible(tmp_path):
    source_config = _cache_config()
    source_components = _cache_components(source_config)
    protocol_hash = sha256_json(source_components)
    cache_dir = tmp_path / protocol_hash
    cache_dir.mkdir()
    protocol_path = cache_dir / "protocol.json"
    protocol_path.write_text(json.dumps(source_components), encoding="utf-8")
    report_path = cache_dir / "evaluation_step_000000.json"
    report_path.write_text('{"step": 0}', encoding="utf-8")
    (cache_dir / "evaluation_noisy_baselines.json").write_text(
        '{"rows": []}', encoding="utf-8"
    )

    current_config = _cache_config()
    current_config["evaluation"]["initial_cache_report_path"] = str(report_path)
    current_components = _cache_components(current_config)
    descriptor = initial_evaluation_cache_descriptor(
        current_config, current_components=current_components
    )
    assert descriptor["source_protocol_hash"] == protocol_hash
    assert descriptor["authorization"] == (
        "immutable_step0_raw_metrics_with_current_reward_recompute"
    )

    # A calibration report changes only a derived scalar.  The importer
    # recomputes that scalar from cached raw component scores, so it must not
    # force expensive step-0 inference/evaluation.
    current_components["reward_calibration"] = {"sha256": "new-calibration"}
    descriptor = initial_evaluation_cache_descriptor(
        current_config, current_components=current_components
    )
    assert descriptor["reward_calibration_changed"] is True

    current_components["config"]["rollout"]["evaluation_nfe"] = 10
    with pytest.raises(ValueError, match="incompatible"):
        initial_evaluation_cache_descriptor(
            current_config, current_components=current_components
        )


def test_training_source_manifest_covers_upstream_flowse_dependencies():
    names = set(training_source_names())
    assert (
        "rl/af/checkpoint.py"
        in names
    )
    assert "flowse/infer.py" in names
    assert "flowse/loader/datareader.py" in names
    assert "flowse/model/modules.py" in names
    assert "flowse/model/model_utils.py" in names
    assert "flowse/model/cfm.py" in names
    assert "flowse/model/backbones/dit.py" in names
    assert all(not name.startswith("grpo/") for name in names)


def test_frozen_source_and_environment_are_verified_exactly():
    sources = source_fingerprint()
    components = {
        "schema_version": 2,
        "source_sha256": sources,
        "runtime_environment": runtime_environment_fingerprint(),
    }
    assert verify_execution_dependencies(components)["passed"]

    changed = {**components, "source_sha256": dict(sources)}
    changed["source_sha256"]["infer.py"] = "0" * 64
    result = verify_execution_dependencies(changed)
    assert result["passed"]
    assert result["enforcement_disabled"]
    assert not result["diagnostic_match"]
    assert not result["criteria"]["source_hashes_match"]

