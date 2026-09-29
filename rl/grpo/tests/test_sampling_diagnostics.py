from pathlib import Path

import pytest
import yaml

from rl.grpo.sampling_diagnostics import (
    _fingerprint_differences,
    analyze_sampling_rows,
    partition_utterances,
    validate_sampling_diagnostic_config,
    variance_decomposition,
)


def _config(*, utterances=2, latents=2, continuations=2):
    return {
        "run": {
            "mode": "sampling_diagnostic",
            "policy_state": "released_base_lora_disabled",
            "seed": 1,
        },
        "flowse_config": "flowse.yaml",
        "dnsmos_official_dir": "dnsmos",
        "output_root": "output",
        "conditioning": {"mode": "wotext", "use_text": False, "drop_text": True},
        "data": {
            "train_manifest": "voicebank_train_16k.json",
            "noisy_dir": "noisy",
            "clean_dir": "clean",
            "selection_seed": 2,
        },
        "diagnostic": {
            "utterance_count": utterances,
            "initial_latents_per_utterance": latents,
            "brownian_continuations_per_latent": continuations,
            "ode_controls_per_latent": 1,
            "repeat_score_audits_per_utterance": 1,
            "near_zero_reward_std": 1.0e-6,
        },
        "sampler": {
            "nfe": 10,
            "window_size": 2,
            "window_starts": [1, 2, 3],
            "diffusion": 0.4,
            "cfg_strength": 0.0,
            "latent_seed_base": 10,
            "brownian_seed_base": 20,
        },
        "resources": {
            "world_size": 2,
            "device_ids": [0, 1],
            "cpu_threads_per_worker": 2,
            "torch_interop_threads_per_worker": 1,
            "enforce_cpu_affinity": True,
        },
        "artifacts": {"keep_all_audio": False, "audit_wavs_per_utterance": 2},
        "normalization": {
            "target_dbfs": -25.0,
            "peak_ceiling": 0.99,
            "output_subtype": "PCM_16",
        },
        "training_reward": {
            "name": "flowse_grpo_public_composite",
            "component_normalization": "frozen_std",
            "weights": {"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
            "calibration": {
                "report_path": "calibration.json",
                "source_nfe": 10,
                "std_ddof": 0,
                "dnsmos_divisor": 4.0,
            },
        },
        "composite_reward_evaluators": {},
        "evaluation": {"paired_metrics": True, "fidelity": {"enabled": False}},
    }


def _row(
    utterance,
    method,
    latent,
    brownian,
    reward,
    *,
    repeat=False,
):
    initial_seed = 1000 + int(utterance[-1]) * 10 + latent
    brownian_seed = None if brownian is None else 2000 + int(utterance[-1]) * 100 + latent * 10 + brownian
    row = {
        "candidate_id": f"{utterance}:{method}:{latent}:{brownian}",
        "utterance": utterance,
        "method": method,
        "latent_index": latent,
        "brownian_index": brownian,
        "initial_latent_seed": initial_seed,
        "brownian_seed": brownian_seed,
        "reward": reward,
        "reward_components": {"raw_components": {"dnsmos": reward / 10.0}},
        "metrics": {
            "dnsmos_ovrl": reward / 2.0,
            "eres2net_speaker_similarity": 0.9 - reward / 1000.0,
            "speechbertscore": 0.8 + reward / 1000.0,
        },
        "terminal_mel_sha256": f"mel-{utterance}-{method}-{latent}-{brownian}",
        "scored_wav_sha256": f"wav-{utterance}-{method}-{latent}-{brownian}",
    }
    if repeat:
        repeated = {
            "reward": reward,
            "reward_components": row["reward_components"],
            "metrics": row["metrics"],
        }
        repeated["numeric_metrics"] = {
            "reward": reward,
            "metrics.dnsmos_ovrl": reward / 2.0,
            "metrics.eres2net_speaker_similarity": 0.9 - reward / 1000.0,
            "metrics.speechbertscore": 0.8 + reward / 1000.0,
            "raw_reward.dnsmos": reward / 10.0,
        }
        row["repeat_score"] = repeated
    else:
        row["repeat_score"] = None
    return row


def test_frozen_two_gpu_config_validates_without_artifacts():
    path = Path("configs/grpo/grpo_sampling_diagnostic_2gpu.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    report = validate_sampling_diagnostic_config(config)
    assert report["world_size"] == 2
    assert report["device_ids"] == [0, 1]
    assert report["maximum_configured_cpu_threads"] == 4
    assert report["sde_candidates"] == 1024
    assert report["ode_candidates"] == 256
    assert report["policy_state"] == "released_base_lora_disabled"


def test_variance_decomposition_obeys_total_variance_law():
    result = variance_decomposition([[1.0, 3.0], [5.0, 7.0]])
    assert result["across_initial_latent_variance"] == pytest.approx(4.0)
    assert result["within_initial_latent_brownian_variance"] == pytest.approx(1.0)
    assert result["total_variance"] == pytest.approx(5.0)
    assert result["initial_latent_variance_fraction"] == pytest.approx(0.8)
    assert result["brownian_variance_fraction"] == pytest.approx(0.2)


def test_partition_keeps_each_utterance_nested_design_on_one_worker():
    tasks = [{"utterance": f"p{i}_001"} for i in range(6)]
    partitions = partition_utterances(tasks, 2)
    assert [[row["utterance"] for row in shard] for shard in partitions] == [
        ["p0_001", "p2_001", "p4_001"],
        ["p1_001", "p3_001", "p5_001"],
    ]


def test_analysis_audits_geometry_and_separates_reward_variance():
    rows = []
    for utterance in ("u0", "u1"):
        matrix = [[1.0, 3.0], [5.0, 7.0]]
        for latent in range(2):
            rows.append(_row(utterance, "ode", latent, None, matrix[latent][0] - 0.5))
            for brownian in range(2):
                rows.append(
                    _row(
                        utterance,
                        "sde",
                        latent,
                        brownian,
                        matrix[latent][brownian],
                        repeat=latent == 0 and brownian == 0,
                    )
                )
    summaries = [
        {
            "utterance": utterance,
            "distance_audit": {
                "within_latent_brownian_terminal_mel_rms": {"mean": 0.2}
            },
        }
        for utterance in ("u0", "u1")
    ]
    report = analyze_sampling_rows(rows, summaries, _config())
    reward = report["metric_variance_decomposition"]["reward"]
    assert reward["ratio_of_aggregate_variance"]["initial_latent_fraction"] == pytest.approx(0.8)
    assert reward["ratio_of_aggregate_variance"]["brownian_fraction"] == pytest.approx(0.2)
    assert report["geometry_audit"]["complete"] is True
    assert report["repeat_score_audit"]["count"] == 2
    assert report["duplicate_audit"]["sde_pcm16_wav_duplicate_count"] == 0


def test_config_rejects_silent_single_gpu_fallback():
    config = _config()
    config["resources"] = {
        "world_size": 1,
        "device_ids": [0],
        "cpu_threads_per_worker": 2,
        "torch_interop_threads_per_worker": 1,
        "enforce_cpu_affinity": True,
    }
    with pytest.raises(ValueError, match="two distinct GPUs"):
        validate_sampling_diagnostic_config(config)


def test_fingerprint_difference_reports_exact_stale_fields():
    differences = _fingerprint_differences(
        {"implementation": "old", "model": {"revision": "a"}},
        {"implementation": "new", "model": {"revision": "b"}},
    )
    assert differences == {
        "implementation": {"calibration": "old", "runtime": "new"},
        "model.revision": {"calibration": "a", "runtime": "b"},
    }
