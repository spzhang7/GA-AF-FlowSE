import pytest
import numpy as np
from pathlib import Path
from types import SimpleNamespace

pytest.importorskip("torch")
import torch

import rl.af.evaluation as checkpoint_module

from rl.af.evaluation import (  # noqa: E402
    paired_policy_analysis,
    paired_state_analysis,
    registered_primary_result,
    speaker_cluster_bootstrap_mean,
    _evaluate_state,
    _load_reward_evaluators,
)
from rl.af.confirmation_evaluation import (  # noqa: E402
    confirmation_decision,
)
from rl.af.protocol import (  # noqa: E402
    sha256_file,
)
from rl.af.trainer import (  # noqa: E402
    _score_evaluation_file,
    _write_wave,
)


def _rows(values):
    utterances = ["p001_001", "p001_002", "p002_001", "p002_002"]
    return [
        {
            "utterance": utterance,
            "latent_seed": 1000 + index,
            "dnsmos_ovrl": value,
        }
        for index, (utterance, value) in enumerate(zip(utterances, values, strict=True))
    ]


def test_speaker_cluster_bootstrap_resamples_whole_speakers():
    result = speaker_cluster_bootstrap_mean(
        ["p001_001", "p001_002", "p002_001", "p002_002"],
        [1.0, 1.0, -1.0, -1.0],
        seed=17,
        samples=2000,
        confidence=0.95,
    )
    assert result["speakers"] == 2
    assert result["mean"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(-1.0)
    assert result["ci_high"] == pytest.approx(1.0)


def test_written_pcm_hash_is_the_exact_scored_file_hash(tmp_path):
    path = tmp_path / "candidate.wav"
    digest = _write_wave(
        path,
        np.linspace(-0.2, 0.2, 2400, dtype=np.float32),
        24000,
        "PCM_16",
    )
    assert digest == sha256_file(path)


def test_legacy_and_composite_checkpoint_state_forward_one_evaluator(
    tmp_path, monkeypatch
):
    captured = []

    def fake_score(**kwargs):
        captured.append(kwargs.get("composite_evaluators"))
        return {"dnsmos_ovrl": 3.0}

    monkeypatch.setattr(checkpoint_module, "load_lora", lambda *args: None)
    monkeypatch.setattr(checkpoint_module, "_score_evaluation_file", fake_score)

    endpoint = SimpleNamespace(
        normalized_waveform=np.zeros(32, dtype=np.float32),
        terminal_mel_sha256="terminal",
        normalized_waveform_sha256="waveform",
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(transformer=object()),
        output_sample_rate=16000,
        generate_group=lambda *args, **kwargs: [endpoint],
    )
    config = {
        "normalization": {
            "target_dbfs": -25.0,
            "peak_ceiling": 0.99,
            "output_subtype": "PCM_16",
        },
        "data": {"noisy_dir": str(tmp_path), "clean_dir": str(tmp_path)},
        "evaluation": {"latent_seed_base": 1, "paired_metrics": False},
        "rollout": {"evaluation_nfe": 1},
    }
    common = {
        "bundle": bundle,
        "state": {"adapter": torch.zeros(1)},
        "policy": "released_model",
        "step": 0,
        "manifest": {"p001_001": "text"},
        "config": config,
        "conditioning": object(),
        "dnsmos": object(),
        "fidelity": None,
        "state_source": {"test": 1},
    }
    _evaluate_state(
        **common,
        state_id="legacy",
        output_dir=tmp_path / "legacy",
    )
    shared = object()
    _evaluate_state(
        **common,
        state_id="composite",
        output_dir=tmp_path / "composite",
        composite_evaluators=shared,
        reward_definition={"name": "flowse_grpo_public_composite"},
    )
    assert captured == [None, shared]


def test_composite_evaluator_is_loaded_once_and_verified(monkeypatch):
    shared = object()
    calls = []
    monkeypatch.setattr(
        checkpoint_module,
        "resolve_training_reward",
        lambda config: {"name": "flowse_grpo_public_composite"},
    )
    monkeypatch.setattr(
        checkpoint_module,
        "load_flowse_grpo_composite_evaluators",
        lambda config: (calls.append("load") or shared, {"frozen": True}),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "verify_reward_calibration",
        lambda config, **kwargs: {"verified": kwargs["evaluator_fingerprint"]},
    )
    definition, evaluator, fingerprint, verification = _load_reward_evaluators({})
    assert calls == ["load"]
    assert evaluator is shared
    assert fingerprint == {"frozen": True}
    assert verification == {"verified": {"frozen": True}}
    assert definition["name"] == "flowse_grpo_public_composite"


def test_heldout_composite_scalar_is_reconstructed_from_reported_components():
    class Composite:
        @staticmethod
        def score(clean_path, audio_path):
            return {
                "eres2net_speaker_similarity": 0.8,
                "speechbertscore": 0.7,
            }

    definition = {
        "name": "flowse_grpo_public_composite",
        "frozen_component_stds": {
            "dnsmos": 0.1,
            "speaker": 0.2,
            "speechbertscore": 0.05,
        },
        "weights": {"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
    }
    metrics = _score_evaluation_file(
        audio_path=Path("estimate.wav"),
        clean_path=Path("clean.wav"),
        transcript="",
        dnsmos=lambda path: {"dnsmos_ovrl": 3.2},
        fidelity=None,
        paired=False,
        composite_evaluators=Composite(),
        reward_definition=definition,
    )
    assert metrics["flowse_grpo_composite_reward"] == pytest.approx(
        0.6 * 8.0 + 4.0 + 14.0
    )


def test_reference_free_scoring_never_reads_clean_dependent_evaluators():
    class Fidelity:
        def speaker(self, clean_path, audio_path):
            raise AssertionError("speaker evaluator must not run")

        def asr(self, transcript, audio_path):
            raise AssertionError("ASR must not run without a transcript")

    class Composite:
        def score(self, clean_path, audio_path):
            raise AssertionError("composite evaluator must not run")

    metrics = _score_evaluation_file(
        audio_path=Path("estimate.wav"),
        clean_path=Path("missing-clean.wav"),
        transcript="",
        dnsmos=lambda path: {"dnsmos_ovrl": 3.2},
        fidelity=Fidelity(),
        paired=False,
        reference_free=True,
        composite_evaluators=Composite(),
        reward_definition={"name": "flowse_grpo_public_composite"},
    )
    assert metrics == {"dnsmos_ovrl": 3.2}


def test_composite_metrics_receive_utterance_and_speaker_cluster_intervals():
    base = _rows([3.0, 3.1, 3.2, 3.3])
    state = _rows([3.1, 3.2, 3.3, 3.4])
    for index, row in enumerate(base):
        row.update(
            {
                "flowse_grpo_composite_reward": 10.0 + index,
                "eres2net_speaker_similarity": 0.80 + index * 0.01,
                "speechbertscore": 0.70 + index * 0.01,
            }
        )
        state[index].update(
            {
                "flowse_grpo_composite_reward": 10.5 + index,
                "eres2net_speaker_similarity": 0.81 + index * 0.01,
                "speechbertscore": 0.72 + index * 0.01,
            }
        )
    result = paired_state_analysis(
        base, state, seed=31, samples=200, confidence=0.95
    )
    for metric in (
        "flowse_grpo_composite_reward",
        "eres2net_speaker_similarity",
        "speechbertscore",
    ):
        assert "utterance_ci" in result["metrics"][metric]
        assert "speaker_ci" in result["metrics"][metric]


def test_utterance_only_analysis_omits_fake_speaker_statistics():
    result = paired_state_analysis(
        _rows([3.0, 3.1, 3.2, 3.3]),
        _rows([3.1, 3.2, 3.3, 3.4]),
        seed=31,
        samples=200,
        confidence=0.95,
        include_speaker_ci=False,
    )
    assert "speaker_ci" not in result["metrics"]["dnsmos_ovrl"]
    assert result["ovrl_concentration"]["speaker_statistics_available"] is False


def test_paired_analysis_rejects_different_latents():
    base = _rows([3.0, 3.1, 3.2, 3.3])
    state = _rows([3.1, 3.2, 3.3, 3.4])
    state[2]["latent_seed"] += 1
    with pytest.raises(ValueError, match="latent seed differs"):
        paired_state_analysis(
            base,
            state,
            seed=19,
            samples=100,
            confidence=0.95,
        )


def test_online_minus_ema_direction_is_online_minus_ema():
    ema = _rows([3.0, 3.1, 3.2, 3.3])
    online = _rows([3.2, 3.3, 3.4, 3.5])
    result = paired_policy_analysis(
        ema,
        online,
        seed=23,
        samples=200,
        confidence=0.95,
    )
    assert result["metrics"]["dnsmos_ovrl"]["delta_mean"] == pytest.approx(0.2)


def test_registered_primary_is_always_ema_step_20():
    def metric(delta, low, high):
        return {
            "delta_mean": delta,
            "utterance_ci": {"ci_low": low, "ci_high": high},
            "speaker_ci": {"ci_low": low, "ci_high": high},
        }

    ovrl = metric(0.012, 0.002, 0.020)
    safe_metrics = {
        "dnsmos_ovrl": ovrl,
        "dnsmos_sig": metric(0.003, -0.001, 0.007),
        "speaker_similarity": metric(0.0, -0.001, 0.001),
        "stoi": metric(0.0, -0.001, 0.001),
        "pesq_wb": metric(-0.01, -0.02, 0.0),
        "wer": metric(-0.002, -0.01, 0.005),
    }
    states = {
        "ema_step_000020": {
            "analysis": {"metrics": safe_metrics}
        },
        "online_step_000015": {
            "analysis": {
                "metrics": {
                    "dnsmos_ovrl": metric(0.5, 0.4, 0.6)
                }
            }
        },
    }
    config = {
        "pilot_decision": {
            "dnsmos_ovrl_minimum_gain": 0.01,
            "require_ci_positive": True,
            "safety": {
                "dnsmos_sig": -0.03,
                "speaker_similarity": -0.02,
                "stoi": -0.005,
                "pesq_wb": -0.05,
                "wer": 0.02,
            },
        }
    }
    result = registered_primary_result(states, config)
    assert result["comparison"] == "ema_step_000020_minus_base"
    assert result["delta_mean"] == pytest.approx(0.012)
    assert result["status"] == "PILOT-SUPPORT"

    states["ema_step_000020"]["analysis"]["metrics"]["wer"] = metric(
        0.2, 0.1, 0.3
    )
    unsafe = registered_primary_result(states, config)
    assert unsafe["status"] == "PILOT-NO-SUPPORT"
    assert not unsafe["criteria"]["wer_safety"]


def test_confirmation_distinguishes_small_effect_from_original_threshold():
    def metric(delta, low, high):
        return {
            "delta_mean": delta,
            "utterance_ci": {"ci_low": low, "ci_high": high},
            "speaker_ci": {"ci_low": low, "ci_high": high},
        }

    analysis = {
        "metrics": {
            "dnsmos_ovrl": metric(0.007, 0.002, 0.012),
            "dnsmos_sig": metric(0.004, -0.001, 0.009),
            "speaker_similarity": metric(0.0, -0.001, 0.001),
            "stoi": metric(0.0, -0.001, 0.001),
            "pesq_wb": metric(-0.01, -0.02, 0.0),
            "wer": metric(-0.002, -0.01, 0.005),
        }
    }
    config = {
        "pilot_decision": {
            "dnsmos_ovrl_minimum_gain": 0.01,
            "safety": {
                "dnsmos_sig": -0.03,
                "speaker_similarity": -0.02,
                "stoi": -0.005,
                "pesq_wb": -0.05,
                "wer": 0.02,
            },
        }
    }
    result = confirmation_decision(analysis, config)
    assert result["decision"] == "CONFIRMATION-SMALL-EFFECT-SUPPORT"
    assert result["medium_scale_training_authorized"]
    assert not result["full_training_authorized"]

    analysis["metrics"]["dnsmos_ovrl"]["speaker_ci"]["ci_low"] = -0.001
    blocked = confirmation_decision(analysis, config)
    assert blocked["decision"] == "CONFIRMATION-NO-SUPPORT"
    assert not blocked["medium_scale_training_authorized"]


def test_registered_primary_uses_frozen_final_optimizer_step():
    def metric(delta, low, high):
        return {
            "delta_mean": delta,
            "utterance_ci": {"ci_low": low, "ci_high": high},
            "speaker_ci": {"ci_low": low, "ci_high": high},
        }

    metrics = {
        "dnsmos_ovrl": metric(0.02, 0.01, 0.03),
        "dnsmos_sig": metric(0.01, 0.0, 0.02),
        "speaker_similarity": metric(0.0, -0.001, 0.001),
        "stoi": metric(0.0, -0.001, 0.001),
        "pesq_wb": metric(0.0, -0.001, 0.001),
        "wer": metric(0.0, -0.001, 0.001),
    }
    config = {
        "run": {"optimizer_steps": 100},
        "evaluation": {},
        "pilot_decision": {
            "dnsmos_ovrl_minimum_gain": 0.01,
            "require_ci_positive": True,
            "safety": {
                "dnsmos_sig": -0.03,
                "speaker_similarity": -0.02,
                "stoi": -0.005,
                "pesq_wb": -0.05,
                "wer": 0.02,
            },
        },
    }
    result = registered_primary_result(
        {"ema_step_000100": {"analysis": {"metrics": metrics}}}, config
    )
    assert result["primary_checkpoint_step"] == 100
    assert result["comparison"] == "ema_step_000100_minus_base"
    assert result["status"] == "PILOT-SUPPORT"

