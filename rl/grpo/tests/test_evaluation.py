import pytest
import torch

from rl.grpo.evaluation import (
    _best_safe_candidate,
    format_validation_comparison_table,
    load_grpo_online_checkpoint,
    reuse_released_base_for_zero_lora_validation,
    select_milestone_checkpoints,
)
from rl.grpo.protocol import lora_state_fingerprint


def _checkpoint_payload(**overrides):
    state = {"block.lora_A": torch.tensor([[1.0, 2.0]])}
    payload = {
        "schema_version": 1,
        "method": "flowse_grpo",
        "policy_kind": "grpo_online",
        "ema_enabled": False,
        "collection_boundary": True,
        "config_sha256": "config-hash",
        "online_lora_state": state,
        "online_lora_fingerprint": lora_state_fingerprint(state),
    }
    payload.update(overrides)
    return payload


def test_validation_comparison_table_matches_training_style():
    table = format_validation_comparison_table(
        base_report={
            "summary": {
                "dnsmos_ovrl": {"mean": 3.2},
                "speechbertscore": {"mean": 0.85},
            }
        },
        state_report={
            "summary": {
                "dnsmos_ovrl": {"mean": 3.25},
                "speechbertscore": {"mean": 0.86},
            }
        },
        optimizer_step=1000,
        collection_index=250,
        percentage=20,
    )

    assert "GRPO validation after optimizer step 1000 (collection 250, 20%)" in table
    assert "Released base" in table
    assert "GRPO online" in table
    assert "DNSMOS OVRL" in table
    assert "+0.05000" in table
    assert "SpeechBERTScore" in table
    assert "+0.01000" in table


@pytest.mark.parametrize(
    "override",
    [
        {"method": "advantageflow"},
        {"policy_kind": "ema"},
        {"ema_enabled": True},
    ],
)
def test_checkpoint_loader_rejects_non_grpo_online_state(tmp_path, override):
    path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(**override), path)
    with pytest.raises(ValueError, match="compatible GRPO-online"):
        load_grpo_online_checkpoint(path, expected_config_hash="config-hash")


def test_zero_lora_validation_reuses_released_base_rows(tmp_path):
    base = {
        "schema_version": 1,
        "state_id": "released_base_cfg0_nfe32",
        "policy_kind": "released_base",
        "percentage": None,
        "collection_index": 0,
        "evaluation_split": "validation",
        "evaluation_setting": "base_cfg0_nfe32",
        "state_source": {"policy_kind": "released_base"},
        "summary": {"dnsmos_ovrl": {"mean": 3.0, "std": 0.0, "count": 1}},
        "rows": [
            {
                "state_id": "released_base_cfg0_nfe32",
                "policy_kind": "released_base",
                "percentage": None,
                "collection_index": 0,
                "utterance": "p001_001",
                "scored_wav_sha256": "wav-hash",
                "terminal_mel_sha256": "mel-hash",
                "dnsmos_ovrl": 3.0,
            }
        ],
        "timing_seconds": 123.0,
    }
    state = {
        "first.lora_A": torch.ones(2, 2),
        "first.lora_B": torch.zeros(2, 2),
        "second.lora_A": torch.ones(2, 2),
        "second.lora_B": torch.zeros(2, 2),
    }
    report, seconds = reuse_released_base_for_zero_lora_validation(
        base_report=base,
        lora_state=state,
        state_id="grpo_online_000pct",
        percentage=0,
        collection_index=0,
        checkpoint_path="milestone-0.pt",
        config={"lora": {"expected_modules": 2}, "evaluation": {"latent_seed_base": 7}},
        output_dir=tmp_path,
        milestone_metadata={"percentage": 0},
    )
    assert seconds == 0.0
    assert report["evaluation_reused"] is True
    assert report["timing_seconds"] == 0.0
    assert report["rows"][0]["scored_wav_sha256"] == "wav-hash"
    assert report["rows"][0]["policy_kind"] == "grpo_online"
    assert report["summary"] == base["summary"]

    nonzero = {**state, "first.lora_B": torch.ones(2, 2)}
    with pytest.raises(ValueError, match="not released-base equivalent"):
        reuse_released_base_for_zero_lora_validation(
            base_report=base,
            lora_state=nonzero,
            state_id="invalid",
            percentage=0,
            collection_index=0,
            checkpoint_path="invalid.pt",
            config={
                "lora": {"expected_modules": 2},
                "evaluation": {"latent_seed_base": 7},
            },
            output_dir=tmp_path,
            milestone_metadata={"percentage": 0},
        )


def test_comparison_selection_tie_breaks_by_ovrl_then_earlier(tmp_path, monkeypatch):
    reports = [
        {
            "percentage": percentage,
            "collection_index": percentage,
            "summary": {
                "flowse_grpo_composite_reward": {"mean": composite},
                "dnsmos_ovrl": {"mean": ovrl},
            },
            "rows": [],
        }
        for percentage, composite, ovrl in [
            (0, 1.0, 2.0),
            (50, 1.1, 2.1),
            (100, 1.1, 2.1),
        ]
    ]
    monkeypatch.setattr(
        "rl.grpo.evaluation._best_safe_candidate",
        lambda base, values, config: (None, []),
    )
    config = {"evaluation": {"selection_milestones": [0, 50, 100]}}
    result = select_milestone_checkpoints(
        base_report={"rows": []},
        milestone_reports=reports,
        checkpoint_paths={value: f"{value}.pt" for value in (0, 50, 100)},
        config=config,
        output_dir=tmp_path,
    )
    assert result["comparison_checkpoint"]["percentage"] == 50


def test_registered_six_node_selection_uses_common_layer_and_safe_callback(
    tmp_path, monkeypatch
):
    percentages = (0, 20, 40, 60, 80, 100)
    reports = [
        {
            "percentage": percentage,
            "collection_index": percentage,
            "summary": {
                "flowse_grpo_composite_reward": {"mean": float(percentage)},
                "dnsmos_ovrl": {"mean": float(percentage)},
            },
            "rows": [],
        }
        for percentage in percentages
    ]
    checkpoint_paths = {}
    for percentage in percentages:
        path = tmp_path / f"{percentage}.pt"
        path.write_bytes(str(percentage).encode())
        checkpoint_paths[percentage] = str(path)
    monkeypatch.setattr(
        "rl.grpo.evaluation._best_safe_candidate",
        lambda base, values, config: ({"percentage": 20}, [{"percentage": 20}]),
    )

    result = select_milestone_checkpoints(
        base_report={"rows": []},
        milestone_reports=reports,
        checkpoint_paths=checkpoint_paths,
        config={"evaluation": {"selection_milestones": list(percentages)}},
        output_dir=tmp_path,
    )
    assert result["selection_opportunities"] == list(percentages)
    assert result["comparison_checkpoint"]["percentage"] == 100
    assert result["best_safe_checkpoint"]["percentage"] == 20
    assert result["best_safe_checkpoint"]["checkpoint"].endswith("20.pt")


def test_best_safe_truthfully_returns_none(monkeypatch):
    def fake_analysis(*args, **kwargs):
        return {
            "metrics": {
                "dnsmos_ovrl": {
                    "utterance_ci": {"ci_low": -0.1, "ci_high": 0.1},
                    "speaker_ci": {"ci_low": -0.1, "ci_high": 0.1},
                }
            }
        }

    monkeypatch.setattr(
        "rl.grpo.evaluation.paired_state_analysis", fake_analysis
    )
    base = {"rows": []}
    reports = [
        {
            "percentage": 0,
            "collection_index": 0,
            "rows": [],
            "summary": {
                "flowse_grpo_composite_reward": {"mean": 0.0},
                "dnsmos_ovrl": {"mean": 0.0},
            },
        }
    ]
    config = {
        "evaluation": {
            "best_safe": {
                "bootstrap_seed": 1,
                "bootstrap_samples": 10,
                "confidence": 0.95,
                "safety": {},
            }
        }
    }
    selected, candidates = _best_safe_candidate(base, reports, config=config)
    assert selected is None
    assert candidates[0]["eligible"] is False
