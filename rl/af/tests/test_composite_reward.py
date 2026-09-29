import hashlib
import math
import json
import sys

import numpy as np
import pytest
import rl.af.reward_evaluators as composite_module

from rl.af.reward_evaluators import (
    ModelScopeERes2NetEvaluator,
    SpeechBERTScoreEvaluator,
    composite_reward,
    evaluator_fingerprint_sha256,
    population_std,
)
from rl.rewards.calibrate_composite import (
    calibration_cache_row_matches,
    select_train_only_rows,
    summarize_calibration,
    summarize_calibration_stability,
    summarize_preflight,
)
from rl.rewards.calibration_audit import (
    summarize_replication_audit,
)
from rl.rewards.calibration_selection import (
    select_conditions,
)
from rl.rewards.specification import (
    FLOWSE_GRPO_COMPOSITE,
    compute_training_reward,
    resolve_training_reward,
    verify_reward_calibration,
)


def _evaluator_fingerprint(revision="frozen"):
    return {
        "schema_version": 1,
        "speaker": {"revision": revision, "implementation": "cosine_v1"},
        "speechbertscore": {"layer": 14, "implementation": "precision_v1"},
    }


def test_paper_composite_uses_published_weights_and_population_stds():
    result = composite_reward(
        {"dnsmos": 0.8, "speaker": 0.9, "speechbertscore": 0.7},
        component_stds={"dnsmos": 0.2, "speaker": 0.3, "speechbertscore": 0.1},
    )
    assert result["normalized_components"] == pytest.approx(
        {"dnsmos": 4.0, "speaker": 3.0, "speechbertscore": 7.0}
    )
    assert result["weighted_components"] == pytest.approx(
        {"dnsmos": 2.4, "speaker": 3.0, "speechbertscore": 7.0}
    )
    assert result["reward"] == pytest.approx(12.4)


def test_population_std_is_ddof_zero():
    assert population_std([1.0, 2.0, 3.0]) == pytest.approx(np.std([1, 2, 3], ddof=0))


def test_calibration_summarizes_three_components_and_candidate_geometry():
    rows = []
    for utterance, offset in (("p1_a", 0.0), ("p2_a", 0.1)):
        for candidate in range(4):
            rows.append(
                {
                    "utterance": utterance,
                    "latent_seed": candidate,
                    "dnsmos_ovrl": 3.0 + offset + candidate * 0.04,
                    "eres2net_speaker_similarity": 0.8 + candidate * 0.01,
                    "speechbertscore": 0.7 + offset + candidate * 0.02,
                }
            )
    summary = summarize_calibration(
        rows,
        weights={"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
    )
    assert set(summary["component_stds"]) == {
        "dnsmos",
        "speaker",
        "speechbertscore",
    }
    assert summary["reward"]["per_condition_range_mean"] > 0.0
    assert summary["reward"]["top2_bottom2_gap_mean"] > 0.0
    assert len(summary["enriched_rows"]) == 8


def test_single_row_preflight_does_not_require_nonzero_component_stds():
    summary = summarize_preflight(
        [
            {
                "utterance": "p1_a",
                "dnsmos_ovrl": 3.2,
                "eres2net_speaker_similarity": 0.8,
                "speechbertscore": 0.7,
            }
        ],
        weights={"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
    )
    assert summary["component_stds"] == {
        "dnsmos": 0.0,
        "speaker": 0.0,
        "speechbertscore": 0.0,
    }
    assert summary["reward"]["computed"] is False


def test_calibration_stability_uses_nested_condition_cluster_bootstrap():
    rows = []
    for condition in range(8):
        for candidate in range(2):
            rows.append(
                {
                    "utterance": f"speaker_{condition}",
                    "condition_index": condition,
                    "candidate_index": candidate,
                    "dnsmos_ovrl": 3.0 + condition * 0.03 + candidate * 0.01,
                    "eres2net_speaker_similarity": (
                        0.7 + condition * 0.02 + candidate * 0.005
                    ),
                    "speechbertscore": (
                        0.6 + condition * 0.01 + candidate * 0.002
                    ),
                }
            )
    stability = summarize_calibration_stability(
        rows,
        prefix_condition_counts=[2, 4, 8],
        candidates_per_condition=2,
        bootstrap_replicates=200,
        bootstrap_seed=7,
        max_last_prefix_relative_change=0.9,
        max_full_relative_ci_width=0.9,
    )
    assert stability["bootstrap_unit"] == "condition"
    assert [row["conditions"] for row in stability["prefix_estimates"]] == [2, 4, 8]
    assert set(stability["checks"]) == {
        "previous_prefix_inside_full_bootstrap_ci",
        "full_prefix_relative_ci_width",
    }
    assert stability["diagnostics"]["last_prefix_relative_change_target"][
        "gating"
    ] is False
    assert stability["prefix_estimates"][-1]["component_stds"] == pytest.approx(
        {
            "dnsmos": np.std(
                [float(row["dnsmos_ovrl"]) / 4.0 for row in rows], ddof=0
            ),
            "speaker": np.std(
                [float(row["eres2net_speaker_similarity"]) for row in rows],
                ddof=0,
            ),
            "speechbertscore": np.std(
                [float(row["speechbertscore"]) for row in rows], ddof=0
            ),
        }
    )


def test_calibration_stability_rejects_incomplete_candidate_group():
    rows = [
        {
            "utterance": f"u{condition}",
            "condition_index": condition,
            "candidate_index": 0,
            "dnsmos_ovrl": 3.0 + condition,
            "eres2net_speaker_similarity": 0.7 + condition * 0.01,
            "speechbertscore": 0.6 + condition * 0.01,
        }
        for condition in range(2)
    ]
    with pytest.raises(ValueError, match="complete K group"):
        summarize_calibration_stability(
            rows,
            prefix_condition_counts=[2],
            candidates_per_condition=2,
            bootstrap_replicates=100,
            bootstrap_seed=7,
            max_last_prefix_relative_change=0.02,
            max_full_relative_ci_width=0.15,
        )


def test_calibration_selection_excludes_non_train_rows_before_statistics():
    rows = [
        {"utterance": "train_a", "nfe": 10, "latent_seed": 1},
        {"utterance": "valid_a", "nfe": 10, "latent_seed": 2},
        {"utterance": "train_a", "nfe": 32, "latent_seed": 3},
        {"utterance": "failed", "nfe": 10, "latent_seed": 4, "error": "x"},
    ]
    selected, audit = select_train_only_rows(
        rows, train_utterances={"train_a"}, source_nfe=10
    )
    assert [row["latent_seed"] for row in selected] == [1]
    assert audit == {
        "eligible_rows_before_train_filter": 2,
        "excluded_outside_train_rows": 1,
        "excluded_outside_train_utterances": ["valid_a"],
        "selected_train_rows": 1,
        "selected_train_utterances": 1,
    }


def test_calibration_selection_rejects_duplicate_source_keys():
    row = {"utterance": "train_a", "nfe": 10, "latent_seed": 1}
    with pytest.raises(ValueError, match="duplicate calibration source row key"):
        select_train_only_rows(
            [row, dict(row)], train_utterances={"train_a"}, source_nfe=10
        )


def test_libritts_calibration_blocks_are_disjoint_permutation_slices():
    manifest = {f"u{index}": "" for index in range(20)}
    block_a = select_conditions(manifest, count=8, seed=17, offset=0)
    block_b = select_conditions(manifest, count=8, seed=17, offset=8)
    assert len(block_a) == len(block_b) == 8
    assert set(block_a).isdisjoint(block_b)
    assert block_a + block_b == select_conditions(
        manifest, count=16, seed=17, offset=0
    )


def test_disjoint_replication_audit_passes_identical_block_distributions():
    rows_by_block = {"A": [], "B": []}
    speakers = {}
    for block in rows_by_block:
        for condition in range(8):
            utterance = f"{block}_{condition}"
            speakers[utterance] = f"speaker_{condition // 2}"
            for candidate in range(4):
                rows_by_block[block].append(
                    {
                        "utterance": utterance,
                        "candidate_index": candidate,
                        "dnsmos_ovrl": 3.0 + condition * 0.03 + candidate * 0.02,
                        "eres2net_speaker_similarity": (
                            0.7 + condition * 0.02 + candidate * 0.01
                        ),
                        "speechbertscore": (
                            0.6 + condition * 0.01 + candidate * 0.015
                        ),
                    }
                )
    audit = summarize_replication_audit(
        rows_by_block,
        speaker_by_utterance=speakers,
        candidates_per_condition=4,
        bootstrap_config={
            "replicates": 100,
            "seed": 3,
            "max_condition_relative_ci_width": 2.0,
            "max_speaker_relative_ci_width": 2.0,
        },
        replication_config={"max_block_relative_std_difference": 0.1},
        sensitivity_config={
            "conditions_per_af_batch": 2,
            "af_batch_replicates": 100,
            "seed": 5,
            "advantage_clip": 1.0,
            "min_reward_spearman_mean": 0.99,
            "min_reward_spearman_p05": 0.99,
            "min_top2_bottom2_exact_agreement": 0.99,
            "min_advantage_pearson_mean": 0.99,
            "min_advantage_pearson_p05": 0.99,
            "min_advantage_sign_agreement": 0.99,
        },
        weights={"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0},
    )
    assert audit["audit_status"] == "AUDIT-PASS"
    assert all(audit["checks"].values())
    assert audit["combined_component_stds"] == pytest.approx(
        audit["block_component_stds"]["A"]
    )


def test_eres2net_embedding_output_is_reduced_to_cosine():
    cosine = ModelScopeERes2NetEvaluator._cosine_from_embeddings(
        {"embs": [[1.0, 0.0], [1.0, 1.0]]}
    )
    assert cosine == pytest.approx(1.0 / math.sqrt(2.0))


def test_composite_loader_honors_explicit_worker_device(monkeypatch):
    observed = []

    class FakeDevice:
        def __init__(self, value):
            value = str(value)
            self.type = value.split(":", 1)[0]
            self.index = int(value.split(":", 1)[1]) if ":" in value else None

        def __str__(self):
            return self.type if self.index is None else f"{self.type}:{self.index}"

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

    class FakeTorch:
        cuda = FakeCuda()
        device = FakeDevice

    class FakeResolved:
        pass

    class FakeModel:
        def to(self, device):
            observed.append(("speaker_model_move", str(device)))
            return self

    class FakePipeline:
        def __init__(self):
            self.model = FakeModel()
            self.device = FakeDevice("cpu")

    class FakeSpeaker:
        def __init__(self, config, *, device):
            observed.append(("speaker", device))
            self._pipeline = FakePipeline()

        def fingerprint(self):
            return {"kind": "speaker"}

    class FakeSpeechBERTScore:
        def __init__(self, resolved, *, device, layer, reference_cache_size):
            observed.append(("speechbertscore", device))

        def fingerprint(self):
            return {"kind": "speechbertscore"}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(
        composite_module._common,
        "resolve_hf_model",
        lambda config: FakeResolved(),
    )
    monkeypatch.setattr(
        composite_module._common, "ModelScopeERes2NetEvaluator", FakeSpeaker
    )
    monkeypatch.setattr(
        composite_module._common, "SpeechBERTScoreEvaluator", FakeSpeechBERTScore
    )
    config = {
        "composite_reward_evaluators": {
            "device": "cuda",
            "speaker": {"model_id": "speaker", "revision": "v1"},
            "speechbertscore": {"repo_id": "wavlm", "revision": "v1"},
        }
    }
    evaluators, fingerprint = composite_module.load_flowse_grpo_composite_evaluators(
        config, device_override="cuda:3"
    )
    assert observed == [
        ("speaker", "cpu"),
        ("speaker_model_move", "cuda:3"),
        ("speechbertscore", "cuda:3"),
    ]
    assert str(evaluators.speaker._pipeline.device) == "cuda:3"
    assert fingerprint == evaluators.fingerprint()


def test_composite_loader_without_override_delegates_to_common(monkeypatch):
    sentinel = (object(), {"fingerprint": "common"})
    config = {"composite_reward_evaluators": {}}
    monkeypatch.setattr(
        composite_module._common,
        "load_flowse_grpo_composite_evaluators",
        lambda received: sentinel if received is config else None,
    )
    assert composite_module.load_flowse_grpo_composite_evaluators(config) is sentinel


def test_speechbertscore_golden_precision_uses_generated_to_reference_direction():
    torch = pytest.importorskip("torch")
    generated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference = torch.tensor([[1.0, 0.0]])
    precision = SpeechBERTScoreEvaluator.precision_from_normalized_features(
        generated, reference
    )
    assert float(precision) == pytest.approx(0.5)


def test_eres2net_local_directory_fingerprint_changes_with_content(tmp_path):
    from rl.rewards.evaluators import sha256_tree

    snapshot = tmp_path / "eres2net"
    snapshot.mkdir()
    weight = snapshot / "model.bin"
    weight.write_bytes(b"first")
    first = sha256_tree(snapshot)
    weight.write_bytes(b"second")
    second = sha256_tree(snapshot)
    assert first != second

    evaluator = ModelScopeERes2NetEvaluator.__new__(ModelScopeERes2NetEvaluator)
    evaluator.model_id = "iic/test"
    evaluator.revision = "v1"
    evaluator.model_source = str(snapshot)
    evaluator.model_source_type = "local_directory"
    evaluator.model_source_sha256 = second
    assert evaluator.fingerprint()["model_source_sha256"] == second


def test_calibration_cache_is_bound_to_complete_evaluator_fingerprint():
    first = evaluator_fingerprint_sha256(_evaluator_fingerprint("a"))
    second = evaluator_fingerprint_sha256(_evaluator_fingerprint("b"))
    cached = {
        "audio_sha256": "audio",
        "clean_sha256": "clean",
        "evaluator_fingerprint_sha256": first,
    }
    assert calibration_cache_row_matches(
        cached,
        audio_sha256="audio",
        clean_sha256="clean",
        evaluator_sha256=first,
    )
    assert not calibration_cache_row_matches(
        cached,
        audio_sha256="audio",
        clean_sha256="clean",
        evaluator_sha256=second,
    )


def test_training_reward_loads_complete_calibration_report(tmp_path):
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text(json.dumps({"p1_a": ""}), encoding="utf-8")
    train_sha256 = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
    report = {
        "status": "CALIBRATION-COMPLETE",
        "source": {
            "source_nfe": 10,
            "rows": 512,
            "std_ddof": 0,
            "dnsmos_divisor": 4.0,
            "train_manifest_sha256": train_sha256,
            "eligible_rows_before_train_filter": 512,
            "selected_train_rows": 512,
            "selected_train_utterances": 1,
            "excluded_outside_train_rows": 0,
            "excluded_outside_train_utterances": [],
        },
        "component_stds": {
            "dnsmos": 0.1,
            "speaker": 0.2,
            "speechbertscore": 0.05,
        },
        "evaluators": _evaluator_fingerprint(),
    }
    report_path = tmp_path / "calibration_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "data": {"train_manifest": str(train_manifest)},
        "training_reward": {
            "name": FLOWSE_GRPO_COMPOSITE,
            "component_normalization": "frozen_std",
            "weights": {
                "dnsmos": 0.6,
                "speaker": 1.0,
                "speechbertscore": 1.0,
            },
            "calibration": {
                "report_path": str(report_path),
                "source_nfe": 10,
                "std_ddof": 0,
                "dnsmos_divisor": 4.0,
            },
        }
    }
    definition = resolve_training_reward(config)
    result = compute_training_reward(
        {
            "dnsmos_ovrl": 3.2,
            "eres2net_speaker_similarity": 0.8,
            "speechbertscore": 0.7,
        },
        definition,
    )
    assert result["reward"] == pytest.approx(0.6 * 8.0 + 4.0 + 14.0)
    verification = verify_reward_calibration(
        config, evaluator_fingerprint=_evaluator_fingerprint()
    )
    assert all(verification["criteria"].values())

    mismatched = verify_reward_calibration(
        config, evaluator_fingerprint=_evaluator_fingerprint("tampered")
    )
    assert mismatched["criteria"]["evaluator_fingerprint_matches"] is False


def test_smoke_manifest_may_be_verified_subset_of_calibration_domain(tmp_path):
    domain_manifest = tmp_path / "domain.json"
    smoke_manifest = tmp_path / "smoke.json"
    domain_manifest.write_text(
        json.dumps({"p1_a": "one", "p2_a": "two"}), encoding="utf-8"
    )
    smoke_manifest.write_text(json.dumps({"p1_a": "one"}), encoding="utf-8")
    domain_sha256 = hashlib.sha256(domain_manifest.read_bytes()).hexdigest()
    report = {
        "status": "CALIBRATION-COMPLETE",
        "source": {
            "source_nfe": 10,
            "rows": 16,
            "std_ddof": 0,
            "dnsmos_divisor": 4.0,
            "train_manifest_sha256": domain_sha256,
            "eligible_rows_before_train_filter": 16,
            "selected_train_rows": 16,
            "selected_train_utterances": 2,
            "excluded_outside_train_rows": 0,
            "excluded_outside_train_utterances": [],
        },
        "component_stds": {
            "dnsmos": 0.1,
            "speaker": 0.2,
            "speechbertscore": 0.05,
        },
        "evaluators": _evaluator_fingerprint(),
    }
    report_path = tmp_path / "calibration_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "data": {
            "train_manifest": str(smoke_manifest),
            "calibration_domain_manifest": str(domain_manifest),
        },
        "training_reward": {
            "name": FLOWSE_GRPO_COMPOSITE,
            "component_normalization": "frozen_std",
            "weights": {
                "dnsmos": 0.6,
                "speaker": 1.0,
                "speechbertscore": 1.0,
            },
            "calibration": {
                "report_path": str(report_path),
                "source_nfe": 10,
                "std_ddof": 0,
                "dnsmos_divisor": 4.0,
            },
        },
    }
    verification = verify_reward_calibration(
        config, evaluator_fingerprint=_evaluator_fingerprint()
    )
    assert all(verification["criteria"].values())
    assert verification["criteria"][
        "calibration_domain_manifest_matches_report"
    ]
    assert verification["criteria"][
        "run_train_manifest_is_subset_of_calibration_domain"
    ]
    smoke_manifest.write_text(json.dumps({"p1_a": "changed"}), encoding="utf-8")
    changed = verify_reward_calibration(
        config, evaluator_fingerprint=_evaluator_fingerprint()
    )
    assert changed["criteria"]["calibration_domain_manifest_matches_report"] is True
    assert changed["criteria"]["run_train_manifest_is_subset_of_calibration_domain"] is False

