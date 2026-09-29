from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from rl.common.lora import (
    inject_lora,
    lora_parameters,
    snapshot_lora,
)
from rl.grpo.protocol import lora_state_fingerprint
from rl.grpo.trainer import (
    CollectionResult,
    _aggregate_mechanism_audits,
    _audit_trajectory_ids,
    _cleanup_worker_rollout_directory,
    _due_selection_milestones,
    _empty_accounting,
    _aggregate_minibatch_diagnostics,
    deterministic_full_buffer_update_batches,
    _gpu_time_budget_match,
    _lora_delta_statistics,
    _load_worker_rollout_shard,
    _mechanism_collection_summary,
    _merge_collection_shards,
    _numeric_distribution,
    _partition_group_indices,
    _record_training_accounting,
    _reconcile_accounting_sidecar,
    _selection_milestone_metadata,
    _worker_rollout_shard_path,
    _write_worker_rollout_shard,
    load_checkpoint,
    optimizer_update_accumulated,
    reporting_checkpoint_collections,
    resolve_physical_gpu_session,
    save_checkpoint,
    stable_seed,
    validate_grpo_config,
)


CONFIG_PATH = Path("configs/grpo/grpo_voicebank_controlled_cfg0_smoke.yaml")
CORRECTNESS_2GPU_CONFIG_PATH = Path(
    "configs/grpo/"
    "grpo_voicebank_controlled_cfg0_2gpu_300step_correctness.yaml"
)
FORMAL_5000_TEMPLATE_PATH = Path(
    "configs/grpo/"
    "grpo_voicebank_controlled_cfg0_4gpu_5000update_scaleup.template.yaml"
)


def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_controlled_smoke_config_is_valid_and_auditable():
    summary = validate_grpo_config(config())
    assert summary["policy_kind"] == "grpo_online"
    assert summary["ema_enabled"] is False
    assert summary["candidate_count_per_collection"] == 4
    assert summary["optimizer_steps"] == 2
    assert config()["optimizer"]["learning_rate"] == pytest.approx(2.0e-4)


def test_2gpu_300step_correctness_pilot_is_narrowly_authorized():
    candidate = yaml.safe_load(
        CORRECTNESS_2GPU_CONFIG_PATH.read_text(encoding="utf-8")
    )
    summary = validate_grpo_config(candidate)
    assert summary["mode"] == "pilot"
    assert summary["rollout_world_size"] == 2
    assert summary["candidate_count_per_collection"] == 720
    assert summary["optimizer_steps"] == 300
    assert summary["training_microbatch_size"] == 12
    assert summary["trajectory_consumption"] == "all_eligible_once_per_collection"
    assert candidate["optimizer"]["learning_rate"] == pytest.approx(2.0e-4)
    assert summary["mechanism_audit_checkpoint_collections"] == [
        0,
        1,
        5,
        10,
        20,
        40,
        60,
        75,
    ]

    wrong_role = deepcopy(candidate)
    wrong_role["comparison"]["budget_role"] = "shared_hyperparameter_pilot"
    with pytest.raises(ValueError, match="four rollout GPUs"):
        validate_grpo_config(wrong_role)

    wrong_horizon = deepcopy(candidate)
    wrong_horizon["run"]["collections"] = 74
    wrong_horizon["optimizer"]["schedule_total_steps"] = 296
    with pytest.raises(ValueError, match="75 collections"):
        validate_grpo_config(wrong_horizon)


def test_4gpu_5000update_scaleup_template_freezes_full_buffer_and_reporting_point():
    candidate = yaml.safe_load(
        FORMAL_5000_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    candidate["lora"]["shared_initial_snapshot"]["expected_state_sha256"] = "0" * 64
    summary = validate_grpo_config(candidate)
    assert summary["mode"] == "secondary"
    assert summary["rollout_world_size"] == 4
    assert summary["optimizer_steps"] == 5000
    assert summary["candidate_count_per_collection"] == 720
    assert summary["trajectory_consumption"] == "all_eligible_once_per_collection"
    assert summary["reporting_checkpoint_collections"] == [889]
    assert candidate["run"]["collections"] == 1250
    assert candidate["resources"]["expected_cuda_visible_devices"] == [4, 2, 0, 5]
    assert candidate["comparison"]["gpu_time_primary_eligible"] is False


def test_reporting_checkpoints_cannot_query_validation_or_enter_selection():
    candidate = yaml.safe_load(
        FORMAL_5000_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    assert reporting_checkpoint_collections(candidate) == [889]
    for field in ("selection_eligible", "validation_queried"):
        invalid = deepcopy(candidate)
        invalid["artifacts"]["reporting_checkpoints"][field] = True
        with pytest.raises(ValueError, match="cannot query validation"):
            reporting_checkpoint_collections(invalid)


def test_formal_resume_can_auditably_remap_only_physical_gpus():
    candidate = yaml.safe_load(
        FORMAL_5000_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    config_before = json.dumps(candidate, sort_keys=True)

    original = resolve_physical_gpu_session(
        candidate,
        resume=None,
        observed_cuda_visible_devices="4,2,0,5",
    )
    assert original["physical_gpu_remapped"] is False

    remapped = resolve_physical_gpu_session(
        candidate,
        resume=Path("checkpoint_latest.pt"),
        requested_resume_physical_gpu_ids="6,7,1,3",
        observed_cuda_visible_devices="6,7,1,3",
    )
    assert remapped["logical_device_ids"] == [0, 1, 2, 3]
    assert remapped["frozen_physical_gpu_ids"] == [4, 2, 0, 5]
    assert remapped["session_physical_gpu_ids"] == [6, 7, 1, 3]
    assert remapped["physical_gpu_remapped"] is True
    assert remapped["explicit_resume_remap_authorization"] is True
    assert json.dumps(candidate, sort_keys=True) == config_before

    unfrozen = deepcopy(candidate)
    unfrozen["resources"].pop("expected_cuda_visible_devices")
    implicit = resolve_physical_gpu_session(
        unfrozen,
        resume=None,
        observed_cuda_visible_devices="",
    )
    assert implicit["session_physical_gpu_ids"] == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="only with --resume"):
        resolve_physical_gpu_session(
            candidate,
            resume=None,
            requested_resume_physical_gpu_ids="6,7,1,3",
            observed_cuda_visible_devices="6,7,1,3",
        )
    with pytest.raises(RuntimeError, match="differs from --resume-physical-gpus"):
        resolve_physical_gpu_session(
            candidate,
            resume=Path("checkpoint_latest.pt"),
            requested_resume_physical_gpu_ids="6,7,1,3",
            observed_cuda_visible_devices="6,7,1,2",
        )
    with pytest.raises(RuntimeError, match="frozen mapping"):
        resolve_physical_gpu_session(
            candidate,
            resume=Path("checkpoint_latest.pt"),
            observed_cuda_visible_devices="6,7,1,3",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_eligible", True, "checkpoint selection"),
        ("run_validation", True, "must not query validation"),
        ("checkpoint_collections", [0, 1, 75], "mechanism checkpoints"),
        (
            "analysis_artifact_io_excluded_from_training_budget",
            False,
            "explicitly excluded",
        ),
    ],
)
def test_correctness_mechanism_audits_cannot_become_selection_opportunities(
    field, value, message
):
    candidate = yaml.safe_load(
        CORRECTNESS_2GPU_CONFIG_PATH.read_text(encoding="utf-8")
    )
    candidate["mechanism_audit"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_grpo_config(candidate)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sampler", "cfg_strength"), 1.0),
        (("evaluation", "nfe"), 10),
        (("ema", "enabled"), True),
        (("objective", "ratio"), "rationorm"),
        (("objective", "loss_dt_scaling"), True),
    ],
)
def test_controlled_config_rejects_unfair_or_nonplain_choices(path, value):
    candidate = deepcopy(config())
    candidate[path[0]][path[1]] = value
    if path == ("objective", "ratio"):
        # This invariant is checked explicitly below as part of this regression.
        with pytest.raises(ValueError):
            validate_grpo_config(candidate)
        return
    if path == ("objective", "loss_dt_scaling"):
        with pytest.raises(ValueError):
            validate_grpo_config(candidate)
        return
    with pytest.raises(ValueError):
        validate_grpo_config(candidate)


def test_topology_and_milestone_configuration_are_frozen():
    wrong_topology = deepcopy(config())
    wrong_topology["resources"]["rollout_world_size"] = 3
    wrong_topology["resources"]["device_ids"] = [0, 1, 2]
    with pytest.raises(ValueError, match="smoke geometry|milestone|topology|rollout"):
        validate_grpo_config(wrong_topology)

    wrong_milestones = deepcopy(config())
    wrong_milestones["evaluation"]["selection_milestones"] = [0, 50, 100]
    with pytest.raises(ValueError, match="selection milestones"):
        validate_grpo_config(wrong_milestones)


def test_audit_wav_selection_is_deterministic_and_bounded():
    trajectory_ids = [f"trajectory-{index}" for index in range(20)]
    first = _audit_trajectory_ids(trajectory_ids, count=3, seed=7)
    second = _audit_trajectory_ids(list(reversed(trajectory_ids)), count=3, seed=7)
    assert first == second
    assert len(first) == 3
    with pytest.raises(ValueError):
        _audit_trajectory_ids(trajectory_ids, count=21, seed=7)


def test_full_buffer_batching_consumes_every_eligible_trajectory_once():
    examples = [SimpleNamespace(trajectory_id=f"trajectory-{index}") for index in range(720)]
    optimizer_batches = deterministic_full_buffer_update_batches(
        examples, seed=19, updates=4, microbatch_size=12
    )
    assert len(optimizer_batches) == 4
    assert [sum(len(batch) for batch in update) for update in optimizer_batches] == [
        180,
        180,
        180,
        180,
    ]
    assert [len(update) for update in optimizer_batches] == [15, 15, 15, 15]
    used = [
        item.trajectory_id
        for update in optimizer_batches
        for microbatch in update
        for item in microbatch
    ]
    assert len(used) == 720
    assert len(set(used)) == 720
    assert set(used) == {item.trajectory_id for item in examples}

    uneven = deterministic_full_buffer_update_batches(
        examples[:710], seed=19, updates=4, microbatch_size=12
    )
    assert [sum(len(batch) for batch in update) for update in uneven] == [
        178,
        178,
        177,
        177,
    ]
    assert sum(len(batch) for update in uneven for batch in update) == 710


def test_accumulated_diagnostics_match_population_statistics():
    first = {
        "loss": 1.0,
        "policy_loss": 0.8,
        "reference_kl": 0.2,
        "weighted_reference_kl": 0.02,
        "ratio_mean": 1.0,
        "ratio_std": 0.0,
        "log_ratio_mean": 0.0,
        "log_ratio_std": 0.0,
        "log_ratio_abs_max": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "positive_clip_fraction": 0.0,
        "negative_clip_fraction": 0.0,
        "overflow_clamp_fraction": 0.0,
    }
    second = {**first, "ratio_mean": 3.0, "log_ratio_mean": 2.0, "log_ratio_abs_max": 2.0}
    combined = _aggregate_minibatch_diagnostics([(1, first), (3, second)])
    assert combined["ratio_mean"] == pytest.approx(2.5)
    assert combined["ratio_std"] == pytest.approx(math.sqrt(0.75))
    assert combined["log_ratio_mean"] == pytest.approx(1.5)
    assert combined["log_ratio_std"] == pytest.approx(math.sqrt(0.75))
    assert combined["log_ratio_abs_max"] == 2.0


def test_accumulated_update_matches_full_macro_batch_and_steps_once(monkeypatch):
    class CountingSGD(torch.optim.SGD):
        def __init__(self, parameters, *, lr):
            super().__init__(parameters, lr=lr)
            self.step_calls = 0
            self.zero_grad_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

        def zero_grad(self, set_to_none=True):
            self.zero_grad_calls += 1
            return super().zero_grad(set_to_none=set_to_none)

    class CountingScheduler:
        def __init__(self):
            self.step_calls = 0

        def step(self):
            self.step_calls += 1

    def objective_output(bundle, examples, *, conditioning, config):
        del conditioning, config
        inputs = torch.tensor([item.input_value for item in examples])
        targets = torch.tensor([item.target_value for item in examples])
        weight = bundle.model.transformer.weight.reshape(())
        loss = ((weight * inputs - targets) ** 2).mean()
        loss_value = float(loss.detach().item())
        diagnostics = {
            "loss": loss_value,
            "policy_loss": loss_value,
            "reference_kl": 0.0,
            "weighted_reference_kl": 0.0,
            "ratio_mean": 1.0,
            "ratio_std": 0.0,
            "log_ratio_mean": 0.0,
            "log_ratio_std": 0.0,
            "log_ratio_abs_max": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "positive_clip_fraction": 0.0,
            "negative_clip_fraction": 0.0,
            "overflow_clamp_fraction": 0.0,
        }
        return SimpleNamespace(loss=loss, diagnostics=diagnostics)

    examples = [
        SimpleNamespace(
            trajectory_id=f"trajectory-{index}",
            input_value=float(index + 1),
            target_value=float(2 * index - 1),
        )
        for index in range(5)
    ]
    microbatches = [examples[:2], examples[2:]]

    actual_model = torch.nn.Linear(1, 1, bias=False)
    reference_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actual_model.weight.fill_(0.25)
        reference_model.weight.copy_(actual_model.weight)

    reference_inputs = torch.tensor([item.input_value for item in examples])
    reference_targets = torch.tensor([item.target_value for item in examples])
    reference_loss = (
        (reference_model.weight.reshape(()) * reference_inputs - reference_targets) ** 2
    ).mean()
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss.backward()
    expected_gradient = reference_model.weight.grad.detach().clone()
    torch.nn.utils.clip_grad_norm_(reference_model.parameters(), max_norm=100.0)
    reference_optimizer.step()

    bundle = SimpleNamespace(
        model=SimpleNamespace(transformer=actual_model),
        device=torch.device("cpu"),
    )
    optimizer = CountingSGD(actual_model.parameters(), lr=0.05)
    scheduler = CountingScheduler()
    monkeypatch.setattr(
        "rl.grpo.trainer.recompute_minibatch_objective",
        objective_output,
    )
    monkeypatch.setattr(
        "rl.grpo.trainer.lora_parameters",
        lambda transformer: list(transformer.parameters()),
    )

    metrics = optimizer_update_accumulated(
        bundle,
        microbatches,
        optimizer=optimizer,
        scheduler=scheduler,
        conditioning=SimpleNamespace(),
        config={"optimizer": {"gradient_clip_norm": 100.0}},
    )

    torch.testing.assert_close(actual_model.weight.grad, expected_gradient)
    torch.testing.assert_close(
        actual_model.weight.detach(), reference_model.weight.detach()
    )
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1
    assert scheduler.step_calls == 1
    assert metrics["effective_batch_size"] == 5
    assert metrics["microbatch_count"] == 2
    assert metrics["microbatch_sizes"] == [2, 3]


def _mechanism_fixture(*, tamper=None):
    candidate = yaml.safe_load(
        CORRECTNESS_2GPU_CONFIG_PATH.read_text(encoding="utf-8")
    )
    group_size = int(candidate["collection"]["group_size"])
    group_count = int(
        candidate["collection"]["prompts_per_mini_batch"]
        * candidate["collection"]["mini_batch_repeats"]
    )
    reward_std = math.sqrt(8.25)
    rows = []
    for group_index in range(group_count):
        for candidate_index in range(group_size):
            trajectory_id = f"1:{group_index}:{candidate_index}"
            reward = float(candidate_index)
            rows.append(
                {
                    "collection_index": 1,
                    "collection_commit_id": "commit-1",
                    "group_id": f"1:{group_index}",
                    "trajectory_id": trajectory_id,
                    "candidate_index": candidate_index,
                    "reward": reward,
                    "advantage": (reward - 4.5) / reward_std,
                    "eligible": True,
                    "nfe": 10,
                    "window_start": 1 + group_index % 3,
                    "initial_latent_seed": group_index * group_size + candidate_index,
                    "brownian_seed": 1000 + group_index * group_size + candidate_index,
                    "terminal_mel_sha256": f"mel-{trajectory_id}",
                    "scored_wav_sha256": f"wav-{trajectory_id}",
                    "reward_components": {
                        "raw_components": {
                            "dnsmos": 0.5 + reward / 100.0,
                            "speaker": 0.6 + reward / 100.0,
                            "speechbertscore": 0.7 + reward / 100.0,
                        }
                    },
                }
            )

    used_ids = [str(row["trajectory_id"]) for row in rows]
    updates = []
    for update_index in range(4):
        update_ids = used_ids[update_index * 180 : (update_index + 1) * 180]
        updates.append(
            {
                "loss": -0.1,
                "policy_loss": -0.11,
                "reference_kl": 0.01,
                "weighted_reference_kl": 0.001,
                "ratio_mean": 1.0 if update_index == 0 else 1.01,
                "ratio_std": 0.0 if update_index == 0 else 0.01,
                "log_ratio_abs_max": 0.0 if update_index == 0 else 0.02,
                "approx_kl": 0.0 if update_index == 0 else 0.001,
                "clip_fraction": 0.0,
                "gradient_norm": 0.5,
                "learning_rate": 2.0e-4,
                "trajectory_ids": update_ids,
                "effective_batch_size": 180,
                "microbatch_count": 15,
                "microbatch_sizes": [12] * 15,
            }
        )
    old_state_hash = "frozen-old-policy"
    collection_row = {
        "collection_index": 1,
        "collection_commit_id": "commit-1",
        "old_lora_state_sha256": old_state_hash,
        "rollout_worker_state_sha256": {"0": old_state_hash, "1": old_state_hash},
        "updates": updates,
        "used_trajectory_ids": used_ids,
        "scored_trajectories": len(rows),
        "eligible_trajectories": len(rows),
        "used_trajectories": len(used_ids),
        "optimizer_batch_shuffle_seed": stable_seed(
            int(candidate["run"]["seed"]), "update_batches", 1
        ),
    }
    if tamper == "first_ratio":
        collection_row["updates"][0]["log_ratio_abs_max"] = 1.0e-3
    elif tamper == "worker_policy":
        collection_row["rollout_worker_state_sha256"]["1"] = "wrong-policy"
    elif tamper == "advantage":
        rows[0]["advantage"] += 0.2
    elif tamper == "latent_seed":
        rows[1]["initial_latent_seed"] = rows[0]["initial_latent_seed"]

    initial_state = {"adapter.weight": torch.tensor([1.0, -1.0])}
    current_state = {"adapter.weight": torch.tensor([1.01, -1.02])}
    checkpoint_payload = {
        "collection_index": 1,
        "optimizer_step": 4,
        "collection_commit_id": "commit-1",
        "online_lora_state": current_state,
        "online_lora_fingerprint": lora_state_fingerprint(current_state),
    }
    summary = _mechanism_collection_summary(
        collection_index=1,
        optimizer_step=4,
        collection_commit_id="commit-1",
        checkpoint_path=Path("mechanism-checkpoint-1.pt"),
        checkpoint_payload=checkpoint_payload,
        initial_lora_state=initial_state,
        config=candidate,
        collection_row=collection_row,
        rollout_rows=rows,
    )
    initial_payload = {
        "collection_index": 0,
        "optimizer_step": 0,
        "collection_commit_id": "initial",
        "online_lora_state": initial_state,
        "online_lora_fingerprint": lora_state_fingerprint(initial_state),
    }
    initial_summary = _mechanism_collection_summary(
        collection_index=0,
        optimizer_step=0,
        collection_commit_id="initial",
        checkpoint_path=Path("mechanism-checkpoint-0.pt"),
        checkpoint_payload=initial_payload,
        initial_lora_state=initial_state,
        config=candidate,
        collection_row=None,
        rollout_rows=[],
    )
    return initial_summary, summary


def test_mechanism_audit_records_full_grpo_geometry_and_passes():
    distribution = _numeric_distribution([1.0, 2.0, 3.0])
    assert distribution["mean"] == 2.0
    assert distribution["std"] == pytest.approx(math.sqrt(2.0 / 3.0))
    delta = _lora_delta_statistics(
        {"value": torch.tensor([1.0, 2.0])},
        {"value": torch.tensor([1.0, 2.5])},
    )
    assert delta["changed_tensor_count"] == 1
    assert delta["delta_l2"] == 0.5

    initial_summary, trained_summary = _mechanism_fixture()
    assert initial_summary["invariants"]["all_passed"] is True
    assert trained_summary["invariants"]["all_passed"] is True
    mechanism = trained_summary["collection_mechanism"]
    assert mechanism["reward"]["count"] == 720
    assert mechanism["advantage"]["count"] == 720
    assert mechanism["valid_groups"] == 72
    assert mechanism["zero_std_groups"] == 0
    assert mechanism["unique_initial_latent_seeds"] == 720
    assert mechanism["unique_brownian_seeds"] == 720
    assert mechanism["used_trajectories"] == 720
    report = _aggregate_mechanism_audits(
        [trained_summary, initial_summary], expected_collections=[0, 1]
    )
    assert report["status"] == "GRPO-MECHANISM-AUDIT-PASS"
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("tamper", "failed_invariant"),
    [
        ("first_ratio", "first_update_replays_old_policy"),
        ("worker_policy", "worker_policy_equals_frozen_old_policy"),
        ("advantage", "group_advantages_match_float32_eq8"),
        ("latent_seed", "independent_initial_latent_seeds"),
    ],
)
def test_mechanism_audit_fails_closed_on_algorithmic_tampering(
    tamper, failed_invariant
):
    initial_summary, trained_summary = _mechanism_fixture(tamper=tamper)
    assert trained_summary["invariants"][failed_invariant] is False
    report = _aggregate_mechanism_audits(
        [initial_summary, trained_summary], expected_collections=[0, 1]
    )
    assert report["status"] == "GRPO-MECHANISM-AUDIT-FAIL"
    assert failed_invariant in report["invariant_failures_by_collection"]["1"]


def _shard(group_index, *, rollout_seconds, reward_seconds):
    rows = tuple(
        {
            "group_index": group_index,
            "mini_batch_id": group_index // 2,
            "condition_slot": group_index % 2,
            "candidate_index": candidate,
            "trajectory_id": f"1:{group_index}:{candidate}",
            "reward": float(group_index * 10 + candidate),
        }
        for candidate in range(2)
    )
    return CollectionResult(
        collection_index=1,
        rewards=torch.tensor([[row["reward"] for row in rows]]),
        examples=(),
        rollout_rows=rows,
        group_rows=({"group_index": group_index},),
        candidate_audio_seconds=1.0,
        phase_seconds={
            "rollout": rollout_seconds,
            "reward": reward_seconds,
            "audio_io": 0.1,
        },
        active_gpu_seconds_by_phase={
            "rollout": rollout_seconds,
            "reward": reward_seconds,
        },
        retained_audio_count=1,
    )


def test_group_partition_and_shard_merge_preserve_canonical_order():
    assert _partition_group_indices(4, 2) == [[0, 1], [2, 3]]
    with pytest.raises(ValueError):
        _partition_group_indices(5, 2)

    merged = _merge_collection_shards(
        [
            _shard(1, rollout_seconds=2.0, reward_seconds=3.0),
            _shard(0, rollout_seconds=1.0, reward_seconds=4.0),
        ],
        collection_index=1,
        expected_groups=2,
        group_size=2,
    )
    assert [row["group_index"] for row in merged.group_rows] == [0, 1]
    assert merged.rewards.tolist() == [[0.0, 1.0], [10.0, 11.0]]
    assert merged.phase_seconds["rollout"] == 2.0
    assert merged.phase_seconds["reward"] == 4.0
    assert merged.active_gpu_seconds_by_phase == {"rollout": 3.0, "reward": 7.0}


def test_worker_rollout_shard_transport_queues_only_scalar_metadata(tmp_path):
    task_id = "collection_000001"
    path = _worker_rollout_shard_path(tmp_path, task_id=task_id, worker_rank=1)
    shard = _shard(1, rollout_seconds=2.0, reward_seconds=3.0)
    metadata = _write_worker_rollout_shard(
        path,
        task_id=task_id,
        worker_rank=1,
        collection_index=1,
        state_sha256="old-policy-hash",
        shard=shard,
    )
    json.dumps(metadata)
    assert path.is_file()
    assert "shard" not in metadata
    assert metadata["shard_bytes"] > 0
    restored = _load_worker_rollout_shard(
        path,
        expected_file_sha256=metadata["shard_sha256"],
        expected_task_id=task_id,
        expected_worker_rank=1,
        expected_collection_index=1,
        expected_state_sha256="old-policy-hash",
    )
    assert restored.collection_index == 1
    assert restored.rollout_rows == shard.rollout_rows
    with pytest.raises(ValueError, match="file SHA256"):
        _load_worker_rollout_shard(
            path,
            expected_file_sha256="wrong-hash",
            expected_task_id=task_id,
            expected_worker_rank=1,
            expected_collection_index=1,
            expected_state_sha256="old-policy-hash",
        )
    _cleanup_worker_rollout_directory(tmp_path)
    assert not path.exists()


def test_checkpoint_round_trip_restores_commit_accounting_and_lora(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4))
    inject_lora(
        model,
        target_patterns=[r"0"],
        rank=2,
        alpha=4.0,
        expected_modules=1,
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(transformer=model), checkpoint_sha256="released-hash"
    )
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    accounting = _empty_accounting(config())
    accounting["completed_collections"] = 3
    expected_state = snapshot_lora(model, device="cpu")
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        bundle=bundle,
        optimizer=optimizer,
        scheduler=scheduler,
        collection_index=3,
        optimizer_step=6,
        config_hash="config-hash",
        collection_commit_id="commit-3",
        cumulative_accounting=accounting,
        completed_milestones=[0, 50],
    )
    with torch.no_grad():
        for parameter in lora_parameters(model):
            parameter.add_(10.0)

    restored = load_checkpoint(
        path,
        bundle=bundle,
        optimizer=optimizer,
        scheduler=scheduler,
        config_hash="config-hash",
    )
    assert restored == (3, 6, accounting, [0, 50], "commit-3")
    for name, value in snapshot_lora(model).items():
        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)


def test_gpu_accounting_separates_allocated_and_active_time():
    pilot = yaml.safe_load(
        Path("configs/grpo/grpo_voicebank_controlled_cfg0_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    accounting = _empty_accounting(pilot)
    _record_training_accounting(
        accounting,
        collection_wall_seconds=10.0,
        phase_seconds={
            "rollout": 4.0,
            "reward": 2.0,
            "recompute": 1.0,
            "backward": 0.5,
            "optimizer": 0.25,
        },
        active_gpu_seconds_by_phase={
            "rollout": 15.0,
            "reward": 7.0,
            "recompute": 1.0,
            "backward": 0.5,
            "optimizer": 0.25,
        },
        optimizer_step=4,
    )
    assert accounting["allocated_training_gpu_seconds"] == 40.0
    assert accounting["active_gpu_seconds_by_phase"]["rollout"] == 15.0
    assert accounting["active_gpu_seconds_by_phase"]["reward"] == 7.0
    assert accounting["completed_optimizer_steps"] == 4


def _gpu_hour_config():
    value = config()
    value["run"]["mode"] = "train"
    value["run"]["collections"] = 100
    value["comparison"]["target_training_gpu_hours"] = 1.0
    value["evaluation"]["milestone_basis"] = "allocated_training_gpu_hours"
    value["evaluation"]["selection_milestones"] = [0, 20, 40, 60, 80, 100]
    return value


def test_gpu_hour_milestone_uses_first_observed_threshold_crossing():
    candidate = _gpu_hour_config()
    accounting = _empty_accounting(candidate)
    accounting["allocated_training_gpu_seconds"] = 720.0
    assert _due_selection_milestones(
        candidate,
        collection_index=7,
        cumulative_accounting=accounting,
        completed_milestones=[0],
    ) == [20]
    metadata = _selection_milestone_metadata(
        candidate,
        percentage=20,
        collection_index=7,
        cumulative_accounting=accounting,
    )
    assert metadata["threshold_allocated_training_gpu_seconds"] == 720.0
    assert metadata["first_crossing_collection"] == 7
    assert metadata["overshoot_gpu_seconds"] == 0.0


def test_gpu_hour_milestone_rejects_coarse_multi_threshold_crossing():
    candidate = _gpu_hour_config()
    accounting = _empty_accounting(candidate)
    accounting["allocated_training_gpu_seconds"] = 1500.0
    with pytest.raises(RuntimeError, match="multiple registered"):
        _due_selection_milestones(
            candidate,
            collection_index=1,
            cumulative_accounting=accounting,
            completed_milestones=[0],
        )


def test_formal_budget_match_requires_exact_cross_resume_accounting():
    candidate = _gpu_hour_config()
    candidate["comparison"]["gpu_time_relative_tolerance"] = 0.01
    accounting = _empty_accounting(candidate)
    accounting["allocated_training_gpu_seconds"] = 3600.0
    assert _gpu_time_budget_match(candidate, accounting)["passed"] is True
    accounting["unmeasured_checkpoint_commit_count"] = 1
    result = _gpu_time_budget_match(candidate, accounting)
    assert result["accounting_exact"] is False
    assert result["passed"] is False


def _sidecar_payload(collection, commit, accounting):
    return {
        "schema_version": 1,
        "config_sha256": "config-hash",
        "collection_index": collection,
        "collection_commit_id": commit,
        "cumulative_accounting": accounting,
    }


def test_accounting_sidecar_recovers_one_commit_crash_window(tmp_path):
    path = tmp_path / "accounting.json"
    stale = _empty_accounting(config())
    stale["completed_collections"] = 2
    path.write_text(
        json.dumps(_sidecar_payload(2, "commit-2", stale)), encoding="utf-8"
    )
    embedded = _empty_accounting(config())
    embedded["completed_collections"] = 3
    recovered = _reconcile_accounting_sidecar(
        path,
        config_hash="config-hash",
        completed_collection=3,
        collection_commit_id="commit-3",
        checkpoint_accounting=embedded,
    )
    assert recovered["completed_collections"] == 3
    assert recovered["unmeasured_checkpoint_commit_count"] == 1
    assert recovered["accounting_recovery_events"][0]["collection_index"] == 3
    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert repaired["collection_index"] == 3
    assert repaired["collection_commit_id"] == "commit-3"


@pytest.mark.parametrize(
    ("sidecar_collection", "sidecar_commit"),
    [(3, "wrong-commit"), (1, "commit-1")],
)
def test_accounting_sidecar_rejects_unrecoverable_mismatch(
    tmp_path, sidecar_collection, sidecar_commit
):
    path = tmp_path / "accounting.json"
    path.write_text(
        json.dumps(
            _sidecar_payload(
                sidecar_collection, sidecar_commit, _empty_accounting(config())
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be reconciled"):
        _reconcile_accounting_sidecar(
            path,
            config_hash="config-hash",
            completed_collection=3,
            collection_commit_id="commit-3",
            checkpoint_accounting=_empty_accounting(config()),
        )


def test_accounting_sidecar_missing_after_checkpoint_is_recreated(tmp_path):
    path = tmp_path / "accounting.json"
    embedded = _empty_accounting(config())
    embedded["completed_collections"] = 1
    recovered = _reconcile_accounting_sidecar(
        path,
        config_hash="config-hash",
        completed_collection=1,
        collection_commit_id="commit-1",
        checkpoint_accounting=embedded,
    )
    assert path.is_file()
    assert recovered["unmeasured_checkpoint_commit_count"] == 1
