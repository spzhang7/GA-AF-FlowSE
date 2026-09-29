import json

import pytest

from rl.af.checkpoint import (
    append_jsonl_batch_durable,
    append_jsonl_durable,
    atomic_write_json,
    commit_step_transaction,
    recover_step_transaction,
    step_commit_id,
    truncate_jsonl_after_step,
)


PROTOCOL_HASH = "protocol-hash"


class InjectedCrash(RuntimeError):
    pass


def test_batch_append_writes_all_rows_with_one_fsync(tmp_path, monkeypatch):
    fsync_calls = []
    monkeypatch.setattr(
        "rl.af.checkpoint.os.fsync",
        lambda descriptor: fsync_calls.append(descriptor),
    )
    path = tmp_path / "rollout_metrics.jsonl"
    rows = [
        {"step": 1, "candidate": 0, "text": "第一条"},
        {"step": 1, "candidate": 1, "text": "second"},
        {"step": 1, "candidate": 2, "text": "third"},
    ]

    append_jsonl_batch_durable(path, rows)

    persisted = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert persisted == rows
    assert len(fsync_calls) == 1


def test_rollout_batch_tail_is_checkpoint_truncated_on_resume(tmp_path):
    path = tmp_path / "rollout_metrics.jsonl"
    committed = [{"step": 1, "candidate": index} for index in range(3)]
    uncommitted = [{"step": 2, "candidate": index} for index in range(3)]
    append_jsonl_batch_durable(path, committed)
    append_jsonl_batch_durable(path, uncommitted)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 2, "candidate": 3')

    truncate_jsonl_after_step(path, completed_step=1)

    persisted = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert persisted == committed


def _initial_extra():
    return {
        "step_transaction": {
            "schema_version": 1,
            "step": 0,
            "step_commit_id": step_commit_id(PROTOCOL_HASH, 0),
            "commit_protocol": "initial_checkpoint",
        }
    }


def _write_initial_checkpoint(output_dir):
    path = output_dir / "checkpoint_latest.pt"
    atomic_write_json(path, {"step": 0, "extra": _initial_extra()})
    return path


def _record(step):
    return {
        "step": step,
        "protocol_hash": PROTOCOL_HASH,
        "timing_seconds": {"through_update": 1.0},
        "allocated_rollout_world_size": 4,
    }


def _commit(output_dir, *, fault_stage=None):
    latest = output_dir / "checkpoint_latest.pt"

    def checkpoint_writer(path, extra):
        atomic_write_json(path, {"step": 1, "extra": extra})

    def fault_injector(stage):
        if stage == fault_stage:
            raise InjectedCrash(stage)

    return commit_step_transaction(
        output_dir=output_dir,
        latest_checkpoint=latest,
        step_record=_record(1),
        step=1,
        protocol_hash=PROTOCOL_HASH,
        checkpoint_interval=1,
        total_steps=1,
        checkpoint_writer=checkpoint_writer,
        checkpoint_started=10.0,
        now=lambda: 12.0,
        step_started=8.0,
        fault_injector=fault_injector if fault_stage is not None else None,
    )


def _checkpoint_extra(path):
    return json.loads(path.read_text(encoding="utf-8"))["extra"]


@pytest.mark.parametrize(
    "fault_stage",
    [
        "after_pending_checkpoint",
        "after_training_log",
        "after_accounting_sidecar",
    ],
)
def test_fault_before_latest_rolls_back_the_entire_step(tmp_path, fault_stage):
    latest = _write_initial_checkpoint(tmp_path)
    with pytest.raises(InjectedCrash, match=fault_stage):
        _commit(tmp_path, fault_stage=fault_stage)

    assert json.loads(latest.read_text(encoding="utf-8"))["step"] == 0
    rows = recover_step_transaction(
        output_dir=tmp_path,
        latest_checkpoint=latest,
        completed_step=0,
        checkpoint_extra=_initial_extra(),
        protocol_hash=PROTOCOL_HASH,
        checkpoint_interval=1,
        total_steps=1,
    )
    assert rows == []
    assert not (tmp_path / "checkpoint_pending.pt").exists()
    assert not list((tmp_path / "accounting_steps").glob("step_*.json"))
    assert not list(tmp_path.glob("checkpoint_step_*.pt"))


@pytest.mark.parametrize(
    "fault_stage", ["after_latest_checkpoint", "after_periodic_checkpoint"]
)
def test_fault_after_latest_recovers_the_committed_step(tmp_path, fault_stage):
    latest = _write_initial_checkpoint(tmp_path)
    with pytest.raises(InjectedCrash, match=fault_stage):
        _commit(tmp_path, fault_stage=fault_stage)

    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["step"] == 1
    rows = recover_step_transaction(
        output_dir=tmp_path,
        latest_checkpoint=latest,
        completed_step=1,
        checkpoint_extra=payload["extra"],
        protocol_hash=PROTOCOL_HASH,
        checkpoint_interval=1,
        total_steps=1,
    )
    assert [row["step"] for row in rows] == [1]
    assert rows[0]["step_commit_id"] == step_commit_id(PROTOCOL_HASH, 1)
    accounting = json.loads(
        (tmp_path / "accounting_steps" / "step_000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert accounting["step_commit_id"] == rows[0]["step_commit_id"]
    assert (tmp_path / "checkpoint_step_000001.pt").is_file()


def test_recovery_removes_a_partial_jsonl_tail(tmp_path):
    latest = _write_initial_checkpoint(tmp_path)
    _commit(tmp_path)
    with (tmp_path / "training_steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"step": 2')

    rows = recover_step_transaction(
        output_dir=tmp_path,
        latest_checkpoint=latest,
        completed_step=1,
        checkpoint_extra=_checkpoint_extra(latest),
        protocol_hash=PROTOCOL_HASH,
        checkpoint_interval=1,
        total_steps=1,
    )
    assert [row["step"] for row in rows] == [1]
    persisted = (tmp_path / "training_steps.jsonl").read_text(encoding="utf-8")
    assert len([line for line in persisted.splitlines() if line]) == 1


@pytest.mark.parametrize("missing", ["log", "sidecar"])
def test_new_transaction_checkpoint_fails_closed_when_metadata_is_missing(
    tmp_path, missing
):
    latest = _write_initial_checkpoint(tmp_path)
    _commit(tmp_path)
    if missing == "log":
        (tmp_path / "training_steps.jsonl").unlink()
    else:
        (tmp_path / "accounting_steps" / "step_000001.json").unlink()

    with pytest.raises(ValueError, match="missing|not exact-resumable"):
        recover_step_transaction(
            output_dir=tmp_path,
            latest_checkpoint=latest,
            completed_step=1,
            checkpoint_extra=_checkpoint_extra(latest),
            protocol_hash=PROTOCOL_HASH,
            checkpoint_interval=1,
            total_steps=1,
        )


@pytest.mark.parametrize("log_was_committed", [False, True])
def test_legacy_checkpoint_recovers_old_log_and_sidecar_windows(
    tmp_path, log_was_committed
):
    latest = tmp_path / "checkpoint_latest.pt"
    legacy_record = _record(1)
    legacy_extra = {"last_step_record": legacy_record}
    atomic_write_json(latest, {"step": 1, "extra": legacy_extra})
    if log_was_committed:
        append_jsonl_durable(tmp_path / "training_steps.jsonl", legacy_record)

    rows = recover_step_transaction(
        output_dir=tmp_path,
        latest_checkpoint=latest,
        completed_step=1,
        checkpoint_extra=legacy_extra,
        protocol_hash=PROTOCOL_HASH,
        checkpoint_interval=10,
        total_steps=10,
    )
    assert [row["step"] for row in rows] == [1]
    accounting = json.loads(
        (tmp_path / "accounting_steps" / "step_000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert accounting["exact_accounting"] is False
    assert "legacy_recovery" in accounting


def test_transaction_id_mismatch_is_rejected(tmp_path):
    latest = _write_initial_checkpoint(tmp_path)
    _commit(tmp_path)
    sidecar_path = tmp_path / "accounting_steps" / "step_000001.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["step_commit_id"] = "tampered"
    atomic_write_json(sidecar_path, sidecar)

    with pytest.raises(ValueError, match="transaction IDs differ"):
        recover_step_transaction(
            output_dir=tmp_path,
            latest_checkpoint=latest,
            completed_step=1,
            checkpoint_extra=_checkpoint_extra(latest),
            protocol_hash=PROTOCOL_HASH,
            checkpoint_interval=1,
            total_steps=1,
        )

