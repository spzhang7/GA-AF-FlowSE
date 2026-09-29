"""Crash-consistent step transactions for AdvantageFlow training artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Mapping, Sequence


PENDING_CHECKPOINT_NAME = "checkpoint_pending.pt"


def _fsync_parent(path: Path) -> None:
    """Best-effort directory sync on platforms that support directory handles."""

    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_parent(path)


def atomic_write_jsonl(path: Path, rows: list[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_parent(path)


def append_jsonl_durable(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl_batch_durable(path: Path, values: Sequence[Mapping]) -> None:
    """Append one logical batch with a single write, flush, and fsync."""

    if not values:
        return
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        for value in values
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_allow_partial_tail(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty = nonempty[-1] if nonempty else -1
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == final_nonempty:
                break
            raise ValueError(f"malformed non-tail JSONL row: {path}:{index + 1}")
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{index + 1}")
        rows.append(row)
    return rows


def truncate_jsonl_after_step(path: Path, completed_step: int) -> None:
    """Drop partial and uncommitted tail rows using latest checkpoint authority."""

    retained = []
    for row in _read_jsonl_allow_partial_tail(path):
        if "step" not in row:
            raise ValueError(f"missing step in {path}")
        if int(row["step"]) <= completed_step:
            retained.append(row)
    if path.is_file() or retained:
        atomic_write_jsonl(path, retained)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    _fsync_parent(destination)


def step_commit_id(protocol_hash: str, step: int) -> str:
    return hashlib.sha256(f"{protocol_hash}:{int(step)}".encode()).hexdigest()


def _fault(fault_injector: Callable[[str], None] | None, stage: str) -> None:
    if fault_injector is not None:
        fault_injector(stage)


def commit_step_transaction(
    *,
    output_dir: Path,
    latest_checkpoint: Path,
    step_record: dict,
    step: int,
    protocol_hash: str,
    checkpoint_interval: int,
    total_steps: int,
    checkpoint_writer: Callable[[Path, Mapping], None],
    checkpoint_started: float,
    now: Callable[[], float],
    step_started: float,
    fault_injector: Callable[[str], None] | None = None,
) -> dict:
    """Persist one step and atomically expose it by replacing latest last."""

    if int(step_record.get("step", -1)) != int(step):
        raise ValueError("step record does not match transaction step")
    if checkpoint_interval < 1 or total_steps < step:
        raise ValueError("invalid checkpoint transaction geometry")
    commit_id = step_commit_id(protocol_hash, step)
    step_record["step_commit_id"] = commit_id
    transaction = {
        "schema_version": 1,
        "step": int(step),
        "step_commit_id": commit_id,
        "commit_protocol": "pending_then_log_then_accounting_then_atomic_latest",
    }
    pending = output_dir / PENDING_CHECKPOINT_NAME
    checkpoint_writer(
        pending,
        {
            "last_step_record": json.loads(json.dumps(step_record)),
            "step_transaction": transaction,
        },
    )
    if not pending.is_file():
        raise RuntimeError("checkpoint writer did not create the pending checkpoint")
    _fault(fault_injector, "after_pending_checkpoint")

    step_record["timing_seconds"]["checkpoint_artifact_io"] = float(
        now() - checkpoint_started
    )
    step_record["timing_seconds"]["training_wall_excluding_validation"] = float(
        now() - step_started
    )
    step_record["allocated_rollout_world_size"] = int(
        step_record["allocated_rollout_world_size"]
    )
    step_record["accounting_boundary"] = (
        "after_checkpoint_candidate_before_commit_metadata_and_latest_rename"
    )
    training_log = output_dir / "training_steps.jsonl"
    append_jsonl_durable(training_log, step_record)
    _fault(fault_injector, "after_training_log")

    accounting = {
        "schema_version": 2,
        "step": int(step),
        "protocol_hash": str(protocol_hash),
        "step_commit_id": commit_id,
        "timing_seconds": step_record["timing_seconds"],
        "allocated_rollout_world_size": step_record[
            "allocated_rollout_world_size"
        ],
        "accounting_boundary": step_record["accounting_boundary"],
    }
    atomic_write_json(
        output_dir / "accounting_steps" / f"step_{step:06d}.json",
        accounting,
    )
    _fault(fault_injector, "after_accounting_sidecar")

    pending.replace(latest_checkpoint)
    _fsync_parent(latest_checkpoint)
    _fault(fault_injector, "after_latest_checkpoint")

    if step % checkpoint_interval == 0 or step == total_steps:
        _atomic_copy(latest_checkpoint, output_dir / f"checkpoint_step_{step:06d}.pt")
    _fault(fault_injector, "after_periodic_checkpoint")
    return step_record


def _legacy_record(extra: Mapping, completed_step: int) -> dict | None:
    record = extra.get("last_step_record")
    if not isinstance(record, Mapping) or int(record.get("step", -1)) != completed_step:
        return None
    recovered = json.loads(json.dumps(record))
    recovered["legacy_recovery"] = (
        "checkpoint_embedded_lower_bound_after_pretransaction_crash_window"
    )
    return recovered


def _accounting_from_legacy_record(
    record: Mapping, *, completed_step: int, protocol_hash: str
) -> dict:
    return {
        "schema_version": 1,
        "step": int(completed_step),
        "protocol_hash": str(protocol_hash),
        "timing_seconds": dict(record.get("timing_seconds") or {}),
        "allocated_rollout_world_size": int(
            record.get("allocated_rollout_world_size", 1)
        ),
        "legacy_recovery": record["legacy_recovery"],
        "exact_accounting": False,
    }


def recover_step_transaction(
    *,
    output_dir: Path,
    latest_checkpoint: Path,
    completed_step: int,
    checkpoint_extra: Mapping,
    protocol_hash: str,
    checkpoint_interval: int,
    total_steps: int,
) -> list[dict]:
    """Roll back uncommitted tails and validate every latest-committed artifact."""

    pending = output_dir / PENDING_CHECKPOINT_NAME
    pending.unlink(missing_ok=True)
    pending.with_suffix(pending.suffix + ".tmp").unlink(missing_ok=True)

    training_log = output_dir / "training_steps.jsonl"
    rows = [
        row
        for row in _read_jsonl_allow_partial_tail(training_log)
        if int(row.get("step", -1)) <= completed_step
    ]
    observed = [int(row.get("step", -1)) for row in rows]
    transaction = checkpoint_extra.get("step_transaction")
    new_transaction = (
        isinstance(transaction, Mapping)
        and int(transaction.get("schema_version", -1)) == 1
        and int(transaction.get("step", -1)) == completed_step
    )
    expected = list(range(1, completed_step + 1))
    if observed != expected:
        if new_transaction:
            raise ValueError(
                "committed AF checkpoint is missing its transaction log row: "
                f"expected={expected}, observed={observed}"
            )
        legacy = _legacy_record(checkpoint_extra, completed_step)
        if observed == expected[:-1] and legacy is not None:
            rows.append(legacy)
            observed.append(completed_step)
        else:
            raise ValueError(
                "AF checkpoint/training log cannot be reconciled: "
                f"expected={expected}, observed={observed}"
            )
    atomic_write_jsonl(training_log, rows)

    accounting_dir = output_dir / "accounting_steps"
    accounting_dir.mkdir(parents=True, exist_ok=True)
    accounting_paths: dict[int, Path] = {}
    for path in accounting_dir.glob("step_*.json"):
        try:
            artifact_step = int(path.stem.rsplit("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid AF accounting artifact name: {path}") from exc
        if artifact_step > completed_step:
            path.unlink()
        else:
            accounting_paths[artifact_step] = path
    for path in accounting_dir.glob("step_*.json.tmp"):
        path.unlink()
    legacy = None if new_transaction else _legacy_record(
        checkpoint_extra, completed_step
    )
    if completed_step not in accounting_paths and legacy is not None:
        path = accounting_dir / f"step_{completed_step:06d}.json"
        accounting_record = json.loads(json.dumps(rows[-1] if rows else legacy))
        accounting_record["legacy_recovery"] = legacy["legacy_recovery"]
        atomic_write_json(
            path,
            _accounting_from_legacy_record(
                accounting_record,
                completed_step=completed_step,
                protocol_hash=protocol_hash,
            ),
        )
        accounting_paths[completed_step] = path
    expected_set = set(range(1, completed_step + 1))
    if set(accounting_paths) != expected_set:
        raise ValueError(
            "AF checkpoint/accounting artifact crash window is not exact-resumable: "
            f"missing={sorted(expected_set - set(accounting_paths))}, "
            f"extra={sorted(set(accounting_paths) - expected_set)}"
        )

    for row in rows:
        artifact_step = int(row["step"])
        accounting = json.loads(
            accounting_paths[artifact_step].read_text(encoding="utf-8")
        )
        if int(accounting.get("step", -1)) != artifact_step:
            raise ValueError("AF accounting artifact step differs from its file name")
        if str(accounting.get("protocol_hash", "")) != str(protocol_hash):
            raise ValueError("AF accounting artifact protocol hash differs")
        log_commit = row.get("step_commit_id")
        accounting_commit = accounting.get("step_commit_id")
        if log_commit is None and accounting_commit is None:
            continue
        expected_commit = step_commit_id(protocol_hash, artifact_step)
        if str(log_commit or "") != expected_commit or str(
            accounting_commit or ""
        ) != expected_commit:
            raise ValueError("AF committed step transaction IDs differ")

    if new_transaction and completed_step > 0:
        expected_commit = step_commit_id(protocol_hash, completed_step)
        if str(transaction.get("step_commit_id", "")) != expected_commit:
            raise ValueError("AF latest checkpoint transaction ID differs")

    for path in output_dir.glob("checkpoint_step_*.pt"):
        try:
            artifact_step = int(path.stem.rsplit("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid AF checkpoint artifact name: {path}") from exc
        if artifact_step > completed_step:
            path.unlink()
    for path in output_dir.glob("checkpoint_step_*.pt.tmp"):
        path.unlink()
    if completed_step > 0 and (
        completed_step % checkpoint_interval == 0 or completed_step == total_steps
    ):
        periodic = output_dir / f"checkpoint_step_{completed_step:06d}.pt"
        if not periodic.is_file():
            _atomic_copy(latest_checkpoint, periodic)
    return rows
