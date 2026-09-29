"""Crash-safe artifact primitives for collection-boundary GRPO training."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import torch


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def _flush_and_replace(temporary: Path, destination: Path) -> None:
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _flush_and_replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def append_jsonl_batch(path: str | Path, rows: Iterable[Mapping]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ]
    if not serialized:
        return
    with destination.open("a", encoding="utf-8", newline="") as handle:
        handle.writelines(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path: str | Path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _flush_and_replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reconcile_collection_logs(
    *,
    collection_log: str | Path,
    rollout_log: str | Path,
    completed_collection: int,
    expected_last_commit_id: str | None = None,
    expected_rollout_rows_per_collection: int | None = None,
) -> dict:
    """Atomically drop every log row beyond the committed checkpoint boundary."""

    if completed_collection < 0:
        raise ValueError("completed_collection cannot be negative")
    collection_path = Path(collection_log)
    rollout_path = Path(rollout_log)
    collection_rows = [
        row
        for row in read_jsonl(collection_path)
        if int(row.get("collection_index", -1)) <= completed_collection
    ]
    observed = [int(row["collection_index"]) for row in collection_rows]
    if len(observed) != len(set(observed)):
        raise ValueError("committed collection log contains duplicate collection rows")
    if observed and observed != list(range(1, completed_collection + 1)):
        raise ValueError(
            "committed collection log is not contiguous: "
            f"completed={completed_collection}, observed={observed}"
        )
    commits_by_collection = {}
    for row in collection_rows:
        collection_index = int(row["collection_index"])
        commit_id = row.get("collection_commit_id")
        if not isinstance(commit_id, str) or not commit_id:
            raise ValueError("committed collection row lacks a commit ID")
        commits_by_collection[collection_index] = commit_id
    commit_ids = list(commits_by_collection.values())
    if len(commit_ids) != len(set(commit_ids)):
        raise ValueError("committed collection log reuses a collection commit ID")
    if completed_collection > 0 and expected_last_commit_id is not None:
        if commits_by_collection.get(completed_collection) != expected_last_commit_id:
            raise ValueError("latest collection log commit differs from checkpoint")
    rollout_rows = [
        row
        for row in read_jsonl(rollout_path)
        if int(row.get("collection_index", -1)) <= completed_collection
    ]
    trajectory_ids = [str(row["trajectory_id"]) for row in rollout_rows]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("committed rollout log contains duplicate trajectory IDs")
    for row in rollout_rows:
        collection_index = int(row["collection_index"])
        if row.get("collection_commit_id") != commits_by_collection.get(
            collection_index
        ):
            raise ValueError("rollout row commit differs from its collection commit")
    if expected_rollout_rows_per_collection is not None:
        expected = int(expected_rollout_rows_per_collection)
        if expected < 1:
            raise ValueError("expected rollout rows per collection must be positive")
        counts = Counter(int(row["collection_index"]) for row in rollout_rows)
        incorrect = {
            collection_index: counts.get(collection_index, 0)
            for collection_index in range(1, completed_collection + 1)
            if counts.get(collection_index, 0) != expected
        }
        if incorrect:
            raise ValueError(
                "committed rollout row counts differ from frozen geometry: "
                f"expected={expected}, observed={incorrect}"
            )
    if collection_path.exists() or collection_rows:
        atomic_write_jsonl(collection_path, collection_rows)
    if rollout_path.exists() or rollout_rows:
        atomic_write_jsonl(rollout_path, rollout_rows)
    return {
        "completed_collection": int(completed_collection),
        "collection_rows": len(collection_rows),
        "rollout_rows": len(rollout_rows),
    }
