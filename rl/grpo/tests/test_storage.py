import json

import pytest
import torch

from rl.grpo import storage as storage


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.pt"
    torch.save({"version": "previous"}, destination)

    def failing_save(payload, handle):
        handle.write(b"partial checkpoint")
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(storage.torch, "save", failing_save)
    with pytest.raises(RuntimeError, match="simulated"):
        storage.atomic_torch_save(destination, {"version": "new"})

    assert torch.load(destination, weights_only=False) == {"version": "previous"}
    assert list(tmp_path.glob(".checkpoint.pt.tmp-*")) == []


def test_resume_truncates_uncommitted_log_tails(tmp_path):
    collection_log = tmp_path / "collections.jsonl"
    rollout_log = tmp_path / "rollouts.jsonl"
    _write_rows(
        collection_log,
        [
            {"collection_index": 1, "collection_commit_id": "commit-1"},
            {"collection_index": 2, "collection_commit_id": "uncommitted"},
        ],
    )
    _write_rows(
        rollout_log,
        [
            {
                "collection_index": 1,
                "collection_commit_id": "commit-1",
                "trajectory_id": "1:0",
            },
            {
                "collection_index": 2,
                "collection_commit_id": "uncommitted",
                "trajectory_id": "2:0",
            },
        ],
    )

    result = storage.reconcile_collection_logs(
        collection_log=collection_log,
        rollout_log=rollout_log,
        completed_collection=1,
        expected_last_commit_id="commit-1",
    )

    assert result["collection_rows"] == 1
    assert result["rollout_rows"] == 1
    assert storage.read_jsonl(collection_log)[0]["collection_index"] == 1
    assert storage.read_jsonl(rollout_log)[0]["trajectory_id"] == "1:0"


@pytest.mark.parametrize("corruption", ["duplicate_collection", "wrong_rollout_commit"])
def test_resume_rejects_committed_log_corruption(tmp_path, corruption):
    collection_log = tmp_path / "collections.jsonl"
    rollout_log = tmp_path / "rollouts.jsonl"
    collection_rows = [{"collection_index": 1, "collection_commit_id": "commit-1"}]
    rollout_rows = [
        {
            "collection_index": 1,
            "collection_commit_id": "commit-1",
            "trajectory_id": "1:0",
        }
    ]
    if corruption == "duplicate_collection":
        collection_rows.append(dict(collection_rows[0]))
    else:
        rollout_rows[0]["collection_commit_id"] = "wrong"
    _write_rows(collection_log, collection_rows)
    _write_rows(rollout_log, rollout_rows)

    with pytest.raises(ValueError):
        storage.reconcile_collection_logs(
            collection_log=collection_log,
            rollout_log=rollout_log,
            completed_collection=1,
            expected_last_commit_id="commit-1",
        )


@pytest.mark.parametrize("row_count", [719, 721])
def test_resume_rejects_incomplete_or_extra_rollout_rows(tmp_path, row_count):
    collection_log = tmp_path / "collections.jsonl"
    rollout_log = tmp_path / "rollouts.jsonl"
    _write_rows(
        collection_log,
        [{"collection_index": 1, "collection_commit_id": "commit-1"}],
    )
    _write_rows(
        rollout_log,
        [
            {
                "collection_index": 1,
                "collection_commit_id": "commit-1",
                "trajectory_id": f"1:{index}",
            }
            for index in range(row_count)
        ],
    )
    with pytest.raises(ValueError, match="frozen geometry"):
        storage.reconcile_collection_logs(
            collection_log=collection_log,
            rollout_log=rollout_log,
            completed_collection=1,
            expected_last_commit_id="commit-1",
            expected_rollout_rows_per_collection=720,
        )
