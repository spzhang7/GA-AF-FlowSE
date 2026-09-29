import pytest

from rl.rewards.evaluators import (
    normalize_wer_text,
    resolve_hf_model,
    word_error_rate,
)


def test_wer_normalization_and_distance():
    assert normalize_wer_text("Hello,  WORLD!") == "hello world"
    assert word_error_rate("one two three", "one four three") == pytest.approx(1 / 3)
    assert word_error_rate("one two", "one two extra") == pytest.approx(0.5)


def test_local_hf_model_directory_needs_no_hub_cache(tmp_path):
    snapshot = tmp_path / "wavlm-large"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    revision = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
    resolved = resolve_hf_model(
        {
            "repo_id": "microsoft/wavlm-large",
            "revision": revision,
            "local_model_dir": str(snapshot),
            "local_files_only": True,
        }
    )
    assert resolved.snapshot_path == snapshot.resolve()
    assert resolved.resolved_revision == revision
    assert len(resolved.snapshot_sha256) == 64

