from pathlib import Path

import pytest

from rl.common.dataset_validation import (
    validate_paired_audio_dataset,
)

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


def _write_pair(root: Path, utterance: str, *, sample_rate: int = 16000) -> None:
    waveform = np.linspace(-0.1, 0.1, 320, dtype=np.float32)
    for kind in ("clean", "noisy"):
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(directory / f"{utterance}.wav", waveform, sample_rate)


def _validate(root: Path) -> dict:
    return validate_paired_audio_dataset(
        train_manifest={"p001_001": "", "p002_001": ""},
        evaluation_manifest={"p003_001": "reference"},
        noisy_dir=root / "noisy",
        clean_dir=root / "clean",
        expected_train_utterances=2,
        expected_evaluation_utterances=1,
        expected_sample_rate=16000,
        expected_channels=1,
        conditions_per_step=2,
        optimizer_steps=1,
        full_decode=True,
    )


def test_preflight_fully_validates_exact_paired_dataset(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    report = _validate(tmp_path)
    assert report["passed"] is True
    assert report["observed"]["audio_files"] == 6
    assert report["observed"]["sample_rates"] == {"16000": 6}
    assert report["observed"]["channels"] == {"1": 6}
    assert report["coverage"]["next_epoch_tail_repeats"] == 0


def test_preflight_rejects_non_16khz_audio(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(
            tmp_path,
            utterance,
            sample_rate=8000 if utterance == "p002_001" else 16000,
        )
    with pytest.raises(ValueError, match="unexpected sample rate"):
        _validate(tmp_path)


def test_preflight_rejects_missing_pair(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    (tmp_path / "clean" / "p002_001.wav").unlink()
    with pytest.raises(ValueError, match="directory membership mismatch"):
        _validate(tmp_path)


def test_preflight_rejects_unexpected_extra_wav(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001", "extra"):
        _write_pair(tmp_path, utterance)
    with pytest.raises(ValueError, match="directory membership mismatch"):
        _validate(tmp_path)


def test_multi_epoch_preflight_allows_extra_heldout_audio(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001", "heldout_test"):
        _write_pair(tmp_path, utterance)
    report = validate_paired_audio_dataset(
        train_manifest={"p001_001": "", "p002_001": ""},
        evaluation_manifest={"p003_001": "reference"},
        noisy_dir=tmp_path / "noisy",
        clean_dir=tmp_path / "clean",
        expected_train_utterances=2,
        expected_evaluation_utterances=1,
        expected_sample_rate=16000,
        expected_channels=1,
        conditions_per_step=2,
        optimizer_steps=5,
        full_decode=True,
        require_single_coverage=False,
        allow_unlisted_audio=True,
    )
    assert report["passed"] is True
    assert report["requirements"]["exact_directory_membership"] is False
    assert report["coverage"]["complete_training_coverages"] == 5


def test_multi_epoch_preflight_requires_one_complete_coverage(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    with pytest.raises(ValueError, match="at least once"):
        validate_paired_audio_dataset(
            train_manifest={"p001_001": "", "p002_001": ""},
            evaluation_manifest={"p003_001": "reference"},
            noisy_dir=tmp_path / "noisy",
            clean_dir=tmp_path / "clean",
            expected_train_utterances=2,
            expected_evaluation_utterances=1,
            expected_sample_rate=16000,
            expected_channels=1,
            conditions_per_step=1,
            optimizer_steps=1,
            full_decode=True,
            require_single_coverage=False,
            allow_unlisted_audio=False,
        )


def test_explicit_partial_coverage_is_allowed_for_short_screen(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    report = validate_paired_audio_dataset(
        train_manifest={"p001_001": "", "p002_001": ""},
        evaluation_manifest={"p003_001": "reference"},
        noisy_dir=tmp_path / "noisy",
        clean_dir=tmp_path / "clean",
        expected_train_utterances=2,
        expected_evaluation_utterances=1,
        expected_sample_rate=16000,
        expected_channels=1,
        conditions_per_step=1,
        optimizer_steps=1,
        full_decode=True,
        require_single_coverage=False,
        allow_partial_coverage=True,
    )
    assert report["passed"] is True
    assert report["requirements"]["allow_partial_coverage"] is True
    assert report["coverage"]["complete_training_coverages"] == 0
    assert report["coverage"]["unvisited_train_conditions"] == 1


def test_preflight_rejects_stereo_audio(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    mono = np.linspace(-0.1, 0.1, 320, dtype=np.float32)
    sf.write(
        tmp_path / "clean" / "p002_001.wav",
        np.column_stack((mono, mono)),
        16000,
    )
    with pytest.raises(ValueError, match="unexpected channel count"):
        _validate(tmp_path)


def test_preflight_rejects_clean_noisy_frame_mismatch(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    sf.write(
        tmp_path / "noisy" / "p002_001.wav",
        np.zeros(160, dtype=np.float32),
        16000,
    )
    with pytest.raises(ValueError, match="clean/noisy frame mismatch"):
        _validate(tmp_path)


def test_preflight_rejects_empty_audio(tmp_path):
    for utterance in ("p001_001", "p002_001", "p003_001"):
        _write_pair(tmp_path, utterance)
    sf.write(
        tmp_path / "clean" / "p002_001.wav",
        np.empty(0, dtype=np.float32),
        16000,
    )
    with pytest.raises(ValueError, match="empty audio file"):
        _validate(tmp_path)

