"""Hard data-integrity checks for paired VoiceBank-DEMAND experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping


def _audio_metadata(
    path: Path,
    *,
    expected_sample_rate: int,
    expected_channels: int,
    full_decode: bool,
) -> dict[str, int]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - server runtime requirement
        raise RuntimeError(
            "dataset preflight requires numpy and soundfile in the training environment"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with sf.SoundFile(path) as handle:
            sample_rate = int(handle.samplerate)
            channels = int(handle.channels)
            frames = int(handle.frames)
            if sample_rate != expected_sample_rate:
                raise ValueError(
                    f"unexpected sample rate for {path}: "
                    f"expected={expected_sample_rate}, got={sample_rate}"
                )
            if channels != expected_channels:
                raise ValueError(
                    f"unexpected channel count for {path}: "
                    f"expected={expected_channels}, got={channels}"
                )
            if frames < 1:
                raise ValueError(f"empty audio file: {path}")
            if full_decode:
                decoded_frames = 0
                while True:
                    block = handle.read(
                        frames=262144, dtype="float32", always_2d=True
                    )
                    if block.size == 0:
                        break
                    if not np.isfinite(block).all():
                        raise ValueError(f"non-finite audio samples in {path}")
                    decoded_frames += int(block.shape[0])
                if decoded_frames != frames:
                    raise ValueError(
                        f"incomplete audio decode for {path}: "
                        f"metadata_frames={frames}, decoded_frames={decoded_frames}"
                    )
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot decode audio file {path}: {exc}") from exc
    return {"sample_rate": sample_rate, "channels": channels, "frames": frames}


def validate_paired_audio_dataset(
    *,
    train_manifest: Mapping[str, str],
    evaluation_manifest: Mapping[str, str],
    noisy_dir: str | Path,
    clean_dir: str | Path,
    expected_train_utterances: int,
    expected_evaluation_utterances: int,
    expected_sample_rate: int,
    expected_channels: int,
    conditions_per_step: int,
    optimizer_steps: int,
    full_decode: bool = True,
    require_single_coverage: bool = True,
    allow_partial_coverage: bool = False,
    allow_unlisted_audio: bool = False,
) -> dict:
    """Validate exact membership and fully decode every clean/noisy WAV pair."""

    train_ids = set(train_manifest)
    evaluation_ids = set(evaluation_manifest)
    if len(train_ids) != expected_train_utterances:
        raise ValueError(
            "unexpected training utterance count: "
            f"expected={expected_train_utterances}, got={len(train_ids)}"
        )
    if len(evaluation_ids) != expected_evaluation_utterances:
        raise ValueError(
            "unexpected evaluation utterance count: "
            f"expected={expected_evaluation_utterances}, got={len(evaluation_ids)}"
        )
    overlap = train_ids & evaluation_ids
    if overlap:
        raise ValueError(f"train/evaluation IDs overlap: {sorted(overlap)[:3]}")
    if conditions_per_step < 1 or optimizer_steps < 1:
        raise ValueError("conditions_per_step and optimizer_steps must be positive")
    expected_steps = (
        expected_train_utterances + conditions_per_step - 1
    ) // conditions_per_step
    if require_single_coverage and allow_partial_coverage:
        raise ValueError(
            "dataset preflight cannot require single coverage and allow partial "
            "coverage simultaneously"
        )
    if require_single_coverage and optimizer_steps != expected_steps:
        raise ValueError(
            "optimizer steps do not provide exactly one full-data coverage: "
            f"expected={expected_steps}, got={optimizer_steps}"
        )
    if (
        not require_single_coverage
        and not allow_partial_coverage
        and optimizer_steps < expected_steps
    ):
        raise ValueError(
            "multi-epoch training must cover the complete training manifest at least once: "
            f"minimum={expected_steps}, got={optimizer_steps}"
        )

    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    expected_ids = train_ids | evaluation_ids
    expected_files_per_kind = len(expected_ids)
    for kind, directory in (("noisy", noisy_dir), ("clean", clean_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        actual_ids = {path.stem for path in directory.glob("*.wav") if path.is_file()}
        missing = expected_ids - actual_ids
        unexpected = actual_ids - expected_ids
        unexpected_is_error = bool(unexpected) and not allow_unlisted_audio
        count_is_error = (
            len(actual_ids) != expected_files_per_kind and not allow_unlisted_audio
        )
        if missing or unexpected_is_error or count_is_error:
            raise ValueError(
                f"{kind} directory membership mismatch: "
                f"expected={expected_files_per_kind}, got={len(actual_ids)}, "
                f"missing={sorted(missing)[:3]}, unexpected={sorted(unexpected)[:3]}"
            )

    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    total_frames = 0
    minimum_frames = None
    maximum_frames = 0
    for utterance in sorted(expected_ids):
        clean = _audio_metadata(
            clean_dir / f"{utterance}.wav",
            expected_sample_rate=expected_sample_rate,
            expected_channels=expected_channels,
            full_decode=full_decode,
        )
        noisy = _audio_metadata(
            noisy_dir / f"{utterance}.wav",
            expected_sample_rate=expected_sample_rate,
            expected_channels=expected_channels,
            full_decode=full_decode,
        )
        if clean["frames"] != noisy["frames"]:
            raise ValueError(
                f"clean/noisy frame mismatch for {utterance}: "
                f"clean={clean['frames']}, noisy={noisy['frames']}"
            )
        for metadata in (clean, noisy):
            sample_rates[metadata["sample_rate"]] += 1
            channels[metadata["channels"]] += 1
            frames = metadata["frames"]
            total_frames += frames
            minimum_frames = frames if minimum_frames is None else min(minimum_frames, frames)
            maximum_frames = max(maximum_frames, frames)

    paired_utterances = len(expected_ids)
    audio_files = paired_utterances * 2
    scheduled_conditions = optimizer_steps * conditions_per_step
    return {
        "passed": True,
        "requirements": {
            "expected_train_utterances": expected_train_utterances,
            "expected_evaluation_utterances": expected_evaluation_utterances,
            "expected_sample_rate": expected_sample_rate,
            "expected_channels": expected_channels,
            "full_decode": full_decode,
            "require_single_coverage": require_single_coverage,
            "allow_partial_coverage": allow_partial_coverage,
            "exact_directory_membership": not allow_unlisted_audio,
            "allow_unlisted_audio": allow_unlisted_audio,
            "clean_noisy_frame_parity": True,
        },
        "observed": {
            "train_utterances": len(train_ids),
            "evaluation_utterances": len(evaluation_ids),
            "paired_utterances": paired_utterances,
            "audio_files": audio_files,
            "sample_rates": {str(key): value for key, value in sorted(sample_rates.items())},
            "channels": {str(key): value for key, value in sorted(channels.items())},
            "minimum_frames": minimum_frames,
            "maximum_frames": maximum_frames,
            "total_frames": total_frames,
        },
        "coverage": {
            "conditions_per_step": conditions_per_step,
            "optimizer_steps": optimizer_steps,
            "scheduled_conditions": scheduled_conditions,
            "next_epoch_tail_repeats": max(0, scheduled_conditions - len(train_ids)),
            "unvisited_train_conditions": max(0, len(train_ids) - scheduled_conditions),
            "complete_training_coverages": scheduled_conditions // len(train_ids),
        },
    }
