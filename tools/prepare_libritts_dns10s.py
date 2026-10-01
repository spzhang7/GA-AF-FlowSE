"""Build a reproducible LibriTTS + noise + RIR enhancement corpus.

The script intentionally keeps dataset acquisition separate from corpus
construction: users download LibriTTS, noise, and room impulse responses under
their respective licenses, then point this command at those directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


TARGET_SAMPLE_RATE = 16_000


def audio_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in {".wav", ".flac"}
    )


def read_mono(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"audio must be a non-empty mono/stereo waveform: {path}")
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        waveform = resample_poly(
            waveform, TARGET_SAMPLE_RATE, int(sample_rate)
        ).astype(np.float32)
    return np.asarray(waveform, dtype=np.float32)


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))


def fit_length(waveform: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if waveform.size == 0:
        raise ValueError("cannot fit an empty waveform")
    if waveform.size < length:
        repeats = (length + waveform.size - 1) // waveform.size
        waveform = np.tile(waveform, repeats)
    if waveform.size == length:
        return waveform.copy()
    start = int(rng.integers(0, waveform.size - length + 1))
    return waveform[start : start + length].copy()


def apply_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32)
    rir = rir / max(float(np.max(np.abs(rir))), 1.0e-8)
    reverberant = fftconvolve(clean, rir, mode="full")[: clean.size]
    clean_rms = rms(clean)
    reverberant_rms = rms(reverberant)
    if clean_rms > 0.0 and reverberant_rms > 0.0:
        reverberant *= clean_rms / reverberant_rms
    return reverberant.astype(np.float32)


def utterance_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return relative.as_posix().replace("/", "__")


def load_transcripts(path: Path | None) -> Mapping[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(text, str)
        for key, text in value.items()
    ):
        raise ValueError("--transcripts must be a JSON object mapping IDs to strings")
    return value


def write_audio(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.99:
        waveform = waveform * (0.99 / peak)
    sf.write(path, np.asarray(waveform, dtype=np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paired 16-kHz LibriTTS/noisy data with deterministic noise and RIR mixing."
    )
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--rir-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/libritts_dns10s/audio"))
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("artifacts/af/manifests/libritts_dns10s"),
    )
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--seed", type=int, default=260810)
    parser.add_argument("--snr-min-db", type=float, default=0.0)
    parser.add_argument("--snr-max-db", type=float, default=20.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-utterances", type=int)
    args = parser.parse_args()

    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in [0, 1)")
    if args.snr_min_db > args.snr_max_db:
        raise ValueError("--snr-min-db must not exceed --snr-max-db")
    clean_paths = audio_files(args.clean_dir)
    noise_paths = audio_files(args.noise_dir)
    rir_paths = audio_files(args.rir_dir)
    if not clean_paths or not noise_paths or not rir_paths:
        raise FileNotFoundError("clean, noise, and RIR directories must contain audio files")
    if args.max_utterances is not None:
        if args.max_utterances < 1:
            raise ValueError("--max-utterances must be positive")
        clean_paths = clean_paths[: args.max_utterances]

    transcripts = load_transcripts(args.transcripts)
    rng = np.random.default_rng(args.seed)
    clean_out = args.output_dir / "clean"
    noisy_out = args.output_dir / "noisy"
    rows: dict[str, str] = {}
    for index, clean_path in enumerate(clean_paths):
        key = utterance_id(clean_path, args.clean_dir)
        clean = read_mono(clean_path)
        noise = fit_length(read_mono(noise_paths[index % len(noise_paths)]), clean.size, rng)
        rir = read_mono(rir_paths[index % len(rir_paths)])
        reverberant = apply_rir(clean, rir)
        clean_level = max(rms(reverberant), 1.0e-8)
        noise_level = max(rms(noise), 1.0e-8)
        snr_db = float(rng.uniform(args.snr_min_db, args.snr_max_db))
        noise *= clean_level / (noise_level * (10.0 ** (snr_db / 20.0)))
        write_audio(clean_out / f"{key}.wav", clean)
        write_audio(noisy_out / f"{key}.wav", reverberant + noise)
        rows[key] = str(transcripts.get(key, ""))

    keys = sorted(rows)
    validation_count = int(round(len(keys) * args.validation_fraction))
    validation_keys = set(keys[-validation_count:]) if validation_count else set()
    train = {key: rows[key] for key in keys if key not in validation_keys}
    validation = {key: rows[key] for key in keys if key in validation_keys}
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in (
        ("libritts_dns10s_train_exposures.json", train),
        ("libritts_dns10s_validation.json", validation),
    ):
        (args.manifest_dir / name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Generated {len(train)} training and {len(validation)} validation utterances")
    print(f"Audio: {args.output_dir}")
    print(f"Manifests: {args.manifest_dir}")


if __name__ == "__main__":
    main()
