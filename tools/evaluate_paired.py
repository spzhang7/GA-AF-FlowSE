"""Evaluate paired enhanced/reference WAV files with common SE metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm


METRICS = (
    "pesq_wb",
    "stoi_raw",
    "stoi_aligned",
    "si_sdr_raw",
    "si_sdr_aligned",
    "delay_ms",
)


def load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=source_rate,
            target_sr=sample_rate,
        )
    return np.asarray(audio, dtype=np.float64)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    eps = np.finfo(np.float64).eps
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = scale * reference
    noise = estimate - target
    return float(
        10
        * np.log10(
            (np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)
        )
    )


def energy_envelope(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if len(audio) < frame_length:
        return np.asarray([np.sqrt(np.mean(audio**2) + 1e-12)])
    squared = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = (
        cumulative[frame_length:] - cumulative[:-frame_length]
    ) / frame_length
    return np.sqrt(np.maximum(energy[::hop_length], 1e-12))


def align_by_energy_envelope(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    max_delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate a global delay from 25 ms RMS envelopes at 5 ms resolution."""
    frame_length = max(1, int(0.025 * sample_rate))
    hop_length = max(1, int(0.005 * sample_rate))
    reference_envelope = energy_envelope(reference, frame_length, hop_length)
    estimate_envelope = energy_envelope(estimate, frame_length, hop_length)
    reference_envelope -= np.mean(reference_envelope)
    estimate_envelope -= np.mean(estimate_envelope)

    if np.linalg.norm(reference_envelope) < 1e-12 or np.linalg.norm(
        estimate_envelope
    ) < 1e-12:
        lag_samples = 0
    else:
        correlation = correlate(
            estimate_envelope,
            reference_envelope,
            mode="full",
            method="fft",
        )
        lags = correlation_lags(
            len(estimate_envelope),
            len(reference_envelope),
            mode="full",
        )
        max_lag_frames = max(
            1,
            int(max_delay_seconds * sample_rate / hop_length),
        )
        valid = np.abs(lags) <= max_lag_frames
        lag_frames = int(lags[valid][np.argmax(correlation[valid])])
        lag_samples = lag_frames * hop_length

    if lag_samples > 0:
        estimate = estimate[lag_samples:]
    elif lag_samples < 0:
        reference = reference[-lag_samples:]
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length], lag_samples


def evaluate_one(task: tuple[str, str, str, int, float]) -> dict:
    utterance, reference_path, estimate_path, sample_rate, max_delay_seconds = task
    row = {
        "utterance": utterance,
        "reference": reference_path,
        "estimate": estimate_path,
        "error": "",
    }
    try:
        reference = load_mono(reference_path, sample_rate)
        estimate = load_mono(estimate_path, sample_rate)
        length_difference = abs(len(reference) - len(estimate))
        length = min(len(reference), len(estimate))
        if length == 0:
            raise ValueError("empty audio")
        reference = reference[:length]
        estimate = estimate[:length]
        if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
            raise ValueError("NaN or Inf in audio")

        aligned_reference, aligned_estimate, delay_samples = align_by_energy_envelope(
            reference,
            estimate,
            sample_rate,
            max_delay_seconds,
        )
        if len(aligned_reference) == 0:
            raise ValueError("alignment produced empty audio")

        row.update(
            {
                "sample_rate": sample_rate,
                "samples": length,
                "duration": length / sample_rate,
                "length_difference": length_difference,
                "pesq_wb": float(pesq(sample_rate, reference, estimate, "wb")),
                "stoi_raw": float(
                    stoi(reference, estimate, sample_rate, extended=False)
                ),
                "stoi_aligned": float(
                    stoi(
                        aligned_reference,
                        aligned_estimate,
                        sample_rate,
                        extended=False,
                    )
                ),
                "si_sdr_raw": si_sdr(reference, estimate),
                "si_sdr_aligned": si_sdr(aligned_reference, aligned_estimate),
                "delay_ms": 1000 * delay_samples / sample_rate,
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        for metric in METRICS:
            row[metric] = math.nan
    return row


def index_wavs(directory: Path) -> dict[str, Path]:
    result = {}
    for path in directory.rglob("*.wav"):
        if path.stem in result:
            raise ValueError(f"Duplicate WAV stem: {path.stem}")
        result[path.stem] = path
    return result


def summarize(rows: list[dict]) -> dict:
    summary = {
        "total": len(rows),
        "successful": sum(not row["error"] for row in rows),
        "failed": sum(bool(row["error"]) for row in rows),
    }
    for metric in METRICS:
        values = np.asarray(
            [row[metric] for row in rows if not row["error"]],
            dtype=np.float64,
        )
        if values.size == 0:
            summary[metric] = None
            continue
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def evaluate_directories(
    reference_dir: Path,
    estimate_dir: Path,
    output_prefix: Path,
    sample_rate: int = 16000,
    workers: int = min(8, os.cpu_count() or 1),
    max_delay_seconds: float = 1.0,
) -> tuple[list[dict], dict]:
    references = index_wavs(reference_dir)
    estimates = index_wavs(estimate_dir)
    missing = sorted(set(references) - set(estimates))
    extra = sorted(set(estimates) - set(references))
    if missing or extra:
        raise ValueError(
            f"Pair mismatch: missing estimates={len(missing)}, "
            f"extra estimates={len(extra)}"
        )

    tasks = [
        (
            utterance,
            str(references[utterance]),
            str(estimates[utterance]),
            sample_rate,
            max_delay_seconds,
        )
        for utterance in sorted(references)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(
            tqdm(
                executor.map(evaluate_one, tasks),
                total=len(tasks),
                desc=f"Evaluating {estimate_dir.name}",
            )
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".summary.json")
    fieldnames = [
        "utterance",
        "reference",
        "estimate",
        "sample_rate",
        "samples",
        "duration",
        "length_difference",
        *METRICS,
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Per-utterance CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--estimate-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    _, summary = evaluate_directories(
        reference_dir=args.reference_dir,
        estimate_dir=args.estimate_dir,
        output_prefix=args.output_prefix,
        sample_rate=args.sample_rate,
        workers=args.workers,
        max_delay_seconds=args.max_delay_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
