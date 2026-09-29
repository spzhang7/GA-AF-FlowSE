"""Method-neutral waveform rewards and independent safety metrics.

SI-SDR is intentionally absent: the feasibility protocol excludes it from the
reward, Go/No-Go gates and reported claims for this experiment.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly


def load_mono(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float64)
    if source_rate != sample_rate:
        divisor = math.gcd(int(source_rate), int(sample_rate))
        mono = resample_poly(
            mono, sample_rate // divisor, source_rate // divisor
        )
    return np.asarray(mono, dtype=np.float64)


def audio_properties(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        raise ValueError("empty audio")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains NaN or Inf")
    absolute = np.abs(audio)
    return {
        "rms": float(np.sqrt(np.mean(audio**2) + 1e-12)),
        "peak": float(np.max(absolute)),
        "clipped_samples": int(np.count_nonzero(absolute >= 0.999)),
        "clipping_fraction": float(np.mean(absolute >= 0.999)),
        "samples": int(audio.size),
        "duration_seconds": float(audio.size / sample_rate),
    }


def saved_audio_properties(path: str | Path) -> dict[str, float | int | str]:
    """Inspect the exact file that waveform evaluators will read."""
    info = sf.info(path)
    audio = load_mono(path, int(info.samplerate))
    properties = audio_properties(audio, int(info.samplerate))
    return {
        "sample_rate": int(info.samplerate),
        "subtype": str(info.subtype),
        **properties,
    }


def _energy_envelope(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.size < frame:
        return np.asarray([np.sqrt(np.mean(audio**2) + 1e-12)])
    squared = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = (cumulative[frame:] - cumulative[:-frame]) / frame
    return np.sqrt(np.maximum(energy[::hop], 1e-12))


def align_by_energy_envelope(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
    max_delay_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return globally delay-aligned signals for diagnostic aligned STOI."""
    frame = max(1, round(0.025 * sample_rate))
    hop = max(1, round(0.005 * sample_rate))
    reference_envelope = _energy_envelope(reference, frame, hop)
    estimate_envelope = _energy_envelope(estimate, frame, hop)
    reference_envelope -= np.mean(reference_envelope)
    estimate_envelope -= np.mean(estimate_envelope)
    if np.linalg.norm(reference_envelope) < 1e-12 or np.linalg.norm(
        estimate_envelope
    ) < 1e-12:
        lag_samples = 0
    else:
        correlation = correlate(
            estimate_envelope, reference_envelope, mode="full", method="fft"
        )
        lags = correlation_lags(
            estimate_envelope.size, reference_envelope.size, mode="full"
        )
        max_lag_frames = max(1, int(max_delay_seconds * sample_rate / hop))
        valid = np.abs(lags) <= max_lag_frames
        lag_samples = int(lags[valid][np.argmax(correlation[valid])]) * hop
    if lag_samples > 0:
        estimate = estimate[lag_samples:]
    elif lag_samples < 0:
        reference = reference[-lag_samples:]
    length = min(reference.size, estimate.size)
    return reference[:length], estimate[:length], lag_samples


def paired_metrics(
    reference_path: str | Path,
    estimate_path: str | Path,
    *,
    sample_rate: int = 16000,
    max_delay_seconds: float = 1.0,
) -> dict[str, float]:
    """Evaluate PESQ-WB and STOI without any SI-SDR calculation."""
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError as exc:  # pragma: no cover - dependency checked on server
        raise RuntimeError("install requirements-eval.txt before Gate A") from exc
    reference = load_mono(reference_path, sample_rate)
    estimate = load_mono(estimate_path, sample_rate)
    reference_samples = int(reference.size)
    estimate_samples = int(estimate.size)
    length_difference = abs(reference_samples - estimate_samples)
    length = min(reference.size, estimate.size)
    if length == 0:
        raise ValueError("empty paired audio")
    reference = reference[:length]
    estimate = estimate[:length]
    aligned_reference, aligned_estimate, lag = align_by_energy_envelope(
        reference, estimate, sample_rate, max_delay_seconds
    )
    return {
        "pesq_wb": float(pesq(sample_rate, reference, estimate, "wb")),
        "stoi": float(stoi(reference, estimate, sample_rate, extended=False)),
        "stoi_aligned_diagnostic": float(
            stoi(
                aligned_reference,
                aligned_estimate,
                sample_rate,
                extended=False,
            )
        ),
        "energy_delay_ms": float(1000.0 * lag / sample_rate),
        "paired_reference_samples_before_truncation": reference_samples,
        "paired_estimate_samples_before_truncation": estimate_samples,
        "paired_length_difference_before_truncation": int(length_difference),
        "paired_relative_length_error": float(
            length_difference / max(1, reference_samples)
        ),
    }


class DNSMOSScorer:
    """Load Microsoft's official DNSMOS ONNX models once and score many files."""

    def __init__(self, official_dir: str | Path, personalized: bool = False):
        from tools.evaluate_dnsmos import load_official_module

        official_dir = Path(official_dir)
        script = official_dir / "dnsmos_local.py"
        p808_model = official_dir / "DNSMOS/model_v8.onnx"
        primary_model = official_dir / (
            "pDNSMOS/sig_bak_ovr.onnx"
            if personalized
            else "DNSMOS/sig_bak_ovr.onnx"
        )
        for path in (script, p808_model, primary_model):
            if not path.is_file():
                raise FileNotFoundError(path)
        module = load_official_module(script)
        self._scorer = module.ComputeScore(str(primary_model), str(p808_model))
        self._personalized = personalized

    def __call__(self, audio_path: str | Path) -> dict[str, float]:
        raw = self._scorer(str(audio_path), 16000, self._personalized)
        return {
            "dnsmos_sig": float(raw["SIG"]),
            "dnsmos_bak": float(raw["BAK"]),
            "dnsmos_ovrl": float(raw["OVRL"]),
            "dnsmos_p808": float(raw["P808_MOS"]),
        }
