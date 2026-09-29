"""Method-neutral linear deployment normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationResult:
    waveform: np.ndarray
    input_rms: float
    rms_gain: float
    peak_safety_gain: float
    peak_before_safety: float
    peak_limited: bool


def rms_then_peak_safe(
    audio: np.ndarray,
    *,
    target_dbfs: float,
    peak_ceiling: float,
    silence_rms: float = 1e-8,
) -> NormalizationResult:
    """Apply target RMS, then a whole-waveform linear peak ceiling.

    No clipping or nonlinear limiting is used. If the RMS-normalized waveform
    exceeds the ceiling, the entire waveform is attenuated by one scalar.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        raise ValueError("cannot normalize empty audio")
    if not np.isfinite(audio).all():
        raise ValueError("cannot normalize audio containing NaN or Inf")
    if not 0 < peak_ceiling < 1:
        raise ValueError("peak_ceiling must lie in (0, 1)")
    input_rms = float(np.sqrt(np.mean(audio**2)))
    if input_rms <= silence_rms:
        raise ValueError(f"audio RMS {input_rms:.3g} is at or below silence threshold")
    target_rms = 10.0 ** (float(target_dbfs) / 20.0)
    rms_gain = target_rms / input_rms
    normalized = audio * rms_gain
    peak_before_safety = float(np.max(np.abs(normalized)))
    peak_safety_gain = (
        peak_ceiling / peak_before_safety
        if peak_before_safety > peak_ceiling
        else 1.0
    )
    normalized = normalized * peak_safety_gain
    result = np.asarray(normalized, dtype=np.float32)
    if float(np.max(np.abs(result))) > peak_ceiling + 1e-6:
        raise AssertionError("peak-safe normalization exceeded its ceiling")
    return NormalizationResult(
        waveform=result,
        input_rms=input_rms,
        rms_gain=float(rms_gain),
        peak_safety_gain=float(peak_safety_gain),
        peak_before_safety=peak_before_safety,
        peak_limited=bool(peak_safety_gain < 1.0),
    )
