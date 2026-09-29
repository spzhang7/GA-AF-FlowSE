import numpy as np
import pytest

from rl.common.normalization import (
    rms_then_peak_safe,
)


def test_peak_safe_normalization_preserves_target_rms_when_possible():
    time = np.linspace(0, 1, 16000, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * time)
    result = rms_then_peak_safe(audio, target_dbfs=-25.0, peak_ceiling=0.99)
    expected_rms = 10 ** (-25 / 20)
    assert np.sqrt(np.mean(result.waveform.astype(np.float64) ** 2)) == pytest.approx(
        expected_rms, rel=1e-6
    )
    assert result.peak_limited is False
    assert result.peak_safety_gain == 1.0


def test_peak_safe_normalization_linearly_attenuates_impulse():
    audio = np.ones(16000, dtype=np.float32) * 0.01
    audio[100] = 10.0
    result = rms_then_peak_safe(audio, target_dbfs=-25.0, peak_ceiling=0.99)
    assert result.peak_limited is True
    assert result.peak_safety_gain < 1.0
    assert np.max(np.abs(result.waveform)) == pytest.approx(0.99, abs=1e-6)
    # The operation is a single scalar: sample ratios remain unchanged.
    assert result.waveform[100] / result.waveform[0] == pytest.approx(1000.0)


def test_peak_safe_normalization_rejects_silence():
    with pytest.raises(ValueError, match="silence threshold"):
        rms_then_peak_safe(
            np.zeros(16000), target_dbfs=-25.0, peak_ceiling=0.99
        )


