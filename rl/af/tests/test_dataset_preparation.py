from pathlib import Path

import numpy as np
import pytest

from rl.common.dataset_preparation import (
    _rir_files,
    _write_jsonl,
    build_recipe_rows,
    sample_without_replacement,
    summarize_recipe,
    synthesize_recipe,
)


def _clean_rows(count=200):
    return [
        {
            "utterance": f"{100 + index % 20}_{index:04d}_0000_0000",
            "speaker": str(100 + index % 20),
            "split": "train-clean-100",
            "clean_source": f"/clean/{index}.wav",
            "transcript": f"text {index}",
        }
        for index in range(count)
    ]


def _recipe(rows, seed=7):
    return build_recipe_rows(
        rows,
        split="train",
        seed=seed,
        noise_pools={
            "demand": [Path("/noise/demand/a.wav")],
            "dns2021": [Path("/noise/dns/a.wav"), Path("/noise/dns/b.wav")],
            "wham_tr": [Path("/noise/wham/a.wav")],
        },
        noise_source_weights={"demand": 0.2, "dns2021": 0.5, "wham_tr": 0.3},
        rir_pools={
            "openslr26": [Path("/rir/26/a.wav")],
            "openslr28": [Path("/rir/28/a.wav")],
        },
        rir_probability=0.30,
    )


def test_exposure_sampling_is_deterministic_without_replacement():
    rows = _clean_rows()
    first = sample_without_replacement(rows, count=80, seed=11)
    second = sample_without_replacement(rows, count=80, seed=11)
    assert first == second
    assert len({row["utterance"] for row in first}) == 80
    assert first != sample_without_replacement(rows, count=80, seed=12)


def test_recipe_file_is_frozen_and_rejects_drift(tmp_path):
    path = tmp_path / "condition_recipe.jsonl"
    rows = [{"utterance": "100_1_1_1", "seed": 7}]
    _write_jsonl(path, rows)
    first = path.read_bytes()
    _write_jsonl(path, rows)
    assert path.read_bytes() == first
    with pytest.raises(ValueError, match="frozen recipe differs"):
        _write_jsonl(path, [{"utterance": "100_1_1_1", "seed": 8}])


def test_recipe_is_deterministic_and_records_all_random_choices():
    rows = _clean_rows(100)
    first = _recipe(rows)
    assert first == _recipe(rows)
    assert first != _recipe(rows, seed=8)
    for row in first:
        assert len(row["noise_paths"]) == row["noise_count"]
        assert len(row["snr_db"]) == row["noise_count"]
        assert -35.0 <= row["output_dbfs"] <= -15.0
        assert row["rir_enabled"] == (row["rir_path"] is not None)
        assert row["selection_seed"] == 7
        assert isinstance(row["recipe_seed"], int)
        assert isinstance(row["mixture_seed"], int)


def test_recipe_probabilities_follow_frozen_design_at_scale():
    rows = _clean_rows(10_000)
    recipe = _recipe(rows)
    two_noise = sum(row["noise_count"] == 2 for row in recipe) / len(recipe)
    reverb = sum(row["rir_enabled"] for row in recipe) / len(recipe)
    assert two_noise == pytest.approx(0.25, abs=0.015)
    assert reverb == pytest.approx(0.30, abs=0.015)
    summary = summarize_recipe(recipe)
    assert summary["unique_clean_utterances"] == len(recipe)
    assert summary["two_noise_fraction"] == pytest.approx(0.25, abs=0.015)
    assert summary["rir_fraction"] == pytest.approx(0.30, abs=0.015)
    assert sum(summary["noise_source_terms"].values()) == sum(
        row["noise_count"] for row in recipe
    )


def test_recipe_rejects_missing_rir_source():
    with pytest.raises(ValueError, match="two non-empty source pools"):
        build_recipe_rows(
            _clean_rows(1),
            split="train",
            seed=7,
            noise_pools={
                "wham": [Path("/noise/a.wav"), Path("/noise/b.wav")]
            },
            noise_source_weights={"wham": 1.0},
            rir_pools={"openslr26": [Path("/rir/a.wav")]},
            rir_probability=0.3,
        )


def test_two_noise_recipe_rejects_a_single_unique_noise_file():
    with pytest.raises(ValueError, match="two distinct noise files"):
        build_recipe_rows(
            _clean_rows(1),
            split="train",
            seed=7,
            noise_pools={"only": [Path("/noise/a.wav")]},
            noise_source_weights={"only": 1.0},
            rir_pools={},
            rir_probability=0.0,
        )


def test_openslr28_mixed_directory_excludes_bundled_noises(tmp_path):
    mixed = tmp_path / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    simulated = tmp_path / "RIRS_NOISES" / "simulated_rirs" / "smallroom"
    point = tmp_path / "RIRS_NOISES" / "pointsource_noises"
    mixed.mkdir(parents=True)
    simulated.mkdir(parents=True)
    point.mkdir(parents=True)
    real_rir = mixed / "air_type1_air_binaural_booth_1_3.wav"
    isotropic_noise = mixed / "RVB2014_type2_noise_simroom3_4.wav"
    simulated_rir = simulated / "rir_001.wav"
    point_noise = point / "babble.wav"
    for path in (real_rir, isotropic_noise, simulated_rir, point_noise):
        path.touch()
    assert _rir_files(tmp_path) == sorted(
        [real_rir.resolve(), simulated_rir.resolve()]
    )


def test_recipe_audio_materialization_is_atomic_and_resumable(tmp_path):
    sf = pytest.importorskip("soundfile")
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    clean = 0.2 * np.sin(2 * np.pi * 220 * time)
    noise = 0.1 * np.sin(2 * np.pi * 937 * time)
    rir = np.zeros(256, dtype=np.float64)
    rir[0], rir[80] = 1.0, 0.25
    clean_source = tmp_path / "clean_source.wav"
    noise_source = tmp_path / "noise_source.wav"
    rir_source = tmp_path / "rir.wav"
    sf.write(clean_source, clean, sample_rate)
    sf.write(noise_source, noise, sample_rate)
    sf.write(rir_source, rir, sample_rate)
    noisy_output = tmp_path / "view" / "noisy" / "100_1_1_1.wav"
    clean_output = tmp_path / "view" / "clean" / "100_1_1_1.wav"
    row = {
        "clean_source": str(clean_source),
        "noise_paths": [str(noise_source)],
        "snr_db": [5.0],
        "rir_enabled": True,
        "rir_path": str(rir_source),
        "mixture_seed": 9,
        "target_dbfs": -25.0,
        "output_dbfs": -20.0,
        "peak_ceiling": 0.99,
        "noisy_output": str(noisy_output),
        "clean_output": str(clean_output),
    }
    assert synthesize_recipe(row)[1] == "written"
    assert synthesize_recipe(row)[1] == "cached"
    noisy_audio, noisy_sr = sf.read(noisy_output, always_2d=True)
    clean_audio, clean_sr = sf.read(clean_output, always_2d=True)
    assert noisy_sr == clean_sr == sample_rate
    assert noisy_audio.shape == clean_audio.shape == (sample_rate, 1)
    assert np.max(np.abs(noisy_audio)) <= 0.99 + 1.0e-4

    # An interrupted/corrupt final file must be regenerated, not silently cached.
    noisy_output.write_bytes(b"not a wav")
    assert synthesize_recipe(row)[1] == "written"
    assert sf.info(noisy_output).samplerate == sample_rate

