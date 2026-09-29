"""Prepare deterministic LibriTTS post-training mixtures and DNS2020 views.

The complete LibriTTS-960 training pool is scanned first.  A frozen exposure
schedule is then sampled from that pool and written as JSONL before audio is
materialized.  This keeps data augmentation reproducible across AF/GRPO and
across interrupted runs without pretending that the exposure schedule is the
underlying clean dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EXPECTED_LIBRITTS_COUNTS = {
    "train-clean-100": 33_236,
    "train-clean-360": 116_500,
    "train-other-500": 205_044,
    "dev-clean": 5_736,
    "dev-other": 4_613,
}
TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
DEV_SPLITS = ("dev-clean", "dev-other")
SNR_BINS = (
    (-5.0, 0.0, 0.05),
    (0.0, 5.0, 0.30),
    (5.0, 10.0, 0.45),
    (10.0, 20.0, 0.20),
)
FILE_ID = re.compile(r"fileid_(\d+)", re.IGNORECASE)
EPS = np.finfo(np.float64).eps


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """Freeze a recipe atomically and reject any later recipe drift."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(
                f"frozen recipe differs from the requested data design: {path}"
            )
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _audio_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    )


def _rir_files(root: Path) -> list[Path]:
    """Select impulse responses while excluding OpenSLR28 bundled noises."""

    blocked_parts = {"isotropic_noises", "pointsource_noises"}
    output = []
    for path in _audio_files(root):
        relative_parts = tuple(part.lower() for part in path.relative_to(root).parts)
        relative_text = "/".join(relative_parts)
        if set(relative_parts) & blocked_parts:
            continue
        # OpenSLR28 stores real AIR RIRs and RVB isotropic noises in the same
        # real_rirs_isotropic_noises directory. The latter are named
        # *_noise_* and must never enter the RIR pool.
        if "noise" in path.stem.lower():
            continue
        if "rir" not in relative_text:
            continue
        output.append(path)
    if not output:
        raise ValueError(f"no RIR WAV files found below {root}")
    return output


def scan_libritts(root: Path, splits: Sequence[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    counts = {}
    seen = set()
    for split in splits:
        audio = _audio_files(root / split)
        counts[split] = len(audio)
        expected = EXPECTED_LIBRITTS_COUNTS.get(split)
        if expected is not None and len(audio) != expected:
            raise ValueError(
                f"incomplete LibriTTS split {split}: expected={expected}, got={len(audio)}"
            )
        for path in audio:
            utterance = path.stem
            if utterance in seen:
                raise ValueError(f"duplicate LibriTTS utterance: {utterance}")
            seen.add(utterance)
            text_path = path.with_suffix(".normalized.txt")
            if not text_path.is_file():
                raise FileNotFoundError(text_path)
            transcript = " ".join(text_path.read_text(encoding="utf-8").split())
            if not transcript:
                raise ValueError(f"empty transcript: {text_path}")
            speaker = utterance.split("_", 1)[0]
            if not speaker.isdigit():
                raise ValueError(f"invalid LibriTTS utterance ID: {utterance}")
            rows.append(
                {
                    "utterance": utterance,
                    "speaker": speaker,
                    "split": split,
                    "clean_source": str(path),
                    "transcript": transcript,
                }
            )
    return rows, counts


def sample_without_replacement(rows: Sequence[dict], *, count: int, seed: int) -> list[dict]:
    if count < 1 or count > len(rows):
        raise ValueError(f"invalid exposure count {count} for pool of {len(rows)}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [dict(rows[index]) for index in indices[:count]]


def _weighted_choice(rng: random.Random, values: Sequence, weights: Sequence[float]):
    if len(values) != len(weights) or not values:
        raise ValueError("weighted choice requires equally sized non-empty inputs")
    total = float(sum(weights))
    if total <= 0 or any(float(value) < 0 for value in weights):
        raise ValueError(f"invalid weights: {weights}")
    threshold = rng.random() * total
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += float(weight)
        if threshold <= cumulative:
            return value
    return values[-1]


def _sample_snr(rng: random.Random) -> float:
    selected = _weighted_choice(rng, SNR_BINS, [item[2] for item in SNR_BINS])
    return rng.uniform(selected[0], selected[1])


def build_recipe_rows(
    clean_rows: Sequence[dict],
    *,
    split: str,
    seed: int,
    noise_pools: Mapping[str, Sequence[Path]],
    noise_source_weights: Mapping[str, float],
    rir_pools: Mapping[str, Sequence[Path]],
    rir_probability: float,
    two_noise_probability: float = 0.25,
) -> list[dict]:
    if not 0.0 <= rir_probability <= 1.0:
        raise ValueError("rir_probability must lie in [0, 1]")
    if not 0.0 <= two_noise_probability <= 1.0:
        raise ValueError("two_noise_probability must lie in [0, 1]")
    noise_sources = sorted(noise_pools)
    if set(noise_sources) != set(noise_source_weights):
        raise ValueError("noise source weights do not match noise pools")
    if any(not noise_pools[name] for name in noise_sources):
        raise ValueError("every noise source must contain audio")
    unique_noise_files = {
        str(path) for paths in noise_pools.values() for path in paths
    }
    if two_noise_probability > 0 and len(unique_noise_files) < 2:
        raise ValueError("two-noise augmentation requires two distinct noise files")
    rir_sources = sorted(rir_pools)
    if rir_probability > 0 and (
        len(rir_sources) != 2 or any(not rir_pools[name] for name in rir_sources)
    ):
        raise ValueError("RIR augmentation requires two non-empty source pools")

    output = []
    for exposure_index, clean in enumerate(clean_rows):
        recipe_seed = stable_seed(seed, split, exposure_index, clean["utterance"])
        mixture_seed = stable_seed(seed, "audio", split, exposure_index)
        rng = random.Random(recipe_seed)
        noise_count = 2 if rng.random() < two_noise_probability else 1
        noise_paths = []
        noise_sources_used = []
        while len(noise_paths) < noise_count:
            source = _weighted_choice(
                rng,
                noise_sources,
                [noise_source_weights[name] for name in noise_sources],
            )
            candidate = str(rng.choice(list(noise_pools[source])))
            if candidate not in noise_paths:
                noise_paths.append(candidate)
                noise_sources_used.append(source)
        use_rir = rng.random() < rir_probability
        rir_source = None
        rir_path = None
        if use_rir:
            # Source-balanced OpenSLR26/OpenSLR28 sampling.
            rir_source = rng.choice(rir_sources)
            rir_path = str(rng.choice(list(rir_pools[rir_source])))
        output.append(
            {
                **clean,
                "recipe_schema": 1,
                "dataset_split": split,
                "exposure_index": exposure_index,
                "selection_seed": int(seed),
                "recipe_seed": recipe_seed,
                "mixture_seed": mixture_seed,
                "noise_count": noise_count,
                "noise_sources": noise_sources_used,
                "noise_paths": noise_paths,
                "snr_db": [_sample_snr(rng) for _ in range(noise_count)],
                "output_dbfs": rng.uniform(-35.0, -15.0),
                "rir_enabled": use_rir,
                "rir_source": rir_source,
                "rir_path": rir_path,
                "target_dbfs": -25.0,
                "peak_ceiling": 0.99,
                "reference_policy": "dry_clean_scaled_with_mixture_gain",
            }
        )
    return output


def summarize_recipe(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty recipe")
    noise_counts = Counter(int(row["noise_count"]) for row in rows)
    noise_sources = Counter(
        str(source) for row in rows for source in row["noise_sources"]
    )
    rir_sources = Counter(
        str(row["rir_source"]) for row in rows if bool(row["rir_enabled"])
    )
    rir_count = sum(bool(row["rir_enabled"]) for row in rows)
    noise_terms = sum(noise_counts[count] * count for count in noise_counts)
    return {
        "conditions": len(rows),
        "unique_clean_utterances": len({str(row["utterance"]) for row in rows}),
        "noise_count_conditions": {
            str(count): int(value) for count, value in sorted(noise_counts.items())
        },
        "two_noise_fraction": float(noise_counts[2] / len(rows)),
        "noise_source_terms": dict(sorted(noise_sources.items())),
        "noise_source_term_fractions": {
            source: float(value / noise_terms)
            for source, value in sorted(noise_sources.items())
        },
        "rir_conditions": int(rir_count),
        "rir_fraction": float(rir_count / len(rows)),
        "rir_source_conditions": dict(sorted(rir_sources.items())),
    }


def _resample(audio: np.ndarray, source_sr: int, target_sr: int = 16_000) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float64, copy=False)
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source_sr), int(target_sr))
    return resample_poly(audio, target_sr // divisor, source_sr // divisor).astype(
        np.float64, copy=False
    )


def _read_mono(path: str | Path, target_sr: int = 16_000) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    # Match the public FlowSE loader, which consistently reads channel zero.
    mono = audio[:, 0]
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError(f"invalid audio: {path}")
    return _resample(mono, int(sr), target_sr)


def _fit_noise(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if noise.size < length:
        start = rng.randrange(length + 1 - noise.size)
        output = np.zeros(length, dtype=np.float64)
        output[start : start + noise.size] = noise
        return output
    if noise.size > length:
        start = rng.randrange(noise.size + 1 - length)
        return noise[start : start + length]
    return noise


def _peak_then_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    normalized = audio / (peak + EPS)
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    return normalized * (10.0 ** (target_dbfs / 20.0) / (rms + EPS))


def synthesize_recipe(row: Mapping) -> tuple[str, str]:
    """Materialize one recipe atomically; safe to repeat after interruption."""

    from scipy.signal import oaconvolve
    import soundfile as sf

    noisy_path = Path(str(row["noisy_output"]))
    clean_path = Path(str(row["clean_output"]))
    if _valid_cached_pair(noisy_path, clean_path):
        return str(noisy_path), "cached"
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(row["mixture_seed"]))
    dry_clean = _read_mono(str(row["clean_source"]))
    speech = dry_clean
    if bool(row["rir_enabled"]):
        rir = _read_mono(str(row["rir_path"]))
        speech = oaconvolve(speech, rir, mode="full")[: speech.size]
    speech = _peak_then_rms(speech, float(row["target_dbfs"]))
    dry_reference = _peak_then_rms(dry_clean, float(row["target_dbfs"]))
    noise_terms = []
    for path, snr_db in zip(row["noise_paths"], row["snr_db"]):
        noise = _fit_noise(_read_mono(str(path)), speech.size, rng)
        noise = _peak_then_rms(noise, float(row["target_dbfs"]))
        speech_rms = float(np.sqrt(np.mean(np.square(speech))))
        noise_rms = float(np.sqrt(np.mean(np.square(noise))))
        scalar = speech_rms / (10.0 ** (float(snr_db) / 20.0)) / (noise_rms + EPS)
        noise_terms.append(noise * scalar)
    noisy = speech + sum(noise_terms, np.zeros_like(speech))
    noisy_rms = float(np.sqrt(np.mean(np.square(noisy))))
    output_gain = 10.0 ** (float(row["output_dbfs"]) / 20.0) / (noisy_rms + EPS)
    noisy *= output_gain
    dry_reference *= output_gain
    peak = float(np.max(np.abs(noisy)))
    ceiling = float(row["peak_ceiling"])
    if peak > ceiling:
        limiter = peak / (ceiling - EPS)
        noisy /= limiter
        dry_reference /= limiter

    noisy_tmp = noisy_path.with_name(noisy_path.stem + ".tmp.wav")
    clean_tmp = clean_path.with_name(clean_path.stem + ".tmp.wav")
    sf.write(str(noisy_tmp), noisy.astype(np.float32), 16_000, subtype="PCM_16")
    sf.write(
        str(clean_tmp), dry_reference.astype(np.float32), 16_000, subtype="PCM_16"
    )
    os.replace(noisy_tmp, noisy_path)
    os.replace(clean_tmp, clean_path)
    return str(noisy_path), "written"


def _valid_cached_pair(noisy_path: Path, clean_path: Path) -> bool:
    """Only resume past a pair whose two final WAVs are complete and compatible."""

    if not noisy_path.is_file() or not clean_path.is_file():
        return False
    try:
        import soundfile as sf

        noisy = sf.info(str(noisy_path))
        clean = sf.info(str(clean_path))
    except Exception:
        return False
    expected = {
        "samplerate": 16_000,
        "channels": 1,
        "subtype": "PCM_16",
    }
    return (
        noisy.samplerate == expected["samplerate"]
        and clean.samplerate == expected["samplerate"]
        and noisy.channels == expected["channels"]
        and clean.channels == expected["channels"]
        and noisy.subtype == expected["subtype"]
        and clean.subtype == expected["subtype"]
        and noisy.frames > 0
        and noisy.frames == clean.frames
    )


def _materialize(rows: Sequence[dict], *, jobs: int) -> dict[str, int]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    counts = {"written": 0, "cached": 0}
    if jobs == 1:
        results = map(synthesize_recipe, rows)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(synthesize_recipe, rows, chunksize=8)
    try:
        for index, (_, status) in enumerate(results, start=1):
            counts[status] += 1
            if index % 500 == 0 or index == len(rows):
                print(f"Materialized {index}/{len(rows)} mixtures", flush=True)
    finally:
        if jobs != 1:
            pool.shutdown(wait=True)
    return counts


def _with_outputs(rows: Sequence[dict], audio_root: Path) -> list[dict]:
    output = []
    for row in rows:
        utterance = str(row["utterance"])
        output.append(
            {
                **row,
                "noisy_output": str((audio_root / "noisy" / f"{utterance}.wav").resolve()),
                "clean_output": str((audio_root / "clean" / f"{utterance}.wav").resolve()),
            }
        )
    return output


def _file_id(path: Path) -> int:
    match = FILE_ID.search(path.stem)
    if match is None:
        raise ValueError(f"DNS2020 filename lacks fileid: {path}")
    return int(match.group(1))


def _safe_symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise ValueError(f"audio-view collision: {target}")
        return
    os.symlink(source, target)


def prepare_dns2020_views(dns_root: Path, audio_root: Path, output_dir: Path) -> dict:
    test_root = dns_root / "datasets" / "test_set"
    manifests = {}
    metadata = {}
    for short, folder in (("no_reverb", "no_reverb"), ("with_reverb", "with_reverb")):
        clean_by_id = {_file_id(path): path for path in _audio_files(test_root / "synthetic" / folder / "clean")}
        noisy_by_id = {_file_id(path): path for path in _audio_files(test_root / "synthetic" / folder / "noisy")}
        if set(clean_by_id) != set(noisy_by_id) or len(clean_by_id) != 150:
            raise ValueError(f"invalid DNS2020 {folder} pairing")
        manifest = {}
        for file_id in sorted(clean_by_id):
            utterance = f"dns_{short}_fileid_{file_id:06d}"
            _safe_symlink(noisy_by_id[file_id], audio_root / "noisy" / f"{utterance}.wav")
            _safe_symlink(clean_by_id[file_id], audio_root / "clean" / f"{utterance}.wav")
            manifest[utterance] = ""
        path = output_dir / f"dns2020_{short}.json"
        _write_json(path, manifest)
        manifests[short] = str(path)
        metadata[short] = {"utterances": len(manifest), "paired": True}

    real_manifest = {}
    for index, source in enumerate(_audio_files(test_root / "real_recordings")):
        utterance = f"dns_real_{index:06d}"
        _safe_symlink(source, audio_root / "noisy" / f"{utterance}.wav")
        real_manifest[utterance] = ""
    if len(real_manifest) != 300:
        raise ValueError(f"expected 300 DNS real recordings, got {len(real_manifest)}")
    real_path = output_dir / "dns2020_real_recordings.json"
    _write_json(real_path, real_manifest)
    manifests["real_recordings"] = str(real_path)
    metadata["real_recordings"] = {"utterances": 300, "paired": False}
    return {"manifests": manifests, "subsets": metadata}


def _manifest(rows: Sequence[dict]) -> dict[str, str]:
    return {str(row["utterance"]): str(row["transcript"]) for row in rows}


def run(args: argparse.Namespace) -> dict:
    libritts_root = args.libritts_root.resolve()
    output_dir = args.output_dir.resolve()
    audio_root = args.audio_root.resolve()
    train_pool, train_counts = scan_libritts(libritts_root, TRAIN_SPLITS)
    dev_pool, dev_counts = scan_libritts(libritts_root, DEV_SPLITS)
    overlap = {row["utterance"] for row in train_pool} & {
        row["utterance"] for row in dev_pool
    }
    if overlap:
        raise ValueError(f"LibriTTS train/dev overlap: {sorted(overlap)[:3]}")
    train_selected = sample_without_replacement(
        train_pool, count=args.train_exposures, seed=args.seed
    )
    dev_selected = sample_without_replacement(
        dev_pool, count=args.validation_utterances, seed=stable_seed(args.seed, "dev")
    )

    train_noise_pools = {
        "demand": _audio_files(args.demand_root),
        "dns2021": _audio_files(args.dns_noise_root),
        "wham_tr": _audio_files(args.wham_train_root),
    }
    valid_noise_pools = {"wham_cv": _audio_files(args.wham_valid_root)}
    rir_pools = {
        "openslr26": _rir_files(args.rir26_root),
        "openslr28": _rir_files(args.rir28_root),
    }
    train_recipe = build_recipe_rows(
        train_selected,
        split="train",
        seed=args.seed,
        noise_pools=train_noise_pools,
        noise_source_weights={
            "dns2021": args.dns_noise_weight,
            "wham_tr": args.wham_noise_weight,
            "demand": args.demand_noise_weight,
        },
        rir_pools=rir_pools,
        rir_probability=args.rir_probability,
    )
    # Half of validation is dry and half reverberant by construction.
    midpoint = len(dev_selected) // 2
    valid_recipe = build_recipe_rows(
        dev_selected[:midpoint],
        split="valid_dry",
        seed=stable_seed(args.seed, "valid_dry"),
        noise_pools=valid_noise_pools,
        noise_source_weights={"wham_cv": 1.0},
        rir_pools=rir_pools,
        rir_probability=0.0,
    ) + build_recipe_rows(
        dev_selected[midpoint:],
        split="valid_reverb",
        seed=stable_seed(args.seed, "valid_reverb"),
        noise_pools=valid_noise_pools,
        noise_source_weights={"wham_cv": 1.0},
        rir_pools=rir_pools,
        rir_probability=1.0,
    )
    train_recipe = _with_outputs(train_recipe, audio_root)
    valid_recipe = _with_outputs(valid_recipe, audio_root)
    recipe_path = output_dir / "libritts_5000step_condition_recipe.jsonl"
    valid_recipe_path = output_dir / "libritts_validation_recipe.jsonl"
    _write_jsonl(recipe_path, train_recipe)
    _write_jsonl(valid_recipe_path, valid_recipe)
    _write_json(output_dir / "libritts_train_exposures.json", _manifest(train_recipe))
    _write_json(output_dir / "libritts_validation.json", _manifest(valid_recipe))
    _write_json(
        output_dir / "libritts_validation_dry.json", _manifest(valid_recipe[:midpoint])
    )
    _write_json(
        output_dir / "libritts_validation_reverb.json",
        _manifest(valid_recipe[midpoint:]),
    )
    _write_json(
        output_dir / "libritts_smoke_train.json", _manifest(train_recipe[:32])
    )
    _write_json(
        output_dir / "libritts_smoke_validation.json", _manifest(valid_recipe[:16])
    )

    materialized = None
    if not args.recipe_only:
        materialized = _materialize(train_recipe + valid_recipe, jobs=args.jobs)
    dns = prepare_dns2020_views(args.dns2020_root.resolve(), audio_root, output_dir)
    report = {
        "schema_version": 1,
        "status": "RECIPE-ONLY" if args.recipe_only else "PREPARATION-COMPLETE",
        "libritts_root": str(libritts_root),
        "complete_clean_pool": {
            "train_counts": train_counts,
            "train_utterances": len(train_pool),
            "dev_counts": dev_counts,
            "dev_utterances": len(dev_pool),
        },
        "exposure_schedule": {
            "seed": args.seed,
            "train_exposures": len(train_recipe),
            "validation_utterances": len(valid_recipe),
            "sampling": "seeded_without_replacement_from_complete_clean_pool",
            "recipe_path": str(recipe_path),
            "validation_recipe_path": str(valid_recipe_path),
            "validation_strata": {
                "dry": str(output_dir / "libritts_validation_dry.json"),
                "reverb": str(output_dir / "libritts_validation_reverb.json"),
            },
        },
        "augmentation": {
            "single_noise_probability": 0.75,
            "two_noise_probability": 0.25,
            "rir_probability": args.rir_probability,
            "rir_source_sampling": "source_balanced_50_50",
            "noise_source_weights": {
                "dns2021": args.dns_noise_weight,
                "wham_tr": args.wham_noise_weight,
                "demand": args.demand_noise_weight,
            },
            "snr_bins": SNR_BINS,
            "output_dbfs": [-35.0, -15.0],
            "target_dbfs": -25.0,
            "peak_ceiling": 0.99,
            "observed_train_recipe": summarize_recipe(train_recipe),
            "observed_validation_recipe": summarize_recipe(valid_recipe),
        },
        "noise_pool_counts": {
            name: len(paths) for name, paths in train_noise_pools.items()
        },
        "rir_pool_counts": {name: len(paths) for name, paths in rir_pools.items()},
        "materialized": materialized,
        "audio_root": str(audio_root),
        "dns2020": dns,
    }
    _write_json(output_dir / "libritts_dns_preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic LibriTTS mixtures and DNS2020 manifests"
    )
    data = Path(os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--libritts-root", type=Path, default=data / "LibriTTS")
    parser.add_argument(
        "--dns-noise-root",
        type=Path,
        default=data / "DNS-Challenge-2021/datasets/wideband/noise_wideband",
    )
    parser.add_argument(
        "--wham-train-root", type=Path, default=data / "WHAM/wham_noise/tr"
    )
    parser.add_argument(
        "--wham-valid-root", type=Path, default=data / "WHAM/wham_noise/cv"
    )
    parser.add_argument("--demand-root", type=Path, default=data / "DEMAND_16k")
    parser.add_argument("--rir26-root", type=Path, default=data / "RIR/OpenSLR26")
    parser.add_argument("--rir28-root", type=Path, default=data / "RIR/OpenSLR28")
    parser.add_argument(
        "--dns2020-root", type=Path, default=data / "DNS-Challenge-2020"
    )
    parser.add_argument(
        "--audio-root", type=Path, default=data / "AF_LibriTTS_DNS/audio"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/af/manifests/libritts_dns"
        ),
    )
    parser.add_argument("--train-exposures", type=int, default=80_000)
    parser.add_argument("--validation-utterances", type=int, default=512)
    parser.add_argument("--rir-probability", type=float, default=0.30)
    parser.add_argument("--dns-noise-weight", type=float, default=0.50)
    parser.add_argument("--wham-noise-weight", type=float, default=0.30)
    parser.add_argument("--demand-noise-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=260806)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--recipe-only", action="store_true")
    args = parser.parse_args()
    if args.config is not None:
        import yaml

        values = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("data preparation config must be a mapping")
        path_fields = {
            "libritts_root",
            "dns_noise_root",
            "wham_train_root",
            "wham_valid_root",
            "demand_root",
            "rir26_root",
            "rir28_root",
            "dns2020_root",
            "audio_root",
            "output_dir",
        }
        unknown = set(values) - (set(vars(args)) - {"config"})
        if unknown:
            raise ValueError(f"unknown data preparation config fields: {sorted(unknown)}")
        explicit_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        for name, value in values.items():
            flag = "--" + name.replace("_", "-")
            if flag not in explicit_flags:
                setattr(args, name, Path(value) if name in path_fields else value)
    report = run(args)
    print("\nLibriTTS/DNS preparation")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(
        "Complete train clean pool: "
        f"{report['complete_clean_pool']['train_utterances']}"
    )
    print(f"Frozen train exposures: {report['exposure_schedule']['train_exposures']}")
    print(f"RIR probability: {report['augmentation']['rir_probability']:.2%}")
    print(f"Audio root: {report['audio_root']}")


if __name__ == "__main__":
    main()

