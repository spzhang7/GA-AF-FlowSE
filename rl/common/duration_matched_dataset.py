"""Prepare a duration-matched LibriTTS/DNS post-training corpus.

The clean side is frozen before degradation.  Short utterances may only be
joined with their true neighbours from the same speaker/chapter, using a
deterministic 100--300 ms silence.  Long training utterances are cropped with
an ID/seed-derived offset; validation excludes crops so its transcript remains
exact.  The existing FlowSE-style noise/SNR/RIR mixture semantics are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .dataset_preparation import (
    DEV_SPLITS,
    SNR_BINS,
    TRAIN_SPLITS,
    _audio_files,
    _materialize,
    _read_mono,
    _rir_files,
    _with_outputs,
    _write_json,
    _write_jsonl,
    build_recipe_rows,
    prepare_dns2020_views,
    scan_libritts,
    stable_seed,
    summarize_recipe,
)


SAMPLE_RATE = 16_000
SCHEMA_VERSION = 1


def _chapter_and_sequence(utterance: str) -> tuple[str, tuple[int, ...]]:
    parts = utterance.split("_")
    if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"invalid LibriTTS utterance ID: {utterance}")
    try:
        sequence = tuple(int(part) for part in parts[2:])
    except ValueError as error:
        raise ValueError(f"invalid LibriTTS sequence ID: {utterance}") from error
    return parts[1], sequence


def _audio_header(path: str) -> tuple[int, int, int]:
    import soundfile as sf

    info = sf.info(path)
    if info.frames < 1 or info.samplerate < 1 or info.channels < 1:
        raise ValueError(f"invalid audio header: {path}")
    canonical_samples = int(math.ceil(info.frames * SAMPLE_RATE / info.samplerate))
    return int(info.frames), int(info.samplerate), canonical_samples


def attach_audio_headers(rows: Sequence[dict], *, workers: int = 16) -> list[dict]:
    """Attach exact source and canonical-16k lengths without decoding PCM."""

    if workers < 1:
        raise ValueError("header workers must be positive")
    paths = [str(row["clean_source"]) for row in rows]
    output = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        headers = pool.map(_audio_header, paths, chunksize=64)
        for index, (row, header) in enumerate(zip(rows, headers), start=1):
            frames, source_rate, canonical_samples = header
            chapter, sequence = _chapter_and_sequence(str(row["utterance"]))
            output.append(
                {
                    **row,
                    "chapter": chapter,
                    "chapter_sequence": list(sequence),
                    "source_frames": frames,
                    "source_sample_rate": source_rate,
                    "canonical_samples": canonical_samples,
                    "duration_seconds": canonical_samples / SAMPLE_RATE,
                }
            )
            if index % 5_000 == 0 or index == len(rows):
                print(f"Clean audio header scan: {index}/{len(rows)}", flush=True)
    return output


def _gap_samples(seed: int, left: str, right: str, low: int, high: int) -> int:
    if low < 0 or high < low:
        raise ValueError("invalid gap sample interval")
    rng = random.Random(stable_seed(seed, "dns10s_gap", left, right))
    return rng.randint(low, high)


def _condition_digest(source_utterances: Sequence[str], mode: str) -> str:
    payload = json.dumps(
        {"mode": mode, "sources": list(source_utterances)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _whole_condition(
    selected: Sequence[dict], gaps: Sequence[int], *, dataset_split: str
) -> dict:
    source_utterances = [str(row["utterance"]) for row in selected]
    samples = sum(int(row["canonical_samples"]) for row in selected) + sum(gaps)
    return {
        "speaker": str(selected[0]["speaker"]),
        "chapter": str(selected[0]["chapter"]),
        "split": str(selected[0]["split"]),
        "source_split": str(selected[0]["split"]),
        "dataset_split": dataset_split,
        "construction_mode": "adjacent_whole_utterances",
        "source_utterances": source_utterances,
        "source_chapter_positions": [int(row["chapter_position"]) for row in selected],
        "source_segments": [
            {
                "utterance": str(row["utterance"]),
                "path": str(row["clean_source"]),
                "start_sample_16k": 0,
                "samples_16k": int(row["canonical_samples"]),
                "transcript": str(row["transcript"]),
            }
            for row in selected
        ],
        "gap_samples_16k": [int(value) for value in gaps],
        "condition_samples_16k": int(samples),
        "condition_seconds": float(samples / SAMPLE_RATE),
        "transcript": " ".join(str(row["transcript"]) for row in selected),
        "transcript_alignment": "exact_ordered_full_utterances",
    }


def _cropped_condition(
    row: Mapping,
    *,
    dataset_split: str,
    target_samples: int,
    seed: int,
) -> dict:
    available = int(row["canonical_samples"]) - target_samples
    if available < 0:
        raise ValueError("cannot crop a source shorter than the target")
    rng = random.Random(stable_seed(seed, "dns10s_crop", row["utterance"]))
    start = rng.randint(0, available)
    return {
        "speaker": str(row["speaker"]),
        "chapter": str(row["chapter"]),
        "split": str(row["split"]),
        "source_split": str(row["split"]),
        "dataset_split": dataset_split,
        "construction_mode": "deterministic_single_utterance_crop",
        "source_utterances": [str(row["utterance"])],
        "source_chapter_positions": [int(row["chapter_position"])],
        "source_segments": [
            {
                "utterance": str(row["utterance"]),
                "path": str(row["clean_source"]),
                "start_sample_16k": int(start),
                "samples_16k": int(target_samples),
                "source_transcript": str(row["transcript"]),
            }
        ],
        "gap_samples_16k": [],
        "condition_samples_16k": int(target_samples),
        "condition_seconds": float(target_samples / SAMPLE_RATE),
        # The audio-only AF/GRPO reward uses the cropped clean waveform, not
        # text.  Empty text prevents a future caller from treating the full
        # source transcript as a time-aligned crop transcript.
        "transcript": "",
        "transcript_alignment": "unavailable_for_crop__audio_reference_only",
        "crop_seed": int(stable_seed(seed, "dns10s_crop", row["utterance"])),
        "crop_start_sample_16k": int(start),
    }


def build_duration_matched_conditions(
    rows: Sequence[dict],
    *,
    count: int,
    dataset_split: str,
    seed: int,
    minimum_seconds: float = 8.0,
    target_seconds: float = 10.0,
    maximum_seconds: float = 10.0,
    gap_min_seconds: float = 0.1,
    gap_max_seconds: float = 0.3,
    allow_long_crops: bool,
) -> list[dict]:
    """Build non-overlapping, chapter-adjacent duration-matched conditions."""

    minimum_samples = int(round(minimum_seconds * SAMPLE_RATE))
    target_samples = int(round(target_seconds * SAMPLE_RATE))
    maximum_samples = int(round(maximum_seconds * SAMPLE_RATE))
    gap_low = int(round(gap_min_seconds * SAMPLE_RATE))
    gap_high = int(round(gap_max_seconds * SAMPLE_RATE))
    if not 0 < minimum_samples <= target_samples <= maximum_samples:
        raise ValueError(
            "duration bounds must satisfy 0 < minimum <= target <= maximum"
        )
    if count < 1:
        raise ValueError("condition count must be positive")

    chapters: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        required = {
            "utterance",
            "speaker",
            "split",
            "chapter",
            "chapter_sequence",
            "canonical_samples",
            "clean_source",
            "transcript",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"duration row misses {sorted(missing)}")
        chapters[(str(row["split"]), str(row["speaker"]), str(row["chapter"]))].append(
            dict(row)
        )

    candidates = []
    used_sources: set[str] = set()
    for chapter_key in sorted(chapters):
        chapter_rows = sorted(
            chapters[chapter_key],
            key=lambda row: (tuple(row["chapter_sequence"]), str(row["utterance"])),
        )
        for chapter_position, row in enumerate(chapter_rows):
            row["chapter_position"] = chapter_position
        position = 0
        while position < len(chapter_rows):
            first = chapter_rows[position]
            first_samples = int(first["canonical_samples"])
            if first_samples > maximum_samples:
                if allow_long_crops:
                    condition = _cropped_condition(
                        first,
                        dataset_split=dataset_split,
                        target_samples=target_samples,
                        seed=seed,
                    )
                    candidates.append(condition)
                    used_sources.add(str(first["utterance"]))
                position += 1
                continue

            selected = [first]
            gaps: list[int] = []
            samples = first_samples
            best: tuple[int, list[int]] | None = (
                (0, []) if minimum_samples <= samples <= maximum_samples else None
            )
            cursor = position
            while cursor + 1 < len(chapter_rows):
                following = chapter_rows[cursor + 1]
                gap = _gap_samples(
                    seed,
                    str(chapter_rows[cursor]["utterance"]),
                    str(following["utterance"]),
                    gap_low,
                    gap_high,
                )
                proposed = samples + gap + int(following["canonical_samples"])
                if proposed > maximum_samples:
                    break
                selected.append(following)
                gaps.append(gap)
                samples = proposed
                cursor += 1
                if samples >= minimum_samples:
                    best = (cursor - position, list(gaps))
                if samples == target_samples:
                    break

            if best is None:
                position += 1
                continue
            relative_stop, selected_gaps = best
            selected_rows = chapter_rows[position : position + relative_stop + 1]
            source_ids = [str(row["utterance"]) for row in selected_rows]
            if used_sources.intersection(source_ids):
                raise AssertionError(
                    "clean source interval reused across DNS10s conditions"
                )
            candidates.append(
                _whole_condition(
                    selected_rows, selected_gaps, dataset_split=dataset_split
                )
            )
            used_sources.update(source_ids)
            position += relative_stop + 1

    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} duration-matched conditions are available; "
            f"requested {count}"
        )
    random.Random(stable_seed(seed, dataset_split, "condition_selection")).shuffle(
        candidates
    )
    selected_conditions = candidates[:count]
    output = []
    selected_sources: set[str] = set()
    for exposure_index, condition in enumerate(selected_conditions):
        sources = list(condition["source_utterances"])
        if selected_sources.intersection(sources):
            raise AssertionError("selected DNS10s conditions reuse source utterances")
        selected_sources.update(sources)
        digest = _condition_digest(sources, str(condition["construction_mode"]))
        utterance = (
            f"{condition['speaker']}_dns10s_{dataset_split}_"
            f"{exposure_index:06d}_{digest}"
        )
        output.append(
            {
                **condition,
                "utterance": utterance,
                "exposure_index": exposure_index,
                "clean_condition_schema": SCHEMA_VERSION,
                "duration_design": {
                    "minimum_seconds": float(minimum_seconds),
                    "target_seconds": float(target_seconds),
                    "maximum_seconds": float(maximum_seconds),
                    "gap_min_seconds": float(gap_min_seconds),
                    "gap_max_seconds": float(gap_max_seconds),
                    "adjacency": "same_speaker_same_chapter_contiguous_sorted_sequence",
                    "source_interval_reuse": False,
                },
            }
        )
    return output


def _valid_clean_condition(path: Path, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        import soundfile as sf

        info = sf.info(str(path))
    except Exception:
        return False
    return (
        info.samplerate == SAMPLE_RATE
        and info.channels == 1
        and info.subtype == "PCM_16"
        and info.frames == expected_frames
    )


def materialize_clean_condition(row: Mapping) -> tuple[str, str]:
    """Materialize one clean 8--10 s condition atomically."""

    import soundfile as sf

    output = Path(str(row["clean_source"]))
    expected = int(row["condition_samples_16k"])
    if _valid_clean_condition(output, expected):
        return str(output), "cached"
    pieces = []
    gaps = list(row["gap_samples_16k"])
    segments = list(row["source_segments"])
    if len(gaps) != max(0, len(segments) - 1):
        raise ValueError("clean condition gap/segment count mismatch")
    for index, segment in enumerate(segments):
        audio = _read_mono(str(segment["path"]), target_sr=SAMPLE_RATE)
        start = int(segment["start_sample_16k"])
        stop = start + int(segment["samples_16k"])
        if start < 0 or stop > audio.size:
            raise ValueError(
                f"frozen clean segment exceeds decoded audio: {segment['utterance']}"
            )
        pieces.append(audio[start:stop])
        if index < len(gaps):
            pieces.append(np.zeros(int(gaps[index]), dtype=np.float64))
    condition = np.concatenate(pieces)
    if condition.size != expected:
        raise ValueError(
            f"clean condition length drift for {row['utterance']}: "
            f"expected={expected}, got={condition.size}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.wav")
    sf.write(
        str(temporary), condition.astype(np.float32), SAMPLE_RATE, subtype="PCM_16"
    )
    os.replace(temporary, output)
    return str(output), "written"


def _materialize_clean(rows: Sequence[dict], *, jobs: int) -> dict[str, int]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    counts = {"written": 0, "cached": 0}
    if jobs == 1:
        results = map(materialize_clean_condition, rows)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(materialize_clean_condition, rows, chunksize=8)
    try:
        for index, (_, status) in enumerate(results, start=1):
            counts[status] += 1
            if index % 500 == 0 or index == len(rows):
                print(
                    f"Materialized {index}/{len(rows)} clean DNS10s conditions",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return counts


def _assign_clean_outputs(rows: Sequence[dict], root: Path) -> list[dict]:
    return [
        {
            **row,
            "clean_source": str(
                (root / str(row["dataset_split"]) / f"{row['utterance']}.wav").resolve()
            ),
        }
        for row in rows
    ]


def split_rir_pool(
    paths: Sequence[Path], *, seed: int, source: str, valid_fraction: float
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < valid_fraction < 1.0 or len(paths) < 2:
        raise ValueError("RIR split requires at least two files and 0 < fraction < 1")
    ordered = sorted(
        paths,
        key=lambda path: (stable_seed(seed, "rir_split", source, str(path)), str(path)),
    )
    valid_count = max(1, min(len(ordered) - 1, round(len(ordered) * valid_fraction)))
    valid = ordered[:valid_count]
    train = ordered[valid_count:]
    if set(train) & set(valid):
        raise AssertionError("RIR train/validation split overlaps")
    return train, valid


def _file_sha256(path: str | Path) -> tuple[str, str]:
    resolved = str(Path(path).resolve())
    digest = hashlib.sha256()
    with Path(resolved).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return resolved, digest.hexdigest()


def enforce_selected_rir_content_isolation(
    train_rows: Sequence[Mapping],
    valid_rows: Sequence[Mapping],
    *,
    seed: int,
    workers: int = 8,
    valid_candidate_pools: Mapping[str, Sequence[Path]] | None = None,
) -> tuple[list[dict], dict]:
    """Repair selected validation RIRs whose bytes duplicate a train RIR.

    OpenSLR28 bundles copies of some OpenSLR26 simulated RIRs under different
    paths. Path-wise splitting therefore cannot guarantee content isolation.
    Replacements come from content-safe validation-pool RIRs of the same
    source, preserving source balance and all clean/noise draws.
    """

    if workers < 1:
        raise ValueError("RIR fingerprint workers must be positive")
    selected_paths = {
        str(Path(str(row["rir_path"])).resolve())
        for row in [*train_rows, *valid_rows]
        if row.get("rir_path")
    }
    if valid_candidate_pools is None:
        candidate_paths = {
            source: {
                str(Path(str(row["rir_path"])).resolve())
                for row in valid_rows
                if row.get("rir_path") and str(row["rir_source"]) == source
            }
            for source in {
                str(row["rir_source"]) for row in valid_rows if row.get("rir_path")
            }
        }
    else:
        candidate_paths = {
            str(source): {str(Path(path).resolve()) for path in paths}
            for source, paths in valid_candidate_pools.items()
        }
    paths = sorted(selected_paths | set().union(*candidate_paths.values()))
    path_digests = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (path, digest) in enumerate(pool.map(_file_sha256, paths), start=1):
            path_digests[path] = digest
            if index % 1_000 == 0 or index == len(paths):
                print(f"Selected RIR SHA-256: {index}/{len(paths)}", flush=True)
    train_digests = {
        path_digests[str(Path(str(row["rir_path"])).resolve())]
        for row in train_rows
        if row.get("rir_path")
    }
    safe_by_source = {
        source: {path for path in candidates if path_digests[path] not in train_digests}
        for source, candidates in candidate_paths.items()
    }

    ordered_candidates = {
        source: sorted(
            candidates,
            key=lambda path: (
                stable_seed(seed, "rir_content_safe", source, path),
                path,
            ),
        )
        for source, candidates in safe_by_source.items()
    }
    offsets: dict[str, int] = defaultdict(int)
    used_valid_digests = {
        path_digests[str(Path(str(row["rir_path"])).resolve())]
        for row in valid_rows
        if row.get("rir_path")
        and path_digests[str(Path(str(row["rir_path"])).resolve())] not in train_digests
    }
    repaired = []
    changes = []
    collision_digests = set()
    for row in valid_rows:
        updated = dict(row)
        if row.get("rir_path"):
            old_path = str(Path(str(row["rir_path"])).resolve())
            old_digest = path_digests[old_path]
            if old_digest in train_digests:
                source = str(row["rir_source"])
                candidates = ordered_candidates.get(source, [])
                if not candidates:
                    raise ValueError(
                        f"no content-safe validation RIR replacement for {source}"
                    )
                replacement = None
                for _ in range(len(candidates)):
                    candidate = candidates[offsets[source] % len(candidates)]
                    offsets[source] += 1
                    if path_digests[candidate] not in used_valid_digests:
                        replacement = candidate
                        break
                if replacement is None:
                    replacement = candidates[offsets[source] % len(candidates)]
                    offsets[source] += 1
                used_valid_digests.add(path_digests[replacement])
                updated["rir_path"] = replacement
                updated["rir_content_isolation_repair"] = {
                    "policy": "same_source_validation_pool_rir_v1",
                    "original_path": old_path,
                    "original_sha256": old_digest,
                    "replacement_sha256": path_digests[replacement],
                }
                collision_digests.add(old_digest)
                changes.append(
                    {
                        "utterance": str(row["utterance"]),
                        "rir_source": source,
                        "original_path": old_path,
                        "original_sha256": old_digest,
                        "replacement_path": replacement,
                        "replacement_sha256": path_digests[replacement],
                    }
                )
        repaired.append(updated)

    final_valid_digests = {
        path_digests.get(str(Path(str(row["rir_path"])).resolve()))
        for row in repaired
        if row.get("rir_path")
    }
    final_valid_digests.discard(None)
    remaining = sorted(train_digests & final_valid_digests)
    if remaining:
        raise AssertionError("selected train/validation RIR content still overlaps")
    return repaired, {
        "status": "RIR-CONTENT-ISOLATION-PASS",
        "policy": "repair_validation_with_same_source_pool_safe_rir_v1",
        "collision_digests_before": len(collision_digests),
        "repaired_validation_conditions": len(changes),
        "collision_digests_after": 0,
        "unique_safe_validation_rir_digests_after": len(used_valid_digests),
        "changes": changes,
    }


def _manifest(rows: Sequence[Mapping]) -> dict[str, str]:
    return {str(row["utterance"]): str(row["transcript"]) for row in rows}


def _source_summary(rows: Sequence[Mapping]) -> dict:
    modes: dict[str, int] = defaultdict(int)
    source_ids = set()
    durations = []
    gaps = []
    for row in rows:
        modes[str(row["construction_mode"])] += 1
        source_ids.update(str(value) for value in row["source_utterances"])
        durations.append(float(row["condition_seconds"]))
        gaps.extend(int(value) / SAMPLE_RATE for value in row["gap_samples_16k"])
    values = np.asarray(durations, dtype=np.float64)
    return {
        "conditions": len(rows),
        "unique_source_utterances": len(source_ids),
        "construction_modes": dict(sorted(modes.items())),
        "duration_seconds": {
            "min": float(values.min()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        },
        "gap_seconds": {
            "count": len(gaps),
            "min": float(min(gaps)) if gaps else None,
            "max": float(max(gaps)) if gaps else None,
        },
        "empty_transcripts": sum(not str(row["transcript"]).strip() for row in rows),
    }


def run(args: argparse.Namespace) -> dict:
    libritts_root = args.libritts_root.resolve()
    output_dir = args.output_dir.resolve()
    audio_root = args.audio_root.resolve()
    clean_condition_root = args.clean_condition_root.resolve()
    train_pool, train_counts = scan_libritts(libritts_root, TRAIN_SPLITS)
    dev_pool, dev_counts = scan_libritts(libritts_root, DEV_SPLITS)
    train_headers = attach_audio_headers(train_pool, workers=args.header_workers)
    dev_headers = attach_audio_headers(dev_pool, workers=args.header_workers)
    train_clean = build_duration_matched_conditions(
        train_headers,
        count=args.train_exposures,
        dataset_split="train",
        seed=args.seed,
        minimum_seconds=args.minimum_seconds,
        target_seconds=args.target_seconds,
        maximum_seconds=args.maximum_seconds,
        gap_min_seconds=args.gap_min_seconds,
        gap_max_seconds=args.gap_max_seconds,
        allow_long_crops=True,
    )
    valid_clean = build_duration_matched_conditions(
        dev_headers,
        count=args.validation_utterances,
        dataset_split="valid",
        seed=stable_seed(args.seed, "valid"),
        minimum_seconds=args.minimum_seconds,
        target_seconds=args.target_seconds,
        maximum_seconds=args.maximum_seconds,
        gap_min_seconds=args.gap_min_seconds,
        gap_max_seconds=args.gap_max_seconds,
        allow_long_crops=False,
    )
    train_clean = _assign_clean_outputs(train_clean, clean_condition_root)
    valid_clean = _assign_clean_outputs(valid_clean, clean_condition_root)

    train_noise_pools = {
        "demand": _audio_files(args.demand_root),
        "dns2021": _audio_files(args.dns_noise_root),
        "wham_tr": _audio_files(args.wham_train_root),
    }
    valid_noise_pools = {"wham_cv": _audio_files(args.wham_valid_root)}
    if {str(path) for paths in train_noise_pools.values() for path in paths} & {
        str(path) for paths in valid_noise_pools.values() for path in paths
    }:
        raise ValueError("train/validation noise paths overlap")
    all_rir = {
        "openslr26": _rir_files(args.rir26_root),
        "openslr28": _rir_files(args.rir28_root),
    }
    train_rir_pools, valid_rir_pools = {}, {}
    for source, paths in all_rir.items():
        train_rir_pools[source], valid_rir_pools[source] = split_rir_pool(
            paths,
            seed=args.seed,
            source=source,
            valid_fraction=args.rir_validation_fraction,
        )

    clean_materialized = None
    mixtures_materialized = None
    if not args.recipe_only:
        clean_materialized = _materialize_clean(
            train_clean + valid_clean, jobs=args.jobs
        )

    train_recipe = build_recipe_rows(
        train_clean,
        split="train",
        seed=args.seed,
        noise_pools=train_noise_pools,
        noise_source_weights={
            "dns2021": args.dns_noise_weight,
            "wham_tr": args.wham_noise_weight,
            "demand": args.demand_noise_weight,
        },
        rir_pools=train_rir_pools,
        rir_probability=args.rir_probability,
    )
    midpoint = len(valid_clean) // 2
    valid_recipe = build_recipe_rows(
        valid_clean[:midpoint],
        split="valid_dry",
        seed=stable_seed(args.seed, "valid_dry"),
        noise_pools=valid_noise_pools,
        noise_source_weights={"wham_cv": 1.0},
        rir_pools=valid_rir_pools,
        rir_probability=0.0,
    ) + build_recipe_rows(
        valid_clean[midpoint:],
        split="valid_reverb",
        seed=stable_seed(args.seed, "valid_reverb"),
        noise_pools=valid_noise_pools,
        noise_source_weights={"wham_cv": 1.0},
        rir_pools=valid_rir_pools,
        rir_probability=1.0,
    )
    valid_recipe, rir_content_isolation = enforce_selected_rir_content_isolation(
        train_recipe,
        valid_recipe,
        seed=args.seed,
        workers=args.header_workers,
        valid_candidate_pools=valid_rir_pools,
    )
    train_recipe = _with_outputs(train_recipe, audio_root)
    valid_recipe = _with_outputs(valid_recipe, audio_root)

    recipe_path = output_dir / "libritts_dns10s_5000step_condition_recipe.jsonl"
    valid_recipe_path = output_dir / "libritts_dns10s_validation_recipe.jsonl"
    _write_jsonl(recipe_path, train_recipe)
    _write_jsonl(valid_recipe_path, valid_recipe)
    _write_json(
        output_dir / "libritts_dns10s_train_exposures.json", _manifest(train_recipe)
    )
    _write_json(output_dir / "libritts_dns10s_validation.json", _manifest(valid_recipe))
    _write_json(
        output_dir / "libritts_dns10s_validation_dry.json",
        _manifest(valid_recipe[:midpoint]),
    )
    _write_json(
        output_dir / "libritts_dns10s_validation_reverb.json",
        _manifest(valid_recipe[midpoint:]),
    )
    _write_json(
        output_dir / "libritts_dns10s_smoke_train.json", _manifest(train_recipe[:32])
    )
    _write_json(
        output_dir / "libritts_dns10s_smoke_validation.json",
        _manifest(valid_recipe[:16]),
    )
    if not args.recipe_only:
        mixtures_materialized = _materialize(
            train_recipe + valid_recipe, jobs=args.jobs
        )

    dns = prepare_dns2020_views(args.dns2020_root.resolve(), audio_root, output_dir)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "RECIPE-ONLY" if args.recipe_only else "PREPARATION-COMPLETE",
        "design": "dns_duration_matched_8_to_10s_v1",
        "libritts_root": str(libritts_root),
        "complete_clean_pool": {
            "train_counts": train_counts,
            "train_utterances": len(train_headers),
            "dev_counts": dev_counts,
            "dev_utterances": len(dev_headers),
        },
        "duration_matching": {
            "minimum_seconds": args.minimum_seconds,
            "target_seconds": args.target_seconds,
            "maximum_seconds": args.maximum_seconds,
            "gap_seconds": [args.gap_min_seconds, args.gap_max_seconds],
            "train_long_policy": "deterministic_id_seed_crop",
            "validation_long_policy": "exclude_to_preserve_exact_transcript",
            "train": _source_summary(train_clean),
            "validation": _source_summary(valid_clean),
        },
        "exposure_schedule": {
            "seed": args.seed,
            "train_exposures": len(train_recipe),
            "validation_utterances": len(valid_recipe),
            "recipe_path": str(recipe_path),
            "validation_recipe_path": str(valid_recipe_path),
            "clean_source_interval_reuse": False,
        },
        "augmentation": {
            "single_noise_probability": 0.75,
            "two_noise_probability": 0.25,
            "rir_probability": args.rir_probability,
            "rir_source_sampling": "source_balanced_50_50",
            "rir_file_split": {
                "validation_fraction": args.rir_validation_fraction,
                "train_counts": {
                    key: len(value) for key, value in train_rir_pools.items()
                },
                "validation_counts": {
                    key: len(value) for key, value in valid_rir_pools.items()
                },
                "path_overlap": 0,
            },
            "selected_rir_content_isolation": rir_content_isolation,
            "noise_source_weights": {
                "dns2021": args.dns_noise_weight,
                "wham_tr": args.wham_noise_weight,
                "demand": args.demand_noise_weight,
            },
            "snr_bins": SNR_BINS,
            "observed_train_recipe": summarize_recipe(train_recipe),
            "observed_validation_recipe": summarize_recipe(valid_recipe),
        },
        "materialized": {
            "clean_conditions": clean_materialized,
            "paired_mixtures": mixtures_materialized,
        },
        "clean_condition_root": str(clean_condition_root),
        "audio_root": str(audio_root),
        "dns2020": dns,
    }
    _write_json(output_dir / "libritts_dns10s_preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--audio-root", type=Path, default=data / "AF_LibriTTS_DNS10s/audio"
    )
    parser.add_argument(
        "--clean-condition-root",
        type=Path,
        default=data / "AF_LibriTTS_DNS10s/clean_conditions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/af/manifests/libritts_dns10s"
        ),
    )
    parser.add_argument("--train-exposures", type=int, default=80_000)
    parser.add_argument("--validation-utterances", type=int, default=512)
    parser.add_argument("--minimum-seconds", type=float, default=8.0)
    parser.add_argument("--target-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-seconds", type=float, default=10.0)
    parser.add_argument("--gap-min-seconds", type=float, default=0.1)
    parser.add_argument("--gap-max-seconds", type=float, default=0.3)
    parser.add_argument("--rir-probability", type=float, default=0.30)
    parser.add_argument("--rir-validation-fraction", type=float, default=0.10)
    parser.add_argument("--dns-noise-weight", type=float, default=0.50)
    parser.add_argument("--wham-noise-weight", type=float, default=0.30)
    parser.add_argument("--demand-noise-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=260810)
    parser.add_argument("--header-workers", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=8)
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
            "clean_condition_root",
            "output_dir",
        }
        unknown = set(values) - (set(vars(args)) - {"config"})
        if unknown:
            raise ValueError(f"unknown DNS10s preparation fields: {sorted(unknown)}")
        explicit_flags = {
            token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")
        }
        for name, value in values.items():
            flag = "--" + name.replace("_", "-")
            if flag not in explicit_flags:
                setattr(args, name, Path(value) if name in path_fields else value)
    report = run(args)
    print("\nLibriTTS/DNS10s preparation")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Frozen train exposures: {report['exposure_schedule']['train_exposures']}")
    duration = report["duration_matching"]["train"]["duration_seconds"]
    print(
        "Train duration: "
        f"p50={duration['p50']:.3f}s p95={duration['p95']:.3f}s "
        f"max={duration['max']:.3f}s"
    )
    print(f"Audio root: {report['audio_root']}")


if __name__ == "__main__":
    main()

