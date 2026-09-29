"""Method-neutral hashing, manifest, sampling, and paired-evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from rl.rewards.evaluators import FidelityEvaluators, normalize_wer_text, resolve_hf_model
from rl.rewards.composite import FlowSEGRPOCompositeEvaluators
from rl.rewards.metrics import DNSMOSScorer, paired_metrics
from rl.rewards.specification import (
    FLOWSE_GRPO_COMPOSITE,
    compute_training_reward,
)


REPORT_METRICS = (
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
    "dnsmos_p808",
    "speaker_similarity",
    "eres2net_speaker_similarity",
    "speechbertscore",
    "flowse_grpo_composite_reward",
    "stoi",
    "pesq_wb",
    "wer",
)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_manifest(path: str | Path) -> dict[str, str]:
    source = Path(path)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate utterance {key!r} in {source}")
            result[key] = value
        return result

    value = json.loads(
        source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest is empty or not an object: {source}")
    if not all(
        isinstance(utterance, str) and isinstance(transcript, str)
        for utterance, transcript in value.items()
    ):
        raise ValueError("manifest must map utterance IDs to transcript strings")
    return value


def voicebank_speaker(utterance: str) -> str:
    parts = utterance.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"cannot extract speaker from {utterance!r}")
    speaker = parts[0].lower()
    # VoiceBank IDs are pXXX_*, while frozen LibriTTS/DNS10s IDs begin with
    # the numeric LibriTTS reader ID.  Preserve the VoiceBank result exactly
    # and use the same LibriTTS rule as the AdvantageFlow trainer.
    if speaker.startswith("p") and speaker[1:].isdigit():
        return speaker
    if speaker.isdigit():
        return speaker
    raise ValueError(f"cannot extract speaker from {utterance!r}")


def speaker_balanced_epoch(
    utterances: Sequence[str], *, seed: int, epoch: int
) -> list[str]:
    """Round-robin a frozen per-speaker shuffle; never sort globally by ID."""

    by_speaker: dict[str, list[str]] = {}
    for utterance in utterances:
        by_speaker.setdefault(voicebank_speaker(utterance), []).append(utterance)
    speakers = sorted(by_speaker)
    random.Random(stable_seed(seed, "speaker_order", epoch)).shuffle(speakers)
    for speaker, items in by_speaker.items():
        items.sort()
        random.Random(stable_seed(seed, "utterances", epoch, speaker)).shuffle(items)
    output = []
    position = 0
    while True:
        progressed = False
        for speaker in speakers:
            if position < len(by_speaker[speaker]):
                output.append(by_speaker[speaker][position])
                progressed = True
        if not progressed:
            break
        position += 1
    if len(output) != len(utterances) or len(set(output)) != len(output):
        raise AssertionError("speaker-balanced epoch is not a permutation")
    return output


def utterances_for_step(
    utterances: Sequence[str],
    *,
    step: int,
    conditions_per_step: int,
    seed: int,
    stride_per_step: int | None = None,
) -> list[str]:
    if step < 1 or conditions_per_step < 1:
        raise ValueError("step and conditions_per_step must be positive")
    count = len(utterances)
    if count < conditions_per_step:
        raise ValueError("training manifest is smaller than one logical batch")
    stride = conditions_per_step if stride_per_step is None else int(stride_per_step)
    if stride < conditions_per_step:
        raise ValueError("stride_per_step must be at least conditions_per_step")
    cursor = (step - 1) * stride
    selected: list[str] = []
    cached_epochs: dict[int, list[str]] = {}
    while len(selected) < conditions_per_step:
        epoch = cursor // count
        position = cursor % count
        if epoch not in cached_epochs:
            cached_epochs[epoch] = speaker_balanced_epoch(
                utterances, seed=seed, epoch=epoch
            )
        candidate = cached_epochs[epoch][position]
        cursor += 1
        if candidate not in selected:
            selected.append(candidate)
    return selected


def load_fidelity(
    config: Mapping, *, lazy_asr: bool = False
) -> tuple[FidelityEvaluators | None, dict | None]:
    fidelity = config["evaluation"].get("fidelity", {"enabled": False})
    if not fidelity.get("enabled", False):
        return None, None
    locked_path = Path(fidelity["source_locked_config"])
    locked = yaml.safe_load(locked_path.read_text(encoding="utf-8"))
    evaluator_config = locked["evaluators"]
    speaker = resolve_hf_model(evaluator_config["speaker"])
    asr = resolve_hf_model(evaluator_config["asr"])
    evaluators = FidelityEvaluators.load(
        speaker,
        asr,
        device=str(fidelity.get("device", evaluator_config["device"])),
        lazy_asr=lazy_asr,
    )
    return evaluators, {
        "speaker": speaker.fingerprint(),
        "asr": asr.fingerprint(),
    }


def score_evaluation_file(
    *,
    audio_path: Path,
    clean_path: Path,
    transcript: str,
    dnsmos: DNSMOSScorer,
    fidelity: FidelityEvaluators | None,
    paired: bool,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None = None,
    reward_definition: Mapping | None = None,
) -> dict:
    metrics = dnsmos(audio_path)
    if paired:
        pair = paired_metrics(clean_path, audio_path)
        metrics.update({"pesq_wb": pair["pesq_wb"], "stoi": pair["stoi"]})
    if fidelity is not None:
        metrics["speaker_similarity"] = fidelity.speaker(clean_path, audio_path)
        # Some official DNS test clips do not carry a usable English
        # transcript (for example, a placeholder or punctuation-only field).
        # WER is undefined for those rows.  Keep all other metrics and omit
        # only WER for that row; paired summaries will use the rows with a
        # valid reference instead of aborting the complete endpoint test.
        if normalize_wer_text(transcript):
            wer, hypothesis = fidelity.asr(transcript, audio_path)
            metrics["wer"] = wer
            metrics["asr_hypothesis"] = hypothesis
        else:
            metrics["wer_excluded_reason"] = "empty_reference_after_normalization"
    if composite_evaluators is not None:
        metrics.update(composite_evaluators.score(clean_path, audio_path))
    if (
        reward_definition is not None
        and reward_definition["name"] == FLOWSE_GRPO_COMPOSITE
    ):
        if composite_evaluators is None:
            raise ValueError(
                "composite held-out reward requires the frozen composite evaluators"
            )
        reconstructed = compute_training_reward(metrics, reward_definition)
        metrics["flowse_grpo_composite_reward"] = float(reconstructed["reward"])
    return metrics


def _bootstrap_mean(
    values: Sequence[float], *, seed: int, samples: int, confidence: float
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    estimates = np.mean(array[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def speaker_cluster_bootstrap_mean(
    utterances: Sequence[str],
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> dict[str, float]:
    if len(utterances) != len(values) or not utterances:
        raise ValueError("utterances and values must be non-empty and aligned")
    grouped: dict[str, list[float]] = {}
    for utterance, value in zip(utterances, values, strict=True):
        grouped.setdefault(voicebank_speaker(utterance), []).append(float(value))
    speakers = sorted(grouped)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = generator.integers(0, len(speakers), size=len(speakers))
        draw = [value for item in selected for value in grouped[speakers[item]]]
        estimates[index] = np.mean(draw)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "speakers": len(speakers),
    }


def paired_state_analysis(
    base_rows: Sequence[Mapping],
    state_rows: Sequence[Mapping],
    *,
    seed: int,
    samples: int,
    confidence: float,
    allow_partial_metrics: bool = False,
    include_speaker_ci: bool = True,
) -> dict:
    """Paired state-minus-base metrics with utterance and speaker intervals.

    Official DNS2020 contains 300 real recordings without clean references.
    When ``allow_partial_metrics`` is enabled, metrics that require a clean
    reference are computed on their paired subset while OVRL concentration
    remains computed over the full manifest.  The strict default preserves
    the original behavior for training/validation audits.
    """

    base = {str(row["utterance"]): row for row in base_rows}
    state = {str(row["utterance"]): row for row in state_rows}
    if set(base) != set(state):
        raise ValueError("base/state utterance sets differ")
    utterances = sorted(base)
    for utterance in utterances:
        if int(base[utterance]["latent_seed"]) != int(state[utterance]["latent_seed"]):
            raise ValueError(f"latent seed differs for {utterance}")
    metrics = {}
    for metric in REPORT_METRICS:
        metric_utterances = [
            key
            for key in utterances
            if metric in base[key] and metric in state[key]
        ]
        if not metric_utterances:
            continue
        if not allow_partial_metrics and len(metric_utterances) != len(utterances):
            continue
        deltas = np.asarray(
            [
                float(state[key][metric]) - float(base[key][metric])
                for key in metric_utterances
            ],
            dtype=np.float64,
        )
        metric_seed = stable_seed(seed, "checkpoint_eval", metric)
        metrics[metric] = {
            "base_mean": float(
                np.mean([base[key][metric] for key in metric_utterances])
            ),
            "state_mean": float(
                np.mean([state[key][metric] for key in metric_utterances])
            ),
            "delta_mean": float(deltas.mean()),
            "count": len(metric_utterances),
            "utterance_ci": _bootstrap_mean(
                deltas, seed=metric_seed, samples=samples, confidence=confidence
            ),
            "speaker_ci": (
                speaker_cluster_bootstrap_mean(
                    metric_utterances,
                    deltas,
                    seed=stable_seed(metric_seed, "speaker"),
                    samples=samples,
                    confidence=confidence,
                )
                if include_speaker_ci
                else None
            ),
            "positive_utterance_fraction": float(np.mean(deltas > 0.0)),
        }
    ovrl_deltas = np.asarray(
        [
            float(state[key]["dnsmos_ovrl"]) - float(base[key]["dnsmos_ovrl"])
            for key in utterances
        ],
        dtype=np.float64,
    )
    descending = np.sort(ovrl_deltas)[::-1]
    positive_total = float(np.clip(ovrl_deltas, 0.0, None).sum())
    concentration = {
        "median_delta": float(np.median(ovrl_deltas)),
        "top1_share_of_positive_gain": (
            float(max(descending[0], 0.0) / positive_total)
            if positive_total > 0
            else 0.0
        ),
        "top3_share_of_positive_gain": (
            float(np.clip(descending[:3], 0.0, None).sum() / positive_total)
            if positive_total > 0
            else 0.0
        ),
        "mean_after_removing_top4": (
            float(descending[4:].mean()) if descending.size > 4 else float("nan")
        ),
    }
    if include_speaker_ci:
        speaker_means: dict[str, list[float]] = {}
        for utterance, value in zip(utterances, ovrl_deltas, strict=True):
            speaker_means.setdefault(voicebank_speaker(utterance), []).append(
                float(value)
            )
        speaker_values = {
            speaker: float(np.mean(values))
            for speaker, values in speaker_means.items()
        }
        concentration.update(
            {
                "speaker_statistics_available": True,
                "positive_speaker_fraction": float(
                    np.mean(np.asarray(list(speaker_values.values())) > 0.0)
                ),
                "positive_speakers": int(
                    sum(value > 0.0 for value in speaker_values.values())
                ),
                "speakers": len(speaker_values),
                "speaker_mean_deltas": speaker_values,
            }
        )
    else:
        concentration.update(
            {
                "speaker_statistics_available": False,
                "speaker_statistics_reason": "speaker_ids_unavailable",
            }
        )
    return {"metrics": metrics, "ovrl_concentration": concentration}
