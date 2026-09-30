"""Independent online/EMA evaluation for saved speech AdvantageFlow checkpoints.

This program is intentionally evaluation-only.  It never constructs an
optimizer, never performs backward, and never writes a model checkpoint.  The
registered primary comparison remains EMA step 20 versus the released-model
step-0 baseline; online policies and intermediate steps are diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import yaml

from .advantage_estimation import cluster_bootstrap_mean
from rl.common.conditioning import ConditioningProtocol
from rl.common.flowse_interface import load_flowse_bundle
from .reward_evaluators import (
    FlowSEGRPOCompositeEvaluators,
    load_flowse_grpo_composite_evaluators,
)
from rl.common.lora import AdapterState, inject_lora, load_lora, snapshot_lora
from rl.rewards.metrics import DNSMOSScorer
from .advantage_flow import stable_seed
from .protocol import (
    sha256_file,
    sha256_json,
    strict_manifest,
    verify_execution_dependencies,
)
from .trainer import (
    COMPOSITE_REPORT_KEYS,
    DNSMOS_KEYS,
    _load_fidelity,
    _metric_summary,
    _score_evaluation_file,
)
from rl.rewards.specification import (
    FLOWSE_GRPO_COMPOSITE,
    resolve_training_reward,
    verify_reward_calibration,
)


REPORT_METRICS = list(DNSMOS_KEYS) + [
    "speaker_similarity",
    "stoi",
    "pesq_wb",
    "wer",
    *COMPOSITE_REPORT_KEYS,
]


def _load_reward_evaluators(config: Mapping) -> tuple[dict, object | None, dict | None, dict | None]:
    """Load one shared composite evaluator instance and bind it to calibration."""

    definition = resolve_training_reward(config)
    if definition["name"] != FLOWSE_GRPO_COMPOSITE:
        return definition, None, None, verify_reward_calibration(config)
    evaluators, fingerprint = load_flowse_grpo_composite_evaluators(config)
    verification = verify_reward_calibration(
        config, evaluator_fingerprint=fingerprint
    )
    return definition, evaluators, fingerprint, verification


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _speaker(utterance: str) -> str:
    return utterance.split("_", 1)[0].lower()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_path(run_dir: Path, step: int) -> Path:
    path = run_dir / f"checkpoint_step_{step:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_checkpoint(run_dir: Path, step: int) -> tuple[dict, Path, str]:
    path = _checkpoint_path(run_dir, step)
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    protocol_path = run_dir / "protocol.json"
    expected_protocol = (
        sha256_json(json.loads(protocol_path.read_text(encoding="utf-8")))
        if protocol_path.is_file()
        else None
    )
    criteria = {
        "schema": payload.get("schema_version") == 1,
        "step": int(payload.get("completed_step", -1)) == step,
        "protocol": expected_protocol is None
        or str(payload.get("protocol_hash", "")) == expected_protocol,
        "current": bool(payload.get("current_lora")),
        "rollout": bool(payload.get("rollout_lora")),
    }
    if not all(criteria.values()):
        raise ValueError(f"invalid checkpoint {path}: {criteria}")
    return payload, path, digest


def speaker_cluster_bootstrap_mean(
    utterances: Sequence[str],
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> dict[str, float]:
    """Bootstrap speakers, retaining all utterances of each sampled speaker."""

    if len(utterances) != len(values) or not utterances:
        raise ValueError("utterances and values must be non-empty and aligned")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("speaker bootstrap values must be finite")
    grouped: dict[str, list[float]] = {}
    for utterance, value in zip(utterances, values, strict=True):
        grouped.setdefault(_speaker(utterance), []).append(float(value))
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
    include_speaker_ci: bool = True,
) -> dict:
    """Paired state-minus-base metrics with utterance and speaker intervals."""

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
        if not all(metric in base[key] and metric in state[key] for key in utterances):
            continue
        deltas = np.asarray(
            [float(state[key][metric]) - float(base[key][metric]) for key in utterances],
            dtype=np.float64,
        )
        metric_seed = stable_seed(seed, "checkpoint_eval", metric)
        utterance_ci = cluster_bootstrap_mean(
            deltas,
            seed=metric_seed,
            samples=samples,
            confidence=confidence,
        )
        metrics[metric] = {
            "base_mean": float(np.mean([base[key][metric] for key in utterances])),
            "state_mean": float(np.mean([state[key][metric] for key in utterances])),
            "delta_mean": float(deltas.mean()),
            "utterance_ci": utterance_ci,
            "positive_utterance_fraction": float(np.mean(deltas > 0.0)),
        }
        if include_speaker_ci:
            metrics[metric]["speaker_ci"] = speaker_cluster_bootstrap_mean(
                utterances,
                deltas,
                seed=stable_seed(metric_seed, "speaker"),
                samples=samples,
                confidence=confidence,
            )

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
            speaker_means.setdefault(_speaker(utterance), []).append(float(value))
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


def paired_policy_analysis(
    ema_rows: Sequence[Mapping],
    online_rows: Sequence[Mapping],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> dict:
    """Online-minus-EMA paired comparison at one optimizer step."""

    return paired_state_analysis(
        ema_rows,
        online_rows,
        seed=seed,
        samples=samples,
        confidence=confidence,
    )


def registered_primary_result(states: Mapping, config: Mapping) -> dict:
    """Apply the complete pilot gate to the frozen final optimizer step."""

    primary_step = int(
        config.get("evaluation", {}).get(
            "primary_checkpoint_step",
            config.get("run", {}).get("optimizer_steps", 20),
        )
    )
    state_id = f"ema_step_{primary_step:06d}"
    primary_metric = states[state_id]["analysis"]["metrics"][
        "dnsmos_ovrl"
    ]
    decision = config["pilot_decision"]
    minimum = float(decision["dnsmos_ovrl_minimum_gain"])
    require_positive_ci = bool(decision.get("require_ci_positive", True))
    ci_low = float(primary_metric["utterance_ci"]["ci_low"])
    speaker_ci_low = float(primary_metric["speaker_ci"]["ci_low"])
    criteria = {
        "dnsmos_ovrl_effect_size": float(primary_metric["delta_mean"]) >= minimum,
        "dnsmos_ovrl_utterance_ci_positive": (
            not require_positive_ci or ci_low > 0.0
        ),
        "dnsmos_ovrl_speaker_ci_positive": (
            not require_positive_ci or speaker_ci_low > 0.0
        ),
    }
    metrics = states[state_id]["analysis"]["metrics"]
    for metric, tolerance in decision["safety"].items():
        if metric not in metrics:
            criteria[f"{metric}_safety"] = False
            continue
        utterance_interval = metrics[metric]["utterance_ci"]
        speaker_interval = metrics[metric]["speaker_ci"]
        if metric == "wer":
            passed = (
                float(utterance_interval["ci_high"]) <= float(tolerance)
                and float(speaker_interval["ci_high"]) <= float(tolerance)
            )
        else:
            passed = (
                float(utterance_interval["ci_low"]) >= float(tolerance)
                and float(speaker_interval["ci_low"]) >= float(tolerance)
            )
        criteria[f"{metric}_safety"] = passed
    primary_supported = all(criteria.values())
    return {
        "comparison": f"ema_step_{primary_step:06d}_minus_base",
        "primary_checkpoint_step": primary_step,
        "status": "PILOT-SUPPORT" if primary_supported else "PILOT-NO-SUPPORT",
        "minimum_effect": minimum,
        "require_ci_positive": require_positive_ci,
        "delta_mean": float(primary_metric["delta_mean"]),
        "ci_low": ci_low,
        "ci_high": float(primary_metric["utterance_ci"]["ci_high"]),
        "speaker_ci_low": speaker_ci_low,
        "speaker_ci_high": float(primary_metric["speaker_ci"]["ci_high"]),
        "criteria": criteria,
        "thresholds": decision,
        "note": "The final optimizer step is fixed by the frozen training config.",
    }


def _evaluate_state(
    *,
    bundle,
    state: Mapping[str, torch.Tensor],
    state_id: str,
    policy: str,
    step: int,
    manifest: Mapping[str, str],
    config: Mapping,
    conditioning: ConditioningProtocol,
    dnsmos: DNSMOSScorer,
    fidelity,
    output_dir: Path,
    state_source: Mapping,
    composite_evaluators: FlowSEGRPOCompositeEvaluators | None = None,
    reward_definition: Mapping | None = None,
    paired_metrics: bool | None = None,
    reference_free: bool = False,
) -> dict:
    if reward_definition is None:
        reward_definition = resolve_training_reward(config)
    report_path = output_dir / "states" / f"{state_id}.json"
    if report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("state_source") == state_source:
            print(f"reuse {state_id}: {report_path}")
            return cached

    load_lora(bundle.model.transformer, state)
    normalization = config["normalization"]
    paired = (
        bool(config["evaluation"]["paired_metrics"])
        if paired_metrics is None
        else bool(paired_metrics)
    )
    rows = []
    audio_dir = output_dir / "audio" / state_id
    for index, (utterance, transcript) in enumerate(manifest.items(), 1):
        noisy = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
        clean = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
        seed = stable_seed(
            int(config["evaluation"]["latent_seed_base"]), "evaluation", utterance
        )
        endpoint = bundle.generate_group(
            noisy,
            "",
            [seed],
            nfe=int(config["rollout"]["evaluation_nfe"]),
            cfg_strength=0.0,
            conditioning=conditioning,
            target_dbfs=float(normalization["target_dbfs"]),
            peak_ceiling=float(normalization["peak_ceiling"]),
        )[0]
        audio_path = audio_dir / f"{utterance}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            audio_path,
            endpoint.normalized_waveform,
            bundle.output_sample_rate,
            subtype=str(normalization["output_subtype"]),
        )
        scored_wav_sha256 = sha256_file(audio_path)
        metrics = _score_evaluation_file(
            audio_path=audio_path,
            clean_path=clean,
            transcript=transcript,
            dnsmos=dnsmos,
            fidelity=fidelity,
            paired=paired,
            reference_free=bool(reference_free),
            composite_evaluators=composite_evaluators,
            reward_definition=reward_definition,
        )
        if sha256_file(audio_path) != scored_wav_sha256:
            raise RuntimeError(f"evaluation WAV changed while scoring: {audio_path}")
        rows.append(
            {
                "state_id": state_id,
                "policy": policy,
                "step": step,
                "utterance": utterance,
                "latent_seed": int(seed),
                "terminal_mel_sha256": endpoint.terminal_mel_sha256,
                "prewrite_waveform_sha256": endpoint.normalized_waveform_sha256,
                "scored_wav_sha256": scored_wav_sha256,
                "audio_path": str(audio_path),
                **metrics,
            }
        )
        if index % 8 == 0 or index == len(manifest):
            print(f"{state_id}: {index}/{len(manifest)}")
    report = {
        "state_id": state_id,
        "policy": policy,
        "step": step,
        "state_source": state_source,
        "state_sha256": _state_sha256(state),
        "summary": _metric_summary(rows, REPORT_METRICS),
        "rows": rows,
    }
    _write_json(report_path, report)
    return report


def _print_table(report: Mapping) -> None:
    def summary_mean(summary: Mapping, key: str) -> float:
        return float(summary.get(key, {}).get("mean", float("nan")))

    def delta(metrics: Mapping, key: str) -> float:
        return float(metrics.get(key, {}).get("delta_mean", float("nan")))

    print("\nCheckpoint policy comparison: fixed utterances and latents")
    print("\nAbsolute metric means")
    print("=" * 105)
    print(
        f"{'State':<20}{'SIG':>9}{'BAK':>9}{'OVRL':>9}{'P808':>9}"
        f"{'STOI':>9}{'PESQ-WB':>10}{'WER':>9}{'SPK':>9}"
    )
    print("-" * 105)
    summaries = [("base", report["base"]["summary"])] + [
        (state_id, report["states"][state_id]["summary"])
        for state_id in report["state_order"]
    ]
    for state_id, summary in summaries:
        print(
            f"{state_id:<20}{summary_mean(summary, 'dnsmos_sig'):>9.4f}"
            f"{summary_mean(summary, 'dnsmos_bak'):>9.4f}"
            f"{summary_mean(summary, 'dnsmos_ovrl'):>9.4f}"
            f"{summary_mean(summary, 'dnsmos_p808'):>9.4f}"
            f"{summary_mean(summary, 'stoi'):>9.4f}"
            f"{summary_mean(summary, 'pesq_wb'):>10.4f}"
            f"{summary_mean(summary, 'wer'):>9.4f}"
            f"{summary_mean(summary, 'speaker_similarity'):>9.4f}"
        )

    print("\nPaired state-minus-base deltas")
    print("=" * 105)
    print(
        f"{'State':<20}{'dSIG':>9}{'dBAK':>9}{'dOVRL':>9}{'dP808':>9}"
        f"{'dSTOI':>9}{'dPESQ':>10}{'dWER':>9}{'dSPK':>9}"
    )
    print("-" * 105)
    for state_id in report["state_order"]:
        metrics = report["states"][state_id]["analysis"]["metrics"]
        print(
            f"{state_id:<20}{delta(metrics, 'dnsmos_sig'):>+9.4f}"
            f"{delta(metrics, 'dnsmos_bak'):>+9.4f}"
            f"{delta(metrics, 'dnsmos_ovrl'):>+9.4f}"
            f"{delta(metrics, 'dnsmos_p808'):>+9.4f}"
            f"{delta(metrics, 'stoi'):>+9.4f}"
            f"{delta(metrics, 'pesq_wb'):>+10.4f}"
            f"{delta(metrics, 'wer'):>+9.4f}"
            f"{delta(metrics, 'speaker_similarity'):>+9.4f}"
        )

    print("\nPaired DNSMOS OVRL confidence intervals")
    print("=" * 72)
    composite_metrics = [
        metric
        for metric in COMPOSITE_REPORT_KEYS
        if any(
            metric in report["states"][state_id]["analysis"]["metrics"]
            for state_id in report["state_order"]
        )
    ]
    if composite_metrics:
        labels = {
            "flowse_grpo_composite_reward": "Composite reward",
            "eres2net_speaker_similarity": "ERes2Net speaker",
            "speechbertscore": "SpeechBERTScore",
        }
        print("\nOptimized composite reward diagnostics")
        print("=" * 124)
        print(
            f"{'State':<20}{'Metric':<22}{'Base':>11}{'State':>11}{'Delta':>12}"
            f"{'utterance 95% CI':>22}{'speaker 95% CI':>22}"
        )
        print("-" * 124)
        for state_id in report["state_order"]:
            state_metrics = report["states"][state_id]["analysis"]["metrics"]
            for metric in composite_metrics:
                if metric not in state_metrics:
                    continue
                row = state_metrics[metric]
                utt = row["utterance_ci"]
                spk = row["speaker_ci"]
                interval_u = f"[{utt['ci_low']:+.4f},{utt['ci_high']:+.4f}]"
                interval_s = f"[{spk['ci_low']:+.4f},{spk['ci_high']:+.4f}]"
                print(
                    f"{state_id:<20}{labels[metric]:<22}{row['base_mean']:>11.5f}"
                    f"{row['state_mean']:>11.5f}{row['delta_mean']:>+12.5f}"
                    f"{interval_u:>22}{interval_s:>22}"
                )
        print("=" * 124)
    print(f"{'State':<20}{'mean delta':>12}{'utterance 95% CI':>20}{'speaker 95% CI':>20}")
    print("-" * 72)
    for state_id in report["state_order"]:
        ovrl = report["states"][state_id]["analysis"]["metrics"]["dnsmos_ovrl"]
        utt = ovrl["utterance_ci"]
        spk = ovrl["speaker_ci"]
        interval_u = f"[{utt['ci_low']:+.4f},{utt['ci_high']:+.4f}]"
        interval_s = f"[{spk['ci_low']:+.4f},{spk['ci_high']:+.4f}]"
        print(
            f"{state_id:<20}{ovrl['delta_mean']:>+12.4f}"
            f"{interval_u:>20}{interval_s:>20}"
        )
    print("=" * 72)
    primary = report["registered_primary"]
    primary_step = int(primary["primary_checkpoint_step"])
    print(
        f"Primary endpoint: EMA step {primary_step} -> {primary['status']} "
        f"(OVRL {primary['delta_mean']:+.5f}, "
        f"CI [{primary['ci_low']:+.5f}, {primary['ci_high']:+.5f}])"
    )
    print("Other requested checkpoints are descriptive; no best-checkpoint selection.")


def run(args: argparse.Namespace) -> tuple[dict, Path]:
    if args.bootstrap_samples < 1 or not 0 < args.confidence < 1:
        raise ValueError("invalid bootstrap settings")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    if conditioning.fingerprint() != {
        "mode": "wotext",
        "use_text": False,
        "drop_text": True,
    }:
        raise ValueError("checkpoint evaluation is permanently audio-only")
    run_dir = args.run_dir.resolve()
    primary_step = int(
        config.get("evaluation", {}).get(
            "primary_checkpoint_step", config["run"]["optimizer_steps"]
        )
    )
    steps = tuple(
        int(step) for step in (args.steps if args.steps is not None else [primary_step])
    )
    if not steps or any(step <= 0 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("steps must be unique positive integers")
    if primary_step not in steps:
        raise ValueError(f"primary checkpoint step {primary_step} must be included")

    training_protocol_path = run_dir / "protocol.json"
    training_protocol = json.loads(training_protocol_path.read_text(encoding="utf-8"))
    if training_protocol.get("config") != config:
        raise ValueError("evaluation config differs from the frozen training config")
    execution_dependencies = verify_execution_dependencies(training_protocol)
    if not execution_dependencies["passed"]:
        raise ValueError(
            "training execution dependencies no longer match the frozen protocol: "
            f"{execution_dependencies['criteria']}"
        )

    checkpoint_payloads = {}
    checkpoint_paths = {}
    checkpoint_hashes = {}
    for step in steps:
        payload, path, digest = _load_checkpoint(run_dir, step)
        checkpoint_payloads[step] = payload
        checkpoint_paths[step] = path
        checkpoint_hashes[step] = digest

    bundle = load_flowse_bundle(config["flowse_config"], deterministic=True)
    torch.manual_seed(int(config["lora"]["initialization_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["lora"]["initialization_seed"]))
    injection = inject_lora(
        bundle.model.transformer,
        target_patterns=config["lora"]["target_patterns"],
        rank=int(config["lora"]["rank"]),
        alpha=float(config["lora"]["alpha"]),
        dropout=float(config["lora"]["dropout"]),
        expected_modules=int(config["lora"]["expected_modules"]),
    )
    base_state = snapshot_lora(bundle.model.transformer, device="cpu")
    fidelity, fidelity_fingerprint = _load_fidelity(config)
    (
        reward_definition,
        composite_evaluators,
        composite_fingerprint,
        reward_calibration,
    ) = _load_reward_evaluators(config)
    evaluator_fingerprint = {
        "fidelity": fidelity_fingerprint,
        "training_reward": composite_fingerprint,
    }
    manifest_path = Path(config["data"]["evaluation_manifest"])
    manifest = strict_manifest(manifest_path)
    if len(manifest) != 64:
        raise ValueError(
            f"registered pilot evaluation requires 64 utterances, got {len(manifest)}"
        )

    source_path = Path(__file__).resolve()
    protocol = {
        "schema_version": 1,
        "method": "speech_advantageflow_checkpoint_online_ema_evaluation",
        "training_protocol_hash": run_dir.name,
        "config_sha256": sha256_file(args.config),
        "training_protocol_sha256": sha256_file(training_protocol_path),
        "training_execution_dependencies": execution_dependencies,
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "evaluation_utterances": len(manifest),
        "steps": list(steps),
        "policies": ["online", "ema"],
        "evaluation_nfe": int(config["rollout"]["evaluation_nfe"]),
        "latent_seed_base": int(config["evaluation"]["latent_seed_base"]),
        "checkpoint_sha256": {str(key): value for key, value in checkpoint_hashes.items()},
        "released_checkpoint_sha256": bundle.checkpoint_sha256,
        "vocoder_sha256": bundle.vocoder_sha256,
        "evaluator_fingerprint": evaluator_fingerprint,
        "reward_calibration": reward_calibration,
        "source_sha256": sha256_file(source_path),
        "registered_primary": f"ema_step_{primary_step:06d}_minus_base",
    }
    evaluation_hash = sha256_json(protocol)
    output_dir = run_dir / "checkpoint_comparison" / evaluation_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "protocol.json", protocol)

    dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
    base_report = _evaluate_state(
        bundle=bundle,
        state=base_state,
        state_id="base",
        policy="released_model",
        step=0,
        manifest=manifest,
        config=config,
        conditioning=conditioning,
        dnsmos=dnsmos,
        fidelity=fidelity,
        output_dir=output_dir,
        state_source={"released_checkpoint_sha256": bundle.checkpoint_sha256},
        composite_evaluators=composite_evaluators,
        reward_definition=reward_definition,
    )

    states = {}
    state_order = []
    raw_reports = {}
    for step in steps:
        payload = checkpoint_payloads[step]
        for policy, payload_key in (("online", "current_lora"), ("ema", "rollout_lora")):
            state_id = f"{policy}_step_{step:06d}"
            state: AdapterState = {
                str(name): value.detach().cpu().clone()
                for name, value in payload[payload_key].items()
            }
            raw = _evaluate_state(
                bundle=bundle,
                state=state,
                state_id=state_id,
                policy=policy,
                step=step,
                manifest=manifest,
                config=config,
                conditioning=conditioning,
                dnsmos=dnsmos,
                fidelity=fidelity,
                output_dir=output_dir,
                state_source={
                    "checkpoint": str(checkpoint_paths[step]),
                    "checkpoint_sha256": checkpoint_hashes[step],
                    "payload_key": payload_key,
                },
                composite_evaluators=composite_evaluators,
                reward_definition=reward_definition,
            )
            analysis = paired_state_analysis(
                base_report["rows"],
                raw["rows"],
                seed=stable_seed(args.bootstrap_seed, state_id),
                samples=args.bootstrap_samples,
                confidence=args.confidence,
            )
            states[state_id] = {
                "policy": policy,
                "step": step,
                "summary": raw["summary"],
                "analysis": analysis,
            }
            raw_reports[state_id] = raw
            state_order.append(state_id)

    online_vs_ema = {}
    for step in steps:
        ema_id = f"ema_step_{step:06d}"
        online_id = f"online_step_{step:06d}"
        online_vs_ema[str(step)] = paired_policy_analysis(
            raw_reports[ema_id]["rows"],
            raw_reports[online_id]["rows"],
            seed=stable_seed(args.bootstrap_seed, "online_vs_ema", step),
            samples=args.bootstrap_samples,
            confidence=args.confidence,
        )

    registered_primary = registered_primary_result(states, config)
    checkpoint_hashes_after = {
        step: sha256_file(path) for step, path in checkpoint_paths.items()
    }
    released_checkpoint_after = sha256_file(bundle.checkpoint_path)
    immutability = {
        "passed": (
            checkpoint_hashes_after == checkpoint_hashes
            and released_checkpoint_after == bundle.checkpoint_sha256
        ),
        "training_checkpoints": {
            "before": checkpoint_hashes,
            "after": checkpoint_hashes_after,
        },
        "released_checkpoint": {
            "path": str(bundle.checkpoint_path),
            "before": bundle.checkpoint_sha256,
            "after": released_checkpoint_after,
        },
    }
    report = {
        "status": "EVALUATION-COMPLETE" if immutability["passed"] else "INCOMPLETE",
        "authorization": "diagnostic_only_never_selects_best_checkpoint_post_hoc",
        "evaluation_protocol_hash": evaluation_hash,
        "training_protocol_hash": run_dir.name,
        "lora_modules": len(injection.module_names),
        "state_order": state_order,
        "base": {"summary": base_report["summary"]},
        "states": states,
        "online_minus_ema": online_vs_ema,
        "registered_primary": registered_primary,
        "checkpoint_immutability": immutability,
        "evaluator_fingerprint": evaluator_fingerprint,
        "reward_calibration": reward_calibration,
    }
    report_path = output_dir / "checkpoint_comparison_report.json"
    _write_json(report_path, report)
    _write_json(
        run_dir / "latest_checkpoint_comparison.json",
        {
            "evaluation_protocol_hash": evaluation_hash,
            "report": str(report_path),
        },
    )
    _print_table(report)
    print(f"Report: {report_path}")
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate online and EMA policies from saved checkpoints"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=92117)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    report, _ = run(args)
    if report["status"] != "EVALUATION-COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
