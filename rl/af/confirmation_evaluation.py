"""Evaluate a saved EMA checkpoint on 256 speaker-disjoint utterances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import yaml

from rl.common.conditioning import ConditioningProtocol
from rl.common.flowse_interface import load_flowse_bundle
from rl.common.lora import AdapterState, inject_lora, snapshot_lora
from rl.rewards.metrics import DNSMOSScorer
from .evaluation import (
    REPORT_METRICS,
    _evaluate_state,
    _load_checkpoint,
    _load_reward_evaluators,
    paired_state_analysis,
)
from .advantage_flow import stable_seed
from .protocol import (
    sha256_file,
    sha256_json,
    strict_manifest,
    verify_execution_dependencies,
)
from .trainer import _load_fidelity


PROTOCOL_SOURCE = "rl/af/protocol.py"


def _protocol_source_override(
    execution_dependencies: Mapping, *, explicitly_allowed: bool
) -> dict:
    recorded = execution_dependencies.get("recorded_source_sha256", {})
    current = execution_dependencies.get("current_source_sha256", {})
    mismatches = sorted(
        name
        for name in set(recorded) | set(current)
        if recorded.get(name) != current.get(name)
    )
    criteria = execution_dependencies.get("criteria", {})
    non_source_criteria_pass = all(
        bool(value)
        for name, value in criteria.items()
        if name != "source_hashes_match"
    )
    authorized = (
        explicitly_allowed
        and mismatches == [PROTOCOL_SOURCE]
        and non_source_criteria_pass
    )
    return {
        "requested": explicitly_allowed,
        "authorized": authorized,
        "allowed_mismatch": PROTOCOL_SOURCE,
        "observed_mismatches": mismatches,
        "non_source_criteria_pass": non_source_criteria_pass,
        "scope": (
            "configuration/fingerprinting source only; no FlowSE, LoRA, vocoder, "
            "DNSMOS, fidelity-evaluator, or model source mismatch is allowed"
        ),
    }


LABELS = {
    "dnsmos_sig": "DNSMOS SIG",
    "dnsmos_bak": "DNSMOS BAK",
    "dnsmos_ovrl": "DNSMOS OVRL",
    "dnsmos_p808": "DNSMOS P808",
    "speaker_similarity": "Speaker similarity",
    "eres2net_speaker_similarity": "ERes2Net speaker",
    "speechbertscore": "SpeechBERTScore",
    "flowse_grpo_composite_reward": "Composite reward",
    "stoi": "STOI",
    "pesq_wb": "PESQ-WB",
    "wer": "WER",
}


def _speaker(utterance: str) -> str:
    return utterance.split("_", 1)[0].lower()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def confirmation_decision(analysis: Mapping, config: Mapping) -> dict:
    """Separate scientific support from the original pilot's fixed threshold."""

    metrics = analysis["metrics"]
    thresholds = config["pilot_decision"]
    ovrl = metrics["dnsmos_ovrl"]
    minimum = float(thresholds["dnsmos_ovrl_minimum_gain"])
    criteria = {
        "dnsmos_ovrl_mean_positive": float(ovrl["delta_mean"]) > 0.0,
        "dnsmos_ovrl_utterance_ci_positive": (
            float(ovrl["utterance_ci"]["ci_low"]) > 0.0
        ),
        "dnsmos_ovrl_speaker_ci_positive": (
            float(ovrl["speaker_ci"]["ci_low"]) > 0.0
        ),
        "dnsmos_ovrl_original_minimum_effect": (
            float(ovrl["delta_mean"]) >= minimum
        ),
    }
    for metric, tolerance in thresholds["safety"].items():
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

    safety_pass = all(
        value for name, value in criteria.items() if name.endswith("_safety")
    )
    positive_support = bool(
        criteria["dnsmos_ovrl_mean_positive"]
        and criteria["dnsmos_ovrl_utterance_ci_positive"]
        and criteria["dnsmos_ovrl_speaker_ci_positive"]
        and safety_pass
    )
    strong_support = bool(
        positive_support and criteria["dnsmos_ovrl_original_minimum_effect"]
    )
    if strong_support:
        decision = "CONFIRMATION-STRONG-SUPPORT"
    elif positive_support:
        decision = "CONFIRMATION-SMALL-EFFECT-SUPPORT"
    else:
        decision = "CONFIRMATION-NO-SUPPORT"
    return {
        "decision": decision,
        "criteria": criteria,
        "original_minimum_effect": minimum,
        "medium_scale_training_authorized": positive_support,
        "full_training_authorized": False,
        "interpretation": (
            "This classification summarizes the fixed checkpoint on the "
            "speaker-disjoint evaluation set; it does not select a checkpoint."
        ),
    }


def _print_report(report: Mapping) -> None:
    step = int(report["checkpoint_step"])
    state_label = f"EMA-{step}"
    print(f"\nSpeaker-disjoint evaluation: EMA step {step} minus released base")
    print("=" * 112)
    print(
        f"{'Metric':<20}{'Base':>10}{state_label:>10}{'Delta':>11}"
        f"{'utterance 95% CI':>25}{'speaker 95% CI':>25}"
    )
    print("-" * 112)
    for metric in REPORT_METRICS:
        if metric not in report["analysis"]["metrics"]:
            continue
        row = report["analysis"]["metrics"][metric]
        utterance = row["utterance_ci"]
        speaker = row["speaker_ci"]
        utterance_ci = f"[{utterance['ci_low']:+.5f},{utterance['ci_high']:+.5f}]"
        speaker_ci = f"[{speaker['ci_low']:+.5f},{speaker['ci_high']:+.5f}]"
        print(
            f"{LABELS[metric]:<20}{row['base_mean']:>10.5f}"
            f"{row['state_mean']:>10.5f}{row['delta_mean']:>+11.5f}"
            f"{utterance_ci:>25}{speaker_ci:>25}"
        )
    print("=" * 112)
    decision = report["confirmation_decision"]
    print(f"Decision: {decision['decision']}")


def _validate_confirmation_manifest(
    *,
    manifest_path: Path,
    metadata_path: Path,
    config: Mapping,
) -> tuple[dict[str, str], dict]:
    manifest = strict_manifest(manifest_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(manifest) != 256:
        raise ValueError(f"confirmation requires 256 utterances, got {len(manifest)}")
    criteria = {
        "schema": metadata.get("schema_version") == 1,
        "purpose": metadata.get("purpose")
        == "fresh_speaker_disjoint_base_vs_ema_step20_confirmation_v2",
        "count": int(metadata.get("utterances", -1)) == len(manifest),
        "manifest_sha256": metadata.get("manifest_sha256")
        == sha256_file(manifest_path),
        "selected_ids": sorted(metadata.get("selected_ids", [])) == sorted(manifest),
    }
    source_path = Path(str(metadata.get("source_manifest", "")))
    if not source_path.is_file():
        criteria["source_exists"] = False
    else:
        criteria["source_exists"] = True
        criteria["source_sha256"] = metadata.get("source_sha256") == sha256_file(
            source_path
        )
        source = strict_manifest(source_path)
        criteria["source_rows"] = all(
            utterance in source and source[utterance] == transcript
            for utterance, transcript in manifest.items()
        )

    pilot = strict_manifest(Path(config["data"]["evaluation_manifest"]))
    training = strict_manifest(Path(config["data"]["train_manifest"]))
    criteria["pilot_disjoint"] = not bool(set(manifest) & set(pilot))
    criteria["training_disjoint"] = not bool(set(manifest) & set(training))
    confirmation_speakers = {_speaker(item) for item in manifest}
    training_speakers = {_speaker(item) for item in training}
    pilot_speakers = {_speaker(item) for item in pilot}
    criteria["training_speaker_disjoint"] = not bool(
        confirmation_speakers & training_speakers
    )
    criteria["pilot_speaker_disjoint"] = not bool(
        confirmation_speakers & pilot_speakers
    )
    expected_counts = {
        speaker: sum(_speaker(item) == speaker for item in manifest)
        for speaker in sorted(confirmation_speakers)
    }
    criteria["speaker_counts_match_metadata"] = (
        metadata.get("speaker_counts") == expected_counts
    )
    criteria["multiple_confirmation_speakers"] = len(confirmation_speakers) >= 2
    if not all(criteria.values()):
        raise ValueError(f"invalid confirmation manifest: {criteria}")
    return manifest, metadata


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
        raise ValueError("confirmation evaluation is permanently audio-only")

    run_dir = args.run_dir.resolve()
    training_protocol_path = run_dir / "protocol.json"
    training_protocol = json.loads(training_protocol_path.read_text(encoding="utf-8"))
    if sha256_json(training_protocol) != run_dir.name:
        raise ValueError("training protocol hash does not match the run directory")
    if training_protocol.get("config") != config:
        raise ValueError("evaluation config differs from the frozen training config")
    execution_dependencies = verify_execution_dependencies(training_protocol)
    source_override = _protocol_source_override(
        execution_dependencies,
        explicitly_allowed=bool(
            getattr(args, "allow_protocol_source_mismatch", False)
        ),
    )
    execution_dependencies["protocol_source_override"] = source_override
    if not execution_dependencies["passed"] and not source_override["authorized"]:
        raise ValueError(
            "training execution dependencies no longer match the frozen protocol: "
            f"criteria={execution_dependencies['criteria']}, "
            f"source_mismatches={source_override['observed_mismatches']}"
        )
    manifest, metadata = _validate_confirmation_manifest(
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        config=config,
    )

    step = int(args.step)
    if step < 1:
        raise ValueError("step must be positive")
    payload, checkpoint_path, checkpoint_hash = _load_checkpoint(run_dir, step)
    ema_state: AdapterState = {
        str(name): value.detach().cpu().clone()
        for name, value in payload["rollout_lora"].items()
    }
    del payload

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

    source_path = Path(__file__).resolve()
    shared_source = source_path.with_name(
        "evaluation.py"
    )
    protocol = {
        "schema_version": 1,
        "method": "speech_advantageflow_speaker_disjoint_256_checkpoint_evaluation_v1",
        "comparison": f"ema_step_{step:06d}_minus_released_base",
        "checkpoint_step": step,
        "training_protocol_hash": run_dir.name,
        "training_protocol_sha256": sha256_file(training_protocol_path),
        "training_execution_dependencies": execution_dependencies,
        "config_sha256": sha256_file(args.config),
        "confirmation_manifest_sha256": sha256_file(args.manifest),
        "confirmation_metadata_sha256": sha256_file(args.metadata),
        "confirmation_utterances": len(manifest),
        "selection_seed": metadata["seed"],
        "evaluation_nfe": int(config["rollout"]["evaluation_nfe"]),
        "latent_seed_base": int(config["evaluation"]["latent_seed_base"]),
        "checkpoint_sha256": checkpoint_hash,
        "released_checkpoint_sha256": bundle.checkpoint_sha256,
        "vocoder_sha256": bundle.vocoder_sha256,
        "evaluator_fingerprint": evaluator_fingerprint,
        "reward_calibration": reward_calibration,
        "source_sha256": {
            source_path.name: sha256_file(source_path),
            shared_source.name: sha256_file(shared_source),
        },
    }
    evaluation_hash = sha256_json(protocol)
    output_dir = run_dir / "confirmation_speaker_disjoint_evaluation" / evaluation_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "protocol.json", protocol)

    dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
    base_report = _evaluate_state(
        bundle=bundle,
        state=base_state,
        state_id="confirmation_base",
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
    ema_report = _evaluate_state(
        bundle=bundle,
        state=ema_state,
        state_id=f"confirmation_ema_step_{step:06d}",
        policy="ema",
        step=step,
        manifest=manifest,
        config=config,
        conditioning=conditioning,
        dnsmos=dnsmos,
        fidelity=fidelity,
        output_dir=output_dir,
        state_source={
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "payload_key": "rollout_lora",
        },
        composite_evaluators=composite_evaluators,
        reward_definition=reward_definition,
    )
    analysis = paired_state_analysis(
        base_report["rows"],
        ema_report["rows"],
        seed=stable_seed(args.bootstrap_seed, "speaker_disjoint_evaluation", step),
        samples=args.bootstrap_samples,
        confidence=args.confidence,
    )
    decision = confirmation_decision(analysis, config)

    checkpoint_after = sha256_file(checkpoint_path)
    released_after = sha256_file(bundle.checkpoint_path)
    immutability = {
        "passed": checkpoint_after == checkpoint_hash
        and released_after == bundle.checkpoint_sha256,
        "training_checkpoint": {
            "path": str(checkpoint_path),
            "before": checkpoint_hash,
            "after": checkpoint_after,
        },
        "released_checkpoint": {
            "path": str(bundle.checkpoint_path),
            "before": bundle.checkpoint_sha256,
            "after": released_after,
        },
    }
    report = {
        "status": "CONFIRMATION-COMPLETE" if immutability["passed"] else "INCOMPLETE",
        "evaluation_protocol_hash": evaluation_hash,
        "comparison": f"ema_step_{step:06d}_minus_released_base",
        "checkpoint_step": step,
        "utterances": len(manifest),
        "lora_modules": len(injection.module_names),
        "base_summary": base_report["summary"],
        "ema_summary": ema_report["summary"],
        "analysis": analysis,
        "confirmation_decision": decision,
        "checkpoint_immutability": immutability,
        "evaluator_fingerprint": evaluator_fingerprint,
        "reward_calibration": reward_calibration,
    }
    report_path = output_dir / "confirmation_report.json"
    _write_json(report_path, report)
    _write_json(
        run_dir / "latest_confirmation_evaluation.json",
        {"evaluation_protocol_hash": evaluation_hash, "report": str(report_path)},
    )
    _print_report(report)
    print(f"Report: {report_path}")
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an EMA checkpoint on 256 speaker-disjoint utterances"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=260723)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--allow-protocol-source-mismatch",
        action="store_true",
        help=(
            "Allow only protocol.py to differ from the "
            "training snapshot; model, inference, vocoder, and evaluator sources "
            "must still match"
        ),
    )
    args = parser.parse_args()
    report, _ = run(args)
    if report["status"] != "CONFIRMATION-COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
