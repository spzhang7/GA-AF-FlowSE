"""Evaluate one completed GRPO milestone on the configured official test."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch
import yaml

from rl.common.conditioning import ConditioningProtocol
from rl.common.lora import inject_lora
from rl.common.protocol import paired_state_analysis, stable_seed
from rl.rewards.specification import (
    resolve_training_reward,
    verify_reward_calibration,
)

from .evaluation import (
    METRIC_LABELS,
    _metric_summary,
    evaluate_validation_state,
    load_grpo_online_checkpoint,
)
from .protocol import strict_manifest
from .storage import atomic_write_json


def milestone_checkpoint_for_step(
    config: Mapping, *, run_dir: Path, optimizer_step: int
) -> tuple[Path, int, int]:
    """Map one registered optimizer step to its immutable milestone file."""
    updates_per_collection = int(config["collection"]["optimizer_updates"])
    total_steps = int(config["run"]["collections"]) * updates_per_collection
    if optimizer_step < 1 or optimizer_step > total_steps:
        raise ValueError("optimizer step is outside the completed training horizon")
    if optimizer_step % updates_per_collection != 0:
        raise ValueError("optimizer step is not a GRPO collection boundary")
    numerator = optimizer_step * 100
    if numerator % total_steps != 0:
        raise ValueError("optimizer step is not a saved evaluation milestone")
    percentage = numerator // total_steps
    registered = {
        int(value) for value in config["evaluation"]["selection_milestones"]
    }
    if percentage not in registered:
        raise ValueError("optimizer step is not a registered evaluation milestone")
    collection_index = optimizer_step // updates_per_collection
    checkpoint = (
        run_dir
        / "milestones"
        / f"checkpoint_milestone_{percentage:03d}pct.pt"
    ).resolve()
    return checkpoint, percentage, collection_index


def _format_table(
    base: Mapping,
    state: Mapping,
    *,
    optimizer_step: int,
    split_name: str | None = None,
) -> str:
    base_summary = base["summary"]
    state_summary = state["summary"]
    base_label = (
        "SFT-20k base"
        if str(base.get("policy_kind", "")) == "sft20k_base"
        else "Released base"
    )
    lines = [
        "=" * 72,
        (
            f"GRPO official test [{split_name}] after optimizer step {optimizer_step}"
            if split_name is not None
            else f"GRPO official test after optimizer step {optimizer_step}"
        ),
        f"{'Metric':<24}{base_label:>15}{'GRPO online':>15}{'Delta':>15}",
        "-" * 72,
    ]
    for key, label in METRIC_LABELS:
        if key not in base_summary or key not in state_summary:
            continue
        base_mean = float(base_summary[key]["mean"])
        state_mean = float(state_summary[key]["mean"])
        lines.append(
            f"{label:<24}{base_mean:>15.5f}{state_mean:>15.5f}"
            f"{state_mean - base_mean:>+15.5f}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def _format_endpoint_table(
    state: Mapping, *, optimizer_step: int, split_name: str
) -> str:
    lines = [
        "=" * 44,
        f"GRPO endpoint [{split_name}] after step {optimizer_step}",
        f"{'Metric':<26}{'GRPO online':>18}",
        "-" * 44,
    ]
    summary = state["summary"]
    for key, label in METRIC_LABELS:
        if key in summary:
            lines.append(f"{label:<26}{float(summary[key]['mean']):>18.5f}")
    lines.append("=" * 44)
    return "\n".join(lines)


def _official_test_split(utterance: str) -> str:
    value = str(utterance)
    if value.startswith("dns_no_reverb_fileid_"):
        return "no_reverb"
    if value.startswith("dns_with_reverb_fileid_"):
        return "with_reverb"
    if value.startswith("dns_real_"):
        return "real_recordings"
    raise ValueError(f"unknown DNS2020 official-test utterance: {value!r}")


def _speaker(utterance: str) -> str:
    return str(utterance).split("_", 1)[0].lower()


def _per_speaker_analysis(
    base_rows: Sequence[Mapping],
    state_rows: Sequence[Mapping],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence: float,
) -> dict:
    speakers = sorted({_speaker(str(row["utterance"])) for row in base_rows})
    result = {}
    for speaker in speakers:
        base = [row for row in base_rows if _speaker(row["utterance"]) == speaker]
        state = [
            row for row in state_rows if _speaker(row["utterance"]) == speaker
        ]
        if len(base) != len(state):
            raise ValueError(f"base/GRPO test rows differ for speaker {speaker}")
        result[speaker] = paired_state_analysis(
            base,
            state,
            seed=stable_seed(bootstrap_seed, "grpo_test", speaker),
            samples=bootstrap_samples,
            confidence=confidence,
            allow_partial_metrics=True,
            include_speaker_ci=False,
        )
    return result


def run_official_test(
    config: dict,
    *,
    run_dir: Path,
    optimizer_step: int,
    endpoint_only: bool = False,
    keep_audio: bool = False,
    rerun_tag: str | None = None,
    latent_seed_base: int | None = None,
    latent_seed_namespace: str = "evaluation",
) -> tuple[dict, Path]:
    from rl.common.flowse_interface import load_flowse_bundle
    from rl.rewards.composite import (
        load_flowse_grpo_composite_evaluators,
    )
    from rl.common.protocol import load_fidelity
    from rl.rewards.metrics import DNSMOSScorer
    from .trainer import _canonical_hash

    run_dir = run_dir.resolve()
    if rerun_tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rerun_tag):
        raise ValueError("rerun tag may contain only letters, digits, dot, dash and underscore")
    if keep_audio and rerun_tag is None:
        raise ValueError("--keep-audio requires --rerun-tag so cached reports are not reused")
    if not latent_seed_namespace:
        raise ValueError("latent seed namespace must be non-empty")
    evaluation_root = (
        run_dir / "official_test_reruns" / rerun_tag
        if rerun_tag is not None
        else run_dir
    )
    checkpoint, percentage, collection_index = milestone_checkpoint_for_step(
        config,
        run_dir=run_dir,
        optimizer_step=optimizer_step,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_hash = _canonical_hash(config)
    payload = load_grpo_online_checkpoint(
        checkpoint, expected_config_hash=config_hash
    )
    if (
        int(payload.get("optimizer_step", -1)) != optimizer_step
        or int(payload.get("collection_index", -1)) != collection_index
    ):
        raise ValueError("checkpoint payload does not match the requested step")

    manifest = strict_manifest(config["data"]["official_test_manifest"])
    if not manifest:
        raise ValueError("official-test manifest is empty")
    bundle = load_flowse_bundle(
        config["flowse_config"], deterministic=True, compute_artifact_hashes=False
    )
    torch.manual_seed(int(config["lora"]["initialization_seed"]))
    inject_lora(
        bundle.model.transformer,
        target_patterns=config["lora"]["target_patterns"],
        rank=int(config["lora"]["rank"]),
        alpha=float(config["lora"]["alpha"]),
        dropout=float(config["lora"]["dropout"]),
        expected_modules=int(config["lora"]["expected_modules"]),
    )
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    composite, evaluator_fingerprint = load_flowse_grpo_composite_evaluators(config)
    verify_reward_calibration(config, evaluator_fingerprint=evaluator_fingerprint)
    reward_definition = resolve_training_reward(config)
    # DNS2020 manifests are transcript-free.  Keep the speaker evaluator but
    # do not allocate the ASR model for an undefined WER calculation.
    fidelity, _ = load_fidelity(config, lazy_asr=True)
    dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
    common = {
        "bundle": bundle,
        "manifest": manifest,
        "config": config,
        "conditioning": conditioning,
        "dnsmos": dnsmos,
        "fidelity": fidelity,
        "composite_evaluators": composite,
        "reward_definition": reward_definition,
        "output_dir": evaluation_root,
        "evaluation_split": "official_test",
        "latent_seed_base_override": latent_seed_base,
        "latent_seed_namespace": latent_seed_namespace,
    }
    sft20k_base = str(config["flowse_config"]).replace("\\", "/").endswith(
        "flowse_libritts_sft20k_wotext.yaml"
    )
    base = None
    if not endpoint_only:
        base, _ = evaluate_validation_state(
            lora_state=None,
            state_id=(
                "sft20k_base_cfg0_nfe32"
                if sft20k_base
                else "released_base_cfg0_nfe32"
            ),
            policy_kind="sft20k_base" if sft20k_base else "released_base",
            percentage=None,
            collection_index=0,
            checkpoint_path=None,
            state_source_extra={"reporting_only": True},
            retain_audio_override=keep_audio,
            **common,
        )
    state_id = f"grpo_online_step_{optimizer_step:06d}"
    state, _ = evaluate_validation_state(
        lora_state=payload["online_lora_state"],
        state_id=state_id,
        policy_kind="grpo_online",
        percentage=percentage,
        collection_index=collection_index,
        checkpoint_path=str(checkpoint),
        state_source_extra={
            "optimizer_step": optimizer_step,
            "reporting_only": True,
            **({"rerun_tag": rerun_tag} if rerun_tag is not None else {}),
        },
        retain_audio_override=keep_audio,
        **common,
    )

    decision = config["evaluation"].get("best_safe", {})
    bootstrap_seed = int(decision.get("bootstrap_seed", 41123))
    bootstrap_samples = int(decision.get("bootstrap_samples", 5000))
    confidence = float(decision.get("confidence", 0.95))
    paired = None
    per_speaker = None
    if base is not None:
        paired = paired_state_analysis(
            base["rows"],
            state["rows"],
            seed=stable_seed(bootstrap_seed, "grpo_test", optimizer_step),
            samples=bootstrap_samples,
            confidence=confidence,
            allow_partial_metrics=True,
            include_speaker_ci=False,
        )
        per_speaker = _per_speaker_analysis(
            base["rows"],
            state["rows"],
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        )
    split_reports = {}
    for split_name in ("no_reverb", "with_reverb", "real_recordings"):
        split_utterances = {
            str(row["utterance"])
            for row in state["rows"]
            if _official_test_split(str(row["utterance"])) == split_name
        }
        state_split_rows = [
            row for row in state["rows"] if str(row["utterance"]) in split_utterances
        ]
        split_reports[split_name] = {
            "utterances": len(split_utterances),
            "grpo_online": {
                "policy_kind": "grpo_online",
                "summary": _metric_summary(state_split_rows),
            },
            "reference_available_rows": int(
                sum(bool(row.get("reference_available", False)) for row in state_split_rows)
            ),
        }
        if base is not None:
            base_split_rows = [
                row
                for row in base["rows"]
                if str(row["utterance"]) in split_utterances
            ]
            split_reports[split_name].update(
                {
                    "base": {
                        "policy_kind": (
                            "sft20k_base" if sft20k_base else "released_base"
                        ),
                        "summary": _metric_summary(base_split_rows),
                    },
                    "paired_grpo_minus_base": paired_state_analysis(
                        base_split_rows,
                        state_split_rows,
                        seed=stable_seed(
                            bootstrap_seed,
                            "grpo_test",
                            optimizer_step,
                            split_name,
                        ),
                        samples=bootstrap_samples,
                        confidence=confidence,
                        allow_partial_metrics=True,
                        include_speaker_ci=False,
                    ),
                }
            )
    output_path = evaluation_root / "official_test" / f"{state_id}_summary.json"
    report = {
        "schema_version": 1,
        "status": "GRPO-OFFICIAL-TEST-COMPLETE",
        "optimizer_step": optimizer_step,
        "collection_index": collection_index,
        "percentage": percentage,
        "checkpoint": str(checkpoint),
        "policy_kind": "grpo_online",
        "evaluation_split": "official_test",
        "utterances": len(manifest),
        "speakers": sorted({_speaker(value) for value in manifest}),
        "nfe": int(config["evaluation"]["nfe"]),
        "cfg_strength": float(config["evaluation"]["cfg_strength"]),
        "base_policy": (
            None
            if endpoint_only
            else "sft20k_base" if sft20k_base else "released_base"
        ),
        "grpo_online": state["summary"],
        "paired_grpo_minus_base": paired,
        "per_speaker_grpo_minus_base": per_speaker,
        "split_reports": split_reports,
        "endpoint_only": bool(endpoint_only),
        "audio_retained": bool(keep_audio),
        "rerun_tag": rerun_tag,
        "latent_seed_base": int(
            config["evaluation"]["latent_seed_base"]
            if latent_seed_base is None
            else latent_seed_base
        ),
        "latent_seed_namespace": latent_seed_namespace,
        "reporting_only": True,
        "af_comparison_pending": True,
    }
    if base is not None:
        report[
            "sft20k_base" if sft20k_base else "released_base"
        ] = base["summary"]
    atomic_write_json(output_path, report)
    if hasattr(fidelity, "release_asr"):
        fidelity.release_asr()
    for split_name in ("no_reverb", "with_reverb", "real_recordings"):
        split = split_reports[split_name]
        if endpoint_only:
            print(
                _format_endpoint_table(
                    split["grpo_online"],
                    optimizer_step=optimizer_step,
                    split_name=split_name,
                )
            )
        else:
            print(
                _format_table(
                    split["base"],
                    split["grpo_online"],
                    optimizer_step=optimizer_step,
                    split_name=split_name,
                )
            )
    return report, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one GRPO test milestone")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument(
        "--endpoint-only",
        action="store_true",
        help="generate and score only the GRPO endpoint; skip the SFT/released base",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="retain every generated endpoint WAV (requires --rerun-tag)",
    )
    parser.add_argument(
        "--rerun-tag",
        help="write into official_test_reruns/TAG so generation is run again",
    )
    parser.add_argument(
        "--latent-seed-base",
        type=int,
        help="optional reporting-only latent seed override",
    )
    parser.add_argument(
        "--latent-seed-namespace",
        default="evaluation",
        help="seed namespace; use dns2020_zero_shot to match the AF evaluator",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, output_path = run_official_test(
        config,
        run_dir=args.run_dir,
        optimizer_step=args.optimizer_step,
        endpoint_only=args.endpoint_only,
        keep_audio=args.keep_audio,
        rerun_tag=args.rerun_tag,
        latent_seed_base=args.latent_seed_base,
        latent_seed_namespace=args.latent_seed_namespace,
    )
    print(f"Status: {report['status']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
