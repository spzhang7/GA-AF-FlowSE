"""Shared validation path and preregistered checkpoint selection for GRPO online."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm.auto import tqdm

from rl.common.conditioning import ConditioningProtocol
from rl.common.lora import (
    inject_lora,
    load_lora,
    lora_enabled,
    snapshot_lora,
)
from rl.common.milestone_selection import (
    select_registered_milestones,
)
from rl.common.protocol import (
    paired_state_analysis,
    score_evaluation_file,
    stable_seed,
)
from .protocol import (
    audit_data_splits,
    lora_state_fingerprint,
    strict_manifest,
)
from .storage import atomic_write_json


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

METRIC_LABELS = (
    ("dnsmos_sig", "DNSMOS SIG"),
    ("dnsmos_bak", "DNSMOS BAK"),
    ("dnsmos_ovrl", "DNSMOS OVRL"),
    ("dnsmos_p808", "DNSMOS P808"),
    ("speaker_similarity", "Speaker similarity"),
    ("eres2net_speaker_similarity", "ERes2Net speaker"),
    ("speechbertscore", "SpeechBERTScore"),
    ("flowse_grpo_composite_reward", "Composite reward"),
    ("stoi", "STOI"),
    ("pesq_wb", "PESQ-WB"),
    ("wer", "WER"),
)


def _metric_summary(rows: Sequence[Mapping]) -> dict:
    summary = {}
    for key in REPORT_METRICS:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row], dtype=np.float64
        )
        if values.size:
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "count": int(values.size),
            }
    return summary


def format_validation_comparison_table(
    *,
    base_report: Mapping,
    state_report: Mapping,
    optimizer_step: int,
    collection_index: int,
    percentage: int,
) -> str:
    """Render an AF-style terminal table for one GRPO validation milestone."""

    base_summary = base_report.get("summary")
    state_summary = state_report.get("summary")
    if not isinstance(base_summary, Mapping) or not isinstance(state_summary, Mapping):
        raise ValueError("validation reports must contain metric summaries")
    base_label = (
        "SFT-20k base"
        if str(base_report.get("policy_kind", "")) == "sft20k_base"
        else "Released base"
    )

    lines = [
        "=" * 68,
        (
            f"GRPO validation after optimizer step {int(optimizer_step)} "
            f"(collection {int(collection_index)}, {int(percentage)}%)"
        ),
        f"{'Metric':<24}{base_label:>13}{'GRPO online':>13}{'Delta':>13}",
        "-" * 68,
    ]
    rendered_metrics = 0
    for key, label in METRIC_LABELS:
        base_row = base_summary.get(key)
        state_row = state_summary.get(key)
        if not isinstance(base_row, Mapping) or not isinstance(state_row, Mapping):
            continue
        base_mean = float(base_row["mean"])
        state_mean = float(state_row["mean"])
        lines.append(
            f"{label:<24}{base_mean:>13.5f}{state_mean:>13.5f}"
            f"{state_mean - base_mean:>+13.5f}"
        )
        rendered_metrics += 1
    if rendered_metrics == 0:
        raise ValueError("validation reports share no printable metrics")
    lines.append("=" * 68)
    return "\n".join(lines)


def load_grpo_online_checkpoint(path: str | Path, *, expected_config_hash: str) -> dict:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    criteria = {
        "schema": payload.get("schema_version") == 1,
        "method": payload.get("method") == "flowse_grpo",
        "policy_kind": payload.get("policy_kind") == "grpo_online",
        "ema_disabled": payload.get("ema_enabled") is False,
        "collection_boundary": payload.get("collection_boundary") is True,
        "config": payload.get("config_sha256") == expected_config_hash,
        "online_state": isinstance(payload.get("online_lora_state"), Mapping),
    }
    if not all(criteria.values()):
        raise ValueError(f"not a compatible GRPO-online checkpoint: {criteria}")
    observed_fingerprint = lora_state_fingerprint(payload["online_lora_state"])
    if observed_fingerprint != payload.get("online_lora_fingerprint"):
        raise ValueError("GRPO-online checkpoint LoRA fingerprint is invalid")
    return payload


def _validation_state_source(
    *,
    lora_state: Mapping[str, torch.Tensor] | None,
    policy_kind: str,
    percentage: int | None,
    collection_index: int,
    checkpoint_path: str | None,
    config: Mapping,
    milestone_metadata: Mapping | None,
    evaluation_split: str,
    state_source_extra: Mapping | None,
    latent_seed_base_override: int | None = None,
    latent_seed_namespace: str = "evaluation",
) -> dict:
    state_fingerprint = (
        {"state_sha256": "released_checkpoint_lora_disabled"}
        if lora_state is None
        else lora_state_fingerprint(lora_state)
    )
    return {
        "policy_kind": policy_kind,
        "percentage": percentage,
        "collection_index": int(collection_index),
        "checkpoint_path": checkpoint_path,
        "latent_seed_namespace": str(latent_seed_namespace),
        "latent_seed_base": int(
            config["evaluation"]["latent_seed_base"]
            if latent_seed_base_override is None
            else latent_seed_base_override
        ),
        "evaluation_split": evaluation_split,
        "milestone_metadata": (
            dict(milestone_metadata) if milestone_metadata is not None else None
        ),
        **state_fingerprint,
        **dict(state_source_extra or {}),
    }


def reuse_released_base_for_zero_lora_validation(
    *,
    base_report: Mapping,
    lora_state: Mapping[str, torch.Tensor],
    state_id: str,
    percentage: int,
    collection_index: int,
    checkpoint_path: str,
    config: Mapping,
    output_dir: Path,
    milestone_metadata: Mapping,
) -> tuple[dict, float]:
    """Materialize the 0% policy report without regenerating identical audio."""

    if int(percentage) != 0 or int(collection_index) != 0:
        raise ValueError("released-base validation reuse is restricted to step 0")
    base_policy_kind = str(base_report.get("policy_kind", ""))
    if base_policy_kind not in {"released_base", "sft20k_base"}:
        raise ValueError("step-0 reuse requires a frozen-base validation report")
    if base_report.get("evaluation_split") != "validation":
        raise ValueError("step-0 reuse requires a validation-split base report")
    expected_rows = len(base_report.get("rows", []))
    if expected_rows < 1:
        raise ValueError("released-base validation report contains no rows")
    b_tensors = {
        str(name): value
        for name, value in lora_state.items()
        if str(name).endswith(".lora_B")
    }
    expected_modules = int(config["lora"]["expected_modules"])
    criteria = {
        "all_lora_tensors_finite": all(
            bool(torch.isfinite(value).all().item()) for value in lora_state.values()
        ),
        "lora_B_count": len(b_tensors) == expected_modules,
        "all_lora_B_exactly_zero": bool(b_tensors)
        and all(int(torch.count_nonzero(value).item()) == 0 for value in b_tensors.values()),
    }
    if not all(criteria.values()):
        raise ValueError(f"step-0 LoRA is not released-base equivalent: {criteria}")
    equivalence = {
        "kind": "exact_zero_lora_B_residual",
        "base_policy_kind": base_policy_kind,
        "base_state_id": str(base_report["state_id"]),
        "released_base_state_id": str(base_report["state_id"]),
        "released_base_row_count": expected_rows,
        "lora_B_tensor_count": len(b_tensors),
        "lora_state_sha256": lora_state_fingerprint(lora_state)["state_sha256"],
    }
    state_source = _validation_state_source(
        lora_state=lora_state,
        policy_kind="grpo_online",
        percentage=0,
        collection_index=0,
        checkpoint_path=checkpoint_path,
        config=config,
        milestone_metadata=milestone_metadata,
        evaluation_split="validation",
        state_source_extra={"evaluation_reuse": equivalence},
    )
    report_path = output_dir / "validation" / f"{state_id}.json"
    if report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("state_source") == state_source:
            return cached, 0.0
        raise ValueError(f"cached validation state source differs: {report_path}")
    rows = [
        {
            **dict(row),
            "state_id": state_id,
            "policy_kind": "grpo_online",
            "percentage": 0,
            "collection_index": 0,
        }
        for row in base_report["rows"]
    ]
    report = {
        **{
            key: json.loads(json.dumps(value))
            for key, value in base_report.items()
            if key not in {"state_id", "policy_kind", "percentage", "state_source", "rows", "timing_seconds"}
        },
        "state_id": state_id,
        "policy_kind": "grpo_online",
        "percentage": 0,
        "collection_index": 0,
        "state_source": state_source,
        "milestone_metadata": dict(milestone_metadata),
        "rows": rows,
        "timing_seconds": 0.0,
        "evaluation_reused": True,
        "evaluation_reuse": equivalence,
    }
    atomic_write_json(report_path, report)
    return report, 0.0


def evaluate_validation_state(
    *,
    bundle,
    lora_state: Mapping[str, torch.Tensor] | None,
    state_id: str,
    policy_kind: str,
    percentage: int | None,
    collection_index: int,
    checkpoint_path: str | None,
    manifest: Mapping[str, str],
    config: Mapping,
    conditioning: ConditioningProtocol,
    dnsmos,
    fidelity,
    composite_evaluators,
    reward_definition: Mapping,
    output_dir: Path,
    milestone_metadata: Mapping | None = None,
    evaluation_split: str = "validation",
    state_source_extra: Mapping | None = None,
    retain_audio_override: bool | None = None,
    latent_seed_base_override: int | None = None,
    latent_seed_namespace: str = "evaluation",
) -> tuple[dict, float]:
    """Generate fixed-latent validation or official-test rows."""

    supported = {
        "released_base",
        "sft20k_base",
        "grpo_online",
        "advantageflow_online",
        "advantageflow_ema",
    }
    if policy_kind not in supported:
        raise ValueError("unsupported evaluation policy kind")
    if evaluation_split not in {"validation", "official_test"}:
        raise ValueError("evaluation_split must be validation or official_test")
    report_path = output_dir / evaluation_split / f"{state_id}.json"
    state_source = _validation_state_source(
        lora_state=lora_state,
        policy_kind=policy_kind,
        percentage=percentage,
        collection_index=collection_index,
        checkpoint_path=checkpoint_path,
        config=config,
        milestone_metadata=milestone_metadata,
        evaluation_split=evaluation_split,
        state_source_extra=state_source_extra,
        latent_seed_base_override=latent_seed_base_override,
        latent_seed_namespace=latent_seed_namespace,
    )
    if report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("state_source") == state_source:
            return cached, 0.0
        raise ValueError(f"cached validation state source differs: {report_path}")

    previous = snapshot_lora(bundle.model.transformer, device="cpu")
    if lora_state is not None:
        load_lora(bundle.model.transformer, lora_state)
    evaluation_device = bundle.device
    evaluation_is_cuda = torch.cuda.is_available() and (
        isinstance(evaluation_device, int)
        or torch.device(evaluation_device).type == "cuda"
    )
    if evaluation_is_cuda:
        torch.cuda.synchronize(device=evaluation_device)
    started = time.perf_counter()
    rows = []
    reference_unavailable_rows = 0
    wer_excluded_rows = 0
    wer_exclusion_reasons: dict[str, int] = {}
    audio_dir = output_dir / f"{evaluation_split}_audio" / state_id
    retain_audio = (
        bool(config["artifacts"].get("keep_validation_audio", False))
        if retain_audio_override is None
        else bool(retain_audio_override)
    )
    try:
        context = lora_enabled(bundle.model.transformer, lora_state is not None)
        progress_description = f"{evaluation_split}:{state_id}"
        with context, tqdm(
            manifest.items(),
            total=len(manifest),
            desc=progress_description,
            unit="utt",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        ) as progress:
            for utterance, transcript in progress:
                noisy_path = Path(config["data"]["noisy_dir"]) / f"{utterance}.wav"
                clean_path = Path(config["data"]["clean_dir"]) / f"{utterance}.wav"
                # DNS2020 real recordings are reference-free: the frozen
                # preparation view intentionally contains only their noisy
                # waveform.  Keep DNSMOS for those rows, while restricting
                # intrusive metrics (PESQ/STOI/speaker/composite/WER) to
                # rows that actually have a clean reference.
                reference_available = clean_path.is_file()
                if not reference_available:
                    reference_unavailable_rows += 1
                    wer_excluded_rows += 1
                    wer_exclusion_reasons["reference_not_available"] = (
                        wer_exclusion_reasons.get("reference_not_available", 0) + 1
                    )
                seed = stable_seed(
                    int(
                        config["evaluation"]["latent_seed_base"]
                        if latent_seed_base_override is None
                        else latent_seed_base_override
                    ),
                    latent_seed_namespace,
                    utterance,
                )
                endpoint = bundle.generate_group(
                    noisy_path,
                    "",
                    [seed],
                    nfe=int(config["evaluation"]["nfe"]),
                    cfg_strength=float(config["evaluation"]["cfg_strength"]),
                    conditioning=conditioning,
                    target_dbfs=float(config["normalization"]["target_dbfs"]),
                    peak_ceiling=float(config["normalization"]["peak_ceiling"]),
                )[0]
                audio_path = audio_dir / f"{utterance}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    audio_path,
                    endpoint.normalized_waveform,
                    bundle.output_sample_rate,
                    subtype=str(config["normalization"]["output_subtype"]),
                )
                metrics = score_evaluation_file(
                    audio_path=audio_path,
                    clean_path=clean_path,
                    transcript=transcript,
                    dnsmos=dnsmos,
                    fidelity=fidelity if reference_available else None,
                    paired=reference_available,
                    composite_evaluators=(
                        composite_evaluators if reference_available else None
                    ),
                    reward_definition=reward_definition if reference_available else None,
                )
                if metrics.get("wer_excluded_reason") == (
                    "empty_reference_after_normalization"
                ):
                    wer_excluded_rows += 1
                    reason = str(metrics["wer_excluded_reason"])
                    wer_exclusion_reasons[reason] = (
                        wer_exclusion_reasons.get(reason, 0) + 1
                    )
                from .protocol import sha256_file

                wav_hash = sha256_file(audio_path)
                rows.append(
                    {
                        "state_id": state_id,
                        "policy_kind": policy_kind,
                        "percentage": percentage,
                        "collection_index": int(collection_index),
                        "utterance": utterance,
                        "reference_available": reference_available,
                        "latent_seed": int(seed),
                        "terminal_mel_sha256": endpoint.terminal_mel_sha256,
                        "scored_wav_sha256": wav_hash,
                        "audio_path": str(audio_path) if retain_audio else None,
                        **metrics,
                    }
                )
                if not retain_audio:
                    audio_path.unlink()
    finally:
        load_lora(bundle.model.transformer, previous)
    if evaluation_is_cuda:
        torch.cuda.synchronize(device=evaluation_device)
    seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "state_id": state_id,
        "policy_kind": policy_kind,
        "percentage": percentage,
        "collection_index": int(collection_index),
        "state_source": state_source,
        "evaluation_split": evaluation_split,
        "milestone_metadata": (
            dict(milestone_metadata) if milestone_metadata is not None else None
        ),
        "evaluation_setting": "base_cfg0_nfe32",
        "summary": _metric_summary(rows),
        "reference_coverage": {
            "available_rows": int(len(rows) - reference_unavailable_rows),
            "unavailable_rows": int(reference_unavailable_rows),
            "unavailable_reason": "official_test_reference_not_published",
        },
        "wer_coverage": {
            "eligible_rows": int(len(rows) - wer_excluded_rows),
            "excluded_rows": int(wer_excluded_rows),
            "exclusion_reasons": wer_exclusion_reasons,
        },
        "audio_retained": retain_audio,
        "rows": rows,
        "timing_seconds": seconds,
    }
    atomic_write_json(report_path, report)
    return report, seconds


def _best_safe_candidate(
    base: Mapping,
    reports: Sequence[Mapping],
    *,
    config: Mapping,
) -> tuple[dict | None, list[dict]]:
    # This evaluator pulls in the full FlowSE audio stack on some installations.
    # Keep it behind the best-safe path so checkpoint-schema and selection tests
    # remain usable on CPU/login nodes.
    decision = config["evaluation"]["best_safe"]
    candidates = []
    for report in reports:
        percentage = int(report["percentage"])
        analysis = paired_state_analysis(
            base["rows"],
            report["rows"],
            seed=stable_seed(
                int(decision["bootstrap_seed"]), "grpo_best_safe", percentage
            ),
            samples=int(decision["bootstrap_samples"]),
            confidence=float(decision["confidence"]),
        )
        metrics = analysis["metrics"]
        criteria = {}
        ovrl = metrics.get("dnsmos_ovrl")
        criteria["ovrl_utterance_ci_positive"] = bool(
            ovrl and float(ovrl["utterance_ci"]["ci_low"]) > 0.0
        )
        criteria["ovrl_speaker_ci_positive"] = bool(
            ovrl and float(ovrl["speaker_ci"]["ci_low"]) > 0.0
        )
        for metric, threshold in decision["safety"].items():
            row = metrics.get(metric)
            if row is None:
                criteria[f"{metric}_available"] = False
                continue
            if metric == "wer":
                bound = max(
                    float(row["utterance_ci"]["ci_high"]),
                    float(row["speaker_ci"]["ci_high"]),
                )
                criteria[f"{metric}_safety"] = bound <= float(threshold)
            else:
                bound = min(
                    float(row["utterance_ci"]["ci_low"]),
                    float(row["speaker_ci"]["ci_low"]),
                )
                criteria[f"{metric}_safety"] = bound >= float(threshold)
        candidates.append(
            {
                "percentage": percentage,
                "collection_index": int(report["collection_index"]),
                "eligible": all(criteria.values()),
                "criteria": criteria,
                "analysis": analysis,
                "composite_mean": float(
                    report["summary"]["flowse_grpo_composite_reward"]["mean"]
                ),
                "ovrl_mean": float(report["summary"]["dnsmos_ovrl"]["mean"]),
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["composite_mean"],
                row["ovrl_mean"],
                -row["percentage"],
            ),
        )
        if eligible
        else None
    )
    return selected, candidates


def select_milestone_checkpoints(
    *,
    base_report: Mapping,
    milestone_reports: Sequence[Mapping],
    checkpoint_paths: Mapping[int, str],
    config: Mapping,
    output_dir: Path,
) -> dict:
    expected = [int(value) for value in config["evaluation"]["selection_milestones"]]
    observed = sorted(int(report["percentage"]) for report in milestone_reports)
    if observed != expected:
        raise ValueError(
            f"milestone evaluations incomplete: expected={expected}, got={observed}"
        )
    if expected == [0, 20, 40, 60, 80, 100]:
        from .protocol import sha256_file

        checkpoint_sources = {
            int(percentage): {
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "payload_key": "online_lora_state",
            }
            for percentage, path in checkpoint_paths.items()
        }
        report = select_registered_milestones(
            policy_kind="grpo_online",
            milestone_reports=milestone_reports,
            checkpoint_sources=checkpoint_sources,
            base_report=base_report,
            safety_config=config,
            best_safe_selector=_best_safe_candidate,
        )
        atomic_write_json(output_dir / "checkpoint_selection.json", report)
        return report
    rows = []
    for report in milestone_reports:
        percentage = int(report["percentage"])
        rows.append(
            {
                "percentage": percentage,
                "collection_index": int(report["collection_index"]),
                "checkpoint": str(checkpoint_paths[percentage]),
                "milestone_metadata": report.get("milestone_metadata"),
                "composite_mean": float(
                    report["summary"]["flowse_grpo_composite_reward"]["mean"]
                ),
                "ovrl_mean": float(report["summary"]["dnsmos_ovrl"]["mean"]),
            }
        )
    comparison = max(
        rows,
        key=lambda row: (row["composite_mean"], row["ovrl_mean"], -row["percentage"]),
    )
    best_safe, safety_candidates = _best_safe_candidate(
        base_report, milestone_reports, config=config
    )
    if best_safe is not None:
        best_safe = {
            **best_safe,
            "checkpoint": str(checkpoint_paths[int(best_safe["percentage"])]),
        }
    report = {
        "schema_version": 1,
        "status": "CHECKPOINT-SELECTION-COMPLETE",
        "policy_kind": "grpo_online",
        "selection_opportunities": expected,
        "comparison_rule": (
            "max validation composite mean, then OVRL mean, then earlier budget percentage"
        ),
        "comparison_checkpoint": comparison,
        "best_safe_checkpoint": best_safe,
        "fixed_budget_endpoint": next(
            row for row in rows if int(row["percentage"]) == 100
        ),
        "candidates": rows,
        "best_safe_candidates": safety_candidates,
        "selection_input": "validation_only",
        "official_test_selection_forbidden": True,
    }
    atomic_write_json(output_dir / "checkpoint_selection.json", report)
    return report


def run_artifact_evaluation(config: Mapping, run_dir: Path) -> tuple[dict, Path]:
    """Evaluate all immutable GRPO-online milestone checkpoints post-training."""

    from rl.common.flowse_interface import (
        load_flowse_bundle,
    )
    from rl.rewards.composite import (
        load_flowse_grpo_composite_evaluators,
    )
    from rl.common.protocol import load_fidelity
    from rl.rewards.metrics import DNSMOSScorer
    from rl.rewards.specification import (
        resolve_training_reward,
        verify_reward_calibration,
    )
    from .trainer import LORA_TARGET_PATTERNS, _canonical_hash

    audit_data_splits(config)
    config_hash = _canonical_hash(config)
    bundle = load_flowse_bundle(config["flowse_config"], deterministic=True)
    torch.manual_seed(int(config["lora"]["initialization_seed"]))
    inject_lora(
        bundle.model.transformer,
        target_patterns=LORA_TARGET_PATTERNS,
        rank=int(config["lora"]["rank"]),
        alpha=float(config["lora"]["alpha"]),
        dropout=float(config["lora"]["dropout"]),
        expected_modules=int(config["lora"]["expected_modules"]),
    )
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    manifest = strict_manifest(config["data"]["validation_manifest"])
    composite, fingerprint = load_flowse_grpo_composite_evaluators(config)
    verify_reward_calibration(config, evaluator_fingerprint=fingerprint)
    reward_definition = resolve_training_reward(config)
    fidelity, _ = load_fidelity(dict(config))
    dnsmos = DNSMOSScorer(config["dnsmos_official_dir"])
    checkpoint_paths = {}
    reports = []
    base, _ = evaluate_validation_state(
        bundle=bundle,
        lora_state=None,
        state_id="released_base_cfg0_nfe32",
        policy_kind="released_base",
        percentage=None,
        collection_index=0,
        checkpoint_path=None,
        manifest=manifest,
        config=config,
        conditioning=conditioning,
        dnsmos=dnsmos,
        fidelity=fidelity,
        composite_evaluators=composite,
        reward_definition=reward_definition,
        output_dir=run_dir,
    )
    for percentage in config["evaluation"]["selection_milestones"]:
        path = (
            run_dir / "milestones" / f"checkpoint_milestone_{int(percentage):03d}pct.pt"
        )
        payload = load_grpo_online_checkpoint(path, expected_config_hash=config_hash)
        checkpoint_paths[int(percentage)] = str(path)
        report, _ = evaluate_validation_state(
            bundle=bundle,
            lora_state=payload["online_lora_state"],
            state_id=f"grpo_online_{int(percentage):03d}pct",
            policy_kind="grpo_online",
            percentage=int(percentage),
            collection_index=int(payload["collection_index"]),
            checkpoint_path=str(path),
            manifest=manifest,
            config=config,
            conditioning=conditioning,
            dnsmos=dnsmos,
            fidelity=fidelity,
            composite_evaluators=composite,
            reward_definition=reward_definition,
            output_dir=run_dir,
        )
        reports.append(report)
    selection = select_milestone_checkpoints(
        base_report=base,
        milestone_reports=reports,
        checkpoint_paths=checkpoint_paths,
        config=config,
        output_dir=run_dir,
    )
    return selection, run_dir / "checkpoint_selection.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GRPO online milestones")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, path = run_artifact_evaluation(config, args.run_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
