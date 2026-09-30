"""Method-neutral training-reward definitions for audio-only speech."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


DNSMOS_ONLY = "dnsmos_ovrl_div_4"
DNSMOS_OVRL_RAW = "dnsmos_ovrl_raw"
DNSMOS_SPEAKER = "dnsmos_ovrl_plus_wavlm_speaker"
FLOWSE_GRPO_COMPOSITE = "flowse_grpo_public_composite"


def resolve_training_reward(
    config: Mapping, *, validate_artifacts: bool | None = None
) -> dict:
    """Return a validated, explicit reward definition.

    Historical configs omit ``training_reward`` and remain DNSMOS-OVRL-only.
    The two-component experiment uses independently frozen component scales;
    no scale is fitted or updated from a training batch.
    """

    if validate_artifacts is None:
        # The public project uses the frozen component scales checked into the
        # configuration.  Calibration reports are useful provenance, but are
        # not required to start either a smoke run or a normal training run.
        validate_artifacts = False
    raw = config.get("training_reward")
    if raw is None:
        return {
            "name": DNSMOS_ONLY,
            "components": ["dnsmos"],
            "formula": "dnsmos_ovrl / 4",
        }
    if not isinstance(raw, Mapping):
        raise ValueError("training_reward must be a mapping")
    name = str(raw.get("name", ""))
    if name == DNSMOS_ONLY:
        return {
            "name": DNSMOS_ONLY,
            "components": ["dnsmos"],
            "formula": "dnsmos_ovrl / 4",
        }
    if name == DNSMOS_OVRL_RAW:
        allowed = {"name", "auxiliary_composite"}
        if not set(raw).issubset(allowed):
            raise ValueError(
                "raw DNSMOS OVRL reward supports only name and auxiliary_composite"
            )
        definition = {
            "name": DNSMOS_OVRL_RAW,
            "components": ["dnsmos"],
            "formula": "dnsmos_ovrl",
        }
        auxiliary = raw.get("auxiliary_composite")
        if auxiliary is not None:
            if not isinstance(auxiliary, Mapping):
                raise ValueError("auxiliary_composite must be a reward mapping")
            if str(auxiliary.get("name", "")) != FLOWSE_GRPO_COMPOSITE:
                raise ValueError(
                    "raw OVRL auxiliary reward must be FlowSE-GRPO composite"
                )
            definition["auxiliary_composite"] = _resolve_flowse_grpo_composite(
                auxiliary, validate_artifacts=validate_artifacts
            )
        return definition
    if name == FLOWSE_GRPO_COMPOSITE:
        return _resolve_flowse_grpo_composite(
            raw, validate_artifacts=validate_artifacts
        )
    if name != DNSMOS_SPEAKER:
        raise ValueError(f"unsupported training reward: {name!r}")

    if str(raw.get("component_normalization")) != "frozen_std":
        raise ValueError("DNSMOS+speaker reward requires frozen_std normalization")
    weights = raw.get("weights")
    stds = raw.get("frozen_component_stds")
    if not isinstance(weights, Mapping) or set(weights) != {"dnsmos", "speaker"}:
        raise ValueError("reward weights must contain exactly dnsmos and speaker")
    if not isinstance(stds, Mapping) or set(stds) != {"dnsmos", "speaker"}:
        raise ValueError(
            "frozen_component_stds must contain exactly dnsmos and speaker"
        )
    parsed_weights = {name: float(weights[name]) for name in ("dnsmos", "speaker")}
    parsed_stds = {name: float(stds[name]) for name in ("dnsmos", "speaker")}
    if any(not math.isfinite(value) or value < 0.0 for value in parsed_weights.values()):
        raise ValueError("reward weights must be finite and non-negative")
    if not any(value > 0.0 for value in parsed_weights.values()):
        raise ValueError("at least one reward weight must be positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed_stds.values()):
        raise ValueError("frozen component standard deviations must be positive")
    calibration = raw.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("DNSMOS+speaker reward requires calibration mapping")
    required_calibration = {
        "screening_protocol_path",
        "screening_report_path",
        "endpoint_metrics_path",
        "source_nfe",
        "std_ddof",
        "dnsmos_divisor",
    }
    if set(calibration) != required_calibration:
        raise ValueError(
            "calibration must contain exactly "
            + ", ".join(sorted(required_calibration))
        )
    parsed_calibration = {
        "screening_protocol_path": str(calibration["screening_protocol_path"]),
        "screening_report_path": str(calibration["screening_report_path"]),
        "endpoint_metrics_path": str(calibration["endpoint_metrics_path"]),
        "source_nfe": int(calibration["source_nfe"]),
        "std_ddof": int(calibration["std_ddof"]),
        "dnsmos_divisor": float(calibration["dnsmos_divisor"]),
    }
    if parsed_calibration["source_nfe"] < 1:
        raise ValueError("calibration source_nfe must be positive")
    if parsed_calibration["std_ddof"] != 0:
        raise ValueError("component calibration requires std_ddof=0")
    if parsed_calibration["dnsmos_divisor"] != 4.0:
        raise ValueError("DNSMOS calibration requires dnsmos_divisor=4")
    return {
        "name": DNSMOS_SPEAKER,
        "components": ["dnsmos", "speaker"],
        "formula": (
            "w_dnsmos*(dnsmos_ovrl/4)/std_dnsmos + "
            "w_speaker*wavlm_speaker_similarity/std_speaker"
        ),
        "component_normalization": "frozen_std",
        "weights": parsed_weights,
        "frozen_component_stds": parsed_stds,
        "calibration": parsed_calibration,
        "speaker_backend": "frozen_wavlm_base_plus_sv_proxy",
    }


def _resolve_flowse_grpo_composite(
    raw: Mapping, *, validate_artifacts: bool = True
) -> dict:
    if str(raw.get("component_normalization")) != "frozen_std":
        raise ValueError("FlowSE-GRPO composite requires frozen_std normalization")
    component_names = ("dnsmos", "speaker", "speechbertscore")
    weights = raw.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != set(component_names):
        raise ValueError(
            "FlowSE-GRPO composite weights must contain dnsmos, speaker, and speechbertscore"
        )
    parsed_weights = {name: float(weights[name]) for name in component_names}
    if parsed_weights != {"dnsmos": 0.6, "speaker": 1.0, "speechbertscore": 1.0}:
        raise ValueError("FlowSE-GRPO composite requires paper weights 0.6/1.0/1.0")
    calibration = raw.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("FlowSE-GRPO composite requires calibration mapping")
    required = {"report_path", "source_nfe", "std_ddof", "dnsmos_divisor"}
    if set(calibration) != required:
        raise ValueError(
            "FlowSE-GRPO calibration must contain exactly " + ", ".join(sorted(required))
        )
    parsed_calibration = {
        "report_path": str(calibration["report_path"]),
        "source_nfe": int(calibration["source_nfe"]),
        "std_ddof": int(calibration["std_ddof"]),
        "dnsmos_divisor": float(calibration["dnsmos_divisor"]),
    }
    if parsed_calibration["source_nfe"] < 1:
        raise ValueError("composite calibration source_nfe must be positive")
    if parsed_calibration["std_ddof"] != 0:
        raise ValueError("FlowSE-GRPO composite calibration requires std_ddof=0")
    if parsed_calibration["dnsmos_divisor"] != 4.0:
        raise ValueError("FlowSE-GRPO composite requires dnsmos_divisor=4")
    static_definition = {
        "name": FLOWSE_GRPO_COMPOSITE,
        "components": list(component_names),
        "formula": (
            "0.6*(dnsmos_ovrl/4)/std_dnsmos + "
            "eres2net_speaker_similarity/std_speaker + "
            "speechbertscore/std_speechbertscore"
        ),
        "component_normalization": "frozen_std",
        "weights": parsed_weights,
        "calibration": parsed_calibration,
        "speaker_backend": "public_modelscope_eres2net",
        "content_backend": "speechbertscore_wavlm_large_layer14_precision",
    }
    if not validate_artifacts:
        # Public smoke runs still need the frozen component scales to compute
        # the composite reward, but they should not require the formal report
        # provenance (private manifest paths, endpoint hashes, evaluator
        # fingerprints).  Prefer the published report when it is available;
        # otherwise allow a self-contained config to provide the scales.
        report_path = Path(parsed_calibration["report_path"])
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report_stds = report.get("component_stds")
                if isinstance(report_stds, Mapping) and set(report_stds) == set(
                    component_names
                ):
                    parsed_stds = {
                        name: float(report_stds[name]) for name in component_names
                    }
                    if all(
                        math.isfinite(value) and value > 0.0
                        for value in parsed_stds.values()
                    ):
                        return {
                            **static_definition,
                            "frozen_component_stds": parsed_stds,
                            "calibration_evaluator_fingerprint": (
                                json.loads(
                                    json.dumps(
                                        report.get("evaluators", {}),
                                        sort_keys=True,
                                    )
                                )
                            ),
                            "calibration_artifacts_verified": False,
                        }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        configured_stds = raw.get("frozen_component_stds")
        if isinstance(configured_stds, Mapping) and set(configured_stds) == set(
            component_names
        ):
            parsed_stds = {
                name: float(configured_stds[name]) for name in component_names
            }
            if all(
                math.isfinite(value) and value > 0.0 for value in parsed_stds.values()
            ):
                return {
                    **static_definition,
                    "frozen_component_stds": parsed_stds,
                    "calibration_evaluator_fingerprint": {},
                    "calibration_artifacts_verified": False,
                }
        # Configuration-only callers (for example the sampling-diagnostic
        # validator) only need to identify the reward family.  The actual
        # training/scoring path will raise a focused error from
        # ``compute_training_reward`` if frozen scales are still absent.
        return {**static_definition, "calibration_artifacts_verified": False}
    report_path = Path(parsed_calibration["report_path"])
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "CALIBRATION-COMPLETE":
        raise ValueError("composite reward requires a complete, non-preflight calibration")
    report_stds = report.get("component_stds")
    if not isinstance(report_stds, Mapping) or set(report_stds) != set(component_names):
        raise ValueError("calibration report has incomplete component standard deviations")
    parsed_stds = {name: float(report_stds[name]) for name in component_names}
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed_stds.values()):
        raise ValueError("composite calibration stds must be finite and positive")
    source = report.get("source", {})
    if (
        int(source.get("source_nfe", -1)) != parsed_calibration["source_nfe"]
        or int(source.get("std_ddof", -1)) != parsed_calibration["std_ddof"]
        or float(source.get("dnsmos_divisor", -1.0))
        != parsed_calibration["dnsmos_divisor"]
    ):
        raise ValueError("composite calibration report rules do not match the config")
    calibration_evaluators = report.get("evaluators")
    if not isinstance(calibration_evaluators, Mapping):
        raise ValueError("composite calibration report has no evaluator fingerprint")
    return {
        **static_definition,
        "frozen_component_stds": parsed_stds,
        "calibration_evaluator_fingerprint": json.loads(
            json.dumps(calibration_evaluators, sort_keys=True)
        ),
        "calibration_artifacts_verified": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_manifest(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"invalid or empty manifest: {path}")
    if not all(isinstance(key, str) and isinstance(text, str) for key, text in value.items()):
        raise ValueError(f"manifest must map strings to strings: {path}")
    return value


def _population_std(values: list[float]) -> float:
    if not values:
        raise ValueError("calibration component is empty")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def verify_reward_calibration(
    config: Mapping,
    *,
    evaluator_fingerprint: Mapping | None = None,
    atol: float = 1.0e-9,
    lightweight: bool = False,
    strict_provenance: bool = False,
) -> dict | None:
    """Recompute and fingerprint the frozen composite component scales."""

    definition = resolve_training_reward(config)
    if definition["name"] == FLOWSE_GRPO_COMPOSITE:
        calibration = definition["calibration"]
        report_path = Path(calibration["report_path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_stds = {
            name: float(report["component_stds"][name])
            for name in ("dnsmos", "speaker", "speechbertscore")
        }
        source = report["source"]
        data_config = config.get("data", {})
        configured_train_manifest = data_config.get("train_manifest")
        configured_train_hash = None
        calibration_domain_manifest = data_config.get(
            "calibration_domain_manifest", configured_train_manifest
        )
        calibration_domain_hash = None
        explicit_calibration_domain = "calibration_domain_manifest" in data_config
        run_is_domain_subset = False
        if (
            not lightweight
            and configured_train_manifest is not None
            and calibration_domain_manifest is not None
        ):
            configured_train_hash = _sha256_file(Path(configured_train_manifest))
            calibration_domain_hash = _sha256_file(Path(calibration_domain_manifest))
            run_manifest = _strict_manifest(Path(configured_train_manifest))
            domain_manifest = _strict_manifest(Path(calibration_domain_manifest))
            run_is_domain_subset = all(
                utterance in domain_manifest and domain_manifest[utterance] == transcript
                for utterance, transcript in run_manifest.items()
            )
        domain_matches_report = (
            calibration_domain_hash is not None
            and source.get("train_manifest_sha256") == calibration_domain_hash
        )
        run_matches_report_directly = (
            configured_train_hash is not None
            and source.get("train_manifest_sha256") == configured_train_hash
        )
        excluded_utterances = source.get("excluded_outside_train_utterances")
        criteria = {
            "calibration_complete": report.get("status") == "CALIBRATION-COMPLETE",
            "source_nfe_matches": int(source["source_nfe"])
            == calibration["source_nfe"],
            "std_ddof_matches": int(source["std_ddof"])
            == calibration["std_ddof"],
            "dnsmos_divisor_matches": float(source["dnsmos_divisor"])
            == calibration["dnsmos_divisor"],
            "component_stds_match": all(
                abs(report_stds[name] - definition["frozen_component_stds"][name])
                <= atol
                for name in report_stds
            ),
            "evaluator_fingerprint_recorded": isinstance(
                report.get("evaluators"), Mapping
            ),
            "train_filter_accounting_complete": (
                isinstance(excluded_utterances, list)
                and int(source.get("eligible_rows_before_train_filter", -1))
                == int(source.get("selected_train_rows", -2))
                + int(source.get("excluded_outside_train_rows", -3))
                and int(source.get("selected_train_rows", -1))
                == int(source.get("rows", -2))
                and int(source.get("selected_train_utterances", -1)) > 0
            ),
        }
        if not lightweight:
            criteria.update(
                {
                    "train_manifest_fingerprint_recorded": bool(
                        source.get("train_manifest_sha256")
                    ),
                    "train_manifest_matches_run": run_matches_report_directly
                    or (domain_matches_report and run_is_domain_subset),
                }
            )
        if explicit_calibration_domain and not lightweight:
            criteria.update(
                {
                    "calibration_domain_manifest_matches_report": domain_matches_report,
                    "run_train_manifest_is_subset_of_calibration_domain": (
                        run_is_domain_subset
                    ),
                }
            )
        if evaluator_fingerprint is not None:
            criteria["evaluator_fingerprint_matches"] = (
                report.get("evaluators") == dict(evaluator_fingerprint)
            )
        # The manifest and evaluator fingerprints are provenance fields.  They
        # are useful for auditing a published run, but they must not turn a
        # calibration report into a server-specific access token.  A user who
        # downloads this release may use the supplied calibration statistics
        # with a different local checkout, dataset subset, or evaluator cache;
        # the algorithmic calibration rules above remain the actual contract.
        provenance_only = {
            "train_manifest_fingerprint_recorded",
            "train_manifest_matches_run",
            "calibration_domain_manifest_matches_report",
            "run_train_manifest_is_subset_of_calibration_domain",
            "evaluator_fingerprint_matches",
        }
        blocking_criteria = (
            dict(criteria)
            if strict_provenance
            else {
                key: value
                for key, value in criteria.items()
                if key not in provenance_only
            }
        )
        if not all(blocking_criteria.values()):
            raise ValueError(f"reward calibration verification failed: {criteria}")
        return {
            "criteria": criteria,
            "source_rows": int(report["source"]["rows"]),
            "rules": {
                "source_nfe": calibration["source_nfe"],
                "std_ddof": calibration["std_ddof"],
                "dnsmos_divisor": calibration["dnsmos_divisor"],
            },
            "recomputed_component_stds": report_stds,
            "calibration_evaluator_fingerprint": report["evaluators"],
            "verified_evaluator_fingerprint": (
                dict(evaluator_fingerprint)
                if evaluator_fingerprint is not None
                else None
            ),
            "file_sha256": (
                {}
                if lightweight
                else {
                    "calibration_report": _sha256_file(report_path),
                    **(
                        {"run_train_manifest": configured_train_hash}
                        if configured_train_hash is not None
                        else {}
                    ),
                    **(
                        {"calibration_domain_manifest": calibration_domain_hash}
                        if calibration_domain_hash is not None
                        else {}
                    ),
                }
            ),
            "lightweight_startup_validation": bool(lightweight),
        }
    if definition["name"] != DNSMOS_SPEAKER:
        return None
    calibration = definition["calibration"]
    paths = {
        name: Path(calibration[name])
        for name in (
            "screening_protocol_path",
            "screening_report_path",
            "endpoint_metrics_path",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(paths["screening_protocol_path"].read_text(encoding="utf-8"))
    report = json.loads(paths["screening_report_path"].read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in paths["endpoint_metrics_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row for row in rows if int(row.get("nfe", -1)) == calibration["source_nfe"]
    ]
    divisor = float(calibration["dnsmos_divisor"])
    values = {
        "dnsmos": [float(row["dnsmos_ovrl"]) / divisor for row in selected],
        "speaker": [float(row["speaker_similarity"]) for row in selected],
    }
    recomputed = {name: _population_std(component) for name, component in values.items()}
    configured = definition["frozen_component_stds"]
    reported = report.get("component_stds", {})
    frozen_gate_a = protocol.get("frozen_gate_a", {})
    endpoint_hash = _sha256_file(paths["endpoint_metrics_path"])
    criteria = {
        "source_rows_present": bool(selected),
        "screening_source_nfe_matches": int(
            protocol.get("input", {}).get("training_nfe", -1)
        )
        == calibration["source_nfe"],
        "screening_std_ddof_matches": int(
            protocol.get("reward", {})
            .get("normalization", {})
            .get("std_ddof", -1)
        )
        == calibration["std_ddof"],
        "gate_a_endpoint_hash_matches_screening": (
            frozen_gate_a.get("endpoint_sha256") == endpoint_hash
        ),
        "configured_stds_recompute": all(
            abs(float(configured[name]) - recomputed[name]) <= atol
            for name in recomputed
        ),
        "screening_report_stds_recompute": all(
            name in reported
            and abs(float(reported[name]) - recomputed[name]) <= atol
            for name in recomputed
        ),
    }
    if not all(criteria.values()):
        raise ValueError(f"reward calibration verification failed: {criteria}")
    return {
        "criteria": criteria,
        "source_rows": len(selected),
        "rules": {
            "source_nfe": calibration["source_nfe"],
            "std_ddof": calibration["std_ddof"],
            "dnsmos_divisor": divisor,
        },
        "recomputed_component_stds": recomputed,
        "file_sha256": {name: _sha256_file(path) for name, path in paths.items()},
    }


def compute_training_reward(metrics: Mapping, definition: Mapping) -> dict:
    """Compute one scalar reward and retain its auditable components."""

    raw_ovrl = float(metrics["dnsmos_ovrl"])
    if not math.isfinite(raw_ovrl):
        raise ValueError("DNSMOS reward component must be finite")
    if definition["name"] == DNSMOS_OVRL_RAW:
        return {
            "reward": raw_ovrl,
            "raw_components": {"dnsmos": raw_ovrl},
            "normalized_components": {"dnsmos": raw_ovrl},
            "weighted_components": {"dnsmos": raw_ovrl},
        }
    dnsmos = raw_ovrl / 4.0
    if definition["name"] == DNSMOS_ONLY:
        return {
            "reward": dnsmos,
            "raw_components": {"dnsmos": dnsmos},
            "normalized_components": {"dnsmos": dnsmos},
            "weighted_components": {"dnsmos": dnsmos},
        }
    if definition["name"] == FLOWSE_GRPO_COMPOSITE:
        frozen_stds = definition.get("frozen_component_stds")
        if not isinstance(frozen_stds, Mapping):
            raise FileNotFoundError(
                "public composite reward needs calibration component_stds: "
                f"provide {definition['calibration']['report_path']} or "
                "training_reward.frozen_component_stds"
            )
        speaker = float(metrics["eres2net_speaker_similarity"])
        speechbertscore = float(metrics["speechbertscore"])
        if not math.isfinite(speaker) or not math.isfinite(speechbertscore):
            raise ValueError("FlowSE-GRPO composite component must be finite")
        raw = {
            "dnsmos": dnsmos,
            "speaker": speaker,
            "speechbertscore": speechbertscore,
        }
        normalized = {
            name: raw[name] / float(frozen_stds[name])
            for name in raw
        }
        weighted = {
            name: float(definition["weights"][name]) * normalized[name]
            for name in raw
        }
        reward = float(sum(weighted.values()))
        if not math.isfinite(reward):
            raise ValueError("combined training reward must be finite")
        return {
            "reward": reward,
            "raw_components": raw,
            "normalized_components": normalized,
            "weighted_components": weighted,
        }
    if definition["name"] != DNSMOS_SPEAKER:
        raise ValueError(f"unsupported resolved reward: {definition['name']!r}")

    speaker = float(metrics["speaker_similarity"])
    if not math.isfinite(speaker):
        raise ValueError("speaker reward component must be finite")
    raw = {"dnsmos": dnsmos, "speaker": speaker}
    normalized = {
        name: raw[name] / float(definition["frozen_component_stds"][name])
        for name in raw
    }
    weighted = {
        name: float(definition["weights"][name]) * normalized[name]
        for name in raw
    }
    reward = float(sum(weighted.values()))
    if not math.isfinite(reward):
        raise ValueError("combined training reward must be finite")
    return {
        "reward": reward,
        "raw_components": raw,
        "normalized_components": normalized,
        "weighted_components": weighted,
    }
