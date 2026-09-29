"""AdvantageFlow adapter for the method-neutral composite evaluators.

The common evaluator implementation is frozen into the reward-calibration
fingerprint.  AdvantageFlow therefore keeps worker-device placement here,
without changing the evaluator implementation used to produce calibration.
"""

from __future__ import annotations

from typing import Mapping

from rl.rewards import composite as _common
from rl.rewards.composite import *  # noqa: F403


def _move_modelscope_pipeline_to_device(pipeline, device) -> None:
    """Move a CPU-built ModelScope pipeline onto one explicit CUDA device."""

    model = getattr(pipeline, "model", None)
    if model is None or not hasattr(model, "to"):
        raise RuntimeError("ModelScope ERes2Net pipeline exposes no movable model")
    moved = model.to(device)
    if moved is not None and moved is not model:
        pipeline.model = moved
        model = moved

    # ModelScope uses ``pipeline.device`` when moving inputs for inference.
    # Some releases also retain a private device field.  Keep both aligned,
    # while tolerating a read-only compatibility property on the model.
    for owner in (pipeline, model):
        for attribute in ("device", "_device"):
            if not hasattr(owner, attribute):
                continue
            try:
                setattr(owner, attribute, device)
            except (AttributeError, TypeError):
                pass


def load_composite_reward_evaluators(
    config: Mapping,
    *,
    device_override: str | None = None,
) -> tuple[_common.FlowSEGRPOCompositeEvaluators, dict]:
    """Load evaluators, optionally pinning an AF worker to an explicit GPU."""

    if device_override is None:
        return _common.load_composite_reward_evaluators(config)

    import torch

    evaluator_config = config.get("composite_reward_evaluators")
    if not isinstance(evaluator_config, Mapping):
        raise ValueError(
            "FlowSE-GRPO composite reward requires composite_reward_evaluators"
        )
    speaker_config = evaluator_config.get("speaker")
    speechbert_config = evaluator_config.get("speechbertscore")
    if not isinstance(speaker_config, Mapping) or not isinstance(
        speechbert_config, Mapping
    ):
        raise ValueError("composite evaluator config requires speaker and speechbertscore")

    device = torch.device(str(device_override))
    if device.type != "cuda" or device.index is None:
        raise ValueError(
            "AdvantageFlow rollout worker evaluator requires explicit cuda:<index>"
        )
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise ValueError(f"rollout worker evaluator device is unavailable: {device}")

    # ModelScope normalizes every CUDA string to its generic ``gpu`` device,
    # which can silently place all spawned evaluators on logical cuda:0.  A
    # CPU construction followed by an explicit move avoids that normalization.
    speaker = _common.ModelScopeERes2NetEvaluator(speaker_config, device="cpu")
    _move_modelscope_pipeline_to_device(speaker._pipeline, device)

    resolved = _common.resolve_hf_model(dict(speechbert_config))
    speechbertscore = _common.SpeechBERTScoreEvaluator(
        resolved,
        device=str(device),
        layer=int(speechbert_config.get("layer", 14)),
        reference_cache_size=int(speechbert_config.get("reference_cache_size", 64)),
    )
    evaluators = _common.FlowSEGRPOCompositeEvaluators(speaker, speechbertscore)
    return evaluators, evaluators.fingerprint()


def load_flowse_grpo_composite_evaluators(
    config: Mapping,
    *,
    device_override: str | None = None,
) -> tuple[_common.FlowSEGRPOCompositeEvaluators, dict]:
    """Compatibility entry point used by the public AF reward API."""

    if device_override is None:
        return _common.load_flowse_grpo_composite_evaluators(config)
    return load_composite_reward_evaluators(
        config,
        device_override=device_override,
    )


