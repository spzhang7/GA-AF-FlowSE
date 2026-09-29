"""Method-neutral public FlowSE-GRPO composite reward evaluators.

The FlowSE-GRPO paper specifies an ERes2Net speaker cosine and the
SpeechBERTScore precision, but it does not publish immutable evaluator
checkpoints.  This module therefore makes the public evaluator choices
explicit instead of calling the result an exact author-code reproduction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np

from .evaluators import (
    ResolvedHFModel,
    package_version,
    resolve_hf_model,
    sha256_tree,
)


FLOWSE_GRPO_COMPONENTS = ("dnsmos", "speaker", "speechbertscore")
FLOWSE_GRPO_WEIGHTS = {
    "dnsmos": 0.6,
    "speaker": 1.0,
    "speechbertscore": 1.0,
}


def evaluator_fingerprint_sha256(fingerprint: Mapping) -> str:
    """Return the stable identity used to bind calibration rows to evaluators."""

    encoded = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_source_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def population_std(values: list[float]) -> float:
    """Population standard deviation used by the paper-aligned calibration."""

    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("cannot calibrate an empty reward component")
    if not np.all(np.isfinite(array)):
        raise ValueError("reward calibration contains a non-finite value")
    return float(array.std(ddof=0))


def composite_reward(
    raw_components: Mapping[str, float],
    *,
    component_stds: Mapping[str, float],
    weights: Mapping[str, float] = FLOWSE_GRPO_WEIGHTS,
) -> dict:
    """Compute the three-component reward and retain every contribution."""

    if set(raw_components) != set(FLOWSE_GRPO_COMPONENTS):
        raise ValueError("raw components must be DNSMOS, speaker, and SpeechBERTScore")
    if set(component_stds) != set(FLOWSE_GRPO_COMPONENTS):
        raise ValueError("component stds must match the three reward components")
    if set(weights) != set(FLOWSE_GRPO_COMPONENTS):
        raise ValueError("weights must match the three reward components")
    parsed_raw = {name: float(raw_components[name]) for name in FLOWSE_GRPO_COMPONENTS}
    parsed_stds = {name: float(component_stds[name]) for name in FLOWSE_GRPO_COMPONENTS}
    parsed_weights = {name: float(weights[name]) for name in FLOWSE_GRPO_COMPONENTS}
    if any(not math.isfinite(value) for value in parsed_raw.values()):
        raise ValueError("raw reward component must be finite")
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed_stds.values()):
        raise ValueError("reward component standard deviations must be positive")
    if any(not math.isfinite(value) or value < 0.0 for value in parsed_weights.values()):
        raise ValueError("reward weights must be finite and non-negative")
    normalized = {
        name: parsed_raw[name] / parsed_stds[name] for name in FLOWSE_GRPO_COMPONENTS
    }
    weighted = {
        name: parsed_weights[name] * normalized[name]
        for name in FLOWSE_GRPO_COMPONENTS
    }
    return {
        "reward": float(sum(weighted.values())),
        "raw_components": parsed_raw,
        "normalized_components": normalized,
        "weighted_components": weighted,
    }


class ModelScopeERes2NetEvaluator:
    """Cosine similarity from the public ModelScope ERes2Net pipeline."""

    def __init__(self, config: Mapping, *, device: str):
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as exc:  # pragma: no cover - server dependency
            raise RuntimeError(
                "ERes2Net evaluation requires modelscope; install it in the GPU environment"
            ) from exc

        local_model_dir = config.get("local_model_dir")
        if local_model_dir and Path(str(local_model_dir)).is_dir():
            local_snapshot = Path(str(local_model_dir)).resolve()
            model = str(local_snapshot)
            model_source_type = "local_directory"
            model_source_sha256 = sha256_tree(local_snapshot)
        else:
            model = str(config["model_id"])
            model_source_type = "modelscope_model_id"
            model_source_sha256 = None
        revision = str(config["revision"])
        pipeline_device = "gpu" if str(device).startswith("cuda") else "cpu"
        self._pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=model,
            model_revision=revision,
            device=pipeline_device,
        )
        self.model_id = str(config["model_id"])
        self.revision = revision
        self.model_source = model
        self.model_source_type = model_source_type
        self.model_source_sha256 = model_source_sha256

    @staticmethod
    def _cosine_from_embeddings(result: Mapping) -> float | None:
        embeddings = result.get("embs")
        if embeddings is None:
            return None
        array = np.asarray(embeddings, dtype=np.float64)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        if array.ndim != 2 or array.shape[0] != 2:
            return None
        first, second = array
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 0.0:
            raise ValueError("ERes2Net returned a zero-norm embedding")
        return float(np.dot(first, second) / denominator)

    def __call__(self, reference_path: str | Path, estimate_path: str | Path) -> float:
        result = self._pipeline(
            [str(reference_path), str(estimate_path)], output_emb=True
        )
        if not isinstance(result, Mapping):
            raise TypeError("unexpected ModelScope speaker-verification output")
        cosine = self._cosine_from_embeddings(result)
        if cosine is None:
            for key in ("score", "scores"):
                if key in result:
                    value = np.asarray(result[key], dtype=np.float64).reshape(-1)
                    if value.size == 1:
                        cosine = float(value[0])
                        break
        if cosine is None or not math.isfinite(cosine):
            raise ValueError("ModelScope ERes2Net output contains no finite similarity")
        return cosine

    def fingerprint(self) -> dict:
        fingerprint = {
            "backend": "modelscope_speaker_verification_eres2net",
            "implementation": "explicit_pair_embedding_cosine_v1",
            "model_id": self.model_id,
            "revision": self.revision,
            "model_source": self.model_source,
            "model_source_type": self.model_source_type,
            "modelscope_version": package_version("modelscope"),
            "numpy_version": package_version("numpy"),
        }
        if self.model_source_sha256 is not None:
            fingerprint["model_source_sha256"] = self.model_source_sha256
        return fingerprint


class SpeechBERTScoreEvaluator:
    """SpeechBERTScore precision using the official WavLM-large layer-14 setting."""

    def __init__(
        self,
        resolved: ResolvedHFModel,
        *,
        device: str,
        layer: int = 14,
        reference_cache_size: int = 64,
    ):
        import torch
        from transformers import WavLMModel

        if layer < 0:
            raise ValueError("SpeechBERTScore layer must be non-negative")
        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = WavLMModel.from_pretrained(
            resolved.snapshot_path, local_files_only=True
        ).eval().to(self._device)
        self._layer = int(layer)
        self._reference_cache_size = int(reference_cache_size)
        if self._reference_cache_size < 1:
            raise ValueError("reference_cache_size must be positive")
        self._reference_cache: OrderedDict[str, object] = OrderedDict()
        self._resolved = resolved

    def _features(self, path: str | Path):
        from .metrics import load_mono

        waveform = load_mono(path, 16000).astype(np.float32)
        tensor = self._torch.from_numpy(waveform).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            hidden_states = self._model(
                tensor, output_hidden_states=True
            ).hidden_states
            if self._layer >= len(hidden_states):
                raise ValueError(
                    f"SpeechBERTScore layer {self._layer} is unavailable; "
                    f"model returned {len(hidden_states)} hidden states"
                )
            features = hidden_states[self._layer][0]
            return self._torch.nn.functional.normalize(features, dim=1)

    def _reference_features(self, path: str | Path):
        key = str(Path(path).resolve())
        cached = self._reference_cache.pop(key, None)
        if cached is None:
            cached = self._features(path).detach().cpu()
        self._reference_cache[key] = cached
        while len(self._reference_cache) > self._reference_cache_size:
            self._reference_cache.popitem(last=False)
        return cached.to(self._device)

    def __call__(self, reference_path: str | Path, estimate_path: str | Path) -> float:
        reference = self._reference_features(reference_path)
        generated = self._features(estimate_path)
        with self._torch.inference_mode():
            precision = self.precision_from_normalized_features(generated, reference)
        value = float(precision.item())
        if not math.isfinite(value):
            raise ValueError("SpeechBERTScore precision is not finite")
        return value

    @staticmethod
    def precision_from_normalized_features(generated, reference):
        """SpeechBERTScore precision: each generated frame selects a reference frame."""

        similarity = generated @ reference.transpose(0, 1)
        return similarity.max(dim=1).values.mean()

    def fingerprint(self) -> dict:
        return {
            "backend": "discrete_speech_metrics_speechbertscore_precision",
            "implementation": "wavlm_hidden_state_cosine_precision_v1",
            "model": self._resolved.fingerprint(),
            "layer": self._layer,
            "sample_rate": 16000,
            "transformers_version": package_version("transformers"),
            "torch_version": package_version("torch"),
        }


class FlowSEGRPOCompositeEvaluators:
    """The two reference-aware components of the FlowSE-GRPO reward."""

    def __init__(
        self,
        speaker: ModelScopeERes2NetEvaluator,
        speechbertscore: SpeechBERTScoreEvaluator,
    ) -> None:
        self.speaker = speaker
        self.speechbertscore = speechbertscore

    def score(self, reference_path: str | Path, estimate_path: str | Path) -> dict:
        return {
            "eres2net_speaker_similarity": float(
                self.speaker(reference_path, estimate_path)
            ),
            "speechbertscore": float(
                self.speechbertscore(reference_path, estimate_path)
            ),
        }

    def fingerprint(self) -> dict:
        return {
            "schema_version": 1,
            "implementation_source_sha256": _implementation_source_sha256(),
            "speaker": self.speaker.fingerprint(),
            "speechbertscore": self.speechbertscore.fingerprint(),
        }


def load_flowse_grpo_composite_evaluators(
    config: Mapping,
) -> tuple[FlowSEGRPOCompositeEvaluators, dict]:
    """Load the explicitly configured public reward evaluators."""

    evaluator_config = config.get("composite_reward_evaluators")
    if not isinstance(evaluator_config, Mapping):
        raise ValueError(
            "FlowSE-GRPO composite reward requires composite_reward_evaluators"
        )
    device = str(evaluator_config.get("device", "cuda"))
    speaker_config = evaluator_config.get("speaker")
    speechbert_config = evaluator_config.get("speechbertscore")
    if not isinstance(speaker_config, Mapping) or not isinstance(
        speechbert_config, Mapping
    ):
        raise ValueError("composite evaluator config requires speaker and speechbertscore")
    resolved = resolve_hf_model(dict(speechbert_config))
    evaluators = FlowSEGRPOCompositeEvaluators(
        ModelScopeERes2NetEvaluator(speaker_config, device=device),
        SpeechBERTScoreEvaluator(
            resolved,
            device=device,
            layer=int(speechbert_config.get("layer", 14)),
            reference_cache_size=int(
                speechbert_config.get("reference_cache_size", 64)
            ),
        ),
    )
    return evaluators, evaluators.fingerprint()
