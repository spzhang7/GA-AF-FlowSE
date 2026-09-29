"""Method-neutral black-box speaker-similarity and ASR/WER evaluators."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def sha256_tree(root: Path) -> str:
    """Hash file names and bytes in a resolved Hugging Face snapshot."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"model snapshot is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedHFModel:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    snapshot_path: Path
    snapshot_sha256: str

    def fingerprint(self) -> dict[str, str]:
        return {
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "snapshot_sha256": self.snapshot_sha256,
        }


def resolve_hf_model(config: dict) -> ResolvedHFModel:
    """Resolve a mutable revision to a commit before protocol hashing."""
    repo_id = str(config["repo_id"])
    requested_revision = str(config["revision"])
    local_model_dir = config.get("local_model_dir")
    if local_model_dir:
        local_snapshot = Path(str(local_model_dir)).resolve()
        if local_snapshot.is_dir():
            if len(requested_revision) != 40:
                raise ValueError(
                    f"local evaluator {repo_id} requires a 40-character commit revision"
                )
            return ResolvedHFModel(
                repo_id=repo_id,
                requested_revision=requested_revision,
                resolved_revision=requested_revision,
                snapshot_path=local_snapshot,
                snapshot_sha256=sha256_tree(local_snapshot),
            )
        if bool(config.get("local_files_only", False)):
            raise FileNotFoundError(
                f"local evaluator directory does not exist: {local_snapshot}"
            )

    from huggingface_hub import model_info, snapshot_download

    cache_dir = config.get("cache_dir")
    local_files_only = bool(config.get("local_files_only", False))
    if local_files_only and len(requested_revision) != 40:
        raise ValueError(
            f"offline evaluator {repo_id} requires a 40-character commit revision"
        )
    if len(requested_revision) == 40:
        resolved_revision = requested_revision
    else:
        resolved_revision = str(
            model_info(repo_id, revision=requested_revision).sha
        )
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    ).resolve()
    return ResolvedHFModel(
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        snapshot_path=snapshot,
        snapshot_sha256=sha256_tree(snapshot),
    )


def normalize_wer_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard word-level Levenshtein distance divided by reference words."""
    reference_words = normalize_wer_text(reference).split()
    hypothesis_words = normalize_wer_text(hypothesis).split()
    if not reference_words:
        raise ValueError("WER reference is empty after normalization")
    previous = list(range(len(hypothesis_words) + 1))
    for ref_index, reference_word in enumerate(reference_words, 1):
        current = [ref_index]
        for hyp_index, hypothesis_word in enumerate(hypothesis_words, 1):
            substitution = previous[hyp_index - 1] + (
                reference_word != hypothesis_word
            )
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return float(previous[-1] / len(reference_words))


class SpeakerSimilarityEvaluator:
    """Cosine similarity using a fixed WavLM x-vector checkpoint."""

    def __init__(self, resolved: ResolvedHFModel, device: str):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._processor = AutoFeatureExtractor.from_pretrained(
            resolved.snapshot_path, local_files_only=True
        )
        self._model = WavLMForXVector.from_pretrained(
            resolved.snapshot_path, local_files_only=True
        ).eval().to(self._device)
        self._reference_cache: dict[str, object] = {}

    def _embedding(self, path: str | Path):
        from .metrics import load_mono

        waveform = load_mono(path, 16000).astype(np.float32)
        inputs = self._processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            embedding = self._model(**inputs).embeddings[0]
            embedding = self._torch.nn.functional.normalize(embedding, dim=0)
        return embedding

    def __call__(self, reference_path: str | Path, estimate_path: str | Path) -> float:
        key = str(Path(reference_path).resolve())
        reference = self._reference_cache.get(key)
        if reference is None:
            reference = self._embedding(reference_path)
            self._reference_cache[key] = reference
        estimate = self._embedding(estimate_path)
        return float(self._torch.sum(reference * estimate).item())


class ASRWEREvaluator:
    """Ground-truth transcript WER from a fixed Wav2Vec2 CTC checkpoint."""

    def __init__(self, resolved: ResolvedHFModel, device: str):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(
            resolved.snapshot_path, local_files_only=True
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            resolved.snapshot_path, local_files_only=True
        ).eval().to(self._device)

    def transcribe(self, path: str | Path) -> str:
        from .metrics import load_mono

        waveform = load_mono(path, 16000).astype(np.float32)
        inputs = self._processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self._device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        with self._torch.inference_mode():
            logits = self._model(
                input_values=input_values, attention_mask=attention_mask
            ).logits
        predicted = self._torch.argmax(logits, dim=-1)
        return str(self._processor.batch_decode(predicted)[0])

    def __call__(self, reference_text: str, estimate_path: str | Path) -> tuple[float, str]:
        hypothesis = self.transcribe(estimate_path)
        return word_error_rate(reference_text, hypothesis), hypothesis


class FidelityEvaluators:
    """Speaker/ASR evaluators with optional lazy ASR GPU residency."""

    def __init__(
        self,
        speaker_model: ResolvedHFModel,
        asr_model: ResolvedHFModel,
        *,
        device: str,
        lazy_asr: bool,
    ) -> None:
        self._speaker_model = speaker_model
        self._asr_model = asr_model
        self._device = device
        self._speaker = SpeakerSimilarityEvaluator(speaker_model, device)
        self._asr = None if lazy_asr else ASRWEREvaluator(asr_model, device)

    @property
    def speaker(self) -> SpeakerSimilarityEvaluator:
        return self._speaker

    @property
    def asr(self) -> ASRWEREvaluator:
        if self._asr is None:
            self._asr = ASRWEREvaluator(self._asr_model, self._device)
        return self._asr

    def release_asr(self, *, empty_cuda_cache: bool = True) -> None:
        """Remove the ASR model after evaluation; speaker reward stays resident."""

        if self._asr is None:
            return
        self._asr = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available() and empty_cuda_cache:
            torch.cuda.empty_cache()

    @classmethod
    def load(
        cls,
        speaker_model: ResolvedHFModel,
        asr_model: ResolvedHFModel,
        *,
        device: str,
        lazy_asr: bool = False,
    ) -> "FidelityEvaluators":
        return cls(
            speaker_model,
            asr_model,
            device=device,
            lazy_asr=lazy_asr,
        )


def evaluator_package_fingerprint() -> dict[str, str]:
    return {
        "transformers": package_version("transformers"),
        "huggingface-hub": package_version("huggingface-hub"),
        "torch": package_version("torch"),
    }
