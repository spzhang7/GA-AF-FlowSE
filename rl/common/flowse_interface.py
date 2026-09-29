"""Method-neutral fixed-latent adapter for the released FlowSE model.

The upstream ``CFM.sample`` always creates ``torch.randn_like(cond)`` internally.
Gate A needs common random numbers, so this isolated adapter performs the same
Euler integration while accepting explicit latent seeds.  It does not patch or
subclass the official model.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

# FlowSE keeps its original top-level ``model``/``loader`` imports.  Add the
# vendored baseline directory explicitly instead of exposing those names at
# the repository root.
_FLOWSE_ROOT = Path(__file__).resolve().parents[2] / "flowse"
if str(_FLOWSE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOWSE_ROOT))

import infer as upstream_infer  # noqa: E402
from model import CFM, DiT  # noqa: E402
from model.model_utils import (  # noqa: E402
    convert_text,
    get_tokenizer,
    list_str_to_idx,
    list_str_to_tensor,
)

from .conditioning import ConditioningProtocol  # noqa: E402
from .flow_matching import euler_terminal, euler_trajectory  # noqa: E402
from .normalization import rms_then_peak_safe  # noqa: E402


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return sha256_array(tensor.detach().float().cpu().numpy())


@dataclass(frozen=True)
class GeneratedEndpoint:
    latent_seed: int
    terminal_mel: torch.Tensor
    raw_waveform: np.ndarray
    normalized_waveform: np.ndarray
    terminal_mel_sha256: str
    raw_waveform_sha256: str
    normalized_waveform_sha256: str
    rms_normalization_gain: float
    peak_safety_gain: float
    peak_before_safety: float
    peak_limited: bool


@dataclass
class FlowSEBundle:
    model: CFM
    vocoder: torch.nn.Module
    device: torch.device
    tokenizer_name: str
    input_sample_rate: int
    model_sample_rate: int
    output_sample_rate: int
    checkpoint_path: Path
    checkpoint_sha256: str | None
    vocoder_model_path: Path
    vocoder_sha256: str | None

    def load_condition(self, path: str | Path) -> torch.Tensor:
        """Follow upstream's source->mix_fs->model_fs resampling path."""
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio[:, 0]
        if source_rate != self.input_sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=source_rate,
                target_sr=self.input_sample_rate,
            )
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None].to(
            self.device
        )
        if self.input_sample_rate != self.model_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=self.input_sample_rate,
                new_freq=self.model_sample_rate,
            )
        return waveform

    def _tokenize(self, transcript: str | list[str], batch: int) -> torch.Tensor:
        texts = [transcript] * batch
        if self.model.vocab_char_map is not None:
            return list_str_to_idx(texts, self.model.vocab_char_map).to(self.device)
        return list_str_to_tensor(texts).to(self.device)

    def prepare_policy_text(
        self, transcript: str, conditioning: ConditioningProtocol
    ) -> str | list[str]:
        raw_policy_text = conditioning.policy_text(transcript)
        if not conditioning.use_text:
            return raw_policy_text
        return convert_text(raw_policy_text, self.tokenizer_name)

    def _fixed_latents(
        self, shape: tuple[int, ...], seeds: list[int], dtype: torch.dtype
    ) -> torch.Tensor:
        latents = []
        for seed in seeds:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            latents.append(
                torch.randn(
                    shape,
                    generator=generator,
                    device=self.device,
                    dtype=dtype,
                )
            )
        return torch.stack(latents, dim=0)

    @torch.inference_mode()
    def sample_fixed_latents(
        self,
        condition_waveform: torch.Tensor,
        transcript: str,
        seeds: list[int],
        *,
        nfe: int,
        cfg_strength: float = 0.0,
        conditioning: ConditioningProtocol,
    ) -> torch.Tensor:
        """Return terminal mels ``[group, frames, channels]`` using Euler."""
        if nfe < 1:
            raise ValueError("nfe must be positive")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("latent seeds must be non-empty and unique")
        if cfg_strength != 0.0:
            raise ValueError(
                "this experiment is preregistered with cfg_strength=0.0; "
                "change the protocol before enabling CFG"
            )
        self.model.eval()
        condition = self.model.mel_spec(condition_waveform).permute(0, 2, 1)
        condition = condition.to(next(self.model.parameters()).dtype)
        condition = condition.expand(len(seeds), -1, -1).contiguous()
        state = self._fixed_latents(
            tuple(condition.shape[1:]), seeds, condition.dtype
        )

        return self.sample_from_latents(
            condition,
            transcript,
            state,
            nfe=nfe,
            cfg_strength=cfg_strength,
            conditioning=conditioning,
        )

    @torch.inference_mode()
    def sample_from_latents(
        self,
        condition_mel: torch.Tensor,
        transcript: str,
        latents: torch.Tensor,
        *,
        nfe: int,
        cfg_strength: float,
        conditioning: ConditioningProtocol,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Integrate explicit latent tensors under a frozen conditioning mode."""
        conditioning.validate()
        if cfg_strength != 0.0:
            raise ValueError("Gate A is preregistered with cfg_strength=0.0")
        if condition_mel.shape != latents.shape:
            raise ValueError("condition mel and latent tensors must have equal shape")
        text = self._tokenize(
            self.prepare_policy_text(transcript, conditioning),
            condition_mel.shape[0],
        )

        def vector_field(time, state):
            return self.model.transformer(
                x=state,
                cond=condition_mel,
                text=text,
                time=time,
                mask=None,
                drop_audio_cond=False,
                drop_text=conditioning.drop_text,
            )

        integrator = euler_trajectory if return_trajectory else euler_terminal
        return integrator(vector_field, latents, nfe)

    @torch.inference_mode()
    def decode_group(
        self,
        terminal_mels: torch.Tensor,
        seeds: list[int],
        *,
        target_dbfs: float = -25.0,
        peak_ceiling: float = 0.99,
    ) -> list[GeneratedEndpoint]:
        """Decode, resample, then apply the frozen v2 RMS/peak-safe protocol."""
        if terminal_mels.shape[0] != len(seeds):
            raise ValueError("mel batch and seeds differ")
        decoded = self.vocoder.decode(terminal_mels.transpose(-1, -2).float())
        decoded = decoded.detach().float().cpu().numpy()
        if decoded.ndim == 1:
            decoded = decoded[None]
        endpoints = []
        for index, seed in enumerate(seeds):
            raw = np.asarray(decoded[index]).squeeze().astype(np.float32)
            output_rate_waveform = raw
            if self.model_sample_rate != self.output_sample_rate:
                output_rate_waveform = librosa.resample(
                    output_rate_waveform,
                    orig_sr=self.model_sample_rate,
                    target_sr=self.output_sample_rate,
                ).astype(np.float32)
            normalization = rms_then_peak_safe(
                output_rate_waveform,
                target_dbfs=target_dbfs,
                peak_ceiling=peak_ceiling,
            )
            normalized = normalization.waveform
            mel = terminal_mels[index].detach().float().cpu()
            endpoints.append(
                GeneratedEndpoint(
                    latent_seed=int(seed),
                    terminal_mel=mel,
                    raw_waveform=raw,
                    normalized_waveform=normalized,
                    terminal_mel_sha256=sha256_tensor(mel),
                    raw_waveform_sha256=sha256_array(raw),
                    normalized_waveform_sha256=sha256_array(normalized),
                    rms_normalization_gain=normalization.rms_gain,
                    peak_safety_gain=normalization.peak_safety_gain,
                    peak_before_safety=normalization.peak_before_safety,
                    peak_limited=normalization.peak_limited,
                )
            )
        return endpoints

    def generate_group(
        self,
        noisy_path: str | Path,
        transcript: str,
        seeds: list[int],
        *,
        nfe: int,
        cfg_strength: float = 0.0,
        conditioning: ConditioningProtocol,
        target_dbfs: float = -25.0,
        peak_ceiling: float = 0.99,
    ) -> list[GeneratedEndpoint]:
        condition = self.load_condition(noisy_path)
        terminal = self.sample_fixed_latents(
            condition,
            transcript,
            seeds,
            nfe=nfe,
            cfg_strength=cfg_strength,
            conditioning=conditioning,
        )
        return self.decode_group(
            terminal,
            seeds,
            target_dbfs=target_dbfs,
            peak_ceiling=peak_ceiling,
        )


def load_flowse_bundle(
    config_path: str | Path,
    *,
    deterministic: bool = True,
    compute_artifact_hashes: bool = True,
) -> FlowSEBundle:
    """Load the released FlowSE checkpoint and local Vocos from an infer config."""
    config_path = Path(config_path)
    root = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    infer_conf = root["infer"]
    model_conf = infer_conf["nnet_conf"]
    device = torch.device(
        "cuda"
        if infer_conf["test"].get("use_cuda", True) and torch.cuda.is_available()
        else "cpu"
    )
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer_name = root["model"]["tokenizer"]
    vocabulary, vocab_size = get_tokenizer(
        root["model"]["tokenizer_path"], tokenizer_name
    )
    model = CFM(
        transformer=DiT(
            **model_conf["arch"],
            text_num_embeds=vocab_size,
            mel_dim=model_conf["mel_spec"]["n_mel_channels"],
        ),
        mel_spec_kwargs=model_conf["mel_spec"],
        vocab_char_map=vocabulary,
    ).eval().to(device)

    checkpoint_path = (
        Path(infer_conf["test"]["checkpoint"])
        / infer_conf["test"]["pt_name"]
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    vocoder_path = Path(model_conf["vocoder"]["local_path"])
    vocoder_model_path = vocoder_path / "pytorch_model.bin"
    vocoder = upstream_infer.load_vocoder(
        vocoder_name=model_conf["mel_spec"]["mel_spec_type"],
        is_local=model_conf["vocoder"]["is_local"],
        local_path=str(vocoder_path),
        device=device,
    )
    return FlowSEBundle(
        model=model,
        vocoder=vocoder,
        device=device,
        tokenizer_name=tokenizer_name,
        input_sample_rate=int(infer_conf["datareader"]["mix_fs"]),
        model_sample_rate=int(model_conf["mel_spec"]["target_sample_rate"]),
        output_sample_rate=int(infer_conf["save"]["fs"]),
        checkpoint_path=checkpoint_path,
        # Training already reads the complete checkpoint above.  Hashing it here
        # forces a second full read of a several-hundred-MiB file on every main
        # process and rollout worker startup.  Callers that do not use artifact
        # hashes as an execution gate can disable that redundant I/O.
        checkpoint_sha256=(
            sha256_file(checkpoint_path) if compute_artifact_hashes else None
        ),
        vocoder_model_path=vocoder_model_path,
        vocoder_sha256=(
            sha256_file(vocoder_model_path) if compute_artifact_hashes else None
        ),
    )
