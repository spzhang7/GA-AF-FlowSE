"""Compatibility adapter for adzzuki's text-free FlowSE SFT checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.checkpoint import checkpoint
from x_transformers.x_transformers import RotaryEmbedding

# The vendored FlowSE source intentionally retains its upstream top-level
# imports; scope them to the baseline directory for this adapter.
_FLOWSE_ROOT = Path(__file__).resolve().parents[2] / "flowse"
if str(_FLOWSE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOWSE_ROOT))

import infer as upstream_infer  # noqa: E402
from model import CFM  # noqa: E402
from model.modules import (  # noqa: E402
    AdaLayerNormZero_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
)

from .flow_matching import euler_terminal, euler_trajectory  # noqa: E402
from .flowse_interface import FlowSEBundle, sha256_file  # noqa: E402


class AudioOnlyInputEmbedding(nn.Module):
    """The personal implementation concatenates only x_t and noisy mel."""

    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        *,
        drop_audio_cond: bool,
    ) -> torch.Tensor:
        if drop_audio_cond:
            condition = torch.zeros_like(condition)
        embedded = self.proj(torch.cat((x, condition), dim=-1))
        return self.conv_pos_embed(embedded) + embedded


class AudioOnlyDiT(nn.Module):
    """Exact 22-layer text-free DiT geometry used by FlowSE-GRPOGUARD."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 4,
        mel_dim: int = 100,
        conv_layers: int = 0,
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
        **unused,
    ):
        super().__init__()
        del conv_layers, unused
        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = AudioOnlyInputEmbedding(mel_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations

    @staticmethod
    def _checkpoint_forward(module):
        def forward(*inputs):
            return module(*inputs)

        return forward

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        time: torch.Tensor,
        drop_audio_cond: bool,
        mask: torch.Tensor | None = None,
        **unused,
    ) -> torch.Tensor:
        del unused
        batch, sequence = x.shape[:2]
        if time.ndim == 0:
            time = time.repeat(batch)
        timestep = self.time_embed(time)
        x = self.input_embed(x, cond, drop_audio_cond=drop_audio_cond)
        rope = self.rotary_embed.forward_from_seq_len(sequence)
        residual = x if self.long_skip_connection is not None else None
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = checkpoint(
                    self._checkpoint_forward(block),
                    x,
                    timestep,
                    mask,
                    rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, timestep, mask=mask, rope=rope)
        if residual is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        return self.proj_out(self.norm_out(x, timestep))


class AudioOnlyFlowSEBundle(FlowSEBundle):
    """Reuse the frozen FlowSE I/O protocol with the text-free vector field."""

    @torch.inference_mode()
    def sample_from_latents(
        self,
        condition_mel: torch.Tensor,
        transcript: str,
        latents: torch.Tensor,
        *,
        nfe: int,
        cfg_strength: float,
        conditioning,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        del transcript
        conditioning.validate()
        if conditioning.use_text or not conditioning.drop_text:
            raise ValueError("the personal SFT checkpoint is structurally audio-only")
        if cfg_strength != 0.0:
            raise ValueError("personal SFT comparison is fixed to CFG=0")
        if condition_mel.shape != latents.shape:
            raise ValueError("condition mel and latent tensors must have equal shape")

        def vector_field(time, state):
            return self.model.transformer(
                x=state,
                cond=condition_mel,
                time=time,
                mask=None,
                drop_audio_cond=False,
            )

        integrator = euler_trajectory if return_trajectory else euler_terminal
        return integrator(vector_field, latents, nfe)


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(name.startswith("module.") for name in state):
        return {name.removeprefix("module."): value for name, value in state.items()}
    return state


def load_personal_sft_bundle(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    state_kind: str = "ema",
) -> AudioOnlyFlowSEBundle:
    """Load the SFT-66k EMA while rejecting incompatible checkpoint geometry."""

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    root = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    infer = root["infer"]
    model_config = infer["nnet_conf"]
    device = torch.device(
        "cuda"
        if infer["test"].get("use_cuda", True) and torch.cuda.is_available()
        else "cpu"
    )
    model = CFM(
        transformer=AudioOnlyDiT(
            **model_config["arch"],
            mel_dim=int(model_config["mel_spec"]["n_mel_channels"]),
        ),
        mel_spec_kwargs=model_config["mel_spec"],
    ).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_key = "ema_state_dict" if state_kind == "ema" else "model_state_dict"
    if state_key not in payload:
        raise ValueError(f"checkpoint lacks {state_key}")
    state = _strip_module_prefix(dict(payload[state_key]))
    expected_projection = (int(model_config["arch"]["dim"]), 200)
    projection = state.get("transformer.input_embed.proj.weight")
    if projection is None or tuple(projection.shape) != expected_projection:
        raise ValueError(
            f"not the expected text-free FlowSE geometry: projection="
            f"{None if projection is None else tuple(projection.shape)}"
        )
    if any("text_embed" in name for name in state):
        raise ValueError("personal SFT checkpoint unexpectedly contains text parameters")
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {"transformer.rotary_embed.inv_freq"}
    unexpected_missing = set(result.missing_keys) - allowed_missing
    if unexpected_missing or result.unexpected_keys:
        raise ValueError(
            f"personal SFT state mismatch: missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    if int(payload.get("global_step", -1)) != 66_000:
        raise ValueError(f"expected SFT step 66000, got {payload.get('global_step')}")
    model.eval()

    vocoder_path = Path(model_config["vocoder"]["local_path"])
    vocoder = upstream_infer.load_vocoder(
        vocoder_name=model_config["mel_spec"]["mel_spec_type"],
        is_local=model_config["vocoder"]["is_local"],
        local_path=str(vocoder_path),
        device=device,
    )
    vocoder_model_path = vocoder_path / "pytorch_model.bin"
    return AudioOnlyFlowSEBundle(
        model=model,
        vocoder=vocoder,
        device=device,
        tokenizer_name="none",
        input_sample_rate=int(infer["datareader"]["mix_fs"]),
        model_sample_rate=int(model_config["mel_spec"]["target_sample_rate"]),
        output_sample_rate=int(infer["save"]["fs"]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        vocoder_model_path=vocoder_model_path,
        vocoder_sha256=sha256_file(vocoder_model_path),
    )
