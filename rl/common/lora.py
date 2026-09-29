"""Method-neutral, auditable LoRA primitives for controlled experiments.

This module intentionally does not depend on PEFT and never modifies upstream
FlowSE source files.  Gate B injects adapters only after the released checkpoint
has been loaded, freezes every upstream parameter, and snapshots only the LoRA
tensors when constructing virtual plus/minus/random directions.
"""

from __future__ import annotations

import contextlib
import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus one trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must lie in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha / rank)
        self.dropout = nn.Dropout(float(dropout))
        self.enabled = True

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.base.in_features,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.base.out_features,
                self.rank,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.enabled:
            return output
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return output + residual * self.scaling


@dataclass(frozen=True)
class InjectionReport:
    module_names: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int


def _parent_and_child(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, child: str, module: nn.Module) -> None:
    if child.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child)] = module
    else:
        setattr(parent, child, module)


def inject_lora(
    root: nn.Module,
    *,
    target_patterns: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    expected_modules: int | None = None,
) -> InjectionReport:
    """Freeze ``root`` and replace every regex-selected linear with LoRA."""

    if not target_patterns:
        raise ValueError("at least one LoRA target pattern is required")
    compiled = [re.compile(pattern) for pattern in target_patterns]
    matches = [
        name
        for name, module in root.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and any(pattern.fullmatch(name) for pattern in compiled)
    ]
    if not matches:
        raise ValueError("LoRA target patterns matched no linear modules")
    if len(matches) != len(set(matches)):
        raise AssertionError("duplicate LoRA module match")
    if expected_modules is not None and len(matches) != int(expected_modules):
        raise ValueError(
            f"LoRA target count {len(matches)} != expected {expected_modules}"
        )

    for parameter in root.parameters():
        parameter.requires_grad_(False)
    for name in matches:
        parent, child = _parent_and_child(root, name)
        base = parent[int(child)] if child.isdigit() else getattr(parent, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target changed during injection: {name}")
        _set_child(
            parent,
            child,
            LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
        )

    trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in root.parameters() if not p.requires_grad)
    if trainable <= 0:
        raise AssertionError("LoRA injection produced no trainable parameters")
    return InjectionReport(tuple(matches), int(trainable), int(frozen))


def iter_lora_modules(root: nn.Module) -> Iterator[LoRALinear]:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield module


def named_lora_parameters(root: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in root.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            yield name, parameter


def lora_parameters(root: nn.Module) -> list[nn.Parameter]:
    return [parameter for _, parameter in named_lora_parameters(root)]


@contextlib.contextmanager
def lora_enabled(root: nn.Module, enabled: bool) -> Iterator[None]:
    modules = list(iter_lora_modules(root))
    previous = [module.enabled for module in modules]
    for module in modules:
        module.enabled = bool(enabled)
    try:
        yield
    finally:
        for module, value in zip(modules, previous, strict=True):
            module.enabled = value


AdapterState = dict[str, torch.Tensor]


def snapshot_lora(root: nn.Module, *, device: str | torch.device | None = None) -> AdapterState:
    state = {}
    for name, parameter in named_lora_parameters(root):
        value = parameter.detach().clone()
        if device is not None:
            value = value.to(device)
        state[name] = value
    if not state:
        raise ValueError("model contains no LoRA parameters")
    return state


def load_lora(root: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = dict(named_lora_parameters(root))
    if set(current) != set(state):
        missing = sorted(set(current) - set(state))
        extra = sorted(set(state) - set(current))
        raise ValueError(f"LoRA state mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"LoRA shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@contextlib.contextmanager
def temporary_lora_state(root: nn.Module, state: Mapping[str, torch.Tensor]) -> Iterator[None]:
    previous = snapshot_lora(root)
    load_lora(root, state)
    try:
        yield
    finally:
        load_lora(root, previous)


def subtract_states(after: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor]) -> AdapterState:
    if set(after) != set(before):
        raise ValueError("cannot subtract LoRA states with different keys")
    return {name: after[name] - before[name] for name in sorted(before)}


def add_direction(
    base: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    scale: float,
) -> AdapterState:
    if set(base) != set(direction):
        raise ValueError("base and direction keys differ")
    if not math.isfinite(scale):
        raise ValueError("direction scale must be finite")
    return {
        name: base[name] + float(scale) * direction[name]
        for name in sorted(base)
    }


def direction_norm(direction: Mapping[str, torch.Tensor]) -> float:
    squared = sum(
        float(torch.sum(value.detach().double().square()).item())
        for value in direction.values()
    )
    return float(math.sqrt(squared))


def per_tensor_norms(direction: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(torch.linalg.vector_norm(value.detach().double()).item())
        for name, value in direction.items()
    }


def random_direction_like(
    direction: Mapping[str, torch.Tensor], *, seed: int
) -> AdapterState:
    """Create an independent Gaussian direction with each tensor norm matched."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    output = {}
    for name in sorted(direction):
        source = direction[name]
        target_norm = torch.linalg.vector_norm(source.detach().float().cpu())
        if float(target_norm.item()) == 0.0:
            output[name] = torch.zeros_like(source)
            continue
        noise = torch.randn(
            source.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        noise_norm = torch.linalg.vector_norm(noise)
        noise = noise * (target_norm / noise_norm)
        output[name] = noise.to(device=source.device, dtype=source.dtype)
    return output


def rescale_direction(
    direction: Mapping[str, torch.Tensor], target_norm: float
) -> AdapterState:
    current = direction_norm(direction)
    if current <= 0 or target_norm <= 0 or not math.isfinite(target_norm):
        raise ValueError("direction norms must be finite and positive")
    scale = float(target_norm / current)
    return {name: value * scale for name, value in direction.items()}


@dataclass
class RuntimeState:
    adapter: AdapterState
    optimizer: dict
    scheduler: dict | None
    python_rng: object
    numpy_rng: tuple
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None


def capture_runtime_state(
    root: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> RuntimeState:
    return RuntimeState(
        adapter=snapshot_lora(root),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        scheduler=(copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    )


def restore_runtime_state(
    root: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: RuntimeState,
    scheduler=None,
) -> None:
    load_lora(root, state.adapter)
    optimizer.load_state_dict(copy.deepcopy(state.optimizer))
    if (scheduler is None) != (state.scheduler is None):
        raise ValueError("scheduler presence differs from captured state")
    if scheduler is not None:
        scheduler.load_state_dict(copy.deepcopy(state.scheduler))
    random.setstate(state.python_rng)
    np.random.set_state(state.numpy_rng)
    torch.set_rng_state(state.torch_rng)
    if state.cuda_rng is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("captured CUDA RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(state.cuda_rng)
