"""Hash-verified LoRA initialization shared through a neutral package."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .lora import load_lora, snapshot_lora


def _atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lora_state_fingerprint(state: Mapping[str, torch.Tensor]) -> dict:
    tensor_hashes = {name: tensor_sha256(state[name]) for name in sorted(state)}
    digest = hashlib.sha256()
    for name, value in tensor_hashes.items():
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return {
        "state_sha256": digest.hexdigest(),
        "tensor_sha256": tensor_hashes,
        "tensor_count": len(tensor_hashes),
    }


def load_shared_lora_snapshot_payload(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    criteria = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("kind") == "shared_flowse_lora_initialization",
        "state_present": isinstance(payload.get("state"), Mapping),
        "module_names": isinstance(payload.get("module_names"), list),
    }
    if not all(criteria.values()):
        raise ValueError(f"invalid shared LoRA snapshot: {criteria}")
    observed = lora_state_fingerprint(payload["state"])
    if observed["state_sha256"] != payload.get("state_sha256"):
        raise ValueError("shared LoRA snapshot state hash is invalid")
    if observed["tensor_sha256"] != payload.get("tensor_sha256"):
        raise ValueError("shared LoRA snapshot tensor hashes are invalid")
    return {
        **payload,
        "state": {
            str(name): value.detach().cpu().clone()
            for name, value in payload["state"].items()
        },
        "file_sha256": sha256_file(source),
    }


def validate_shared_lora_snapshot_spec(config: Mapping, *, formal: bool = False) -> dict:
    specification = config["lora"].get("shared_initial_snapshot")
    if not isinstance(specification, Mapping) or not specification.get("path"):
        raise ValueError("shared_initial_snapshot.path is required")
    create_if_missing = bool(specification.get("create_if_missing", False))
    expected = specification.get("expected_state_sha256")
    if formal and (create_if_missing or not expected):
        raise ValueError(
            "formal paired training requires an existing shared LoRA snapshot hash"
        )
    if expected is not None and (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected.lower())
    ):
        raise ValueError("expected_state_sha256 must be null or a SHA-256 hex string")
    return {
        "path": str(specification["path"]),
        "create_if_missing": create_if_missing,
        "expected_state_sha256": expected,
    }


def prepare_or_load_shared_lora_snapshot(
    transformer,
    *,
    config: Mapping,
    module_names: Sequence[str],
) -> dict:
    specification = validate_shared_lora_snapshot_spec(config)
    path = Path(specification["path"])
    if not path.is_file():
        if not specification["create_if_missing"]:
            raise FileNotFoundError(
                f"shared paired-seed LoRA snapshot is required: {path}"
            )
        state = snapshot_lora(transformer, device="cpu")
        fingerprint = lora_state_fingerprint(state)
        payload = {
            "schema_version": 1,
            "kind": "shared_flowse_lora_initialization",
            "training_seed": int(config["run"]["seed"]),
            "initialization_seed": int(config["lora"]["initialization_seed"]),
            "rank": int(config["lora"]["rank"]),
            "alpha": float(config["lora"]["alpha"]),
            "module_names": list(module_names),
            "state": state,
            **fingerprint,
        }
        _atomic_torch_save(path, payload)
    payload = load_shared_lora_snapshot_payload(path)
    criteria = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("kind") == "shared_flowse_lora_initialization",
        "training_seed": int(payload.get("training_seed", -1))
        == int(config["run"]["seed"]),
        "initialization_seed": int(payload.get("initialization_seed", -1))
        == int(config["lora"]["initialization_seed"]),
        "rank": int(payload.get("rank", -1)) == int(config["lora"]["rank"]),
        "alpha": math.isclose(
            float(payload.get("alpha", float("nan"))),
            float(config["lora"]["alpha"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "module_names": list(payload.get("module_names", [])) == list(module_names),
        "state_present": isinstance(payload.get("state"), Mapping),
    }
    if not all(criteria.values()):
        raise ValueError(f"shared LoRA snapshot metadata mismatch: {criteria}")
    observed = lora_state_fingerprint(payload["state"])
    expected = specification["expected_state_sha256"]
    if expected is not None and expected != observed["state_sha256"]:
        raise ValueError("shared LoRA snapshot differs from frozen expected hash")
    load_lora(transformer, payload["state"])
    loaded = lora_state_fingerprint(snapshot_lora(transformer, device="cpu"))
    if loaded != observed:
        raise RuntimeError("loaded LoRA state differs from shared snapshot")
    return {
        "path": str(path),
        "file_sha256": payload["file_sha256"],
        "training_seed": int(payload["training_seed"]),
        "rank": int(payload["rank"]),
        "alpha": float(payload["alpha"]),
        **observed,
    }
