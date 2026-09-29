"""Freeze the runnable GRPO scale-up config against the shared LoRA snapshot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from rl.common.shared_initialization import (
    load_shared_lora_snapshot_payload,
)

from rl.grpo.trainer import validate_grpo_config


DEFAULT_TEMPLATE = Path(
    "configs/grpo/"
    "grpo_voicebank_controlled_cfg0_4gpu_5000update_scaleup.template.yaml"
)
DEFAULT_OUTPUT = Path(
    "configs/grpo/"
    "grpo_voicebank_controlled_cfg0_4gpu_5000update_scaleup.yaml"
)


def freeze_config(*, template: Path, output: Path, snapshot: Path) -> dict:
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen GRPO config: {output}"
        )
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    payload = load_shared_lora_snapshot_payload(snapshot)
    checks = {
        "training_seed": int(payload["training_seed"]) == int(config["run"]["seed"]),
        "initialization_seed": int(payload["initialization_seed"])
        == int(config["lora"]["initialization_seed"]),
        "rank": int(payload["rank"]) == int(config["lora"]["rank"]),
        "alpha": float(payload["alpha"]) == float(config["lora"]["alpha"]),
    }
    if not all(checks.values()):
        raise ValueError(f"shared LoRA snapshot differs from GRPO template: {checks}")
    specification = config["lora"]["shared_initial_snapshot"]
    if Path(specification["path"]).resolve() != snapshot.resolve():
        raise ValueError(
            "template shared LoRA path differs from --shared-lora-snapshot"
        )
    specification["create_if_missing"] = False
    specification["expected_state_sha256"] = str(payload["state_sha256"])
    config["comparison"]["shared_initial_lora"] = {
        "path": str(specification["path"]),
        "state_sha256": str(payload["state_sha256"]),
        "file_sha256": str(payload["file_sha256"]),
        "tensor_count": int(payload["tensor_count"]),
    }
    validate_grpo_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the four-GPU 5000-update GRPO scale-up config"
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--shared-lora-snapshot",
        type=Path,
        default=Path("artifacts/shared_initial_lora/seed_260521.pt"),
    )
    args = parser.parse_args()
    config = freeze_config(
        template=args.template,
        output=args.output,
        snapshot=args.shared_lora_snapshot,
    )
    print("GRPO-FORMAL-CONFIG-FROZEN")
    print(f"Config: {args.output}")
    print(
        "Shared LoRA state SHA256: "
        f"{config['lora']['shared_initial_snapshot']['expected_state_sha256']}"
    )
    print(
        "Physical GPU order: "
        + ",".join(
            str(value)
            for value in config["resources"]["expected_cuda_visible_devices"]
        )
    )


if __name__ == "__main__":
    main()
