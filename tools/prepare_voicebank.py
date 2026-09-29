"""Prepare a VoiceBank+DEMAND test manifest and FlowSE inference config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def wav_stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.wav")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datalist/voicebankdemand_test.json"),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config/voicebankdemand.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_out/voicebankdemand"),
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    noisy_dir = root / "noisy_testset_wav"
    clean_dir = root / "clean_testset_wav"
    text_dir = root / "testset_txt"
    for directory in (noisy_dir, clean_dir, text_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    noisy = wav_stems(noisy_dir)
    clean = wav_stems(clean_dir)
    if noisy != clean:
        raise ValueError(
            f"Noisy/clean mismatch: noisy-only={len(noisy-clean)}, "
            f"clean-only={len(clean-noisy)}"
        )

    transcripts = {}
    missing_text = []
    for stem in sorted(noisy):
        text_path = text_dir / f"{stem}.txt"
        if not text_path.is_file():
            missing_text.append(stem)
            continue
        text = " ".join(
            text_path.read_text(encoding="utf-8", errors="replace").split()
        )
        transcripts[stem] = text
    if missing_text:
        raise ValueError(
            f"Missing {len(missing_text)} transcripts, e.g. {missing_text[0]}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(transcripts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["infer"]["datareader"].update(
        {
            "mix_json": str(args.manifest),
            "mix_dir": str(noisy_dir),
            "mix_fs": 16000,
        }
    )
    config["infer"]["save"].update(
        {"dir": str(args.output_dir), "fs": 16000}
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"Prepared {len(transcripts)} paired utterances")
    print(f"Manifest: {args.manifest}")
    print(f"Config: {args.output_config}")
    print(f"Enhanced output: {args.output_dir}")


if __name__ == "__main__":
    main()
