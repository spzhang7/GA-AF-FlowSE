"""Call Microsoft's official DNSMOS implementation as a reusable module."""

from __future__ import annotations

import csv
import json
import math
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm


METRICS = ("SIG", "BAK", "OVRL", "P808_MOS")


def load_official_module(script_path: Path):
    source = script_path.read_text(encoding="utf-8")
    source = source.replace(
        "librosa.resample(aud, input_fs, fs)",
        "librosa.resample(aud, orig_sr=input_fs, target_sr=fs)",
    )
    module = types.ModuleType("flowse_official_dnsmos")
    module.__file__ = str(script_path)
    exec(compile(source, str(script_path), "exec"), module.__dict__)
    return module


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if not row.get("error")]
    result = {
        "total": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
    }
    for metric in METRICS:
        values = np.asarray([row[metric] for row in successful], dtype=np.float64)
        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def evaluate_directory(
    audio_dir: Path,
    official_dir: Path,
    output_prefix: Path,
    workers: int = 8,
    personalized: bool = False,
) -> tuple[list[dict], dict]:
    script_path = official_dir / "dnsmos_local.py"
    p808_model = official_dir / "DNSMOS/model_v8.onnx"
    primary_model = official_dir / (
        "pDNSMOS/sig_bak_ovr.onnx"
        if personalized
        else "DNSMOS/sig_bak_ovr.onnx"
    )
    for path in (script_path, p808_model, primary_model):
        if not path.is_file():
            raise FileNotFoundError(path)

    module = load_official_module(script_path)
    scorer = module.ComputeScore(str(primary_model), str(p808_model))
    files = sorted(audio_dir.rglob("*.wav"))
    if not files:
        raise ValueError(f"No WAV files found in {audio_dir}")

    def score(path: Path) -> dict:
        try:
            row = scorer(str(path), 16000, personalized)
            row["utterance"] = path.stem
            row["error"] = ""
            return row
        except Exception as exc:
            row = {
                "filename": str(path),
                "utterance": path.stem,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for metric in METRICS:
                row[metric] = math.nan
            return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(
            tqdm(
                executor.map(score, files),
                total=len(files),
                desc=f"DNSMOS {audio_dir.name}",
            )
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".summary.json")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"DNSMOS CSV: {csv_path}")
    print(f"DNSMOS summary: {json_path}")
    return rows, summary
