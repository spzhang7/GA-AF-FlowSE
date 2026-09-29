"""Summarize the CSV produced by Microsoft's dnsmos_local.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("SIG", "BAK", "OVRL", "P808_MOS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"Missing DNSMOS columns: {missing}")

    summary = {"count": len(frame)}
    for metric in METRICS:
        values = frame[metric].dropna()
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    content = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    print(content, end="")
    output = args.output or args.csv.with_suffix(".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Summary JSON: {output}")


if __name__ == "__main__":
    main()
