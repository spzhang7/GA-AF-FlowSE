#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-config/train.yaml}"
python infer.py -conf "${config_path}"
