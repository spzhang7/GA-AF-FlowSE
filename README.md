# GA-AF-FlowSE

Reference implementation of Gradient-Aligned AdvantageFlow (GA-AF) for
FlowSE-based speech enhancement. In the code and experiment logs, GA-AF is
also called OGAF. The repository contains the method-side AF/OGAF training,
the controlled OVRL-only AF comparison, evaluation utilities, and a pinned
FlowSE baseline.

This is a source-code release. Training data, FlowSE/metric checkpoints,
calibration artifacts, and experiment outputs are intentionally not stored in
Git. Paths in the public YAML files are relative templates; edit them for the
data and checkpoints available on your machine.

## What is included

- AF and GA-AF/OGAF forward-process training with LoRA and EMA updates.
- A controlled FlowSE-GRPO baseline under `rl/grpo/` for the paper comparison.
- Deterministic ODE rollouts with `L=16` conditions and `K=8` candidates in
  the 5000-step templates (the smoke configuration uses a smaller budget).
- Reward, advantage, protocol, checkpoint, and paired-evaluation utilities.
- A controlled DNSMOS-OVRL-only AF configuration for comparison.
- Unit tests for the method-side math and data-manifest preparation.
- The pinned FlowSE baseline under `flowse/`, with the upstream commit and
  compatibility patch documented in [`docs/FLOWSE_UPSTREAM.md`](docs/FLOWSE_UPSTREAM.md).
- Our method code under `rl/`; this directory contains no upstream FlowSE
  implementation.

## Quick start

Use Python 3.10+ and a PyTorch installation appropriate for your CUDA version.
The single `requirements.txt` file contains runtime, reward/evaluation, test,
and lint dependencies; there are no separate development or evaluation
requirement files.

```bash
python -m pip install -r requirements.txt
python -m pytest -q rl/af/tests rl/grpo/tests
```

For a CUDA 12.1 environment, install the matching PyTorch wheels before the
requirements file if your platform does not select them automatically:

```bash
python -m pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU-only installations can use the ordinary `torch==2.2.0` and
`torchaudio==2.2.0` entries from `requirements.txt`.

PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q rl/af/tests rl/grpo/tests
```

The model implementation under `flowse/` is the verified, pinned FlowSE
source snapshot (with only the repository-local import/layout adaptation
listed in `docs/FLOWSE_UPSTREAM.md`). The reinforcement-learning code never
replaces the FlowSE model; it loads it through the same model, loader, and
inference path as the baseline.

## Running an experiment

1. Prepare LibriTTS/DNS10s and its manifests. The public data builder accepts
   explicit roots, so no machine-specific paths are required:

   ```bash
   python -m rl.common.duration_matched_dataset \
     --libritts-root /path/to/LibriTTS \
     --dns-noise-root /path/to/DNS-Challenge-2021/datasets/wideband/noise_wideband \
     --wham-train-root /path/to/WHAM/wham_noise/tr \
     --wham-valid-root /path/to/WHAM/wham_noise/cv \
     --demand-root /path/to/DEMAND_16k \
     --rir26-root /path/to/OpenSLR26 \
     --rir28-root /path/to/OpenSLR28 \
     --dns2020-root /path/to/DNS-Challenge-2020 \
     --output-dir artifacts/af/manifests/libritts_dns10s
   ```

   This creates the clean/noisy paired audio tree and train/validation/DNS2020
   manifest files expected by the LibriTTS/DNS10s templates. If you already
   have the paired `data/libritts_dns10s` tree, run the builder with the
   existing roots or point the YAML paths at your own manifests.
2. Download the released FlowSE/SFT checkpoint and Vocos weights separately,
   following [`checkpoints/README.md`](checkpoints/README.md), then place them
   under the paths configured by
   `configs/flowse/flowse_libritts_sft20k_wotext.yaml`.
3. Install the reward models (DNSMOS, ModelScope ERes2Net, and WavLM) using
   [`pretrainmodel/README.md`](pretrainmodel/README.md). The composite reward
   uses the DNS10s 8192-condition calibration:

   ```text
   dnsmos          = 0.07032371343108539
   speaker         = 0.13865593039460408
   speechbertscore = 0.09550272361009968
   source_nfe=10, std_ddof=0, dnsmos_divisor=4.0
   ```

   with weights `(0.6, 1.0, 1.0)` for DNSMOS OVRL, ERes2Net, and
   SpeechBERTScore. These values are included in the public YAML templates;
   formal paper configurations additionally verify the calibration report's
   provenance.
4. Edit the `data`, checkpoint, reward-model, and output paths in one of the
   templates, then
   launch the training entry point:

   ```bash
   python scripts/train_af.py --config configs/af/af_smoke.yaml
   ```

For a full 5000-step run, use
`configs/af/af_libritts_dns10s_sft20k_5000step.yaml` for ordinary AF or
`configs/af/gaaf_libritts_dns10s_sft20k_ogaf_0_to_5000.yaml` for GA-AF/OGAF.
The OVRL-only control is in
`configs/af/af_libritts_dns10s_sft20k_ovrl_only_0_to_5000.yaml`.

The two-step AF and GA-AF/OGAF paths are available at
`configs/af/af_smoke.yaml` and `configs/af/gaaf_smoke.yaml`. They use the
same three local composite reward evaluators as the paper configuration.

The controlled LibriTTS/DNS10s GRPO smoke baseline is launched with:

```bash
python scripts/train_grpo.py \
  --config configs/grpo/grpo_libritts_dns10s_4gpu_production_geometry_smoke.yaml
```

For CUDA runs, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) is only needed
for the formal deterministic reproduction configs. It is optional for public
smoke runs; if a platform does not provide deterministic cuBLAS kernels, set
`run.deterministic: false`.

The smoke configs perform only runnable-code checks (configuration schema,
algorithm geometry, finite rewards, and disjoint data manifests). Strict
artifact provenance, calibration fingerprints, fixed comparison paths, and
paper fidelity audits are enabled only by the pilot/formal configurations.

For ordinary inference with the verified FlowSE implementation, use the same
entry point and configuration style as the baseline:

```bash
python flowse/infer.py -conf <your-flowse-config.yaml>
```

The templates are intentionally conservative and may require adapting the
number of GPUs, checkpoint names, and reward-model locations. No checkpoint or
dataset is implied by this repository.

## Third-party baseline and citations

FlowSE is not our original code. Its checked-in source, fixed commit,
compatibility patch, and permission notice are described in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`docs/FLOWSE_UPSTREAM.md`](docs/FLOWSE_UPSTREAM.md). Please cite both this
work and FlowSE when using the repository. The self-authored code is released
under the MIT License; third-party terms remain separate.

## Citation

The citation metadata for this repository is in [`CITATION.cff`](CITATION.cff).

