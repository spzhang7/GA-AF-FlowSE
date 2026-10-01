# Getting started

This is the shortest supported path from a fresh checkout to a full AF,
GA-AF, or GRPO training run. Commands below are run from the repository root.

## 1. Create the environment

Python 3.10 or newer is required. For a CUDA 12.1 machine, install PyTorch
first and then the repository dependencies:

```bash
conda create -n ga-af-flowse python=3.10 -y
conda activate ga-af-flowse
python -m pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU-only development can use the ordinary PyTorch wheels from
`requirements.txt`; full training requires a CUDA GPU.

Check the installation before downloading data:

```bash
python -m pytest -q rl/af/tests rl/grpo/tests
```

## 2. Download the pretrained models

Follow [pretrainmodel/README.md](../pretrainmodel/README.md) for DNSMOS,
ERes2Net, and WavLM. Follow [checkpoints/README.md](../checkpoints/README.md)
for FlowSE and Vocos. The expected local directories are:

```text
pretrainmodel/DNSMOS-official/
pretrainmodel/speech_eres2net_sv_zh-cn_16k-common/
pretrainmodel/wavlm-large/
checkpoints/flowse/wenetspeech4tts-Premium/best.pt.tar
checkpoints/flowse/libritts_sft20k/checkpoint_step_020000.pt
checkpoints/vocos-mel-24khz/pytorch_model.bin
```

Model weights are not committed to Git. Keep them local or download the
published release assets into the paths above.

## 3. Download and construct LibriTTS/DNS10s data

The repository does not redistribute licensed speech, noise, room impulse
responses, or DNS2020 test audio. Download clean LibriTTS speech from
[OpenSLR 60](https://www.openslr.org/60/), download DNS Challenge noise from
the [Microsoft DNS-Challenge repository](https://github.com/microsoft/DNS-Challenge),
and obtain a licensed room-impulse-response (RIR) set. Keep the three source
directories separate, for example:

```text
/datasets/LibriTTS/       # clean speech, recursively organised WAV/FLAC
/datasets/DNS/noise/      # noise recordings
/datasets/RIR/            # room impulse responses
```

Use the supplied [prepare_libritts_dns10s.py](../tools/prepare_libritts_dns10s.py)
script to construct the paired training corpus. It resamples to 16 kHz,
convolves each clean utterance with a deterministic RIR, mixes deterministic
noise at a reproducible SNR, and writes clean/noisy WAV files plus disjoint
training and validation manifests:

```bash
python tools/prepare_libritts_dns10s.py \
  --clean-dir /datasets/LibriTTS \
  --noise-dir /datasets/DNS/noise \
  --rir-dir /datasets/RIR \
  --output-dir data/libritts_dns10s/audio \
  --manifest-dir artifacts/af/manifests/libritts_dns10s \
  --seed 260810 \
  --snr-min-db 0 \
  --snr-max-db 20 \
  --validation-fraction 0.1
```

If you maintain a JSON object mapping generated utterance IDs to transcripts,
pass it with `--transcripts`. Audio-only (`wotext`) configurations can omit
this option; the script writes empty transcript strings while preserving the
required manifest schema. The seed and SNR range are explicit so that a data
build can be reproduced and audited.

The resulting checkout must contain:

The checkout must contain this layout:

```text
data/libritts_dns10s/audio/clean/
data/libritts_dns10s/audio/noisy/
artifacts/af/manifests/libritts_dns10s/
├── libritts_dns10s_train_exposures.json
├── libritts_dns10s_validation.json
└── dns2020_official_test_all.json
```

If a prepared paired corpus already exists elsewhere, symlink it instead of
copying many gigabytes:

```bash
mkdir -p data artifacts/af/manifests
ln -sfn /path/to/AF_LibriTTS_DNS10s/audio data/libritts_dns10s/audio
ln -sfn /path/to/libritts_dns10s_manifests \
  artifacts/af/manifests/libritts_dns10s
```

The manifests map utterance IDs to transcript strings, and clean/noisy files
must be readable at the paths encoded by those IDs. Place the separately
prepared DNS2020 official-test manifest at
`artifacts/af/manifests/libritts_dns10s/dns2020_official_test_all.json`; keep
all official-test audio outside the training split. Do not commit audio or
manifests if their licenses prohibit redistribution.

The repository currently documents and consumes this prepared paired-data
layout; training commands do not silently download data or change the
dataset. This keeps a fresh GitHub checkout reproducible without embedding
machine-specific dataset paths.

## 4. Run the formal training configurations

Ordinary AdvantageFlow and GA-AF use the 5000-step LibriTTS/DNS10s configs:

```bash
python scripts/train_af.py \
  --config configs/af/af_libritts_dns10s_5000step.yaml

python scripts/train_af.py \
  --config configs/af/gaaf_libritts_dns10s_0_to_5000.yaml
```

Controlled GRPO uses four rollout GPUs and 5000 optimizer updates:

```bash
python scripts/train_grpo.py \
  --config configs/grpo/grpo_libritts_dns10s_4gpu_5000update.yaml
```

Set `CUDA_VISIBLE_DEVICES` to the physical GPU order expected by the selected
configuration. The GRPO YAML currently targets four GPUs; the AF YAMLs use
four rollout workers as well.

To continue an interrupted run, pass the checkpoint in that run's ordinary
output directory:

```bash
python scripts/train_af.py \
  --config configs/af/gaaf_libritts_dns10s_0_to_5000.yaml \
  --resume artifacts/af/speech_advantageflow_libritts_dns10s_0_to_5000/checkpoint_latest.pt
```

GRPO resume uses the same pattern with `scripts/train_grpo.py` and the
`checkpoint_latest.pt` under its configured `output_root`.

## 5. Evaluate

After training, use the paired-metric and DNSMOS tools under `tools/`. Keep
the validation and DNS2020 manifests separate from the training manifest;
the training code checks that the splits do not overlap.
