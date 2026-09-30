# Reward and auxiliary model directory

This directory is intentionally kept free of model weights. Downloaded
checkpoints are ignored by Git; place them under the following exact paths:

```text
pretrainmodel/
├── DNSMOS-official/
├── speech_eres2net_sv_zh-cn_16k-common/
└── wavlm-large/
```

The paths are referenced by the public AF, GA-AF, and GRPO templates.
The required model versions are:

| Directory | Source | Version |
| --- | --- | --- |
| `DNSMOS-official` | [Microsoft DNS Challenge/DNSMOS](https://github.com/microsoft/DNS-Challenge) | official DNSMOS models |
| `speech_eres2net_sv_zh-cn_16k-common` | [ModelScope `iic/speech_eres2net_sv_zh-cn_16k-common`](https://modelscope.cn/models/iic/speech_eres2net_sv_zh-cn_16k-common) | `v1.0.5` |
| `wavlm-large` | [Hugging Face `microsoft/wavlm-large`](https://huggingface.co/microsoft/wavlm-large) | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` |

## Download commands

Install the corresponding command-line clients first (`modelscope` and the
Hugging Face Hub CLI). Run these commands from the repository root:

```bash
modelscope download \
  --model iic/speech_eres2net_sv_zh-cn_16k-common \
  --revision v1.0.5 \
  --local_dir pretrainmodel/speech_eres2net_sv_zh-cn_16k-common

hf download microsoft/wavlm-large \
  --revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c \
  --local-dir pretrainmodel/wavlm-large
```

For DNSMOS, download the official DNSMOS release from the Microsoft
DNS-Challenge repository and copy its `DNSMOS` directory so that the following
files exist:

```text
pretrainmodel/DNSMOS-official/DNSMOS/dnsmos_local.py
pretrainmodel/DNSMOS-official/DNSMOS/model_v8.onnx
pretrainmodel/DNSMOS-official/DNSMOS/sig_bak_ovr.onnx
```

The exact DNSMOS distribution may change independently of this repository;
follow its upstream license and release instructions.

For a local ModelScope cache, a symbolic link is sufficient:

```bash
ln -sfn \
  /path/to/modelscope/iic--speech_eres2net_sv_zh-cn_16k-common/snapshots/v1.0.5 \
  pretrainmodel/speech_eres2net_sv_zh-cn_16k-common
```

For Hugging Face, download the model into the directory named `wavlm-large`
and keep the directory local when `local_files_only: true` is enabled.

Do not commit model weights, cache directories, or machine-specific absolute
paths. The DNSMOS directory should contain its official ONNX files, including
`DNSMOS/model_v8.onnx` and `DNSMOS/sig_bak_ovr.onnx`.
