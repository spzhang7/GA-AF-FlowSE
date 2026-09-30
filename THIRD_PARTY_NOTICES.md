# Third-party notices

This file separates third-party components from the GA-AF code authored
for this repository. The repository maintainers have obtained permission to use
and redistribute the FlowSE source listed below. Any final public release
should retain the corresponding written record and update this notice with its
exact terms.

## FlowSE

- Repository: <https://github.com/honee-w/flowse>
- Fixed upstream commit: `cfb81f171d804689bf7607afd0d12a6ee89de547`
- Authors: Ziqian Wang, Zikai Liu, Xinfa Zhu, Yike Zhu, Mingshuai Liu, Jun
  Chen, Longshuai Xiao, Chao Weng, and Lei Xie.
- Source handling: the selected upstream files are checked in under `flowse/`
  and are marked with the fixed commit in `flowse/UPSTREAM_SOURCE.json`.
  `flowse/patches/0001-ga-af-flowse-compatibility.patch` records the local
  compatibility changes. `rl/` contains the self-authored method code.
- The patch changes imports, padding-mask handling, dtype/API compatibility,
  and path/entry-point behavior needed by the GA-AF adapter. It does not turn
  FlowSE into GA-AF-authored code.

Suggested citation:

```bibtex
@misc{wang2025flowseefficienthighqualityspeech,
  title={FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching},
  author={Ziqian Wang and Zikai Liu and Xinfa Zhu and Yike Zhu and Mingshuai Liu and Jun Chen and Longshuai Xiao and Chao Weng and Lei Xie},
  year={2025},
  eprint={2505.19476},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2505.19476}
}
```

## Vocos and evaluation models

The Vocos mel-vocoder, DNSMOS, ERes2Net speaker verifier, WavLM/SpeechBERTScore
and any other evaluator used with the training scripts are external
dependencies. This repository contains configuration and adapter code only;
their weights are not redistributed here. Users must download them from their
respective official sources and follow each model's license and terms.

The optional BigVGAN branch in `flowse/infer.py` retains FlowSE's upstream
external-checkout import convention. No BigVGAN source or `third_party/`
directory is included in this repository; the published templates use Vocos.

## Datasets

LibriTTS, DNS Challenge, VoiceBank+DEMAND, WHAM/DEMAND, RIR collections and
other speech/noise corpora are not included. Download them from the official
sources, follow their terms, and generate local manifests with the preparation
scripts.

## Self-authored code

The self-authored GA-AF, AF, GRPO adapter, reward, and utility code is
released under the MIT License in the repository root. This license does not
supersede the separate terms for FlowSE, Vocos, DNSMOS, ERes2Net, WavLM, or
any dataset and model weights downloaded from their respective sources.
