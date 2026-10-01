# Checkpoint layout

Model weights are not committed to Git because of their size and the separate
licenses of the upstream and experiment checkpoints. Download them into the
following paths before running the matching configuration:

```text
checkpoints/
├── flowse/
│   ├── wenetspeech4tts-Premium/best.pt.tar
│   └── libritts_sft20k/checkpoint_step_020000.pt
├── af/af_libritts_step005000.pt
├── gaaf/gaaf_libritts_step005000.pt
├── grpo/grpo_libritts_step005000.pt
└── vocos-mel-24khz/pytorch_model.bin
```

The original FlowSE checkpoint is available from the upstream Hugging Face
release:

```bash
hf download flowse/wenetspeech4tts_Premium.pt.tar \
  --local-dir checkpoints/flowse/wenetspeech4tts-Premium
mv checkpoints/flowse/wenetspeech4tts-Premium/wenetspeech4tts_Premium.pt.tar \
  checkpoints/flowse/wenetspeech4tts-Premium/best.pt.tar
```

Vocos can be downloaded from its upstream Hugging Face repository:

```bash
hf download charactr/vocos-mel-24khz \
  --local-dir checkpoints/vocos-mel-24khz
```

## Project checkpoints

Download the project checkpoints from
[`spzhang7/GA-AF-FlowSE`](https://huggingface.co/spzhang7/GA-AF-FlowSE):

```bash
hf download spzhang7/GA-AF-FlowSE \
  base_sft20k/flowse_sft20k.pt \
  af/af_libritts_step005000.pt \
  gaaf/gaaf_libritts_step005000.pt \
  grpo/grpo_libritts_step005000.pt \
  --local-dir checkpoints
```

The AF, GA-AF, and GRPO files are available for inference or checkpoint
inspection at the paths shown above. The base checkpoint is named according to
the path expected by the public training configurations:

```bash
mkdir -p checkpoints/flowse/libritts_sft20k
ln -sfn ../../base_sft20k/flowse_sft20k.pt \
  checkpoints/flowse/libritts_sft20k/checkpoint_step_020000.pt
```

Do not commit downloaded checkpoint files to the source repository.
