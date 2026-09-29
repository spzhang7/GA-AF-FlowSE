# Checkpoint layout

Model weights are not committed to Git because of their size and the separate
licenses of the upstream and experiment checkpoints. Download them into the
following paths before running the matching configuration:

```text
checkpoints/
├── flowse/
│   ├── wenetspeech4tts-Premium/best.pt.tar
│   └── libritts_sft20k/checkpoint_step_020000.pt
├── af/af_libritts_dns10s_step005000.pt
├── gaaf/gaaf_libritts_dns10s_ogaf_0_to_5000_step005000.pt
├── grpo/grpo_voicebank_controlled_latest.pt
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

The SFT-20k, AF, GA-AF/OGAF, and GRPO checkpoints are project artifacts.
When they are published, download the corresponding release asset and place
it at the path shown above. Do not commit these files to the source repository.
