# FlowSE: Flow-Matching Model for Speech Enhancement

<div>
    <a href="https://arxiv.org/abs/2505.19476"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
    <a href="https://huggingface.co/flowse/wenetspeech4tts_Premium.pt.tar"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FlowSE-pink"></a>
</div>
<br>


**FlowSE** is the first flow-matching model for Speech Enhancement (SE), designed to address the key challenges faced by existing generative models in SE tasks. Traditional approaches like language model-based SE often degrade timbre and intelligibility due to **quantization loss**, while diffusion models suffer from **complex training** and **high inference latency**. FlowSE provides an efficient and innovative solution to these issues.

## 🔑 Key Features
- **Flow Matching for Speech Enhancement**: FlowSE is trained on noisy mel spectrograms and optional text sequences, optimizing a condition flow matching loss with ground-truth mel spectrograms as labels.
- **Implicit Learning of Temporal-Spectral Structure and Text Alignment**: FlowSE learns the speech’s temporal-spectral structure and text-to-speech alignment implicitly without explicit alignment procedures.
- **Flexible Inference Modes**:
  - Inference with noisy mel spectrograms only
  - Inference with noisy mel spectrograms and additional transcripts, providing enhanced performance

## 📊 Experimental Results
Extensive experiments demonstrate that FlowSE significantly **outperforms state-of-the-art generative SE methods**, establishing a new standard for generative-based SE and highlighting the potential of flow matching in advancing the field.

## 🗃️ Project Structure
```plaintext
FlowSE/
│
├── data/                  # Data preprocessing and loading utilities
├── models/                # FlowSE model code
├── checkpoints/           # Pre-trained model weights
├── utils/                 # Utility functions
├── inference.py           # Inference script
├── train.py               # Training script
└── README.md              # This documentation
```


## 🚀 Quick Start

- **1️⃣ Download environment requirements**


- **2️⃣ Download pretrained weights**

  We provided pretrained weights <a href="https://huggingface.co/flowse/wenetspeech4tts_Premium.pt.tar"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FlowSE-pink"></a>  and audio samples.
  
- **3️⃣ Inference example**


## 📁 Resources
  - Audio samples in FlowSE/static/audio


## 📖 Citation

If you find **FlowSE** useful in your research or work, please consider citing our paper:

```bibtex
@misc{wang2025flowseefficienthighqualityspeech,
      title={FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching}, 
      author={Ziqian Wang and Zikai Liu and Xinfa Zhu and Yike Zhu and Mingshuai Liu and Jun Chen and Longshuai Xiao and Chao Weng and Lei Xie},
      year={2025},
      eprint={2505.19476},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.19476}, 
}
