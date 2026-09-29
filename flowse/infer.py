import argparse
import os
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
from vocos import Vocos

from loader.datareader import DataReader
from model import CFM, DiT
from model.model_utils import convert_text, get_tokenizer

EPS = np.finfo(float).eps
warnings.filterwarnings("ignore", category=FutureWarning)

def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=None, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = Path(local_path) / "config.yaml"
            model_path = Path(local_path) / "pytorch_model.bin"
            missing = [str(path) for path in (config_path, model_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing local Vocos files: {', '.join(missing)}")
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError as exc:
            raise RuntimeError("BigVGAN requires the optional third_party/BigVGAN checkout") from exc
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    else:
        raise ValueError(f"Unsupported vocoder: {vocoder_name}")
    return vocoder

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    return scalar * audio


def run(args):
    with open(args.conf, "r") as f:
        root_conf = yaml.safe_load(f)
    tokenizer = root_conf['model']['tokenizer']
    tokenizer_path = root_conf['model']['tokenizer_path']
    conf = root_conf['infer']
    
    device = torch.device(
        "cuda" if conf["test"]["use_cuda"] and torch.cuda.is_available() else "cpu"
    )

    checkpoint_dir = Path(conf["test"]["checkpoint"])
    cpt_fname = checkpoint_dir / conf["test"]["pt_name"]
    if not cpt_fname.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {cpt_fname}. Download the released checkpoint and update infer.test in the config."
        )
    ckpt = torch.load(cpt_fname, map_location=device, weights_only=True)
    print("checkpoint: ", cpt_fname)
    print("epoch: ", ckpt["epoch"])
    print("last modified: ", time.ctime(os.path.getmtime(cpt_fname)))
    print("decode wav: ", conf["datareader"]["mix_json"])
    print("save path: ", conf["save"]["dir"])
    
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    model_cls = DiT
    nnet = CFM(transformer=model_cls(**conf['nnet_conf']['arch'], text_num_embeds=vocab_size, mel_dim=conf['nnet_conf']['mel_spec']['n_mel_channels']),
        mel_spec_kwargs=conf['nnet_conf']['mel_spec'],vocab_char_map=vocab_char_map).eval().to(device)
    nnet.load_state_dict(ckpt["model_state_dict"])

    save_dir = Path(conf["save"]["dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    vocoder_local_path = conf['nnet_conf']['vocoder']['local_path']
    is_local = conf['nnet_conf']['vocoder']['is_local']
    vocoder_name = conf['nnet_conf']['mel_spec']['mel_spec_type']
    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=is_local,
        local_path=vocoder_local_path,
        device=device,
    )

    input_sample_rate = conf["datareader"]["mix_fs"]
    model_sample_rate = conf['nnet_conf']['mel_spec']['target_sample_rate']
    output_sample_rate = conf["save"]["fs"]
    resampler = None
    if input_sample_rate != model_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=input_sample_rate,
            new_freq=model_sample_rate,
        ).to(device)
    data_reader = DataReader(**conf["datareader"])
    sample_kwargs = {
        "steps": int(conf["test"].get("steps", 32)),
        "cfg_strength": float(conf["test"].get("cfg_strength", 1.0)),
    }
    
    with torch.no_grad():

        for egs in tqdm(data_reader):
            mix = egs["mix"].contiguous().to(device)
            text = convert_text(egs["text"], tokenizer)
            if resampler is not None:
                mix = resampler(mix)
            
            if conf['test']['cond_type'] == 'noisy':
            
                output, _ = nnet.sample(cond=mix, text=[text], **sample_kwargs)
            elif conf['test']['cond_type'] == 'wotext':
                output, _ = nnet.sample(cond=mix, text=[" "], drop_text=True, **sample_kwargs)
            else:
                raise ValueError("infer.test.cond_type must be 'noisy' or 'wotext'")

            utt_id = egs["utt_id"]
            if not utt_id.lower().endswith(".wav"):
                utt_id += ".wav"
            relative_path = Path(utt_id)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe utterance id: {utt_id}")
            output_path = save_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
                    
            output = output.transpose(-1,-2)
            output = output.to(torch.float32)
            generated_wave = vocoder.decode(output)
            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_wave = normalize(generated_wave)
            
            if model_sample_rate != output_sample_rate:
                generated_wave = librosa.resample(
                    generated_wave,
                    orig_sr=model_sample_rate,
                    target_sr=output_sample_rate,
                )
            sf.write(
                output_path,
                generated_wave,
                output_sample_rate,
            )

       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to test model in Pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-conf", type=str, required=True, help="Yaml configuration file for training"
    )
    args = parser.parse_args()
    run(args)
