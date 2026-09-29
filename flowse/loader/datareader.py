import librosa
import numpy as np
import soundfile as sf

import torch
import json
from pathlib import Path

class DataReader(object):
    def __init__(self,
                 mix_json,
                 mix_dir,
                 mix_fs=16000):

        with open(mix_json, "r", encoding="utf-8") as f:
            self.mix_json =  json.load(f) 
        self.utt = list(self.mix_json.keys())
        
        self.mix_dir = mix_dir
        self.mix_fs = mix_fs

    def extract_feature(self, utt):
        relative_path = Path(utt)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe utterance id: {utt}")
        if relative_path.suffix.lower() != ".wav":
            relative_path = Path(f"{relative_path}.wav")
        mix_path = Path(self.mix_dir) / relative_path
        text = self.mix_json[utt]
        mix_data = self.get_firstchannel_read(mix_path, self.mix_fs).astype(np.float32)

        mix_input = np.reshape(mix_data, [1, mix_data.shape[0]])

        mix_input = torch.from_numpy(mix_input)

        egs = {
            'utt_id': utt,
            'mix': mix_input,
            'text':text
        }

        return egs

    def __len__(self):
        return len(self.utt)

    def __getitem__(self, index):
        return self.extract_feature(self.utt[index])
    
        

    def get_firstchannel_read(self, path, fs, channel=0):
        wave_data, sr = sf.read(path)
        if sr != fs:
            if len(wave_data.shape) != 1:
                wave_data = wave_data.transpose((1, 0))
            wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
            if len(wave_data.shape) != 1:
                wave_data = wave_data.transpose((1, 0))
        if len(wave_data.shape) > 1:
            wave_data = wave_data[:, channel]
        return wave_data



