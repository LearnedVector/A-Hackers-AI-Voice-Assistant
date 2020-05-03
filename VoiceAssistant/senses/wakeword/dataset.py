"""download and/or process data"""
import torch
import torchaudio
import pandas as pd


class WakeWordData(torch.utils.data.Dataset):

    def __init__(self, data_json, sample_rate=8000):
        self.data = pd.read_json(data_json)

        self.audio_transform = nn.Sequential(
            torchaudio.transforms.MFCC(sample_rate=sample_rate, log_mels=True)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = id.item()
        
        try:    
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = preprocess(self.data.text.iloc[idx])
