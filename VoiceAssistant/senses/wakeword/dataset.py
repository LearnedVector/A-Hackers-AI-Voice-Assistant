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
            mfcc = self.audio_transform(waveform)
            label = self.data.label.iloc[idx]

        except Exception as e:
            print(e)
        
        return mfcc, label


def collate_fn(data):
    mfccs = []
    labels = []
    for d in data:
        mfcc, label = d
        mfccs.append(mfcc)
        labels.append(label)
    
    # pad mfccs to ensure all tensors are same size in the time dim
    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    return mfcc, mfccs
