"""download and/or process data"""
import torch
import torch.nn as nn
import torchaudio
import pandas as pd


def get_featurizer(sample_rate):
    return torchaudio.transforms.MFCC(sample_rate=sample_rate, log_mels=True)


class WakeWordData(torch.utils.data.Dataset):

    def __init__(self, data_json, sample_rate=8000):
        self.sr = sample_rate
        self.data = pd.read_json(data_json, lines=True)
        self.audio_transform = get_featurizer(sample_rate)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:    
            file_path = self.data.key.iloc[idx]
            waveform, sr = torchaudio.load(file_path)
            if sr > self.sr:
                waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
            mfcc = self.audio_transform(waveform)
            label = self.data.label.iloc[idx]

        except Exception as e:
            print(str(e), file_path)
            return self.__getitem__(torch.randint(0, len(self), (1,)))

        return mfcc, label


def collate_fn(data):
    mfccs = []
    labels = []
    for d in data:
        mfcc, label = d
        mfccs.append(mfcc.squeeze(0).transpose(0, 1))
        labels.append(label)

    # pad mfccs to ensure all tensors are same size in the time dim
    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)  # batch, seq_len, feature
    mfccs = mfccs.transpose(0, 1) # seq_len, batch, feature
    labels = torch.Tensor(labels)
    return mfccs, labels
