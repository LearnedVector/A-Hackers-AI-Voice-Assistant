"""wakeword model"""
import torch
import torch.nn as nn


class LSTMWakeWord(nn.Module):

    def __init__(self, num_classes, feature_size, hidden_size,
                num_layers, dropout, bidirectional, device='cpu'):
        super(LSTMWakeWord, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1
        self.device = device
        self.layernorm = nn.LayerNorm(feature_size)
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_size*self.directions, num_classes)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return (torch.zeros(n*d, batch_size, hs).to(self.device),
                torch.zeros(n*d, batch_size, hs).to(self.device))

    def forward(self, x):
        # x.shape => seq_len, batch, feature
        x = self.layernorm(x)
        hidden = self._init_hidden(x.size()[1])
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn)
        return out


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


# class DepthWiseSeperableCNN(nn.Module):
#     pass


class SiameseWakeWord(nn.Module):

    def __init__(self, num_classes, feature_size, filter_size,
                num_layers, dropout, device='cpu'):
        self.siamese_cnn = nn.Sequential(
            nn.Conv1d(feature_size, filter_size, kernel_size=3, stride=3, padding=3//2),
            nn.BatchNorm1d(filter_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(filter_size, num_classes)

    def forward(self, x):
        print(x.shape)
        x = self.siamese_cnn(x)
        out = self.classifier(x)
        return out
