"""wakeword model"""
import torch.nn as nn


class LSTMWakeWord(nn.Module):

    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, dropout, bidirectional)
        self.classifier = nn.Linear(hidden_size*self.directions, num_classes)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return (torch.randn(n*d, batch_size, hs), torch.randn(n*d, batch_size, hs))

    def forward(self, x):
        batch_size = x.sizes[1]
        hidden = self._init_hidden(batch_size)
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn)
        return out
