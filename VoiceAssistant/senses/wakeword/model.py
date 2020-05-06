"""wakeword model"""
import torch.nn as nn

class LSTMWakeWord(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, dropout, bidirectional):
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, dropout, bidirectional)
    
    def forward(self, x):
        return self.lstm(x)
