"""the interface to interact with wakeword model"""
import pyaudio
import threading
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from model import LSTMWakeWord
from dataset import get_featurizer
import time
import argparse
import sys


class Listener:

    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk , exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.05)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()


class WakeWordEngine:

    def __init__(self, model_checkpoint):
        self.listener = Listener(sample_rate=8000, record_seconds=2)
        checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))
        self.model = LSTMWakeWord(**checkpoint['model_params'], device='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().to('cpu')  #run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()
        self.audio_buffer = []

    def inference(self, audio):
        with torch.no_grad():
            waveform = torch.Tensor([np.frombuffer(a, dtype=np.int16) for a in audio]).flatten()
            mfcc = self.featurizer(waveform).transpose(0, 1).unsqueeze(1)
            out = self.model(mfcc)
            pred = torch.round(F.sigmoid(out))
            print(pred.item())

    def run(self, action):
        self.listener.run(self.audio_q)
        while True:
            if len(self.audio_q) >= 15:  # remove part of stream
                self.audio_q = self.audio_q[len(self.audio_q) - 15:len(self.audio_q)]
                self.inference(self.audio_q)
            elif len(self.audio_q) == 15:
                self.inference(self.audio_q)
            time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the wakeword engine")
    parser.add_argument('--model_checkpoint_path', type=str, default=None, required=True,
                        help='if set to None, then will record forever until keyboard interrupt')
    args = parser.parse_args()

    wakeword_engine = WakeWordEngine(args.model_checkpoint_path)
    action = lambda x: print(x)
    wakeword_engine.run(action)
