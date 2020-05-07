"""the interface to interact with wakeword model"""
import pyaudio
import threading
import torch
from collections import deque
from model import LSTMWakeWord
from dataset import get_featurizer
import time
import argparse


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

    def run(self, queue):
        thread = threading.Thread(listen, args=(queue,), daemon=True)
        thread.start()


class WakeWordEngine:

    def __init__(self, model_checkpoint):
        self.listener = Listener(sample_rate=8000, record_seconds=2)
        checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))
        self.model = LSTMWakeWord(**checkpoint['model_params'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().to('cpu')  #run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = deque()

    def run(self, action):
        self.listener.run(self.audio_q)
        while True:
            action(self.audio_q)
            time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the wakeword engine")
    parser.add_argument('--model_checkpoint_path', type=str, default=None, required=True,
                        help='if set to None, then will record forever until keyboard interrupt')
    args = parser.parse_args()

    wakeword_engine = WakeWordEngine(args.model_checkpoint_path)

    action = lambda x: print(len(x))
    wakeword_engine.run(action)
