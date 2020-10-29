"""the interface to interact with wakeword model"""
import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import numpy as np
from threading import Event
from array import array
from neuralnet.dataset import get_featurizer


class Listener:
    def __init__(self, sample_rate=8000, record_seconds=2, threshold = 300):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.threshold = threshold
        self.silent = False

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)

    def is_silent(self, data):
        "Returns 'True' if below the 'silent' threshold"
        return max(data) < self.threshold

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            if self.is_silent(array('h', data)):
                self.silent = True
            else:
                self.silent = False
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")


class WakeWordEngine:

    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=2, threshold = 150)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  #run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()
        self.prediction = []

    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname


    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname)
            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)

            out = self.model(mfcc)
            value = torch.round(torch.sigmoid(out)).item()
            acc =  np.asanyarray(torch.sigmoid(out)).tolist()[0][0][0]
            return value, acc

    def inference_loop(self, callback, sensitivity):
        while True:
            if not self.listener.silent:
                if len(self.audio_q) > 25:

                    while True:
                        if len(self.audio_q) > 25:
                            self.audio_q.pop(0)
                        else:
                            break

                    value, acc = self.predict(self.audio_q)

                    if value == 1.0:
                        self.prediction.append(acc)
                    else:
                        self.prediction = []

            time.sleep(0.05)

            if self.listener.silent and len(self.prediction) > 2: #Change depending on the length of your model
                avg_acc = 0
                #Calculate sensitivity
                for i in self.prediction:
                    avg_acc += i

                avg_acc = avg_acc / len(self.prediction)
                self.prediction = []

                if avg_acc > sensitivity:
                    callback()

            elif self.listener.silent > 2 and not len(self.prediction):
                self.prediction = []


    def run(self, callback, sensitivity):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args = (callback, sensitivity), daemon=True)
        thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the wakeword engine")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--sensitivty', type=float, default=0.9, required=False,
                        help='lower value is more sensitive to activations')

    print("""\n*** Make sure you have sox installed on your system for the demo to work!!!
    If you don't want to use sox, change the play function in the DemoAction class
    in engine.py module to something that works with your system.\n
    """)

    args = parser.parse_args()
    wakeword_engine = WakeWordEngine(args.model_file)
    action = lambda: print('Wakeword detected!')

    wakeword_engine.run(callback = action, sensitivity = 0.80)
    threading.Event().wait()
