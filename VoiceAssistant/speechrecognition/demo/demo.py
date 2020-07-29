import webbrowser
import sys
import argparse
from os.path import join, realpath
from flask import Flask, render_template, jsonify
sys.path.append(realpath(join(realpath(__file__), '..', '..')))
from engine import SpeechRecognitionEngine


global asr_engine
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('demo.html')


@app.route("/start_asr")
def start():
    action = DemoAction()
    asr_engine.run(action)
    return jsonify("speechrecognition start success!")


@app.route("/get_audio")
def get_audio():
    with open('transcript.txt', 'r') as f:
        transcript = f.read()
    return jsonify(transcript)


class DemoAction:

    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""
    
    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        trascript = " ".join(self.asr_results.split() + results.split())
        self.save_transcript(trascript)
        if current_context_length > 10:
            self.asr_results = trascript

    def save_transcript(self, transcript):
        with open("transcript.txt", 'w+') as f:
            f.write(transcript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the speech recognition engine")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--ken_lm_file', type=str, default=None, required=False,
                        help='If you have an ngram lm use to decode')
    args = parser.parse_args()
    asr_engine = SpeechRecognitionEngine(args.model_file, args.ken_lm_file)
    webbrowser.open_new('http://127.0.0.1:3000/')
    app.run(port=3000)
