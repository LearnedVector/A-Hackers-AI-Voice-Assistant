# A Hackers AI Voice Assistant
Build your own voice ai. This repo is for my [YouTube video series](https://www.youtube.com/playlist?list=PL5rWfvZIL-NpFXM9nFr15RmEEh4F4ePZW) on building an AI voice assistant with PyTorch

TODO:
- [x] wake word model and engine
- [ ] pre-trained wake word model use for fine tuning on your own wakeword
- [ ] speech recognition model, pretrained model, and engine
- [ ] natural langauge underestanding model, pretrained model, and engine
- [ ] speech synthesis model, pretrained model, and engine

## Running on native machine
### dependencies
* python3
* portaudio (for pyaudio to work) 

If you're on mac you can install `portaudio` using `homebrew`

### using virtualenv (recommend)
1. `virtualenv virtualassistant.venv`
2. `source voiceassistant.venv/bin/activate`

### pip packages
`pip install -r requirements.txt` 

## Running with Docker
### setup
If you are running with just the cpu
`docker build -f cpu.Dockerfile -t voiceassistant .`

If you are running on a cuda enabled machine 
`docker build -f cpu.Dockerfile -t voiceassistant .`

## Wake word

### scripts
For more details make sure to visit these files to look at script arguments and description

`neuralnet/train.py` is used to train the model

`neuralnet/optimize_graph.py` is used to create a production ready graph that can be used in `engine.py`

`engine.py` is used to demo the wakeword model

`collect_wakeword_audio.py` - used to collect wakeword and environment data

`split_audio_into_chunks.py` - used to split audio into n second chunks

`split_commonvoice.py` - if you download the common voice dataset, use this script to split it into n second chunks

`create_wakeword_json.py` - used to create the wakeword json for training

### Steps to train and demo your wakeword model

For more details make sure to visit these files to look at script arguments and description

1. collect data
    1. environment and wakeword data can be collected using `python collect_wakeword_audio.py`
    2. be sure to collect other speech data like common voice. split the data into n seconds chunk with `split_audio_into_chunks.py`.
    3. put data into two seperate directory named `0` and `1`. `0` for non wakeword, `1` for wakeword. use `create_wakeword_json.py` to create train and test json
    4. create a train and test json in this format...
        ```
        // make each sample is on a seperate line
        {"key": "/path/to/audio/sample, "label" 0}
        {"key": "/path/to/audio/sample, "label" 1}
        ```

2. train model
    1. use `train.py` to train model
    2. after model training us `optimize_graph.py` to create an optimized pytorch model

3. test
    1. test using the `engine.py` script


## Raspberry pi
documenation to get this running on rpi is in progress...
