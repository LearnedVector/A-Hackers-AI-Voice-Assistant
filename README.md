# A Hackers AI Voice Assistant
Build your own voice ai


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

### data
For more details make sure to visit these files to look at script arguments and description

`collect_wakeword_audio.py` - used to collect wakeword and environment data
`split_audio_into_chunks.py` - used to split audio into n second chunks
`split_commonvoice.py` - if you download the common voice dataset, use this script to split it into n second chunks
`create_wakeword_json.py` - used to create the wakeword json for training

## Raspberry pi
documenation to get this running on rpi is in progress...
