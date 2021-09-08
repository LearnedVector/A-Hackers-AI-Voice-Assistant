# A Hackers AI Voice Assistant

### I am not mantaining this repo anymore. If you want to take over, please shoot me a message.

Build your own voice ai. This repo is for my [YouTube video series](https://www.youtube.com/playlist?list=PL5rWfvZIL-NpFXM9nFr15RmEEh4F4ePZW) on building an AI voice assistant with PyTorch.

## Looking for contributors!
Looking for contributors to help build out the assistant. There is still alot of work to do. This would be a good oppurtunity to learn Machine Learning and how to Engineer an entire ML system from the ground up. If you're interested join the [Discord Server](https://discord.gg/9wSTT4F)

TODO:
- [x] wake word model and engine
- [ ] pre-trained wake word model use for fine tuning on your own wakeword
- [x] speech recognition model, pretrained model, and engine
- [ ] natural langauge understanding model, pretrained model, and engine
- [ ] speech synthesis model, pretrained model, and engine
- [ ] skills framework
- [ ] Core A.I. Voice Assistant logic to integrate wake word, speech recongition, natural language understanding, speech sysnthesis, and the skills framework.

## Running on native machine
### dependencies
* python3
* portaudio (for recording with pyaudio to work)
* [ctcdecode](https://github.com/parlance/ctcdecode) - for speechrecognition

If you're on mac you can install `portaudio` using `homebrew`

**NOTICE: If you are using windows, some things may not work. For example, torchaudio. I suggest trying this on linux or mac, or use wsl2 on windows**

### using virtualenv (recommend)
1. `virtualenv voiceassistant.venv`
2. `source voiceassistant.venv/bin/activate`

### pip packages
`pip install -r requirements.txt` 

## Running with Docker
### setup
If you are running with just the cpu
`docker build -f cpu.Dockerfile -t voiceassistant .`

If you are running on a cuda enabled machine 
`docker build -f Dockerfile -t voiceassistant .`

## Wake word
[Youtube Video For WakeWord](https://www.youtube.com/watch?v=ob0p7G2QoHA&list=PL5rWfvZIL-NpFXM9nFr15RmEEh4F4ePZW)

### scripts
For more details make sure to visit these files to look at script arguments and description

`wakeword/neuralnet/train.py` is used to train the model

`wakeword/neuralnet/optimize_graph.py` is used to create a production ready graph that can be used in `engine.py`

`wakeword/engine.py` is used to demo the wakeword model

`wakeword/scripts/collect_wakeword_audio.py` - used to collect wakeword and environment data

`wakeword/scripts/split_audio_into_chunks.py` - used to split audio into n second chunks

`wakeword/scripts/split_commonvoice.py` - if you download the common voice dataset, use this script to split it into n second chunks

`wakeword/scripts/create_wakeword_jsons.py` - used to create the wakeword json for training

### Steps to train and demo your wakeword model

For more details make sure to visit these files to look at script arguments and description

1. collect data
    1. environment and wakeword data can be collected using `python collect_wakeword_audio.py`
       ```
       cd VoiceAssistant/wakeword/scripts
       mkdir data
       cd data
       mkdir 0 1 wakewords
       python collect_wakeword_audio.py --sample_rate 8000 --seconds 2 --interactive --interactive_save_path ./data/wakewords
       ```
    2. to avoid the imbalanced dataset problem, we can duplicate the wakeword clips with 
       ```
       python replicate_audios.py --wakewords_dir data/wakewords/ --copy_destination data/1/ --copy_number 100
       ```
    3. be sure to collect other speech data like common voice. split the data into n seconds chunk with `split_audio_into_chunks.py`.
    4. put data into two seperate directory named `0` and `1`. `0` for non wakeword, `1` for wakeword. use `create_wakeword_jsons.py` to create train and test json
    5. create a train and test json in this format...
        ```
        // make each sample is on a seperate line
        {"key": "/path/to/audio/sample.wav, "label": 0}
        {"key": "/path/to/audio/sample.wav, "label": 1}
        ```

2. train model
    1. use `train.py` to train model
    2. after model training us `optimize_graph.py` to create an optimized pytorch model

3. test
    1. test using the `engine.py` script


## Speech Recognition
[YouTube Video for Speech Recognition](https://www.youtube.com/watch?v=YereI6Gn3bM&list=PL5rWfvZIL-NpFXM9nFr15RmEEh4F4ePZW&index=2)

### scripts
For more details make sure to visit these files to look at script arguments and description

`speechrecognition/scripts/mimic_create_jsons.py`is used to create the train.json and test.json files with Mimic Recording Studio 

`speechrecognition/scripts/commonvoice_create_jsons.py`is used to convert mp3 into wav and create the train.json and test.json files with the Commonvoice dataset

`spechrecognition/neuralnet/train.py` is used to train the model

`spechrecognition/neuralnet/optimize_graph.py` is used to create a production ready graph that can be used in `engine.py`

`spechrecognition/engine.py` is used to demo the speech recognizer model

`spechrecognition/demo/demo.py` is used to demo the speech recognizer model with a Web GUI


### Steps for pretraining or finetuning speech recognition model

The pretrained model can be found here at this [google drive](https://drive.google.com/drive/folders/14ljfpvisK1tz8fvFYETbdWqR3lOmJ_2Y?usp=sharing)

1. Collect your own data - the pretrain model was trained on common voice. To make this model work for you, you can collect about an hour or so of your own voice using the [Mimic Recording Studio](https://github.com/MycroftAI/mimic-recording-studio). They have prompts that you can read from.
    1. collect data using mimic recording studio, or your own dataset.
    2. be sure to chop up your audio into 5 - 16 seconds chunks max.
    3. create a train and test json in this format...
    ```
        // make each sample is on a seperate line
        {"key": "/path/to/audio/speech.wav, "text": "this is your text"}
        {"key": "/path/to/audio/speech.wav, "text": "another text example"}
    ```
    use `mimic_create_jsons.py` to create train and test json's with the data from Mimic Recording Studio.
        
        python mimic_create_jsons.py --file_folder_directory /dir/to/the/folder/with/the/studio/data --save_json_path /path/where/you/want/them/saved

    (The Mimic Recording Studio files are usually stored in ~/mimic-recording-studio-master/backend/audio_files/[random_string].) 
    
    use `commonvoice_create_jsons.py` to convert from mp3 to wav and to create train and test json's with the data from Commonvoice by Mozilla
        
        python commonvoice_create_jsons.py --file_path /path/to/commonvoice/file/.tsv --save_json_path /path/where/you/want/them/saved 
    
    if you dont want to convert use `--not-convert` 
        
2. Train model
    1. use `train.py` to fine tune. checkout the [train.py](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/blob/master/VoiceAssistant/speechrecognition/neuralnet/train.py#L115) argparse for other arguments
    ```
       python train.py --train_file /path/to/train/json --valid_file /path/to/valid/json --load_model_from /path/to/pretrain/speechrecognition.ckpt
    ```
   2. To train from scratch omit the `--load_model_from` argument in train.py
   3. after model training us `optimize_graph.py` to create a frozen optimized pytorch model. The pretrained optimized torch model can be found in the google drive link as `speechrecognition.zip`


3. test
    1. test using the `engine.py` script

## Raspberry pi
documenation to get this running on rpi is in progress...
