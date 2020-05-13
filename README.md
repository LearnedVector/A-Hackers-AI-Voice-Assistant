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

### run
