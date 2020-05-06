# You can change pytorch image to one that's compatible
# with your system https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

# install utilities
RUN apt update && \
    apt install vim net-tools ffmpeg portaudio19-dev \
    alsa-base alsa-utils sox    \
    -y

WORKDIR /VoiceAssistant
COPY . /VoiceAssistant
RUN pip install -r requirements.txt