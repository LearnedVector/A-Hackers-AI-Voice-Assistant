#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH -t 05:00:00
#SBATCH --job-name=NLUModel-Run1
#SBATCH --mail-type=ALL
#SBATCH --mail-user ncwn67@durham.ac.uk

source /etc/profile
module load cuda/8.0

python /home2/ncwn67/A-Hackers-AI-Voice-Assistant/VoiceAssistant/nlu/neuralnet/train.py