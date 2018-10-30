#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

PYTHONPATH=$(pwd)
export PYTHONPATH

script=main.py

source /h/alexadam/anaconda3/bin/activate dl

python3 $script --network_type=rnn --dataset=ptb