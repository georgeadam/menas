#!/bin/bash

#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mincpus=2

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate pt3

python3 train.py --train-random --seed=$seed --save=EXP-random-arc