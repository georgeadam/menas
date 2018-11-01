#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

PYTHONPATH=$(pwd)
export PYTHONPATH

script=main.py

source /h/alexadam/anaconda3/bin/activate dl

#python3 $script --network_type=rnn --dataset=ptb
python main.py --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001