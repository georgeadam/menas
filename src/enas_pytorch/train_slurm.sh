#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

type=regular

if [ "$type" == "random" ]
then
    script=train_scripts/train_random.py
    python3 $script --network_type=rnn --dataset=ptb
else
    script=train_scripts/train_regular.py
    python3 $script --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001
fi