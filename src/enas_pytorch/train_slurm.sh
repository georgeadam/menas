#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=guppy5

hostname

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

type=ours

script=train_scripts/train_regular.py

if [ "$type" == "random" ]
then
    python3 $script --network_type=rnn --dataset=ptb --train_type=$type
elif [ "$type" == "hardcoded" ]
then
    python3 $script --network_type=rnn --dataset=ptb --architecture=chain --train_type=$type
else
    script=train_scripts/train_regular.py
    python3 $script --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001 --train_type=$type
fi