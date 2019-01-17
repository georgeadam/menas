#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=12G

hostname

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

type=preset

script=train_scripts/train_regular.py

#if [ "$type" == "random" ]
#then
#    python3 $script --network_type=rnn --dataset=ptb --train_type=$type --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001
#elif [ "$type" == "hardcoded" ]
#then
#    python3 $script --network_type=rnn --dataset=ptb --architecture=tree --train_type=$type --shared_optim adam
#    --shared_lr 0.00035 --entropy_coeff 0.0001
#else
#    script=train_scripts/train_regular.py
#    python3 $script --network_type rnn --dataset ptb --controller_optim adam --controller_lr 0.00035 --shared_optim adam --shared_lr 0.00035 --entropy_coeff 0.0001 --train_type=$type --num_blocks 4
#fi

python3 $script --train_type preset --mode train --shared_optim adam --shared_lr 0.00035 --shared_embed 500 --shared_hid 500 --shared_dropoute 0.3 --shared_dropouti 0.80 --shared_dropout 0.60 --shared_l2_reg 0.000005 --max_epoch 400 --load_path ptb_preset_2018-12-15_22-18-27