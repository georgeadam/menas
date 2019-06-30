#!/bin/bash

#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mincpus=2

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate pt3

# train longer
#python3 train_search.py --epochs=200

# search with same model size as the final tuning, i.e. same num neurons
python3 train_search.py --epochs=2000 --nhid=850 --nhidlast=850 --emsize=850 --batch_size=128 --save=EXP-full-capacity


#python3 train.py --arch=FULL_SIZE_NET

