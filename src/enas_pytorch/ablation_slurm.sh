#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=guppy3,guppy9,guppy4

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

ablation_study=$1
load_path=$2

if [ "$ablation_study" == "activation_replacement" ]
then
    script=ablation_studies/activation_replacement.py
elif [ "$ablation_study" == "node_removal" ]
then
    script=ablation_studies/node_removal.py
elif [ "$ablation_study" == "edge_addition" ]
then
    script=ablation_studies/edge_addition.py
fi

python3 $script --load_path=$load_path