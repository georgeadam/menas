#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

PYTHONPATH=$(pwd)
export PYTHONPATH

source /h/alexadam/anaconda3/bin/activate dl

ablation_study=$1
model_path=$2

if [ "$ablation_study" == "activation_replacement" ]
then
    script=ablation_studies/activation_relacement.py
elif [ "$ablation_study" == "node_removal" ]
then
    script=ablation_studies/node_removal.py
fi

python3 $script --model_path=$model_path