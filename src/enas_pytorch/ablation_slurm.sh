#!/bin/bash
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=guppy3,guppy9,guppy4
#SBATCH --mem=12G

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
elif [ "$ablation_study" == "perplexity_speed" ]
then
    script=analysis/model_evaluation_speed.py
elif [ "$ablation_study" == "hidden_state_performance" ]
then
    script=analysis/hidden_state_performance_similarity.py
elif [ "$ablation_study" == "single_activation_replacement" ]
then
    script=ablation_studies/single_activation_replacement.py
elif [ "$ablation_study" == "hidden_state_naive" ]
then
    script=analysis/hidden_state_naive_similarity.py
fi

python3 $script --load_path=$load_path