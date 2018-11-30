#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=myTest
#SBATCH --output=slurm_%j.out

export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
export PATH=/pkgs/anaconda3/bin:$PATH

source activate /u/relu/tfenv

python mysimplepython.py