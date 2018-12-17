#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=%A_%a
#SBATCH --output=slurm_%A_%a.out

export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
#export PATH=/pkgs/anaconda3/bin:$PATH
#

source activate py36-darts

cd rnn
echo ${SLURM_ARRAY_TASK_ID}

if [ "${SLURM_ARRAY_TASK_ID}" == "0" ]
then
    python train_search.py
elif [ "${SLURM_ARRAY_TASK_ID}" == "1" ]
then
    sleep 2
    python train_search.py --unrolled
elif [ "${SLURM_ARRAY_TASK_ID}" == "2" ]
then
    sleep 4
    python train_search.py --unrolled --diff_unrolled
fi

# srun --partion=gpu --gres=gpu:1 --mem=4GB python train_search.py
# sbatch --array=0-2 srun_script.sh