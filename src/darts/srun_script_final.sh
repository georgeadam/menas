#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
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
    python train.py
elif [ "${SLURM_ARRAY_TASK_ID}" == "1" ]
then
    sleep 2
    python train.py --arch ENAS
elif [ "${SLURM_ARRAY_TASK_ID}" == "2" ]
then
    sleep 4
    python train.py --arch OURS
elif [ "${SLURM_ARRAY_TASK_ID}" == "3" ]
then
    sleep 4
    python train.py --arch DARTS_1ST_ORDER
fi

# srun --partion=gpu --gres=gpu:1 --mem=4GB python train_search.py
# sbatch --array=0-2 srun_script.sh