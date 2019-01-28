#!/bin/bash

seeds=$(seq 100 1 109)

for seed in ${seeds[@]}
do
    sbatch --export=seed=$seed train_random_shell.sh
    sleep 2
done



