#!/bin/bash

seeds=$(seq 110 1 119)

for seed in ${seeds[@]}
do
    sbatch --export=seed=$seed search_shell.sh
    sleep 5
done



