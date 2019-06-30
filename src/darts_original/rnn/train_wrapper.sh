#!/bin/bash

arcs=$(seq 1 1 10)

for arc in ${arcs[@]}
do
    sbatch --export=arc=$arc train_shell.sh
    sleep 5
done