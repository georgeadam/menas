#!/bin/bash

dirs=$(ls | grep eval-EXP-random)

for dir in ${dirs[@]}
do
    echo $dir
done
