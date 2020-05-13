#!/bin/bash

formulation=$1
filread=$2
filwrite=$3
seed_sample=1
seed_sample_retest=2
param=150
filinput='./../../../data/generated_inputs/'
sbatch -n 1 -t 01:00:00 --wrap="${formulation}/DPS_retest $seed_sample_retest $param $filinput $filread $filwrite"


