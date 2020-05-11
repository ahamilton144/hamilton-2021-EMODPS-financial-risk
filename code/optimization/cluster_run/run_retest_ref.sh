#!/bin/bash

filread=$1
filwrite=$2
seed_sample=1
seed_sample_retest=2
param=0
filinput='./../'
sbatch -n 1 -t 01:00:00 --wrap="./PortDPS_retest $seed_sample_retest $param $filinput $filread $filwrite"


