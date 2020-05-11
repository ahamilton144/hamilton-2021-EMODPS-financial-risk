#!/bin/bash

seed_sample=1
seed_sample_retest=2
param=0
filinput='./../'
for seed_borg in {1..10}
do
  filread='sets/PortDPS_DPS_maxDebt_samp50000_seedS'$seed_sample'_seedB'$seed_borg'.set'
  filwrite='retest/PortDPS_DPS_maxDebt_samp50000_seedS'$seed_sample'_seedB'$seed_borg'_retestS'$seed_sample_retest'.set'
  sbatch -n 1 -t 01:00:00 --wrap="./PortDPS_retest $seed_sample_retest $param $filinput $filread $filwrite"
done


