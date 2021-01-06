#!/bin/bash

param=150
nseeds=30

# 2-obj dynamic
nobj=2
nrbf=2
formulation=${nobj}obj_${nrbf}rbf
ndv=$(( 4 + 10 * nrbf ))
dir='../../../data/optimization_output/'${formulation}
sh find_refSets.sh $formulation $nobj $ndv NA
sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile

# 2-obj static
nobj=2
formulation=${nobj}obj_2dv
ndv=2
dir='../../../data/optimization_output/'${formulation}
sh find_refSets.sh $formulation $nobj $ndv NA
sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile

# 4-obj static
nobj=4
formulation=${nobj}obj_2dv
ndv=2
dir='../../../data/optimization_output/'${formulation}
sh find_refSets.sh $formulation $nobj $ndv NA
sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile
