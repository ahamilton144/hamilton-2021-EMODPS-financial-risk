#!/bin/bash

nobj=$1
nrbf=$2
nseeds=$3

ndv=$(( 4 + 10 * nrbf )) 
formulation=${nobj}obj_${nrbf}rbf
dir='../../../data/optimization_output/'${formulation}
param=150

sh find_refSets.sh $formulation $nobj $ndv ${dir}/../${nobj}obj_rbf_overall/

sh get_objs.sh $formulation $nobj $ndv $nseeds

sh find_hypervolume.sh ${dir}/DPS_${formulation}_borg.reference ${dir}/DPS_${formulation}_borg.hypervolume

sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile
