#!/bin/bash

nobj=4
nrbf=2
nseeds=30
old_formulation=${nobj}obj_${nrbf}rbf
new_formulation=${old_formulation}_moreSeeds
ndv=$(( 4 + 10 * nrbf ))
old_dir='../../../data/optimization_output/'${old_formulation}
new_dir=${old_dir}_moreSeeds
param=150

cp ${old_dir}/runtime/* ${new_dir}/runtime
cp ${old_dir}/sets/* ${new_dir}/sets

sh find_refSets.sh $new_formulation $nobj $ndv ${dir}/../${nobj}obj_rbf_overall/

sh get_objs.sh $new_formulation $nobj $ndv $nseeds

sh run_retest_ref.sh $new_formulation ${new_dir}/DPS_${new_formulation}_borg.resultfile ${new_dir}/DPS_${new_formulation}_borg_retest.resultfile

