#!/bin/bash

# postprocess for 2 obj, 2rbf formulation 
nobj=2
nrbf=2
formulation=${nobj}obj_${nrbf}rbf
ndv=24
dir='../../../data/optimization_output/'${formulation}

sh find_refSets.sh $formulation $nobj $ndv 'NA'

sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile



# postprocess for 2 obj, 2dv formulation
nobj=2
formulation=${nobj}obj_2dv
ndv=2
dir='../../../data/optimization_output/'${formulation}

sh find_refSets.sh $formulation $nobj $ndv 'NA'

sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile


# postprocess for 4 obj, 2dv formulation
nobj=4
formulation=${nobj}obj_2dv
ndv=2
dir='../../../data/optimization_output/'${formulation}

sh find_refSets.sh $formulation $nobj $ndv 'NA'

sh run_retest_ref.sh $formulation ${dir}/DPS_${formulation}_borg.resultfile ${dir}/DPS_${formulation}_borg_retest.resultfile

