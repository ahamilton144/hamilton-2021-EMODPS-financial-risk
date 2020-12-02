#!/bin/bash

nobj=$1
nseeds=$2
dir=$3

formulation=${nobj}obj_rbf_overall
ref_fil=${dir}/DPS_${formulation}_borg.reference

for nrbf in 1 2 3 4 8 12
do
	formulation_rbf=${nobj}obj_${nrbf}rbf
	dir_rbf=${dir}/../${formulation_rbf}
	mkdir ${dir_rbf}/metrics
	ndv=$(( 4 + 10 * nrbf ))
	sh find_runtime_metrics.sh $formulation_rbf $nobj $ndv $nseeds $ref_fil
done

