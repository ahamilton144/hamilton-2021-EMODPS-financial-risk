#!/bin/bash

nobj=4
nseeds=10
dir=../../../data/optimization_output/${nobj}obj_rbf_overall
mkdir $dir

#for nrbf in 1 2 3 4 8 12 
#do
#	sh rbf_postprocess_1.sh $nobj $nrbf $nseeds
#done

sh rbf_postprocess_2.sh $nobj $nseeds $dir
