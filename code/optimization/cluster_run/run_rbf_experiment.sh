#!/bin/bash

nobj=4
param=150
nseeds=4

for nrbf in 1 2 3 4 8 12 
do
	formulation=${nobj}obj_${nrbf}rbf
	dir='../../../data/optimization_output/'${formulation}
	mkdir $dir
	mkdir ${dir}/runtime
	mkdir ${dir}/sets
	cp -r ex_4obj $formulation
	cd $formulation
	rm dps_*
	sed -i "s/NUM_RBF 4/NUM_RBF $nrbf/" main.cpp
	sed -i "s/4rbf/${nrbf}rbf/" run_borgms.sh
	sed -i "s/formulation_name/${formulation}/" run_borgms.sh
	sh remake.sh
	cp main.cpp $dir
	cp run_borgms.sh $dir
	sbatch run_borgms.sh
	cd ../
done
