#!/bin/bash

nobj=4
seed_start=11
seed_end=30

for nrbf in 2 
do
	old_formulation=${nobj}obj_${nrbf}rbf
	new_formulation=${old_formulation}_moreSeeds
	dir='../../../data/optimization_output/'${new_formulation}
	mkdir $dir
	mkdir ${dir}/runtime
	mkdir ${dir}/sets
	cp -r $old_formulation/ $new_formulation
	cp -r $old_formulation/.gitignore $new_formulation
	cd $new_formulation
	rm dps_*

	sed -i "s+{1\.\.10}+{${seed_start}\.\.${seed_end}}+"  run_borgms.sh
	sed -i "s+-t 12+-t 24+" run_borgms.sh
	sed -i "s+${old_formulation}+${new_formulation}+" run_borgms.sh
	cp main.cpp ../$dir
        cp run_borgms.sh ../$dir
        cp .gitignore ../$dir
	sbatch run_borgms.sh
	cd ../
done

