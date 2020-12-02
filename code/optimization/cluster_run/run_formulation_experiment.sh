#!/bin/bash

### first run 2 objective version of full dps (2 rbf)
old_nobj=4
old_seed_start=11
old_seed_end=30
new_nobj=2
new_seed_start=1
new_seed_end=30
nrbf=2
old_formulation=${old_nobj}obj_${nrbf}rbf_moreSeeds
new_formulation=${new_nobj}obj_${nrbf}rbf
dir='../../../data/optimization_output/'${new_formulation}
mkdir $dir
mkdir ${dir}/runtime
mkdir ${dir}/sets
cp -r $old_formulation/ $new_formulation
cp -r $old_formulation/.gitignore $new_formulation
cd $new_formulation
rm dps_*
sed -i "s+{${old_seed_start}\.\.${old_seed_end}+{${new_seed_start}\.\.${new_seed_end}}+"  run_borgms.sh
sed -i "s+-t 24+-t 16+" run_borgms.sh
sed -i "s+${old_formulation}+${new_formulation}+" run_borgms.sh
sed -i "s+nodes=1+nodes=2+" run_borgms.sh
sed -i "s+NUM_OBJECTIVES ${old_nobj}+NUM_OBJECTIVES ${new_nobj}+" main.cpp
sh remake.sh
cp main.cpp ../$dir
cp run_borgms.sh ../$dir
cp .gitignore ../$dir
sbatch run_borgms.sh
cd ../

### now run 2 objective version of 2dv (static) formulation
nobj=2
old_nrbf=2
old_formulation=${nobj}obj_${old_nrbf}rbf
new_formulation=${nobj}obj_2dv
dir='../../../data/optimization_output/'${new_formulation}
mkdir $dir
mkdir ${dir}/runtime
mkdir ${dir}/sets
cp -r $old_formulation/ $new_formulation
cp -r $old_formulation/.gitignore $new_formulation
cd $new_formulation
rm dps_*
sed -i "s+${old_formulation}+${new_formulation}+" run_borgms.sh
sed -i "s+DPS_RUN_TYPE 1+DPS_RUN_TYPE 0+" main.cpp
sh remake.sh
cp main.cpp ../$dir
cp run_borgms.sh ../$dir
cp .gitignore ../$dir
sbatch run_borgms.sh
cd ../

### now run 4 objective version of 2dv (static) formulation
old_nobj=2
new_nobj=4
old_formulation=${old_nobj}obj_2dv
new_formulation=${new_nobj}obj_2dv
dir='../../../data/optimization_output/'${new_formulation}
mkdir $dir
mkdir ${dir}/runtime
mkdir ${dir}/sets
cp -r $old_formulation/ $new_formulation
cp -r $old_formulation/.gitignore $new_formulation
cd $new_formulation
rm dps_*
sed -i "s+${old_formulation}+${new_formulation}+" run_borgms.sh
sed -i "s+NUM_OBJECTIVES ${old_nobj}+NUM_OBJECTIVES ${new_nobj}+" main.cpp
sh remake.sh
cp main.cpp ../$dir
cp run_borgms.sh ../$dir
cp .gitignore ../$dir
sbatch run_borgms.sh
cd ../

