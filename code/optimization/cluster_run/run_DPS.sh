#!/bin/bash
#SBATCH --job-name="F1D1W1"
#SBATCH --output="DPS_F1D1W1_%j.out"
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -t 20:00:00

NFE=120000
seed_sample=1

wd=$SLURM_SUBMIT_DIR'/'
scratch='/scratch/spec823/d'$SLURM_JOB_ID'/'
mkdir $scratch
cd $scratch
cp ${wd}PortDPS_borgms .
cp ${wd}../HHsamp10132019.txt .
cp ${wd}../param_SFPUC_withLamPremShift.txt .
mkdir runtime
mkdir sets

for seed_borg in {1..10}
do
  time mpirun ${scratch}PortDPS_borgms $seed_borg $seed_sample $NFE ${scratch}'/' ${scratch}'/'
done

mv runtime/* ${wd}'/runtime/'
mv sets/* ${wd}'/sets/'
cd ../
rm -r $scratch

