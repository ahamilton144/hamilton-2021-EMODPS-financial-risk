#!/bin/bash
#SBATCH --job-name="dps_4rbf"
#SBATCH --output="dps_4rbf_%j.out"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -t 5:00:00

NFE=150000
seed_sample=1
formulation=4obj_4rbf
data='./../../../../data/'
output=${data}optimization_output/${formulation}/

time {
  for seed_borg in {9..10}
  do
    echo
    echo "Seed "${seed_borg}
    time mpirun DPS_borgms $seed_borg $seed_sample $NFE ${data}'generated_inputs/' $output
  done
  echo
  echo "Finished all seeds "
}

