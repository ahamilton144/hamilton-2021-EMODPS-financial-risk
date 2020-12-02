#!/bin/bash
#SBATCH --job-name="dps_2rbf"
#SBATCH --output="dps_2rbf_%j.out"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -t 24:00:00

NFE=150000
seed_sample=1
formulation=4obj_2rbf_moreSeeds
data='./../../../../data/'
output=${data}optimization_output/${formulation}/

time {
  for seed_borg in {11..30}
  do
    echo
    echo "Seed "${seed_borg}
    time mpirun DPS_borgms $seed_borg $seed_sample $NFE ${data}'generated_inputs/' $output
  done
  echo
  echo "Finished all seeds "
}

