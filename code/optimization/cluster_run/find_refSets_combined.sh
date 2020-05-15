#/bin/bash
nobj=$1
formulation=$2
dir=../../../data/optimization_output/${formulation}
pareto=../../misc/pareto.py
eps1=0.075
eps2=0.225
eps3=0.05001
eps4=0.225
ndv=0

module load python/3.6.9

if [ $nobj -lt 3 ]; then
  python3 $pareto ${dir}/*.reference -o $((ndv))-$((ndv+nobj-1)) -e $eps1 $eps2 --output ${dir}/DPS_${formulation}_borg.reference --delimiter=" " --comment="#"
else
  python3 $pareto ${dir}/*.reference -o $((ndv))-$((ndv+nobj-1)) -e $eps1 $eps2 $eps3 $eps4 --output ${dir}/DPS_${formulation}_borg.reference --delimiter=" " --comment="#"
fi




