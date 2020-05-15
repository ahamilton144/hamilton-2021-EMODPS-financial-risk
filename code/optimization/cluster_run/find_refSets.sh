#/bin/bash
formulation_slash=$1
formulation=${formulation_slash%/}
nobj=$2
ndv=$3
dir_reference_copy=$4

dir=../../../data/optimization_output/${formulation}
pareto=../../misc/pareto.py
eps1=0.075
eps2=0.225
eps3=0.05001
eps4=0.225
param=150

module load python/3.6.9

if [ $nobj -lt 3 ]; then
  python3 $pareto ${dir}/sets/*.set -o $((ndv))-$((ndv+nobj-1)) -e $eps1 $eps2 --output ${dir}/DPS_${formulation}_borg.resultfile --delimiter=" " --comment="#"
else
  python3 $pareto ${dir}/sets/*.set -o $((ndv))-$((ndv+nobj-1)) -e $eps1 $eps2 $eps3 $eps4 --output ${dir}/DPS_${formulation}_borg.resultfile --delimiter=" " --comment="#"
fi
cut -d ' ' -f $((ndv+1))-$((ndv+nobj)) ${dir}/DPS_${formulation}_borg.resultfile >${dir}/DPS_${formulation}_borg.reference

cp ${dir}/DPS_${formulation}_borg.reference $dir_reference_copy



