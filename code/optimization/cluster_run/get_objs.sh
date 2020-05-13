#!/bin/bash
formulation=$1
nobj=$2
ndv=$3
nseeds=$4

dir=../../../data/optimization_output/${formulation}
param=150
seeds=$(seq 1 ${nseeds})

mkdir $dir/objs

for seed in ${seeds}
do
if [ $nobj -lt 3 ]; then
	awk -v ndv="$ndv" 'BEGIN {FS=" "}; /^#/ {print $0}; /^[^#/]/ {printf("%s %s\n",$(ndv+1),$(ndv+2))}' ${dir}/runtime/DPS_param${param}_seedS1_seedB${seed}.runtime >${dir}/objs/DPS_param${param}_seedS1_seedB${seed}.obj
else
        awk -v ndv="$ndv" 'BEGIN {FS=" "}; /^#/ {print $0}; /^[^#/]/ {printf("%s %s %s %s\n",$(ndv+1),$(ndv+2),$(ndv+3),$(ndv+4))}' ${dir}/runtime/DPS_param${param}_seedS1_seedB${seed}.runtime >${dir}/objs/DPS_param${param}_seedS1_seedB${seed}.obj
fi
done

