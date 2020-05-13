#!/bin/bash
formulation=$1
nobj=$2
ndv=$3
nseeds=$4

dir=../../../data/optimization_output/${formulation}
param=150
seeds=$(seq 1 ${nseeds})

mkdir $dir/metrics

JAVA_ARGS="-cp ../../misc/MOEAFramework-2.12-Demo.jar"

for seed in ${seeds}
do
	objfil=${dir}/objs/DPS_param${param}_seedS1_seedB${seed}.obj
	metricfil=${dir}/metrics/DPS_param${param}_seedS1_seedB${seed}.metrics
	reffil=${dir}/DPS_${formulation}_borg.reference
	#echo $objfil $metricfil $reffil
	sbatch -n 1 -t 12:00:00 --wrap="java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.ResultFileEvaluator -d $nobj -i $objfil -r $reffil -o $metricfil"
done
