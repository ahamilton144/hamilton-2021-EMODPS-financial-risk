NSEEDS=$1
ps=$(seq 1 ${NSEEDS})
NDV=$2
NOBJ=$3
DIR=$4
REF=$5

JAVA_ARGS="-cp MOEAFramework-2.12-Demo.jar"

for p in ${ps}
do
	if [ $NDV -gt 3 ];then
		nameObj="maxDebt"
		nameRef="full"
	else
		nameObj="2dv"
		nameRef="2dv"
	fi
	objfil=$DIR/objs/PortDPS_DPS_${nameObj}_samp50000_seedS1_seedB${p}.obj
	metricfil=$DIR/metrics/PortDPS_DPS_${nameObj}_samp50000_seedS1_seedB${p}_borg.metrics
	reffil=$REF
	sbatch -n 1 -t 12:00:00 --wrap="java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.ResultFileEvaluator -d $NOBJ -i $objfil -r $reffil -o ${metricfil}"
done
