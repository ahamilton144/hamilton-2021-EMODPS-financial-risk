NSEEDS=$1
ps=$(seq 1 ${NSEEDS})
NDV=$2
NOBJ=$3
DIR=$4
SENS=$5

JAVA_ARGS="-cp MOEAFramework-2.12-Demo.jar"

for p in ${ps}
do
	objfil=$DIR/objs/PortDPS_2dv_param${SENS}_samp50000_seed1_seedB${p}.obj
	metricfil=$DIR/metrics/PortDPS_2dv_param${SENS}_samp50000_seed1_seedB${p}_borg.metrics
	reffil=$DIR/${DIR}_param${SENS}_borg.reference
	sbatch -n 1 -t 12:00:00 --wrap="java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.ResultFileEvaluator -d $NOBJ -i $objfil -r $reffil -o ${metricfil}"
done
