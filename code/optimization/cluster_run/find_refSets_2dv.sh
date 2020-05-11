#/bin/bash
dir=$1
dir_clean=${dir%/}
echo $dir_clean
NDV=$2
NOBJ=$3
SENS=$4
eps1=0.075
eps2=0.225

module load python/3.6.9
python3 pareto.py ${dir_clean}/sets${SENS}/*.set -o $((NDV))-$((NDV+NOBJ-1)) -e $eps1 $eps2 --output ${dir_clean}/${dir_clean}_param${SENS}_borg.resultfile --delimiter=" " --comment="#"
cut -d ' ' -f $((NDV+1))-$((NDV+NOBJ)) ${dir_clean}/${dir_clean}_param${SENS}_borg.resultfile >${dir_clean}/${dir_clean}_param${SENS}_borg.reference

#if [ $NOBJ -gt 3 ]; then
#  python3 pareto.py ${dir_clean}/retest/*.set -o $((NDV))-$((NDV+NOBJ-1)) -e $eps1 $eps2 $eps3 $eps4 --output ${dir_clean}/${dir_clean}_retest.resultfile --delimiter=" " --comment="#"
#else
#  python3 pareto.py ${dir_clean}/retest/*.set -o $((NDV))-$((NDV+NOBJ-1)) -e $eps1 $eps2 --output ${dir_clean}/${dir_clean}_retest.resultfile --delimiter=" " --comment="#"
#fi
#cut -d ' ' -f $((NDV+1))-$((NDV+NOBJ)) ${dir_clean}/${dir_clean}_retest.resultfile >${dir_clean}/${dir_clean}_retest.reference


