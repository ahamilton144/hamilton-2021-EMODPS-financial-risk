#/bin/bash
NOBJ=$1
dir=$2
dir_clean=${dir%/}
echo $dir_clean
eps1=0.075
eps2=0.225
eps3=0.05
eps4=0.225

module load python/3.6.9
if [ $NOBJ -gt 3 ]; then
  python3 pareto.py ${dir_clean}/*.reference -o 0-$((NOBJ-1)) -e $eps1 $eps2 $eps3 $eps4 --output ${dir_clean}/overall.reference --delimiter=" " --comment="#"
else
  python3 pareto.py ${dir_clean}/*.reference -o 0-$((NOBJ-1)) -e $eps1 $eps2 --output ${dir_clean}/overall.reference --delimiter=" " --comment="#"
fi

