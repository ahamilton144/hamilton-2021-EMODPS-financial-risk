#/bin/bash

dir='../../../data/optimization_output/4obj_2rbf_moreSeeds'
resultfile=${dir}/DPS_4obj_2rbf_moreSeeds_borg.resultfile
fbeg=${dir}/DPS_4obj_2rbf_
pareto='../../misc/pareto.py'
eps1=0.075
eps2=0.225
eps3=0.05001
eps4=0.225
ndv=24
nobj=4

echo 123
python3 $pareto $resultfile -o $((ndv)) $((ndv+1)) $((ndv+2)) -e $eps1 $eps2 $eps3 --output ${fbeg}123.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '123'
echo 124
python3 $pareto $resultfile -o $((ndv)) $((ndv+1)) $((ndv+3)) -e $eps1 $eps2 $eps4 --output ${fbeg}124.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '124'
echo 134
python3 $pareto $resultfile -o $((ndv)) $((ndv+2)) $((ndv+3)) -e $eps1 $eps3 $eps4 --output ${fbeg}134.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '134'
echo 234
python3 $pareto $resultfile -o $((ndv+1)) $((ndv+2)) $((ndv+3)) -e $eps2 $eps3 $eps4 --output ${fbeg}234.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '234'
echo 12
python3 $pareto $resultfile -o $((ndv)) $((ndv+1)) -e $eps1 $eps2 --output ${fbeg}12.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '12'
echo 13
python3 $pareto $resultfile -o $((ndv)) $((ndv+2)) -e $eps1 $eps3 --output ${fbeg}13.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '13'
echo 14
python3 $pareto $resultfile -o $((ndv)) $((ndv+3)) -e $eps1 $eps4 --output ${fbeg}14.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '14'
echo 23
python3 $pareto $resultfile -o $((ndv+1)) $((ndv+2)) -e $eps2 $eps3 --output ${fbeg}23.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '23'
echo 24
python3 $pareto $resultfile -o $((ndv+1)) $((ndv+3)) -e $eps2 $eps4 --output ${fbeg}24.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '24'
echo 34
python3 $pareto $resultfile -o $((ndv+2)) $((ndv+3)) -e $eps3 $eps4 --output ${fbeg}34.resultfile --delimiter=" " --comment="#"
sh get_linenums_pareto.sh '34'
