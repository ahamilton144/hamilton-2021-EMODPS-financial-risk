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



###### old version that calculated ref sets from retest, not original borg objectives

#dir='../../../data/optimization_output/4obj_2rbf_moreSeeds'
#resultfile_retest=${dir}/DPS_4obj_2rbf_moreSeeds_borg_retest.resultfile
#fbeg=${dir}/DPS_4obj_2rbf_
#pareto='../../misc/pareto.py'
#eps1=0.075
#eps2=0.225
#eps3=0.05001
#eps4=0.225
#ndv=24
#nobj=4

## recalculate pareto set after rerunning with validation seed
#python3 $pareto $resultfile_retest -o $((ndv))-$((ndv+nobj-1)) -e $eps1 $eps2 $eps3 $eps4 --output ${fbeg}1234.resultfile --delimiter=" " --comment="#"
#cut -d ' ' -f $((ndv+1))-$((ndv+nobj)) ${fbeg}1234.resultfile >${fbeg}1234.reference
## get subproblem combinations
##note, this is set up for 4-objective problem only

#fend=.reference

#awk '{print NR " " $s}' ${fbeg}1234${fend} > ${fbeg}1234_pareto${fend}

#module load python/3.6.9
#echo 123
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 2 3 -e $eps1 $eps2 $eps3 --output ${fbeg}123_pareto${fend} --delimiter=" " --comment="#"
#echo 124
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 2 4 -e $eps1 $eps2 $eps4 --output ${fbeg}124_pareto${fend} --delimiter=" " --comment="#"
#echo 134
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 3 4 -e $eps1 $eps3 $eps4 --output ${fbeg}134_pareto${fend} --delimiter=" " --comment="#"
#echo 234
#python3 $pareto ${fbeg}1234_pareto${fend} -o 2 3 4 -e $eps2 $eps3 $eps4 --output ${fbeg}234_pareto${fend} --delimiter=" " --comment="#"
#echo 12
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 2 -e $eps1 $eps2 --output ${fbeg}12_pareto${fend} --delimiter=" " --comment="#"
#echo 13
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 3 -e $eps1 $eps3 --output ${fbeg}13_pareto${fend} --delimiter=" " --comment="#"
#echo 15
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 4 -e $eps1 $eps4 --output ${fbeg}14_pareto${fend} --delimiter=" " --comment="#"
#echo 23
#python3 $pareto ${fbeg}1234_pareto${fend} -o 2 3 -e $eps2 $eps3 --output ${fbeg}23_pareto${fend} --delimiter=" " --comment="#"
#echo 24
#python3 $pareto ${fbeg}1234_pareto${fend} -o 2 4 -e $eps2 $eps4 --output ${fbeg}24_pareto${fend} --delimiter=" " --comment="#"
#echo 34
#python3 $pareto ${fbeg}1234_pareto${fend} -o 3 4 -e $eps3 $eps4 --output ${fbeg}34_pareto${fend} --delimiter=" " --comment="#"
#echo 1
#python3 $pareto ${fbeg}1234_pareto${fend} -o 1 -e $eps1 --output ${fbeg}1_pareto${fend} --delimiter=" " --comment="#"
#echo 2
#python3 $pareto ${fbeg}1234_pareto${fend} -o 2 -e $eps2 --output ${fbeg}2_pareto${fend} --delimiter=" " --comment="#"
#echo 3
#python3 $pareto ${fbeg}1234_pareto${fend} -o 3 -e $eps3 --output ${fbeg}3_pareto${fend} --delimiter=" " --comment="#"
#echo 4
#python3 $pareto ${fbeg}1234_pareto${fend} -o 4 -e $eps4 --output ${fbeg}4_pareto${fend} --delimiter=" " --comment="#"
#

