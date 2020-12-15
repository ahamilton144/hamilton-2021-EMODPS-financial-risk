#!/bin/bash

## arg input is pareto combination (eg '124')
combo=$1
dir=../../../data/optimization_output/4obj_2rbf_moreSeeds/
base=DPS_4obj_2rbf_moreSeeds_borg.resultfile
subset=DPS_4obj_2rbf_${combo}.resultfile
new=DPS_4obj_2rbf_${combo}.linefile
echo 'Pareto subset for objectives '$combo', from '$subset', line nums relative to '$base > $dir/$new

## loop over lines in subset file, get matching line from base file
linenum=1
numlines=$(wc -l < $dir/$subset)
while [[ $linenum -le $numlines ]]
do
	## sed command gets particular line from subset -> grep gets matching line and linenum from subset -> cut trims output to just keep linenum -> append to file new
	sed "${linenum}q;d" $dir/$subset | grep -nf - $dir/$base | cut -f1 -d: >> $dir/$new
	linenum=$((linenum+1))
done
