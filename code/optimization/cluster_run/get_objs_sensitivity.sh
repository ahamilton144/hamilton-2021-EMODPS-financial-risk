#!/bin/bash
NSEEDS=$1
ps=$(seq 1 ${NSEEDS})
NDV=$2
NOBJ=$3
DIR=$4
SENS=$5
for p in ${ps}
do
	awk -v NDV="$NDV" 'BEGIN {FS=" "}; /^#/ {print $0}; /^[^#/]/ {printf("%s %s\n",$(NDV+1),$(NDV+2))}' $DIR/runtime${SENS}/PortDPS_2dv_param${SENS}_samp50000_seed1_seedB${p}.runtime >$DIR/objs/PortDPS_2dv_param${SENS}_samp50000_seed1_seedB${p}.obj
done

