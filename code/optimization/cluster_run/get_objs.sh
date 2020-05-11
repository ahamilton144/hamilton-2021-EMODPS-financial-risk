#!/bin/bash
NSEEDS=$1
ps=$(seq 1 ${NSEEDS})
NDV=$2
NOBJ=$3
DIR=$4

if [ "$NDV" -gt 3 ]; then
	name="maxDebt"
else
	name="2dv"
fi

for p in ${ps}
do
	if [ "$NOBJ" -gt 3 ]; then
		awk -v NDV="$NDV" 'BEGIN {FS=" "}; /^#/ {print $0}; /^[^#/]/ {printf("%s %s %s %s\n",$(NDV+1),$(NDV+2),$(NDV+3),$(NDV+4))}' $DIR/runtime/PortDPS_DPS_${name}_samp50000_seedS1_seedB${p}.runtime >$DIR/objs/PortDPS_DPS_${name}_samp50000_seedS1_seedB${p}.obj
	else
		awk -v NDV="$NDV" 'BEGIN {FS=" "}; /^#/ {print $0}; /^[^#/]/ {printf("%s %s\n",$(NDV+1),$(NDV+2))}' $DIR/runtime/PortDPS_DPS_${name}_samp50000_seedS1_seedB${p}.runtime >$DIR/objs/PortDPS_DPS_${name}_samp50000_seedS1_seedB${p}.obj
	fi
done
