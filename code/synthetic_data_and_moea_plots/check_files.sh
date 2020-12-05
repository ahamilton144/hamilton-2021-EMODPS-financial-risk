#!/bin/bash
cd ../../data/policy_simulation
for i in {1..2044}
do
	if [ ! -f ${i}.pkl ]; then
		echo ${i}.pkl
	fi
done
