#!/bin/bash
module load python/3.6.9

nrun=32
npolperrun=64

run=0
while (( $run < $nrun ))
do
	echo $run
	polstart=$(( $run*npolperrun ))
	polend=$(( $(( $run+1 ))*npolperrun ))
	sbatch -n 1 -t 18:00:00 --wrap="python3 calculate_entropic_SA.py $polstart $polend"
	run=$(($run+1))
done

#npol=1867
#policy=1866
#while (( $policy < $(( $npol+1 )) ))
#do
#        echo $policy
#	if [ ! -f '../../data/policy_simulation/'${policy}'.pkl' ]; then
#        	echo $policy
#		sbatch -n 1 -t 00:30:00 --wrap="python3 calculate_entropic_SA.py $policy $(( $policy+1 ))"
#	fi
#	policy=$(( $policy+1 ))
#done



#sbatch -n 1 -t 18:00:00 --wrap="python3 calculate_entropic_SA.py 1866 1867"
