#!/bin/bash
module load python/3.6.9

### Run entropic SA for 4-objective Pareto set. nrun should be equal to the number of processors you want to use, 
###     and nrun*npolperrun should be >= the number of policies in the solution set.
###     In this case, solution set has 2044 solutions, and 32*64=2048.
nobj=4
nrun=32
npolperrun=64

run=0
while (( $run < $nrun ))
do
	echo $run
	polstart=$(( $run*npolperrun ))
	polend=$(( $(( $run+1 ))*npolperrun ))
	sbatch -n 1 -t 18:00:00 --wrap="python3 calculate_entropic_SA.py $nobj $polstart $polend"
	run=$(($run+1))
done

### Repeat with 2-objective Pareto set. In this case solution set only has 10 policies.
nobj=2 
nrun=10 
npolperrun=1

run=0
while (( $run < $nrun ))
do
	echo $run
	polstart=$(( $run*npolperrun ))
	polend=$(( $(( $run+1 ))*npolperrun ))
	sbatch -n 1 -t 1:00:00 --wrap="python3 calculate_entropic_SA.py $nobj $polstart $polend"
	run=$(($run+1))
done
