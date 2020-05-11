#!/bin/bash

mkdir $2
cp $1'/borgms.o' $2
cp $1'/moeaframework.o' $2
cp $1'/mt19937ar.o' $2
cp $1'/main_2v.cpp' $2
cp $1'/run_DPS'* $2
cp $1'/makefile' $2
cp $1'/remake.sh' $2
cp $1/'param_LHC_sample_withLamPremShift.txt' $2
mkdir $2'/runtime'
mkdir $2'/sets'
