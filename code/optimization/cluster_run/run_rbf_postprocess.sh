#!/bin/bash

nobj=4
param=150
nseeds=10

for nrbf in 3 #1 2 #3 4 8 12 
do
	sh postprocess_output.sh $nobj $nrbf $nseeds
done
