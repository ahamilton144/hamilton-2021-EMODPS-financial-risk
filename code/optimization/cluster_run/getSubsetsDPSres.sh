#!/bin/bash

awk -v var="$1" '$1==var' DPS.resultfile > "DPS_set"$1".resultfile"
