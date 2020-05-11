#!/bin/bash

cp $1'sets/'* 'DPS/sets/'
cp $1'runtime/'* 'DPS/runtime'
rm 'DPS/metrics/'*
rm 'DPS/objs/'*
cp $1'main.cpp' 'DPS/main.cpp'
rm 'DPS.re'*

