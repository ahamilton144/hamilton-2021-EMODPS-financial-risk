#!/bin/bash

module unload gnu8
module load intel/19.0.2.187
module load boost/1.70.0

dos2unix main.cpp
sed -i 's/define BORG_RUN_TYPE .*/define BORG_RUN_TYPE 2/' main.cpp
sed -i 's/borg.h/borgms.h/' main.cpp
sed -i 's+\.\/\.\.\/\.\.\/misc\/++' main.cpp
make DPS_borgms

cp main.cpp main_retest.cpp
sed -i 's/define BORG_RUN_TYPE .*/define BORG_RUN_TYPE 0/' main_retest.cpp
make DPS_retest

