#!/bin/bash

module unload gnu8
module load intel/19.0.2.187
module load boost/1.70.0
#dos2unix ../param_LHC_sample_withLamPremShift.txt
#dos2unix ../HHSamp06262019.txt
dos2unix main.cpp
sed -i 's/define BORG_RUN_TYPE .*/define BORG_RUN_TYPE 2/' main.cpp
sed -i 's/"boostutil.h"/".\/..\/boostutil.h"/' main.cpp
sed -i 's/"moeaframework.h"/".\/..\/borg\/moeaframework.h"/' main.cpp
sed -i 's/"borg.h"/".\/..\/borg\/borgms.h"/' main.cpp
make PortDPS_borgms

cp main.cpp main_retest.cpp
sed -i 's/define BORG_RUN_TYPE .*/define BORG_RUN_TYPE 0/' main_retest.cpp
make PortDPS_retest

