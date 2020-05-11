#!/bin/bash

p=0
echo 'max_fund value_snow_contract avg_ann_revenue debt_over_constraint debt_steal' > 'PortDPS_2dv_paramCombined.set'
for p in `seq 0 999`;
do
	fil='sets/PortDPS_2dv_param'$p'.set'
	cat 'PortDPS_2dv_paramCombined.set' $fil >> 'PortDPS_2dv_paramCombined.set'
done
paste -d' ' 'param_LHC_sample_withLamPremShift.txt' 'PortDPS_2dv_paramCombined.set' > 'PortDPS_2dv_results.set' 
rm 'PortDPS_2dv_paramCombined.set'
