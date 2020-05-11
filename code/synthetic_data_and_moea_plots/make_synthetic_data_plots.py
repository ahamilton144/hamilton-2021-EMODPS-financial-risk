##############################################################################################################
### make_synthetic_data_plots.py - python script to create synthetic data and related plots
### Project started March 2018, last update May 2020
##############################################################################################################

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sbn
import importlib
from datetime import datetime
import matplotlib.pyplot as plt


### Project functions ###
import functions_clean_data
import functions_synthetic_data
import functions_revenues_contracts

sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

eps = 1e-13
startTime = datetime.now()

dir_downloaded_inputs = './../../data/downloaded_inputs/'
dir_generated_inputs = './../../data/generated_inputs/'
dir_figs = './../../figures/'



### Get and clean data
print('Reading in data..., ', datetime.now() - startTime)
# SWE
importlib.reload(functions_clean_data)
swe = functions_clean_data.get_clean_swe(dir_downloaded_inputs)

# hydro generation (GWh/mnth)
gen = functions_clean_data.get_historical_generation(dir_downloaded_inputs, swe).reset_index()

# wholesale power price ($/MWh), inflation adjusted
power = functions_clean_data.get_historical_power(dir_downloaded_inputs)

# SFPUC fin year sales and rates
hp_GWh, hp_dolPerKwh, hp_dolM = functions_clean_data.get_historical_SFPUC_sales()




### Generate synthetic time series
# # SWE, Feb 1 & Apr 1
print('Generating synthetic swe..., ', datetime.now() - startTime)
importlib.reload(functions_synthetic_data)
sweSynth = functions_synthetic_data.synthetic_swe(dir_generated_inputs, swe, redo = True, save = False)

# monthly generation, dependent on swe. Will also create fig S2, showing fitted models (gen as fn of swe) for each month.
print('Generating synthetic hydropower generation..., ', datetime.now() - startTime)
genSynth = functions_synthetic_data.synthetic_generation(dir_generated_inputs, dir_figs, gen, sweSynth, redo = True, save = False)


# monthly power price
print('Generating synthetic power prices..., ', datetime.now() - startTime)
importlib.reload(functions_synthetic_data)
powSynth = functions_synthetic_data.synthetic_power(dir_generated_inputs, power, redo = True, save = False)




### Simulate revenues and hedge payouts
# monthly revenues for SFPUC model
print('Generating simulated revenues..., ', datetime.now() - startTime)
importlib.reload(functions_revenues_contracts)
revHist, revSim = functions_revenues_contracts.simulate_revenue(dir_generated_inputs, gen, hp_GWh, hp_dolPerKwh, genSynth, powSynth, redo = True, save = False)


# get index from swe/revenue relationship.
nYr = int(len(revSim) / 12)
yrSim = np.full((1, nYr * 12), 0)
for i in range(1, nYr):
  yrSim[0, (12 * i):(12 * (i + 1))] = i
revSimWyr = revSim.groupby(yrSim[0, :(nYr * 12)]).sum()
genSynthWyr = genSynth.groupby(yrSim[0, :(nYr * 12)]).sum()

lmRevSWE = sm.ols(formula='rev ~ sweFeb + sweApr', data=pd.DataFrame(
  {'rev': revSimWyr.values, 'sweFeb': sweSynth.danFeb.values,
   'sweApr': sweSynth.danApr.values}))
lmRevSWE = lmRevSWE.fit()
# print(lmRevSWE.summary())

sweWtParams = [lmRevSWE.params[1]/(lmRevSWE.params[1]+lmRevSWE.params[2]), lmRevSWE.params[2]/(lmRevSWE.params[1]+lmRevSWE.params[2])]
sweWtSynth = (sweWtParams[0] * sweSynth.danFeb + sweWtParams[1] * sweSynth.danApr)

genSynth['sweWt'] = (sweWtParams[0] * genSynth.sweFeb + sweWtParams[1] * genSynth.sweApr)
gen['sweWt'] = (sweWtParams[0] * gen.sweFeb + sweWtParams[1] * gen.sweApr)


### fixed cost parameters
meanRevenue = np.mean(revSimWyr)
fixedCostFraction =  0.914


# payout for swe-based capped contract for differences (cfd), centered around 50th percentile
importlib.reload(functions_revenues_contracts)
print('Generating simulated CFD net payouts..., ', datetime.now() - startTime)

payoutCfdSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType = 'cfd',
                                                               lambdaRisk = 0.25, strikeQuantile = 0.5,
                                                               capQuantile = 0.95, redo = True, save = False)


### plot CFD contract (Fig 4)
importlib.reload(functions_revenues_contracts)
print('Plotting CFD contract (Fig 3)..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_contract(dir_figs, sweWtSynth, payoutCfdSim, lambda_shifts=[0., 0.5], plot_type='cfd')


### get power price index 
powerIndex = functions_revenues_contracts.power_price_index(powSynth, genSynth, revSim, hp_GWh)

print(revSimWyr.shape, payoutCfdSim.shape, powerIndex.shape)
print(np.min(powerIndex[~np.isnan(powerIndex)]))

# ### save data to use as inputs to moea for the current study
print('Saving synthetic data..., ', datetime.now() - startTime)
functions_revenues_contracts.save_synthetic_data_moea(dir_generated_inputs, revSimWyr, payoutCfdSim, powerIndex)

print('Finished, ', datetime.now() - startTime)



