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

sbn.set_style('ticks')
sbn.set_context('paper', font_scale=1.55)

eps = 1e-13
startTime = datetime.now()

dir_downloaded_inputs = './../../data/downloaded_inputs/'
dir_generated_inputs = './../../data/generated_inputs/'
dir_figs = './../../figures/'
fig_format = 'jpg'


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
powHistSampleStart = 3600
revHist, powHistSample, revSim = functions_revenues_contracts.simulate_revenue(dir_generated_inputs, gen, hp_GWh,
                                                                           hp_dolPerKwh, genSynth, powSynth, powHistSampleStart,
                                                                           redo = True, save = False)

# get index from swe/revenue relationship.
nYr = int(len(revSim) / 12)
yrSim = np.full((1, nYr * 12), 0)
for i in range(1, nYr):
  yrSim[0, (12 * i):(12 * (i + 1))] = i
revSimWyr = revSim.groupby(yrSim[0, :(nYr * 12)]).sum()
genSynthWyr = genSynth.gen.groupby(yrSim[0, :(nYr * 12)]).sum()
revHistWyr = revHist.groupby('wyear').sum()
genHistWyr = gen.groupby(yrSim[0, :len(powHistSample)]).sum()
powHistWyr = powHistSample.groupby(yrSim[0, :len(powHistSample)]).mean()


## regression for swe index
lmRevSWE = sm.ols(formula='rev ~ sweFeb + sweApr', data=pd.DataFrame(
  {'rev': revSimWyr.values, 'sweFeb': sweSynth.danFeb.values,
   'sweApr': sweSynth.danApr.values}))
lmRevSWE = lmRevSWE.fit()
# print(lmRevSWE.summary())

sweWtParams = [lmRevSWE.params[1]/(lmRevSWE.params[1]+lmRevSWE.params[2]), lmRevSWE.params[2]/(lmRevSWE.params[1]+lmRevSWE.params[2])]
sweWtSynth = (sweWtParams[0] * sweSynth.danFeb + sweWtParams[1] * sweSynth.danApr)
sweWtHist = (sweWtParams[0] * swe.danFeb + sweWtParams[1] * swe.danApr)



### fixed cost parameters
meanRevenue = np.mean(revSimWyr)
fixedCostFraction =  0.914


# payout for swe-based capped contract for differences (cfd), centered around 50th percentile
importlib.reload(functions_revenues_contracts)
print('Generating simulated CFD net payouts..., ', datetime.now() - startTime)

payoutCfdSim = functions_revenues_contracts.snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType = 'cfd',
                                                               lambdaRisk = 0.25, strikeQuantile = 0.5,
                                                               capQuantile = 0.95, redo = True, save = False)
payoutCfdHist = functions_revenues_contracts.snow_contract_payout_hist(sweWtHist, sweWtSynth, payoutCfdSim)

# ### plot CFD structure
# importlib.reload(functions_revenues_contracts)
print('Plotting CFD structure..., ', datetime.now() - startTime)
functions_revenues_contracts.plot_contract(dir_figs, sweWtSynth, payoutCfdSim, fig_format, lambda_shifts=[0.0, 0.5])


### get power price index 
print('Generating power price index..., ', datetime.now() - startTime)
powerIndex, powGenWt = functions_revenues_contracts.power_price_index(powSynth, genSynth, revSim, hp_GWh)

# print(revSimWyr.shape, payoutCfdSim.shape, powerIndex.shape)
# print(np.min(powerIndex[~np.isnan(powerIndex)]))


# # ### get historical swe, gen, power price, revenue, net revenue. Period of record for hydropower = WY 1988-2016
# print('Saving synthetic data..., ', datetime.now() - startTime)
# historical_data = pd.DataFrame({'sweIndex': sweWtHist.loc[revHistWyr.index]})
# historical_data['cfd'] = payoutCfdHist.loc[revHistWyr.index].values
# historical_data['gen'] = genHistWyr.tot.values/1000
# powHistWyr.index = revHistWyr.index
# historical_data['pow'] = powHistWyr
# historical_data['rev'] = revHistWyr.rev
# historical_data.index = np.arange(1988, 2017)

# ### get power price index for sample for historical rerun (starts 1 year before revHistWyr, so can be used for prediction)
# powerIndexHist = powerIndex[powHistSampleStart - 1: powHistSampleStart + revHistWyr.shape[0]]
# powerIndexHist = pd.DataFrame({'powIndex': powerIndexHist}, index=np.arange(1987, 2017))
# historical_data = historical_data.join(powerIndexHist, how='right')

# ## save historical data for simulation of policies
# historical_data.to_csv(dir_generated_inputs + 'historical_data.csv', sep=' ')


# ### get wet, dry, avg example 20-yr periods for plotting
ny = 20
m20AvgSwe = revSimWyr.rolling(ny).mean()
minM20AvgSwe = np.where(m20AvgSwe == np.sort(m20AvgSwe.dropna())[9])[0][0]
maxM20AvgSwe = np.where(m20AvgSwe == np.sort(m20AvgSwe.dropna())[-10])[0][0]
avgM20AvgSwe = np.where(np.abs(m20AvgSwe - m20AvgSwe.mean()) == np.abs(m20AvgSwe - m20AvgSwe.mean()).min())[0][0]

example_data = pd.DataFrame({'sweIndex_wet': sweWtSynth[(maxM20AvgSwe - ny):(maxM20AvgSwe + 1)].values,
                             'sweIndex_avg': sweWtSynth[(avgM20AvgSwe - ny):(avgM20AvgSwe + 1)].values,
                             'sweIndex_dry': sweWtSynth[(minM20AvgSwe - ny):(minM20AvgSwe + 1)].values,
                             'cfd_wet': payoutCfdSim[(maxM20AvgSwe - ny):(maxM20AvgSwe + 1)].values,
                             'cfd_avg': payoutCfdSim[(avgM20AvgSwe - ny):(avgM20AvgSwe + 1)].values,
                             'cfd_dry': payoutCfdSim[(minM20AvgSwe - ny):(minM20AvgSwe + 1)].values,
                             'gen_wet': genSynthWyr[(maxM20AvgSwe - ny):(maxM20AvgSwe + 1)].values/1000,
                             'gen_avg': genSynthWyr[(avgM20AvgSwe - ny):(avgM20AvgSwe + 1)].values/1000,
                             'gen_dry': genSynthWyr[(minM20AvgSwe - ny):(minM20AvgSwe + 1)].values/1000,
                             'powWt_wet': powGenWt[(maxM20AvgSwe - ny):(maxM20AvgSwe + 1)].values,
                             'powWt_avg': powGenWt[(avgM20AvgSwe - ny):(avgM20AvgSwe + 1)].values,
                             'powWt_dry': powGenWt[(minM20AvgSwe - ny):(minM20AvgSwe + 1)].values,
                             'powIndex_wet': powerIndex[(maxM20AvgSwe - ny):(maxM20AvgSwe + 1)],
                             'powIndex_avg': powerIndex[(avgM20AvgSwe - ny):(avgM20AvgSwe + 1)],
                             'powIndex_dry': powerIndex[(minM20AvgSwe - ny):(minM20AvgSwe + 1)],
                             'rev_wet': revSimWyr[(maxM20AvgSwe - ny):(maxM20AvgSwe + 1)].values,
                             'rev_avg': revSimWyr[(avgM20AvgSwe - ny):(avgM20AvgSwe + 1)].values,
                             'rev_dry': revSimWyr[(minM20AvgSwe - ny):(minM20AvgSwe + 1)].values})

## save historical data for simulation of policies
example_data.to_csv(dir_generated_inputs + 'example_data.csv', sep=' ')

# ### save data to use as inputs to moea for the current study
functions_revenues_contracts.save_synthetic_data_moea(dir_generated_inputs, revSimWyr, payoutCfdSim, powerIndex)

print('Finished, ', datetime.now() - startTime)



