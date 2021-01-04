######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from datetime import datetime

### Project functions ###
import functions_moea_output_plots
import functions_entropic_SA

startTime = datetime.now()

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.55)

startTime = datetime.now()


#### Constants
fixed_cost = 0.914
mean_revenue = 127.80086602479503
mean_net_revenue = mean_revenue * (1 - fixed_cost)

ny = 20
# ns = 50000
# nbins_entropy = 50


dir_data = '../../data/'
dir_figs = '../../figures/'
fig_format = 'jpg'

### policies from moea
dps = functions_moea_output_plots.get_set(dir_data + 'optimization_output/4obj_2rbf_moreSeeds/DPS_4obj_2rbf_moreSeeds_borg_retest.resultfile', 4, 1, sort=False)[0]

### stochastic data
samp = pd.read_csv(dir_data + 'generated_inputs/synthetic_data.txt', delimiter=' ')

### data from entropic SA (after running calculate_entropic_SA.py & consolidate_SA_output.py)
mi = pd.read_csv(dir_data + 'policy_simulation/4obj/mi_combined.csv', index_col=0).sort_index()


### view information of policies
lims = {}
# lims['entropy'] = {'min': np.min(mi.hedge_entropy), 'max':np.max(mi.hedge_entropy)}
# lims['mi'] = {'min': min(np.min(mi.fund_hedge_mi), np.min(mi.debt_hedge_mi), np.min(mi.power_hedge_mi)),
#               'max': max(np.max(mi.fund_hedge_mi), np.max(mi.debt_hedge_mi), np.max(mi.power_hedge_mi))}
lims['entropy'] = {'min': 0, 'max': 5}
lims['mi'] = {'min': 0, 'max': 1}

lims3d = {'annRev':[9.4,11.13],'maxDebt':[0.,36.],'maxComplex':[0.,1.]}



## get example policies for each sensitivity index
mi.fund_hedge_mi = mi.fund_hedge_mi.fillna(-1)
mi.debt_hedge_mi = mi.debt_hedge_mi.fillna(-1)
mi.power_hedge_mi = mi.power_hedge_mi.fillna(-1)
mi.fund_withdrawal_mi = mi.fund_withdrawal_mi.fillna(-1)
mi.debt_withdrawal_mi = mi.debt_withdrawal_mi.fillna(-1)
mi.power_withdrawal_mi = mi.power_withdrawal_mi.fillna(-1)
mi.cash_withdrawal_mi = mi.cash_withdrawal_mi.fillna(-1)

fund_hedge_mi_max = [np.array(mi.fund_hedge_mi).argsort()[-2]]
debt_hedge_mi_max = [np.array(mi.debt_hedge_mi).argsort()[-2]]
power_hedge_mi_max = [np.array(mi.power_hedge_mi).argsort()[-4]]
fund_withdrawal_mi_max = []# [np.array(mi.fund_withdrawal_mi).argsort()[-1]]
debt_withdrawal_mi_max = []# [np.array(mi.debt_withdrawal_mi).argsort()[-1]]
power_withdrawal_mi_max = []# [np.array(mi.power_withdrawal_mi).argsort()[-1]]
cash_withdrawal_mi_max = []# [np.array(mi.cash_withdrawal_mi).argsort()[-1]]

### plot mutual info for each sensitivity index
functions_entropic_SA.plot_MI(mi, 'fund_hedge_mi', fund_hedge_mi_max, 'mi_fund_hedge', lims, lims3d, mean_net_revenue, fig_format)
functions_entropic_SA.plot_MI(mi, 'debt_hedge_mi', debt_hedge_mi_max, 'mi_debt_hedge', lims, lims3d, mean_net_revenue, fig_format)
functions_entropic_SA.plot_MI(mi, 'power_hedge_mi', power_hedge_mi_max, 'mi_power_hedge', lims, lims3d, mean_net_revenue, fig_format)
functions_entropic_SA.plot_MI(mi, 'fund_withdrawal_mi', fund_withdrawal_mi_max, 'mi_fund_withdrawal', lims, lims3d, mean_net_revenue, fig_format)
functions_entropic_SA.plot_MI(mi, 'debt_withdrawal_mi', debt_withdrawal_mi_max, 'mi_debt_withdrawal', lims, lims3d, mean_net_revenue, fig_format)
functions_entropic_SA.plot_MI(mi, 'power_withdrawal_mi', power_withdrawal_mi_max, 'mi_power_withdrawal', lims, lims3d, mean_net_revenue, fig_format)
functions_entropic_SA.plot_MI(mi, 'cash_withdrawal_mi', cash_withdrawal_mi_max, 'mi_cash_withdrawal', lims, lims3d, mean_net_revenue, fig_format)


### get dataframe of simulation results, output for parallel coords in R
policy_ranks = [fund_hedge_mi_max[0], debt_hedge_mi_max[0], power_hedge_mi_max[0]]
ny = 20
ns = 20
functions_entropic_SA.get_parallel_coord_data(samp, dps, policy_ranks, ny, ns, fig_format)


### update table2 to add MI and additional policies
functions_entropic_SA.add_mi_table2(mi, policy_ranks)