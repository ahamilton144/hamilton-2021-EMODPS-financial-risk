##################################################################
#### get policy line numbers from file args
##################################################################
import sys
objective_formulation = int(sys.argv[1])  ### this should be 2, to run SA on 2-obj problem output, or 4, to run on 4-obj output
policy_start = int(sys.argv[2])
policy_end = int(sys.argv[3])
policy_ranks = range(policy_start, policy_end)



######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import itertools
from datetime import datetime

### Project functions ###
import functions_moea_output_plots
import functions_entropic_SA

##########################

startTime = datetime.now()

EPS = 1e-10
ny = 20
ns = 500
nbins_entropy = 50


dir_data = '../../data/'
dir_figs = '../../figures/'

### get solutions from moea
if objective_formulation == 2:
  dps = functions_moea_output_plots.get_set(dir_data + 'optimization_output/2obj_2rbf/DPS_2obj_2rbf_borg_retest.resultfile', 4, 1, sort=False)[0]
elif objective_formulation == 4:
  dps = functions_moea_output_plots.get_set(dir_data + 'optimization_output/4obj_2rbf_moreSeeds/DPS_4obj_2rbf_moreSeeds_borg_retest.resultfile', 4, 1, sort=False)[0]


### entropic SA analysis
samp = pd.read_csv(dir_data + 'generated_inputs/synthetic_data.txt', delimiter=' ')
samp_rev = samp.revenue.values
samp_sswp = samp.payoutCfd.values
samp_pow = samp.power.values
samples = np.random.choice([int(x) for x in np.arange(1e6 - 21)], size=ns, replace=True)

### loop over policies in pareto set
for m in policy_ranks:
  name = 'm'+str(m)
  mi_dict = {}
  # get policy params
  dps_choice = dps.iloc[m,:]
  dv_d, dv_c, dv_b, dv_w, dv_a = functions_moea_output_plots.get_dvs(dps_choice)
  # get trajectories through state space
  mi_dict['annRev'] = dps_choice['annRev']
  mi_dict['maxDebt'] = dps_choice['maxDebt']
  mi_dict['maxComplex'] = dps_choice['maxComplex']
  mi_dict['maxFund'] = dps_choice['maxFund']

  ### simulate ns*ny trajectories
  results = np.empty([ns*ny, 9])
  for s in range(ns):
    start, end = s*ny, ny * (s + 1)
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = functions_moea_output_plots.simulate(samp_rev[samples[s]:(samples[s]+ny+1)], samp_sswp[samples[s]:(samples[s]+ny+1)], samp_pow[samples[s]:(samples[s]+ny+1)])
    results[start:end, 0] = fund_hedge
    results[start:end, 1] = fund_withdrawal
    results[start:end, 2] = debt_hedge
    results[start:end, 3] = debt_withdrawal
    results[start:end, 4] = power_hedge
    results[start:end, 5] = power_withdrawal
    results[start:end, 6] = cash_in
    results[start:end, 7] = action_hedge
    results[start:end, 8] = action_withdrawal
    # results[start:end, 9] = adj_rev

  print(name + ' simulation finished', datetime.now() - startTime)
  sys.stdout.flush()

  ### calculate entropic sensitivity indices - hedge action
  atts = ['fund', 'debt', 'power', 'hedge']
  # results columns = fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev
  atts_cols = {'fund': 0, 'debt': 2, 'power': 4, 'hedge': 7}
  dat_temp = {name:{}}
  for i, att in enumerate(atts):
    (dat_temp[name][att + '_binfreq'], dat_temp[name][att + '_bincenter'], dat_temp[name][att + '_binpoint']) = functions_entropic_SA.sort_bins(results[:, atts_cols[att]], nbins_entropy, True)
  dat_temp = functions_entropic_SA.get_joint_probability(dat_temp, name, atts)
  mi_dict['hedge_entropy'] = functions_entropic_SA.get_entropy(dat_temp[name]['hedge_binfreq'])
  tot_mi = 0
  for att in atts[:-1]:
    mi_dict[att + '_hedge_mi'] = functions_entropic_SA.get_mutual_info(dat_temp, name, atts, [[att, 'hedge'], [att], ['hedge']]) / mi_dict['hedge_entropy']
    tot_mi += mi_dict[att + '_hedge_mi']
  mi_dict['hedge_total_mi'] = tot_mi

  ### calculate entropic sensitivity indices - withdrawal action
  atts = ['fund', 'debt', 'power', 'cash', 'withdrawal']
  # results columns = fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev
  atts_cols = {'fund': 1, 'debt': 3, 'power': 5, 'cash': 6, 'withdrawal': 8}
  dat_temp = {name:{}}
  for i, att in enumerate(atts):
    (dat_temp[name][att + '_binfreq'], dat_temp[name][att + '_bincenter'], dat_temp[name][att + '_binpoint']) = functions_entropic_SA.sort_bins(results[:, atts_cols[att]], nbins_entropy, True)
  dat_temp = functions_entropic_SA.get_joint_probability(dat_temp, name, atts)
  mi_dict['withdrawal_entropy'] = functions_entropic_SA.get_entropy(dat_temp[name]['withdrawal_binfreq'])
  tot_mi = 0
  for att in atts[:-1]:
    mi_dict[att + '_withdrawal_mi'] = functions_entropic_SA.get_mutual_info(dat_temp, name, atts, [[att, 'withdrawal'], [att], ['withdrawal']]) / mi_dict['withdrawal_entropy']
    tot_mi += mi_dict[att + '_withdrawal_mi']
  mi_dict['withdrawal_total_mi'] = tot_mi

  print(name + ' entropy finished', datetime.now() - startTime)
  sys.stdout.flush()

  if objective_formulation == 2:
    pd.to_pickle(mi_dict, dir_data + 'policy_simulation/2obj/' + str(m) + '.pkl')
  elif objective_formulation == 4:
    pd.to_pickle(mi_dict, dir_data + 'policy_simulation/4obj/' + str(m) + '.pkl')

#  reread = pd.read_pickle(dir_data + 'policy_simulation/' + str(m) + '.pkl')
#  print(reread)




