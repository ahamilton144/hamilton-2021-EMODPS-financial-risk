##################################################################
#### get policy line numbers from file args
##################################################################
import sys
# policy_start = int(sys.argv[1])
# policy_end = int(sys.argv[2])
# policy_ranks = range(policy_start, policy_end)



######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker, cm, colors
import seaborn as sns
import copy
import itertools
from datetime import datetime

startTime = datetime.now()

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.55)

startTime = datetime.now()

# cmap_vir = cm.get_cmap('viridis')
# col_vir = [cmap_vir(0.1),cmap_vir(0.4),cmap_vir(0.7),cmap_vir(0.85)]
# cmap_blues = cm.get_cmap('Blues_r')
# col_blues = [cmap_blues(0.1),cmap_blues(0.3),cmap_blues(0.5),cmap_blues(0.8)]
# cmap_reds = cm.get_cmap('Reds_r')
# col_reds = [cmap_reds(0.1),cmap_reds(0.3),cmap_reds(0.5),cmap_reds(0.8)]
# cmap_purples = cm.get_cmap('Purples_r')
# col_purples = [cmap_purples(0.1),cmap_purples(0.3),cmap_purples(0.5),cmap_purples(0.8)]
# col_brewerQual4 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

##################################################################
#### Constants
##################################################################
fixed_cost = 0.914
mean_revenue = 127.80086602479503
mean_net_revenue = mean_revenue * (1 - fixed_cost)

ny = 20
# ns = 50000
# nbins_entropy = 50


dir_data = '../../data/'
dir_figs = '../../figures/'


##################################################################
#### Analysis of 2dv vs full DPS. both 2obj & 4obj problems
##################################################################
nseeds = 1
nobj_4 = 4
nobj_2 = 2
ncon = 1

#fn for cleaning set data
def getSet(file, nobj, has_dv = True, has_constraint = True, sort = True):
  # read data
  df = pd.read_csv(file, sep=' ', header=None).dropna(axis=1)
  if (has_dv == True):
    ndv = df.shape[1] - nobj - ncon
  else:
    ndv = 0
  # negate negative objectives
  df.iloc[:, ndv] *= -1
  # get colnames
  if (has_dv == True):
    names = np.array(['dv1'])
    for i in range(2, ndv + 1):
      names = np.append(names, 'dv' + str(i))
    names = np.append(names, 'annRev')
  else:
    names = np.array(['annRev'])
  names = np.append(names, 'maxDebt')
  if (nobj > 2):
    names = np.append(names, 'maxComplex')
    names = np.append(names, 'maxFund')
  if (has_constraint == True):
    names = np.append(names, 'constraint')
  df.columns = names
  # sort based on objective values
  if (sort == True):
    if (nobj == 4):
      df = df.sort_values(by=list(df.columns[-5:-1])).reset_index(drop=True)
    else:
      df = df.sort_values(by=list(df.columns[-5:-1])).reset_index(drop=True)
  return df, ndv


dps = getSet(dir_data + 'optimization_output/4obj_2rbf_moreSeeds/DPS_4obj_2rbf_moreSeeds_borg_retest.resultfile', 4, sort=False)[0]
nsolns = dps.shape[0]

##################################################################
#### Constants
##################################################################
DPS_RUN_TYPE = 1          # 0: 2dv version; 1: full DPS with RBFs, maxDebt formulation; 2: full DPS with RBFs, minRev formulation
#BORG_RUN_TYPE 1       # 0: single run no borg; 1: borg run, serial; 2: borg parallel for cluster;
NUM_YEARS = 20                   #20yr sims
NUM_SAMPLES = 50000
NUM_DECISIONS_TOTAL = 2           # each year, have to choose value snow contract + adjusted revenue
#NUM_LINES_STOCHASTIC_INPUT 999999    #Input file samp.txt has 1M rows, 6 cols.
#NUM_VARIABLES_STOCHASTIC_INPUT 6            #6 cols in input: swe,powIndex,revRetail,revWholesale,sswp,pswp
#INDEX_STOCHASTIC_REVENUE 2   # 2 = revRetail, 3 = revWholesale
#INDEX_STOCHASTIC_SNOW_PAYOUT 4    # 4 = sswp
#INDEX_STOCHASTIC_POWER_INDEX 1  # 1 = power index
#if INDEX_STOCHASTIC_REVENUE == 2
MEAN_REVENUE = 127.80086602479503    # mean revenue in absense of any financial risk mgmt. Make sure this is consistent with current input HHSamp revenue column.
#elif INDEX_STOCHASTIC_REVENUE == 3
#MEAN_REVENUE  70.08967184742373     # mean revenue in absense of any financial risk mgmt. Make sure this is consistent with current input HHSamp revenue column.
#endif
NORMALIZE_SNOW_CONTRACT_SIZE = 4.0
NORMALIZE_REVENUE = 250.0
NORMALIZE_FUND = 150.0
NORMALIZE_POWER_PRICE = 350.0
NORMALIZE_SWE = 150.0
#BUFFER_MAX_SIZE 5000
EPS = 0.0000000000001
NUM_OBJECTIVES = 4
#EPS_ANNREV 0.075
#EPS_MAXDEBT 0.225
#EPS_MINREV 0.225
#EPS_MAXCOMPLEXITY 0.05
#EPS_MAXFUND 0.225
#if DPS_RUN_TYPE<2
#NUM_CONSTRAINTS 1
#else
#NUM_CONSTRAINTS 0
#endif
#EPS_CONS1 0.05
NUM_RBF = 2       # number of radial basis functions
SHARED_RBFS = 1     # 1 = 1 rbf shared between hedge and withdrawal policies. 0 = separate rbf for each. 2 = rbf for hedge, and 2dv formulation for withdrawal.
if (SHARED_RBFS == 2):
  NUM_INPUTS_RBF = 3    # inputs: fund balance, debt, power index
else:
  NUM_INPUTS_RBF = 4    # inputs: fund balance, debt, power index, rev+hedge cash flow
if (DPS_RUN_TYPE > 0):
  if (SHARED_RBFS == 0):
    NUM_DV = (2 * NUM_DECISIONS_TOTAL * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * (NUM_RBF + 2))
  elif (SHARED_RBFS == 1):
    NUM_DV = (2 * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * (NUM_RBF + 2))
  else:
    NUM_DV = (2 * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * 2) + NUM_RBF
else:
  NUM_DV = 2
MIN_SNOW_CONTRACT= 0.05          # DPS_RUN_TYPE==0 only: if contract slope dv < $0.05M/inch, act as if 0.
MIN_MAX_FUND= 0.05               # DPS_RUN_TYPE==0 only: if max fund dv < $0.05M, act as if 0.
#NUM_PARAM 6         # cost_fraction, discount_rate, delta_interest_fund, delta_interest_debt, lambda, lambda_prem_shift
#NUM_PARAM_SAMPLES 1  # number of LHC samples in param file. Last line is values for SFPUC, Oct 2016.

fixed_cost = 0.914
delta = 0.4
Delta_fund = -1.73
Delta_debt = 1
lam = 0.25
lam_prem_shift = 0

samp = pd.read_csv(dir_data + 'generated_inputs/synthetic_data.txt', delimiter=' ')

# get financial params
discount_rate = 1 / (delta/100 + 1)
interest_fund = (Delta_fund + delta)/100 + 1
interest_debt = (Delta_debt + delta)/100 + 1
discount_factor = discount_rate ** np.arange(1, NUM_YEARS+1)
discount_normalization = 1 / np.sum(discount_factor)



# get dv types
def getDV(dvs):
    dv_d = np.zeros(NUM_DECISIONS_TOTAL)
    if (SHARED_RBFS == 0):
      dv_c = np.zeros(NUM_RBF * NUM_INPUTS_RBF * NUM_DECISIONS_TOTAL)
      dv_b = np.zeros(NUM_RBF * NUM_INPUTS_RBF * NUM_DECISIONS_TOTAL)
    else:
      dv_c = np.zeros(NUM_RBF * NUM_INPUTS_RBF)
      dv_b = np.zeros(NUM_RBF * NUM_INPUTS_RBF)
    if (SHARED_RBFS == 2):
      dv_w = np.zeros(NUM_RBF)
    else:
      dv_w = np.zeros(NUM_RBF * NUM_DECISIONS_TOTAL)
    dv_a = np.zeros(NUM_DECISIONS_TOTAL)
    objDPS = np.zeros(NUM_OBJECTIVES)
    for i in range(len(dv_d)):
        dv_d[i] = dvs[i]
    for i in range(len(dv_c)):
        dv_c[i] = dvs[i + len(dv_d)]
    for i in range(len(dv_b)):
        dv_b[i] = max(dvs[i + len(dv_d) + len(dv_c)],EPS)
    for i in range(len(dv_w)):
        dv_w[i] = dvs[i + len(dv_d) + len(dv_c) + len(dv_b)]
    for i in range(len(dv_a)):
        dv_a[i] = dvs[i + len(dv_d) + len(dv_c) + len(dv_b) + len(dv_w)]
    for i in range(NUM_OBJECTIVES):
        objDPS[i] = dvs[i + len(dv_d) + len(dv_c) + len(dv_b) + len(dv_w) + len(dv_a)]
    # normalize weights
    for j in range(NUM_DECISIONS_TOTAL):
        dum = np.sum(dv_w[(j * NUM_RBF):((j + 1) * NUM_RBF)])
        if dum > 0:
          dv_w[(j * NUM_RBF):((j + 1) * NUM_RBF)] = dv_w[(j * NUM_RBF):((j + 1) * NUM_RBF)] / dum
    return (dv_d, dv_c, dv_b, dv_w, dv_a)





def simulate(revenue, payout, power):
  ny = len(revenue) - 1
  net_rev = revenue - MEAN_REVENUE * fixed_cost
  fund = np.zeros(ny + 1)
  debt = np.zeros(ny + 1)
  adj_rev = np.zeros(ny)
  withdrawal = np.zeros(ny)
  value_snow_contract = np.zeros(ny)
  cash_in = np.zeros(ny)
  for i in range(ny):
    value_snow_contract[i] = policySnowContractValue(fund[i], debt[i], power[i])
    net_payout_snow_contract = value_snow_contract[i] * payout[i+1]
    cash_in[i] = net_rev[i+1] + net_payout_snow_contract - debt[i] * interest_debt
    withdrawal[i] = policyWithdrawal(fund[i]*interest_fund, debt[i]*interest_debt, power[i+1], cash_in[i])
    adj_rev[i] = cash_in[i] + withdrawal[i]
    fund[i+1] = fund[i]*interest_fund - withdrawal[i]
    if (adj_rev[i] < -EPS):
      debt[i+1] = -adj_rev[i]
      adj_rev[i] = 0
  return (fund[:-1], fund[:-1]*interest_fund, debt[:-1], debt[:-1]*interest_debt, power[:-1], power[1:], cash_in, value_snow_contract, withdrawal, adj_rev)





@np.vectorize
def policySnowContractValue(f_fund_balance, f_debt, f_power_price_index, useinrbf_fund_hedge=1, useinrbf_debt_hedge=1, useinrbf_power_hedge=1):
  decision_order = 0
  value = 0
  for i in range(NUM_RBF):
    # sum RBFs
    if (SHARED_RBFS == 1):
      arg_exp = -((f_fund_balance * useinrbf_fund_hedge / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i]) ** 2) \
                 / (dv_b[NUM_INPUTS_RBF * i]) ** 2
      arg_exp += -((f_debt * useinrbf_debt_hedge / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i + 1]) ** 2) \
                 / (dv_b[NUM_INPUTS_RBF * i + 1]) ** 2
      arg_exp += -((f_power_price_index * useinrbf_power_hedge / NORMALIZE_POWER_PRICE - dv_c[NUM_INPUTS_RBF * i + 2]) ** 2) \
                 / (dv_b[NUM_INPUTS_RBF * i + 2]) ** 2
    else:
      arg_exp = -((f_fund_balance * useinrbf_fund_hedge / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2) \
                 / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2
      arg_exp += -((f_debt * useinrbf_debt_hedge / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2) \
                 / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2
      arg_exp += -((f_power_price_index * useinrbf_power_hedge / NORMALIZE_POWER_PRICE - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2) \
                 / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2
    value += dv_w[decision_order * NUM_RBF + i] * np.exp(arg_exp)
  # add constant term & scale to [0, NORMALIZE_SNOW_CONTRACT_SIZE]
  value = max(min((value + dv_a[decision_order]) * NORMALIZE_SNOW_CONTRACT_SIZE, NORMALIZE_SNOW_CONTRACT_SIZE), 0)
  # enforce minimum contract size
  if (value < dv_d[decision_order] * NORMALIZE_SNOW_CONTRACT_SIZE):
    value = 0
  return (value)





### policyWithdrawal
@np.vectorize
def policyWithdrawal(f_fund_balance, f_debt, f_power_price_index, f_cash_in, useinrbf_fund_withdrawal=1, useinrbf_debt_withdrawal=1, useinrbf_power_withdrawal=1, useinrbf_cashin_withdrawal=1):
  decision_order = 1
  cash_out = 0
  for i in range(NUM_RBF):
    # sum RBFs
    if (SHARED_RBFS == 0):
      arg_exp = -((f_fund_balance * useinrbf_fund_withdrawal / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2) \
                    / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2\
                -((f_debt * useinrbf_debt_withdrawal / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2) \
                    / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2\
                -((f_power_price_index * useinrbf_power_withdrawal / NORMALIZE_POWER_PRICE - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2) \
                    / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2\
                -(((f_cash_in * useinrbf_cashin_withdrawal + NORMALIZE_REVENUE) / (2 * NORMALIZE_REVENUE) - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 3]) ** 2) \
                    / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 3]) ** 2
      cash_out += dv_w[decision_order * NUM_RBF + i] * np.exp(arg_exp)
    elif (SHARED_RBFS == 1):
      arg_exp = -((f_fund_balance * useinrbf_fund_withdrawal / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i]) ** 2) \
                    / (dv_b[NUM_INPUTS_RBF * i]) ** 2\
                -((f_debt * useinrbf_debt_withdrawal / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i + 1]) ** 2) \
                    / (dv_b[NUM_INPUTS_RBF * i + 1]) ** 2\
                -((f_power_price_index * useinrbf_power_withdrawal / NORMALIZE_POWER_PRICE - dv_c[NUM_INPUTS_RBF * i + 2]) ** 2) \
                    / (dv_b[NUM_INPUTS_RBF * i + 2]) ** 2\
                -(((f_cash_in * useinrbf_cashin_withdrawal + NORMALIZE_REVENUE) / (2 * NORMALIZE_REVENUE) - dv_c[NUM_INPUTS_RBF * i + 3]) ** 2) \
                    / (dv_b[NUM_INPUTS_RBF * i + 3]) ** 2
      cash_out += dv_w[decision_order * NUM_RBF + i] * np.exp(arg_exp)
    else:
      cash_out += 0
  # add constant term
  cash_out += dv_a[decision_order]
  # now scale back to [-NORMALIZE_REVENUE,NORMALIZE_REVENUE]
  cash_out = max(min((cash_out * 2 * NORMALIZE_REVENUE) - NORMALIZE_REVENUE, NORMALIZE_REVENUE), -NORMALIZE_REVENUE)
  # now write as withdrawal for policy return
  withdrawal = cash_out - f_cash_in
  # ensure that cant withdraw more than fund balance
  if (withdrawal > EPS):
    withdrawal = min(withdrawal, f_fund_balance)
  elif (withdrawal < -EPS):
    withdrawal = max(withdrawal, -max(f_cash_in, 0))
  if ((f_fund_balance - withdrawal) > dv_d[decision_order] * NORMALIZE_FUND):
    withdrawal = (f_fund_balance - (dv_d[decision_order] * NORMALIZE_FUND))
  return withdrawal




### vis results from MI analysis(after running calculate_entropic_SA.py & consolidate_SA_output.py)
dat = pd.read_csv(dir_data + 'policy_simulation/mi_combined.csv', index_col=0).sort_index()


### view information of policies
lims = {}
# lims['entropy'] = {'min': np.min(dat.hedge_entropy), 'max':np.max(dat.hedge_entropy)}
# lims['mi'] = {'min': min(np.min(dat.fund_hedge_mi), np.min(dat.debt_hedge_mi), np.min(dat.power_hedge_mi)),
#               'max': max(np.max(dat.fund_hedge_mi), np.max(dat.debt_hedge_mi), np.max(dat.power_hedge_mi))}
lims['entropy'] = {'min': 0, 'max': 5}
lims['mi'] = {'min': 0, 'max': 1}

lims3d = {'annRev':[9.4,11.13],'maxDebt':[0.,36.],'maxComplex':[0.,1.]}





# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# zs = dat.annRev
# ys = dat.maxDebt
# xs = dat.maxComplex
# ss = [20 + 1.3 * x for x in dat.maxFund]
# cs = dat.hedge_entropy
# p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap=cmap_vir, vmin=lims['entropy']['min'], vmax=lims['entropy']['max'], marker='v', alpha=0.4)
# cb = fig.colorbar(p1, ax=ax)
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([12, 24, 36])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[mean_net_revenue+0.05],marker='*',ms=15,c='k')
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# plt.savefig(dir_figs + 'entropy_cb.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)


## get example policies for each sensitivity index
dat.fund_hedge_mi = dat.fund_hedge_mi.fillna(-1)
dat.debt_hedge_mi = dat.debt_hedge_mi.fillna(-1)
dat.power_hedge_mi = dat.power_hedge_mi.fillna(-1)
dat.fund_withdrawal_mi = dat.fund_withdrawal_mi.fillna(-1)
dat.debt_withdrawal_mi = dat.debt_withdrawal_mi.fillna(-1)
dat.power_withdrawal_mi = dat.power_withdrawal_mi.fillna(-1)
dat.cash_withdrawal_mi = dat.cash_withdrawal_mi.fillna(-1)


fund_hedge_mi_max = [np.array(dat.fund_hedge_mi).argsort()[-2]]
debt_hedge_mi_max = [np.array(dat.debt_hedge_mi).argsort()[-2]]
power_hedge_mi_max = [np.array(dat.power_hedge_mi).argsort()[-4]]
fund_withdrawal_mi_max = []# [np.array(dat.fund_withdrawal_mi).argsort()[-1]]
debt_withdrawal_mi_max = []# [np.array(dat.debt_withdrawal_mi).argsort()[-1]]
power_withdrawal_mi_max = []# [np.array(dat.power_withdrawal_mi).argsort()[-1]]
cash_withdrawal_mi_max = []# [np.array(dat.cash_withdrawal_mi).argsort()[-1]]

# print(fund_hedge_mi_max, debt_hedge_mi_max, power_hedge_mi_max)


cmap_vir_mi = cm.get_cmap('viridis')
cmap_vir_mi.set_under('grey')

def plot_MI(dat, mi_name, example_pol, name):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection='3d')
  zs = dat.annRev
  ys = dat.maxDebt
  xs = dat.maxComplex
  ss = [20 + 1.3 * x for x in dat.maxFund]
  cs = dat[mi_name]
  p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap=cmap_vir_mi, vmin=lims['mi']['min'], vmax=lims['mi']['max'], marker='v', alpha=0.4)
  if len(example_pol) > 0:
    zs = [dat.annRev[x] for x in example_pol]
    ys = [dat.maxDebt[x] for x in example_pol]
    xs = [dat.maxComplex[x] for x in example_pol]
    ss = [20 + 1.3 * x for x in [dat.maxFund[x] for x in example_pol]]
    cs = [dat[mi_name][x] for x in example_pol]
    p1 = ax.scatter(xs[0], ys[0], zs[0], s=ss, c=cs, cmap=cmap_vir_mi, vmin=lims['mi']['min'], vmax=lims['mi']['max'], marker='v', edgecolors='k', lw=1.5)
  ax.set_xticks([0,0.25,0.5,0.75])
  ax.set_yticks([12, 24, 36])
  ax.set_zticks([9.5,10,10.5,11])
  ax.view_init(elev=20, azim =-45)
  ax.plot([0.01],[0.01],[mean_net_revenue+0.05],marker='*',ms=15,c='k')
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  plt.savefig(dir_figs + name + '.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)


plot_MI(dat, 'fund_hedge_mi', fund_hedge_mi_max, 'mi_fund_hedge')
plot_MI(dat, 'debt_hedge_mi', debt_hedge_mi_max, 'mi_debt_hedge')
plot_MI(dat, 'power_hedge_mi', power_hedge_mi_max, 'mi_power_hedge')
plot_MI(dat, 'fund_withdrawal_mi', fund_withdrawal_mi_max, 'mi_fund_withdrawal')
plot_MI(dat, 'debt_withdrawal_mi', debt_withdrawal_mi_max, 'mi_debt_withdrawal')
plot_MI(dat, 'power_withdrawal_mi', power_withdrawal_mi_max, 'mi_power_withdrawal')
plot_MI(dat, 'cash_withdrawal_mi', cash_withdrawal_mi_max, 'mi_cash_withdrawal')


### table 2 output
tab2 = pd.read_csv('../../figures/table2.csv', index_col=0)

tab2['fund_hedge_mi'] = np.nan
tab2['debt_hedge_mi'] = np.nan
tab2['power_hedge_mi'] = np.nan

for i in range(dat.shape[0]):
  for j in range(2, 4):
    if (abs(dat.annRev[i] - tab2.annRev[j]) < 1e-6) and (abs(dat.maxDebt[i] - tab2.maxDebt[j]) < 1e-6) and (abs(dat.maxComplex[i] - tab2.maxComplex[j]) < 1e-6) and (abs(dat.maxFund[i] - tab2.maxFund[j]) < 1e-6):
      tab2['fund_hedge_mi'][j] = dat.fund_hedge_mi[i]
      tab2['debt_hedge_mi'][j] = dat.debt_hedge_mi[i]
      tab2['power_hedge_mi'][j] = dat.power_hedge_mi[i]

for i in [fund_hedge_mi_max[0], debt_hedge_mi_max[0], power_hedge_mi_max[0]]:
  tab2 = tab2.append(dat[['annRev','maxDebt','maxComplex','maxFund','fund_hedge_mi','debt_hedge_mi','power_hedge_mi']].iloc[i, :])
tab2.reset_index(inplace=True, drop=True)
tab2.formulation.iloc[-3:] = '4obj_dynamic_MI'

print(tab2)





### get dataframe of simulation results, output for parallel coords in R
policy_ranks = [fund_hedge_mi_max[0], debt_hedge_mi_max[0], power_hedge_mi_max[0]]
ny = 20
ns = 20

### get input samples
samples = np.random.choice([int(x) for x in np.arange(1e6 - 21)], size=ns, replace=True)
samp_rev = samp.revenue.values
samp_sswp = samp.payoutCfd.values
samp_pow = samp.power.values

### loop over policies in pareto set
df = pd.DataFrame({'fund_hedge':[], 'fund_withdrawal':[], 'debt_hedge':[], 'debt_withdrawal':[], 'power_hedge':[],
                   'power_withdrawal':[], 'cash_in':[], 'action_hedge':[], 'action_withdrawal':[], 'policy':[]})
for m in policy_ranks:
  name = 'm'+str(m)
  # get policy params
  dps_choice = dps.iloc[m,:]
  dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
 
  ### simulate ns*ny trajectories
  results = np.empty([ns*ny, 9])
  for s in range(ns):
    start, end = s*ny, ny * (s + 1)
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp_rev[samples[s]:(samples[s]+ny+1)], samp_sswp[samples[s]:(samples[s]+ny+1)], samp_pow[samples[s]:(samples[s]+ny+1)])
    results[start:end, 0] = fund_hedge
    results[start:end, 1] = fund_withdrawal
    results[start:end, 2] = debt_hedge
    results[start:end, 3] = debt_withdrawal
    results[start:end, 4] = power_hedge
    results[start:end, 5] = power_withdrawal
    results[start:end, 6] = cash_in
    results[start:end, 7] = action_hedge
    results[start:end, 8] = action_withdrawal

  df = df.append(pd.DataFrame({'fund_hedge':results[:, 0], 'fund_withdrawal':results[:, 1], 'debt_hedge':results[:, 2], 'debt_withdrawal':results[:, 3],
                               'power_hedge':results[:, 4], 'power_withdrawal':results[:, 5], 'cash_in':results[:, 6], 'action_hedge':results[:, 7],
                               'action_withdrawal':results[:, 8], 'policy':m}))
df.to_csv(dir_data + 'policy_simulation/mi_examples_simulation.csv', index=False)


