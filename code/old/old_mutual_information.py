import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import seaborn as sns
import matplotlib as mpl
from matplotlib import ticker, cm, colors
from mpl_toolkits.mplot3d import Axes3D
import copy
import itertools


sns.set_style('white')
# sns.set_context('paper', font_scale=3)

eps = 1e-10
ny = 20
ns = 10000

cmap_vir = cm.get_cmap('viridis')
col_vir = [cmap_vir(0.1),cmap_vir(0.3),cmap_vir(0.6),cmap_vir(0.8)]
cmap_blues = cm.get_cmap('Blues_r')
col_blues = [cmap_blues(0.1),cmap_blues(0.3),cmap_blues(0.6),cmap_blues(0.8)]
cmap_reds = cm.get_cmap('Reds_r')
col_reds = [cmap_reds(0.1),cmap_reds(0.3),cmap_reds(0.6),cmap_reds(0.8)]

dir_data = 'C:/Users/Andrew/Documents/DataFilesNonGit/DPS_results/DPSManuscript/'
dir_figs = 'C:/Users/Andrew/Documents/MeetingsAbstractsEtc/Hamilton_DPS_formulation/figs/'

# ##################################################################
# #### Analysis of 2dv vs full DPS. both 2obj & 4obj problems
# ##################################################################
# nseeds = 1
# nobj_4 = 4
# nobj_2 = 2
# ncon = 1
#
# #fn for cleaning set data
# def getSet(file, nobj, has_dv = True, has_constraint = True, sort = True):
#   # read data
#   df = pd.read_csv(file, sep=' ', header=None).dropna(axis=1)
#   if (has_dv == True):
#     ndv = df.shape[1] - nobj - ncon
#   else:
#     ndv = 0
#   # negate negative objectives
#   df.iloc[:, ndv] *= -1
#   # get colnames
#   if (has_dv == True):
#     names = np.array(['dv1'])
#     for i in range(2, ndv + 1):
#       names = np.append(names, 'dv' + str(i))
#     names = np.append(names, 'annRev')
#   else:
#     names = np.array(['annRev'])
#   names = np.append(names, 'maxDebt')
#   if (nobj > 2):
#     names = np.append(names, 'maxComplex')
#     names = np.append(names, 'maxFund')
#   if (has_constraint == True):
#     names = np.append(names, 'constraint')
#   df.columns = names
#   # sort based on objective values
#   if (sort == True):
#     if (nobj == 4):
#       df = df.sort_values(by=list(df.columns[-5:-1])).reset_index(drop=True)
#     else:
#       df = df.sort_values(by=list(df.columns[-5:-1])).reset_index(drop=True)
#   return df, ndv
#
#
# dps = getSet(dir_data + 'DPS_2wd_F1D1P1/DPS_2wd_F1D1P1_borg_retest.resultfile', 4, sort=True)[0]
#
#
# ######################################################################
# DPS_RUN_TYPE = 1          # 0: 2dv version; 1: full DPS with RBFs, maxDebt formulation; 2: full DPS with RBFs, minRev formulation
# #BORG_RUN_TYPE 1       # 0: single run no borg; 1: borg run, serial; 2: borg parallel for cluster;
# NUM_YEARS = 20                   #20yr sims
# NUM_SAMPLES = 50000
# NUM_DECISIONS_TOTAL = 2           # each year, have to choose value snow contract + adjusted revenue
# #NUM_LINES_STOCHASTIC_INPUT 999999    #Input file samp.txt has 1M rows, 6 cols.
# #NUM_VARIABLES_STOCHASTIC_INPUT 6            #6 cols in input: swe,powIndex,revRetail,revWholesale,sswp,pswp
# #INDEX_STOCHASTIC_REVENUE 2   # 2 = revRetail, 3 = revWholesale
# #INDEX_STOCHASTIC_SNOW_PAYOUT 4    # 4 = sswp
# #INDEX_STOCHASTIC_POWER_INDEX 1  # 1 = power index
# #if INDEX_STOCHASTIC_REVENUE == 2
# MEAN_REVENUE = 128.48255822159567     # mean revenue in absense of any financial risk mgmt. Make sure this is consistent with current input HHSamp revenue column.
# #elif INDEX_STOCHASTIC_REVENUE == 3
# #MEAN_REVENUE  70.08967184742373     # mean revenue in absense of any financial risk mgmt. Make sure this is consistent with current input HHSamp revenue column.
# #endif
# NORMALIZE_SNOW_CONTRACT_SIZE = 4.0
# NORMALIZE_REVENUE = 250.0
# NORMALIZE_FUND = 150.0
# NORMALIZE_POWER_PRICE = 350.0
# NORMALIZE_SWE = 150.0
# #BUFFER_MAX_SIZE 5000
# EPS = 0.0000000000001
# NUM_OBJECTIVES = 4
# #EPS_ANNREV 0.075
# #EPS_MAXDEBT 0.225
# #EPS_MINREV 0.225
# #EPS_MAXCOMPLEXITY 0.05
# #EPS_MAXFUND 0.225
# #if DPS_RUN_TYPE<2
# #NUM_CONSTRAINTS 1
# #else
# #NUM_CONSTRAINTS 0
# #endif
# #EPS_CONS1 0.05
# NUM_RBF = 4       # number of radial basis functions
# SHARED_RBFS = 2     # 0 = 1 rbf shared between hedge and withdrawal policies. 1 = separate rbf for each. 2 = rbf for hedge, and 2dv formulation for withdrawal.
# if (SHARED_RBFS == 2):
#   NUM_INPUTS_RBF = 3    # inputs: fund balance, debt, power index
# else:
#   NUM_INPUTS_RBF = 4    # inputs: fund balance, debt, power index, rev+hedge cash flow
# if (DPS_RUN_TYPE > 0):
#   if (SHARED_RBFS == 0):
#     NUM_DV = (2 * NUM_DECISIONS_TOTAL * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * (NUM_RBF + 2))
#   elif (SHARED_RBFS == 1):
#     NUM_DV = (2 * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * (NUM_RBF + 2))
#   else:
#     NUM_DV = (2 * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * 2) + NUM_RBF
# else:
#   NUM_DV = 2
# MIN_SNOW_CONTRACT= 0.05          # DPS_RUN_TYPE==0 only: if contract slope dv < $0.05M/inch, act as if 0.
# MIN_MAX_FUND= 0.05               # DPS_RUN_TYPE==0 only: if max fund dv < $0.05M, act as if 0.
# #NUM_PARAM 6         # cost_fraction, discount_rate, delta_interest_fund, delta_interest_debt, lambda, lambda_prem_shift
# #NUM_PARAM_SAMPLES 1  # number of LHC samples in param file. Last line is values for SFPUC, Oct 2016.
#
# fixed_cost = 0.914
# delta = 0.4
# Delta_fund = -1.73
# Delta_debt = 1
# lam = 0.25
# lam_prem_shift = 0
#
# samp = pd.read_csv('../../Documents/DataFilesNonGit/HHsamp10132019.txt', delimiter=' ')
#
# # get financial params
# discount_rate = 1 / (delta/100 + 1)
# interest_fund = (Delta_fund + delta)/100 + 1
# interest_debt = (Delta_debt + delta)/100 + 1
# discount_factor = discount_rate ** np.arange(1, NUM_YEARS+1)
# discount_normalization = 1 / np.sum(discount_factor)
#
# # get dv types
# def getDV(dvs):
#     dv_d = np.zeros(NUM_DECISIONS_TOTAL)
#     if (SHARED_RBFS == 0):
#       dv_c = np.zeros(NUM_RBF * NUM_INPUTS_RBF * NUM_DECISIONS_TOTAL)
#       dv_b = np.zeros(NUM_RBF * NUM_INPUTS_RBF * NUM_DECISIONS_TOTAL)
#     else:
#       dv_c = np.zeros(NUM_RBF * NUM_INPUTS_RBF)
#       dv_b = np.zeros(NUM_RBF * NUM_INPUTS_RBF)
#     if (SHARED_RBFS == 2):
#       dv_w = np.zeros(NUM_RBF)
#     else:
#       dv_w = np.zeros(NUM_RBF * NUM_DECISIONS_TOTAL)
#     dv_a = np.zeros(NUM_DECISIONS_TOTAL)
#     objDPS = np.zeros(NUM_OBJECTIVES)
#     for i in range(len(dv_d)):
#         dv_d[i] = dvs[i]
#     for i in range(len(dv_c)):
#         dv_c[i] = dvs[i + len(dv_d)]
#     for i in range(len(dv_b)):
#         dv_b[i] = max(dvs[i + len(dv_d) + len(dv_c)],EPS)
#     for i in range(len(dv_w)):
#         dv_w[i] = dvs[i + len(dv_d) + len(dv_c) + len(dv_b)]
#     for i in range(len(dv_a)):
#         dv_a[i] = dvs[i + len(dv_d) + len(dv_c) + len(dv_b) + len(dv_w)]
#     for i in range(NUM_OBJECTIVES):
#         objDPS[i] = dvs[i + len(dv_d) + len(dv_c) + len(dv_b) + len(dv_w) + len(dv_a)]
#     # normalize weights
#     for j in range(NUM_DECISIONS_TOTAL):
#         dum = np.sum(dv_w[(j * NUM_RBF):((j + 1) * NUM_RBF)])
#         dv_w[(j * NUM_RBF):((j + 1) * NUM_RBF)] = dv_w[(j * NUM_RBF):((j + 1) * NUM_RBF)] / dum
#     return (dv_d, dv_c, dv_b, dv_w, dv_a)
#
#
#
#
#
# def simulate(revenue, payout, power):
#   ny = len(revenue) - 1
#   net_rev = revenue - MEAN_REVENUE * fixed_cost
#   fund = np.zeros(ny + 1)
#   debt = np.zeros(ny + 1)
#   adj_rev = np.zeros(ny)
#   withdrawal = np.zeros(ny)
#   value_snow_contract = np.zeros(ny)
#   cash_in = np.zeros(ny)
#   for i in range(ny):
#     value_snow_contract[i] = policySnowContractValue(fund[i], debt[i], power[i])
#     net_payout_snow_contract = value_snow_contract[i] * payout[i+1]
#     cash_in[i] = net_rev[i+1] + net_payout_snow_contract - debt[i] * interest_debt
#     withdrawal[i] = policyWithdrawal(fund[i]*interest_fund, debt[i]*interest_debt, power[i+1], cash_in[i])
#     adj_rev[i] = cash_in[i] + withdrawal[i]
#     fund[i+1] = fund[i]*interest_fund - withdrawal[i]
#     if (adj_rev[i] < -EPS):
#       debt[i+1] = -adj_rev[i]
#       adj_rev[i] = 0
#   return (fund[:-1], fund[:-1]*interest_fund, debt[:-1], debt[:-1]*interest_debt, power[:-1], power[1:], cash_in, value_snow_contract, withdrawal, adj_rev)
#
#
#
#
#
# @np.vectorize
# def policySnowContractValue(f_fund_balance, f_debt, f_power_price_index, useinrbf_fund_hedge=1, useinrbf_debt_hedge=1, useinrbf_power_hedge=1):
#   decision_order = 0
#   value = 0
#   for i in range(NUM_RBF):
#     # sum RBFs
#     if (SHARED_RBFS == 1):
#       arg_exp = -((f_fund_balance * useinrbf_fund_hedge / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i]) ** 2) \
#                  / (dv_b[NUM_INPUTS_RBF * i]) ** 2
#       arg_exp += -((f_debt * useinrbf_debt_hedge / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i + 1]) ** 2) \
#                  / (dv_b[NUM_INPUTS_RBF * i + 1]) ** 2
#       arg_exp += -((f_power_price_index * useinrbf_power_hedge / NORMALIZE_POWER_PRICE - dv_c[NUM_INPUTS_RBF * i + 2]) ** 2) \
#                  / (dv_b[NUM_INPUTS_RBF * i + 2]) ** 2
#     else:
#       arg_exp = -((f_fund_balance * useinrbf_fund_hedge / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2) \
#                  / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2
#       arg_exp += -((f_debt * useinrbf_debt_hedge / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2) \
#                  / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2
#       arg_exp += -((f_power_price_index * useinrbf_power_hedge / NORMALIZE_POWER_PRICE - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2) \
#                  / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2
#     value += dv_w[decision_order * NUM_RBF + i] * np.exp(arg_exp)
#   # add constant term & scale to [0, NORMALIZE_SNOW_CONTRACT_SIZE]
#   value = max(min((value + dv_a[decision_order]) * NORMALIZE_SNOW_CONTRACT_SIZE, NORMALIZE_SNOW_CONTRACT_SIZE), 0)
#   # enforce minimum contract size
#   if (value < dv_d[decision_order] * NORMALIZE_SNOW_CONTRACT_SIZE):
#     value = 0
#   return (value)
#
#
#
#
#
# ### policyWithdrawal
# @np.vectorize
# def policyWithdrawal(f_fund_balance, f_debt, f_power_price_index, f_cash_in, useinrbf_fund_withdrawal=1, useinrbf_debt_withdrawal=1, useinrbf_power_withdrawal=1, useinrbf_cashin_withdrawal=1):
#   decision_order = 1
#   cash_out = 0
#   for i in range(NUM_RBF):
#     # sum RBFs
#     if (SHARED_RBFS == 0):
#       arg_exp = -((f_fund_balance * useinrbf_fund_withdrawal / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2) \
#                     / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i]) ** 2\
#                 -((f_debt * useinrbf_debt_withdrawal / NORMALIZE_FUND - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2) \
#                     / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1]) ** 2\
#                 -((f_power_price_index * useinrbf_power_withdrawal / NORMALIZE_POWER_PRICE - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2) \
#                     / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2]) ** 2\
#                 -(((f_cash_in * useinrbf_cashin_withdrawal + NORMALIZE_REVENUE) / (2 * NORMALIZE_REVENUE) - dv_c[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 3]) ** 2) \
#                     / (dv_b[(decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 3]) ** 2
#       cash_out += dv_w[decision_order * NUM_RBF + i] * np.exp(arg_exp)
#     elif (SHARED_RBFS == 1):
#       arg_exp = -((f_fund_balance * useinrbf_fund_withdrawal / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i]) ** 2) \
#                     / (dv_b[NUM_INPUTS_RBF * i]) ** 2\
#                 -((f_debt * useinrbf_debt_withdrawal / NORMALIZE_FUND - dv_c[NUM_INPUTS_RBF * i + 1]) ** 2) \
#                     / (dv_b[NUM_INPUTS_RBF * i + 1]) ** 2\
#                 -((f_power_price_index * useinrbf_power_withdrawal / NORMALIZE_POWER_PRICE - dv_c[NUM_INPUTS_RBF * i + 2]) ** 2) \
#                     / (dv_b[NUM_INPUTS_RBF * i + 2]) ** 2\
#                 -(((f_cash_in * useinrbf_cashin_withdrawal + NORMALIZE_REVENUE) / (2 * NORMALIZE_REVENUE) - dv_c[NUM_INPUTS_RBF * i + 3]) ** 2) \
#                     / (dv_b[NUM_INPUTS_RBF * i + 3]) ** 2
#       cash_out += dv_w[decision_order * NUM_RBF + i] * np.exp(arg_exp)
#     else:
#       cash_out += 0
#   # add constant term
#   cash_out += dv_a[decision_order]
#   # now scale back to [-NORMALIZE_REVENUE,NORMALIZE_REVENUE]
#   cash_out = max(min((cash_out * 2 * NORMALIZE_REVENUE) - NORMALIZE_REVENUE, NORMALIZE_REVENUE), -NORMALIZE_REVENUE)
#   # now write as withdrawal for policy return
#   withdrawal = cash_out - f_cash_in
#   # ensure that cant withdraw more than fund balance
#   if (withdrawal > EPS):
#     withdrawal = min(withdrawal, f_fund_balance)
#   elif (withdrawal < -EPS):
#     withdrawal = max(withdrawal, -max(f_cash_in, 0))
#   if ((f_fund_balance - withdrawal) > dv_d[decision_order] * NORMALIZE_FUND):
#     withdrawal = (f_fund_balance - (dv_d[decision_order] * NORMALIZE_FUND))
#   return withdrawal
#
#
# # # policy_ranks = [9,498,2059]
# # policy_ranks = range(1661,len(dps))
# # # policies = ['LowDebt','MedDebt','HighDebt']
# #
# # zmax = -100
# # zmin = 100
# # # dict = {}
# # samples = np.random.choice([int(x) for x in np.arange(1e6 - 21)], size=ns, replace=True)
# # samp_rev = samp.revRetail.values
# # samp_sswp = samp.sswp.values
# # samp_pow = samp.powIndex.values
# # for m in policy_ranks:
# #   name = 'm'+str(m)
# #   dict = {}
# #   # get policy params
# #   dps_choice = dps.iloc[m,:]
# #   dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
# #   # get trajectories through state space
# #   dict['annRev'] = dps_choice['annRev']
# #   dict['maxDebt'] = dps_choice['maxDebt']
# #   dict['maxComplex'] = dps_choice['maxComplex']
# #   dict['maxFund'] = dps_choice['maxFund']
# #   dict['fund_hedge'] = np.array([])
# #   # dict['fund_withdrawal'] = np.array([])
# #   dict['debt_hedge'] = np.array([])
# #   # dict['debt_withdrawal'] = np.array([])
# #   dict['power_hedge'] = np.array([])
# #   # dict['power_withdrawal'] = np.array([])
# #   # dict['cash_in'] = np.array([])
# #   dict['action_hedge'] = np.array([])
# #   # dict['action_withdrawal'] = np.array([])
# #   for s in range(ns):
# #     fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp_rev[samples[s]:(samples[s]+ny+1)], samp_sswp[samples[s]:(samples[s]+ny+1)], samp_pow[samples[s]:(samples[s]+ny+1)])
# #     dict['fund_hedge'] = np.append(dict['fund_hedge'], fund_hedge)
# #     # dict['fund_withdrawal'] = np.append(dict['fund_withdrawal'], fund_withdrawal)
# #     dict['debt_hedge'] = np.append(dict['debt_hedge'], debt_hedge)
# #     # dict['debt_withdrawal'] = np.append(dict['debt_withdrawal'], debt_withdrawal)
# #     dict['power_hedge'] = np.append(dict['power_hedge'], power_hedge)
# #     # dict['power_withdrawal'] = np.append(dict['power_withdrawal'], power_withdrawal)
# #     # dict['cash_in'] = np.append(dict['cash_in'], cash_in)
# #     dict['action_hedge'] = np.append(dict['action_hedge'], action_hedge)
# #     # dict['action_withdrawal'] = np.append(dict['action_withdrawal'], action_withdrawal)
# #   pd.to_pickle(dict, 'data/save_dps_samples/'+str(m)+'.pkl')
# #   print(m)
#
#
#
#
#
#
#
#
#
#
#
# # policy_ranks = [9,498,2059]
# # policies = ['LowDebt','MedDebt','HighDebt']
#
#
#
# # names = ['m0','m1','m2']
# # ranges = {'fund_hedge':[100,0], 'debt_hedge':[100,0], 'power_hedge':[100,0], 'action_hedge':[4,0]}
# # for name in names:
# #   for att in list(ranges.keys()):
# #     ranges[att][0] = min(ranges[att][0], dict[name][att].min())
# #     ranges[att][1] = max(ranges[att][1], dict[name][att].max())
# # ranges = {'fund_hedge':[0,35], 'debt_hedge':[0,55], 'power_hedge':[30,60], 'action_hedge':[0,2]}
#
# def getProbGrid(xdat, ydat, xdomain, ydomain):
#   xmin = xdomain[0]
#   xmax = xdomain[1]
#   ymin = ydomain[0]
#   ymax = ydomain[1]
#   yrange = ymax - ymin
#   xrange = xmax - xmin
#
#   nstep = 10
#   dy = yrange / nstep
#   dx = xrange / nstep
#   freqMat = np.zeros([nstep, nstep])
#   for i in range(len(ydat)):
#     row = int(np.floor((ydat[i] - ymin) / dy))
#     col = int(np.floor((xdat[i] - xmin) / dx))
#     freqMat[row, col] += 1
#   freqMat /= len(ydat)
#   freqMat = np.log10(freqMat)
#   freqMat[freqMat == -np.inf] = np.nan
#   return freqMat
#
# # freqMat = {}
# # attPairs = ['fund_fund','fund_debt','fund_power','fund_action','debt_fund','debt_debt','debt_power','debt_action','power_fund','power_debt','power_power','power_action','action_fund','action_debt','action_power','action_action']
# # for name in names:
# #   freqMat[name] = {}
# #   for attPair in attPairs:
# #     [att1, att2] = attPair.split(sep='_')
# #     freqMat[name][attPair + '_hedge'] = getProbGrid(dict[name][att1 + '_hedge'], dict[name][att2 + '_hedge'], ranges[att1 + '_hedge'],ranges[att2 + '_hedge'])
# #
# # ranges['freq_hedge'] = [0,-1000]
# # for name in names:
# #   for attPair in attPairs:
# #     ranges['freq_hedge'][0] = min(ranges['freq_hedge'][0], np.nanmin(freqMat[name][attPair + '_hedge']))
# #     ranges['freq_hedge'][1] = max(ranges['freq_hedge'][1], np.nanmax(freqMat[name][attPair + '_hedge']))
# #
# # for name in names:
# #   fig = plt.figure()
# #   col = 0
# #   row = -1
# #   for attPair in attPairs:
# #     row += 1
# #     if row == 4:
# #       row = 0
# #       col += 1
# #     ax = plt.subplot2grid((4, 4), (row, col))  # ,colspan=2)
# #     plt.imshow(freqMat[name][attPair + '_hedge'], cmap='RdYlBu_r', origin='lower',
# #                norm=mpl.colors.Normalize(vmin=ranges['freq_hedge'][0], vmax=ranges['freq_hedge'][1]))
#
#
#
# def sort_bins(d, n_bins, try_sep_bins):
#   separate_min = False
#   separate_max = False
#   if (try_sep_bins == True):
#     if (np.mean(np.abs(d - d.min()) < eps) > 0.05):
#       separate_min = True
#     if (np.mean(np.abs(d - d.max()) < eps) > 0.05):
#       separate_max = True
#   if (separate_min):
#     d_min_val = d.min()
#     d_min_n = np.sum(np.abs(d - d_min_val) < eps)
#     d_interior = d[np.abs(d - d_min_val) > eps]
#     n_bins -= 1
#   if (separate_max):
#     d_max_val = d.max()
#     d_max_n = np.sum(np.abs(d - d_max_val) < eps)
#     if (separate_min):
#       d_interior = d_interior[np.abs(d_interior - d_max_val) > eps]
#     else:
#       d_interior = d[np.abs(d - d_max_val) > eps]
#     n_bins -= 1
#   if (separate_min | separate_max):
#     (d_bins, bins, dum) = plt.hist(d_interior, bins=n_bins)
#   else:
#     (d_bins, bins, dum) = plt.hist(d, bins=n_bins)
#   if (separate_min):
#     bins = np.insert(bins, 0, d_min_val + eps)
#     bins = np.insert(bins, 0, d_min_val - eps)
#     d_bins = np.insert(d_bins, 0, 0)
#     d_bins = np.insert(d_bins, 0, d_min_n)
#     n_bins += 2
#   else:
#     bins[0] -= eps
#   if (separate_max):
#     bins = np.append(bins, d_max_val - eps)
#     bins = np.append(bins, d_max_val + eps)
#     d_bins = np.append(d_bins, 0)
#     d_bins = np.append(d_bins, d_max_n)
#     n_bins += 2
#   else:
#     bins[-1] += eps
#   bincenter = (bins[:-1] + bins[1:]) / 2
#   while (np.sum(d_bins < eps) > 0):
#     ind = np.where(d_bins < eps)[0]
#     d_bins = np.delete(d_bins, ind)
#     bins = np.delete(bins, ind + 1)
#     bincenter = np.delete(bincenter, ind)
#     n_bins -= 1
#   binpoint = np.digitize(d, bins=bins) - 1
#   binfreq = d_bins / len(d)
#   return binfreq, bincenter, binpoint
#
#
#
#
#
# def get_joint_probability(dat, name, atts):
#   nbintot = ns * ny
#   ndim = len(atts)
#   nbin = []
#   d = []
#   p = []
#   for i in range(ndim):
#     d.append(dat[name][atts[i] + '_binpoint'])
#     p.append(np.unique(dat[name][atts[i] + '_binpoint']))
#     nbin.append(len(p[i]))
#   joint_prob = np.histogramdd(np.transpose(d), bins=nbin)[0] / nbintot
#   dat[name]['joint_freq'] = joint_prob
#   return(dat)
#
#
#
# @np.vectorize
# def prob_log_prob(numerator_prob, denominator_prob = -1):
#   if numerator_prob < eps:
#     z = 0
#   else:
#     if (denominator_prob < eps):
#       z = numerator_prob * np.log2(numerator_prob)
#     else:
#       z = numerator_prob * (np.log2(1 / denominator_prob) - np.log2(1 / numerator_prob))
#   return z
#
#
# def get_entropy(probs):
#   entropy = - np.sum(prob_log_prob(probs))
#   return entropy
#
#
# def get_mutual_info(dat, name, atts_full, atts_mi):
#   mutual_info = 0
#   probs = dat[name]['joint_freq']
#   sets_mi = {}
#   for i in range(len(atts_mi)):
#     sets_mi[i] = {}
#     sets_mi[i]['n_att'] = len(atts_mi[i])
#     sets_mi[i]['n_grid'] = {}
#     sets_mi[i]['idx_base'] = {}
#     for att in atts_mi[i]:
#       sets_mi[i]['n_grid'][att] = len(dat[name][att + '_binfreq'])
#       sets_mi[i]['idx_base'][att] = [idx for idx, x in enumerate(list(sets_mi[0]['n_grid'].keys())) if x == att]
#   for j in range(len(sets_mi)):
#     probs_temp = probs.copy()
#     for i in range(len(atts_full)-1, -1, -1):
#       if ((atts_full[i] in atts_mi[j]) == False):
#         probs_temp = probs_temp.sum(i)
#     sets_mi[j]['probs_integrated'] = probs_temp
#   if (len(sets_mi[0]['n_grid'].keys()) == 2):
#     # two-pt mutual info (e.g. I(X1;Y))
#     for i in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[0]]):
#       for j in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[1]]):
#         numerator_prob = sets_mi[0]['probs_integrated'][i,j]
#         denominator_prob = sets_mi[1]['probs_integrated'][i] * sets_mi[2]['probs_integrated'][j]
#         mutual_info += prob_log_prob(numerator_prob, denominator_prob)
#   elif (len(sets_mi[0]['n_grid'].keys()) == 4):
#     if (len(sets_mi.keys())-1 == 4):
#       # 4-pt interaction (e.g. I(X1;X2;X3;Y))
#       for i in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[0]]):
#         for j in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[1]]):
#           for k in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[2]]):
#             for l in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[3]]):
#                 numerator_prob = sets_mi[0]['probs_integrated'][i, j, k, l]
#                 denominator_prob = sets_mi[1]['probs_integrated'][i] * sets_mi[2]['probs_integrated'][j] * sets_mi[3]['probs_integrated'][k] * sets_mi[4]['probs_integrated'][l]
#                 mutual_info += prob_log_prob(numerator_prob, denominator_prob)
#     else:
#       # 3+1 mutual interaction (e.g. I(X1,X2,X3;Y))
#       for i in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[0]]):
#         for j in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[1]]):
#           for k in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[2]]):
#             for l in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[3]]):
#                 numerator_prob = sets_mi[0]['probs_integrated'][i, j, k, l]
#                 denominator_prob = sets_mi[1]['probs_integrated'][i, j, k] * sets_mi[2]['probs_integrated'][l]
#                 mutual_info += prob_log_prob(numerator_prob, denominator_prob)
#   return mutual_info
#
# # h = get_entropy(dat[name]['action_binfreq'])
# # abcde = get_mutual_info(dat, name, ['t', 'fund', 'debt', 'power', 'action'], [['t', 'fund', 'debt', 'power', 'action'],['t', 'fund', 'debt', 'power'],['action']])
# # a=get_mutual_info(dat, name, ['t', 'fund', 'debt', 'power', 'action'], [['t', 'action'],['t'],['action']])
# # b=get_mutual_info(dat, name, ['t', 'fund', 'debt', 'power', 'action'], [['fund', 'action'],['fund'],['action']])
# # c=get_mutual_info(dat, name, ['t', 'fund', 'debt', 'power', 'action'], [['debt', 'action'],['debt'],['action']])
# # d=get_mutual_info(dat, name, ['t', 'fund', 'debt', 'power', 'action'], [['power', 'action'],['power'],['action']])
# # e=get_mutual_info(dat, name, ['t', 'fund', 'debt', 'power', 'action'], [['t', 'fund', 'debt', 'power', 'action'], ['t','fund','debt','power'], ['action']])
# #
#
# # policy_ranks = range(1)
# policy_ranks = range(2062)
#
# nbins_each = 50
# atts = ['fund', 'debt', 'power', 'action']
# dat = {}
# for m in policy_ranks:
#   try:
#     name = 'm'+str(m)
#     dat[name] = {}
#     dat_temp = {name:{}}
#     samples = pd.read_pickle('data/save_dps_samples/'+str(m)+'.pkl')
#     (dat[name]['annRev'], dat[name]['maxDebt'], dat[name]['maxComplex'], dat[name]['maxFund']) = (samples['annRev'], samples['maxDebt'], samples['maxComplex'], samples['maxFund'])
#     # dict[name]['t_hedge'] = np.tile(np.arange(ny), ns)
#     # (dat[name]['t_binfreq'], dat[name]['t_bincenter'], dat[name]['t_binpoint']) = sort_bins(dict[name]['t_hedge'], nbins_each, False)
#     for i, att in enumerate(atts):
#       (dat_temp[name][att + '_binfreq'], dat_temp[name][att + '_bincenter'], dat_temp[name][att + '_binpoint']) = sort_bins(samples[att + '_hedge'], nbins_each, True)
#     dat_temp = get_joint_probability(dat_temp, name, atts)
#     dat[name]['action_entropy'] = get_entropy(dat_temp[name]['action_binfreq'])
#     tot_mi = 0
#     for att in atts[:-1]:
#       dat[name][att + '_action_mi'] = get_mutual_info(dat_temp, name, atts, [[att, 'action'], [att], ['action']]) / dat[name]['action_entropy']
#       tot_mi += dat[name][att + '_action_mi']
#     dat[name]['total_mi'] = tot_mi
#     print(name + ' success')
#   except:
#     print(name + ' not found')
#
# pd.to_pickle(dat, 'data/save_dps_samples/policy_MI.pkl')





### vis results from MI analysis
dat = pd.read_pickle('data/save_dps_samples/policy_MI.pkl')
policy_ranks = range(2062)

annRev = []
maxDebt = []
maxComplex = []
maxFund = []
fund_action_mi = []
debt_action_mi = []
power_action_mi = []
action_entropy = []

for m in policy_ranks:
  try:
    fund_action_mi.append(dat['m' + str(m)]['fund_action_mi'])
    debt_action_mi.append(dat['m' + str(m)]['debt_action_mi'])
    power_action_mi.append(dat['m' + str(m)]['power_action_mi'])
    action_entropy.append(dat['m'+str(m)]['action_entropy'])
    annRev.append(dat['m'+str(m)]['annRev'])
    maxDebt.append(dat['m'+str(m)]['maxDebt'])
    maxComplex.append(dat['m'+str(m)]['maxComplex'])
    maxFund.append(dat['m'+str(m)]['maxFund'])
  except:
    dum = 0




### view information of policies
# lims = {}
# lims['entropy'] = {'min': np.min(action_entropy), 'max':np.max(action_entropy)}
# lims['mi'] = {'min': min(np.min(fund_action_mi), np.min(debt_action_mi), np.min(power_action_mi)),
#               'max': max(np.max(fund_action_mi), np.max(debt_action_mi), np.max(power_action_mi))}
# lims['entropy'] = {'min': 0, 'max': 5}
# lims['mi'] = {'min': 0, 'max': 1}

lims = {'annRev':[9.4,11.1],'maxDebt':[0,40],'maxComplex':[0,1],'maxFund':[0,125],'entropy':[0,5],'mi':[0,1]}
#Find the amount of padding
padding = {'annRev':(lims['annRev'][1] - lims['annRev'][0])/50, 'maxDebt':(lims['maxDebt'][1] - lims['maxDebt'][0])/50,
           'maxComplex':(lims['maxComplex'][1] - lims['maxComplex'][0])/50, 'maxFund':(lims['maxFund'][1] - lims['maxFund'][0])/50}
lims3d = copy.deepcopy(lims)
lims3d['annRev'][0] += padding['annRev']
lims3d['annRev'][1] -= padding['annRev']
lims3d['maxDebt'][0] += padding['maxDebt']
lims3d['maxDebt'][1] -= padding['maxDebt']
lims3d['maxComplex'][0] += padding['maxComplex']
lims3d['maxComplex'][1] -= padding['maxComplex']





fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
zs = annRev
ys = maxDebt
xs = maxComplex
ss = [20 + 1.3 * x for x in maxFund]
cs = action_entropy
p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap='viridis', vmin=lims['entropy'][0], vmax=lims['entropy'][1], marker='v', alpha=0.4)
cb = fig.colorbar(p1, ax=ax)
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'entropy_cb.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)


### get example policies for parallel coord plot
# fund_mi_max = list(np.array(fund_action_mi).argsort()[-5:])
# debt_mi_max = list(np.array(debt_action_mi).argsort()[-5:])
# power_mi_max = list(np.array(power_action_mi).argsort()[-5:])
fund_mi_max = [np.array(fund_action_mi).argsort()[-3]]
debt_mi_max = [np.array(debt_action_mi).argsort()[-2]]
power_mi_max = [np.array(power_action_mi).argsort()[-2]]



fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
zs = annRev
ys = maxDebt
xs = maxComplex
ss = [20 + 1.3 * x for x in maxFund]
cs = fund_action_mi
p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap='viridis', vmin=lims['mi'][0], vmax=lims['mi'][1], marker='v', alpha=0.4)
zs = [annRev[x] for x in fund_mi_max]
ys = [maxDebt[x] for x in fund_mi_max]
xs = [maxComplex[x] for x in fund_mi_max]
ss = [20 + 1.3 * x for x in [maxFund[x] for x in fund_mi_max]]
cs = [fund_action_mi[x] for x in fund_mi_max]
p1 = ax.scatter(xs[0], ys[0], zs[0], s=ss, c=cs, cmap='viridis', vmin=lims['mi'][0], vmax=lims['mi'][1], marker='v', alpha=1)
p1 = ax.text(xs[0], ys[0], zs[0], 'o', color='k', horizontalalignment='center', verticalalignment='center', size=36)
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'mi_fund.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)


fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
zs = annRev
ys = maxDebt
xs = maxComplex
ss = [20 + 1.3 * x for x in maxFund]
cs = debt_action_mi
p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap='viridis', vmin=lims['mi'][0], vmax=lims['mi'][1], marker='v', alpha=0.4)
# cb = fig.colorbar(p1, ax=ax)
zs = [annRev[x] for x in debt_mi_max]
ys = [maxDebt[x] for x in debt_mi_max]
xs = [maxComplex[x] for x in debt_mi_max]
ss = [20 + 1.3 * x for x in [maxFund[x] for x in debt_mi_max]]
cs = [debt_action_mi[x] for x in debt_mi_max]
p1 = ax.scatter(xs[0], ys[0], zs[0], s=ss, c=cs, cmap='viridis', vmin=lims['mi'][0], vmax=lims['mi'][1], marker='v', alpha=1)
p1 = ax.text(xs[0], ys[0], zs[0], 'o', color='k', horizontalalignment='center', verticalalignment='center', size=36)
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'mi_debt.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)


fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
zs = annRev
ys = maxDebt
xs = maxComplex
ss = [20 + 1.3 * x for x in maxFund]
cs = power_action_mi
p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap='viridis', vmin=lims['mi'][0], vmax=lims['mi'][1], marker='v', alpha=0.4)
zs = [annRev[x] for x in power_mi_max]
ys = [maxDebt[x] for x in power_mi_max]
xs = [maxComplex[x] for x in power_mi_max]
ss = [20 + 1.3 * x for x in [maxFund[x] for x in power_mi_max]]
cs = [power_action_mi[x] for x in power_mi_max]
p1 = ax.scatter(xs[0], ys[0], zs[0], s=ss, c=cs, cmap='viridis', vmin=lims['mi'][0], vmax=lims['mi'][1], marker='v', alpha=1)
p1 = ax.text(xs[0], ys[0], zs[0], 'o', color='k', horizontalalignment='center', verticalalignment='center', size=36)
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'mi_power.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)














#
# #### recreate multivariate information decomposition from william & beers 2010
# # Create tree of all information combinations (inputs only)
# def get_info_tree(vars):
#   tree = {}
#   tree['nvar'] = len(vars)
#   if (tree['nvar'] == 2):
#     tree['up_tree_dict'] = {'1.2':['1','2'],
#                     '1':['12'], '2':['12'],
#                     '12':[]}
#     tree['down_tree_dict'] = {'12': ['1', '2'],
#                               '1': ['1.2'], '2': ['1.2'],
#                               '1.2': []}
#     tree['up_tree_keys'] = list(tree['up_tree_dict'].keys())
#     tree['down_tree_keys'] = list(tree['down_tree_dict'].keys())
#     tree['node_sets'] = {'1.2':[[vars[0]], [vars[1]]],
#                  '1': [[vars[0]]], '2':[[vars[1]]],
#                  '12':[[vars[0], vars[1]]]}
#     tree['down_tree_full'] = {'1.2':[]}
#     for node in tree['up_tree_keys'][1:]:
#       down_nodes = tree['down_tree_dict'][node]
#       for down_node in tree['down_tree_dict'][node]:
#         for down_node_down_tree_full in tree['down_tree_full'][down_node]:
#           if (down_node_down_tree_full in down_nodes) == False:
#             down_nodes.append(down_node_down_tree_full)
#       tree['down_tree_full'][node] = down_nodes
#
#   elif (tree['nvar'] == 3):
#     tree['up_tree_dict'] = {'1.2.3':['1.2','1.3','2.3'],
#                     '1.2':['1.23','2.13'], '1.3':['1.23','3.12'], '2.3':['2.13','3.12'],
#                     '1.23':['1','12.13.23'], '2.13':['2','12.13.23'], '3.12':['3','12.13.23'],
#                     '1':['12.13'], '2':['12.23'], '3':['13.23'], '12.13.23':['12.13','12.23','13.23'],
#                     '12.13':['12','13'], '12.23':['12','23'], '13.23':['13','23'],
#                     '12':['123'], '13':['123'], '23':['123'],
#                     '123': []}
#     tree['down_tree_dict'] = {'123':['12','13','23'],
#                       '12':['12.13','12.23'], '13':['12.13','13.23'], '23':['12.23','13.23'],
#                       '12.13':['1','12.13.23'], '12.23':['2','12.13.23'], '13.23':['3','12.13.23'],
#                       '1':['1.23'], '2':['2.13'], '3':['3.12'], '12.13.23':['1.23','2.13','3.12'],
#                       '1.23':['1.2','1.3'], '2.13':['1.2','2.3'], '3.12':['1.3','2.3'],
#                       '1.2':['1.2.3'], '1.3':['1.2.3'], '2.3':['1.2.3'],
#                       '1.2.3':[]}
#     tree['up_tree_keys'] = list(tree['up_tree_dict'].keys())
#     tree['down_tree_keys'] = list(tree['down_tree_dict'].keys())
#     tree['node_sets'] = {'1.2.3':[[vars[0]], [vars[1]], [vars[2]]],
#                  '1.2':[[vars[0]], [vars[1]]], '1.3':[[vars[0]], [vars[2]]], '2.3':[[vars[1]], [vars[2]]],
#                  '1.23': [[vars[0]], [vars[1],vars[2]]], '2.13': [[vars[1]], [vars[0],vars[2]]], '3.12': [[vars[2]], [vars[0],vars[1]]],
#                  '1': [[vars[0]]], '2':[[vars[1]]], '3':[[vars[2]]], '12.13.23':[[vars[0],vars[1]], [vars[0],vars[2]], [vars[1],vars[2]]],
#                  '12.13': [[vars[0], vars[1]], [vars[0], vars[2]]], '12.23':[[vars[0],vars[1]], [vars[1],vars[2]]], '13.23':[[vars[0],vars[2]], [vars[1],vars[2]]],
#                  '12':[[vars[0], vars[1]]], '13':[[vars[0], vars[2]]], '23':[[vars[1],vars[2]]],
#                  '123':[[vars[0], vars[1], vars[2]]]}
#     tree['down_tree_full'] = {'1.2.3':[]}
#     for node in tree['up_tree_keys'][1:]:
#       down_nodes = tree['down_tree_dict'][node]
#       for down_node in tree['down_tree_dict'][node]:
#         for down_node_down_tree_full in tree['down_tree_full'][down_node]:
#           if (down_node_down_tree_full in down_nodes) == False:
#             down_nodes.append(down_node_down_tree_full)
#       tree['down_tree_full'][node] = down_nodes
#   else:
#     print('nvar must be 2 or 3 for now')
#     return
#   return tree
#
#
# # info_tree = get_info_tree(['fund','debt'])
#
#
#
#
#
# # function for redundant info between {A1,...,Ak} (combinations of inputs) and S (action)
# def get_redundant_info(dat, name, atts_full, atts_redund, full_probs):
#   redundant_info = 0
#   probs = full_probs
#   sets_redund = {}
#   for i, atts_set_list in enumerate(atts_redund):
#     sets_redund[i] = {}
#     sets_redund[i]['n_set'] = len(atts_set_list)
#     for j, atts_set in enumerate(atts_set_list):
#       sets_redund[i][j] = {}
#       sets_redund[i][j]['n_att'] = len(atts_set)
#       sets_redund[i][j]['atts'] = []
#       sets_redund[i][j]['n_grid'] = {}
#       sets_redund[i][j]['idx_base'] = {}
#       for k, att in enumerate(atts_set):
#         sets_redund[i][j]['atts'].append(att)
#         sets_redund[i][j]['n_grid'][att] = len(dat[name][att + '_binfreq'])
#         sets_redund[i][j]['idx_base'][att] = [idx for idx, x in enumerate(sets_redund[0][0]['atts']) if x == att]
#   for i, atts_set_list in enumerate(atts_redund):
#     if (i == 0)|(i == 2):    # i==0: full set of all vars included (e.g. A1,...,Ak,S); i==2: action states only (S)
#       for j, atts_set in enumerate(atts_set_list):
#         probs_temp = probs.copy()
#         for k in range(len(atts_full) - 1, -1, -1):
#           if ((atts_full[k] in atts_redund[i][j]) == False):
#             probs_temp = probs_temp.sum(k)
#         sets_redund[i][j]['probs_integrated'] = probs_temp    # probability distribution with all non-included vars integrated out
#     elif (i == 1):  # set of sets of inputs (e.g. (A1),(A2,A3))
#       for j, atts_set in enumerate(atts_set_list):
#         atts_set.append('action')
#         probs_temp = probs.copy()
#         for k in range(len(atts_full) - 1, -1, -1):
#           if ((atts_full[k] in atts_redund[i][j]) == False):
#             probs_temp = probs_temp.sum(k)
#         sets_redund[i][j]['probs_integrated'] = probs_temp
#
#   action_states = sets_redund[2][0]['n_grid']['action']
#   for s in range(action_states):
#     p_s = sets_redund[2][0]['probs_integrated'][s]
#     if (p_s > eps):
#       min_I_wrtA = np.inf
#       for j in range(sets_redund[1]['n_set']):
#         n_a = sets_redund[1][j]['n_att']
#         p_aANDs = sets_redund[1][j]['probs_integrated']
#         keys = sets_redund[1][j]['atts']
#         info_specific = 0
#         for a1 in range(sets_redund[1][j]['n_grid'][keys[0]]):
#           if (n_a == 1):
#             p_a = p_aANDs[a1, :].sum()
#             p_sGIVENa = p_aANDs[a1,s] / p_a
#             p_aGIVENs = p_aANDs[a1,s] / p_s
#             if (p_sGIVENa > eps):
#               info_specific += p_aGIVENs * (np.log2(p_sGIVENa / p_s))
#           else:
#             for a2 in range(sets_redund[1][j]['n_grid'][keys[1]]):
#               if (n_a == 2):
#                 p_a = p_aANDs[a1,a2,:].sum()
#                 p_sGIVENa = p_aANDs[a1,a2,s] / p_a
#                 p_aGIVENs = p_aANDs[a1,a2,s] / p_s
#                 if (p_sGIVENa > eps):
#                   info_specific += p_aGIVENs * (np.log2(1 / p_s) - np.log2(1 / p_sGIVENa))
#               else:
#                 for a3 in range(sets_redund[1][j]['n_grid'][keys[2]]):
#                   if (n_a == 3):
#                     p_a = p_aANDs[a1,a2,a3,:].sum()
#                     p_sGIVENa = p_aANDs[a1,a2,a3,s] / p_a
#                     p_aGIVENs = p_aANDs[a1,a2,a3,s] / p_s
#                     if (p_sGIVENa > eps):
#                       info_specific += p_aGIVENs * (np.log2(p_sGIVENa / p_s))
#                   else:
#                     for a4 in range(sets_redund[1][j]['n_grid'][keys[3]]):
#                       if (n_a == 4):
#                         p_a = p_aANDs[a1,a2,a3,a4,:].sum()
#                         p_sGIVENa = p_aANDs[a1,a2,a3,a4,s] / p_a
#                         p_aGIVENs = p_aANDs[a1,a2,a3,a4,s] / p_s
#                         if (p_sGIVENa > eps):
#                           info_specific += p_aGIVENs * (np.log2(p_sGIVENa / p_s))
#                       else:
#                         print ('only set up for up to four input vars')
#                         return
#
#         min_I_wrtA = min(min_I_wrtA, info_specific)
#         info_specific = 0
#       redundant_info += p_s * min_I_wrtA
#
#   return redundant_info
#
#
#
#
#
# # atts_redund = [[['fund','debt', 'power','action']], [['power']], [['action']]]
# # get_redundant_info(dat, 'm1', atts, atts_redund)
#
#
#
#
#
# ### march up info tree to get partial information
# atts = ['t', 'fund', 'debt', 'power', 'action']
# name = 'm1'
# info_tree = get_info_tree(['fund','debt','power'])
# info_tree['redundant_info'] = {}
# info_tree['partial_info'] = {}
# for node in info_tree['up_tree_keys']:
#   atts_redund = [[['fund', 'debt', 'power', 'action']], info_tree['node_sets'][node], [['action']]]
#   info_tree['redundant_info'][node] = get_redundant_info(dat, name, atts, atts_redund, dat[name]['joint_freq'])
#   info_tree['partial_info'][node] = info_tree['redundant_info'][node]
#   down_tree_info = 0
#   for down_node in info_tree['down_tree_full'][node]:
#     down_tree_info += info_tree['partial_info'][down_node]
#   info_tree['partial_info'][node] -= down_tree_info
#
# # node = '123'
# # down_tree_info = info_tree['partial_info'][node]
# # for down_node in info_tree['down_tree_full'][node]:
# #   down_tree_info += info_tree['partial_info'][down_node]
#
# info_tree['redundant_info']
#
# get_entropy(dat[name]['action_binfreq'])
#
#
#
# ### get time/state-varying partial info
# tree_list = {}
# for name in names:
#   tree_list[name] = {}
#   # slice by t
#   tree_list[name]['t'] = {'center': dat[name]['t_bincenter'], 'action_entropy': []}
#   atts_slice = ['fund', 'debt', 'power', 'action']
#   for slice in range(len(dat[name]['t_binfreq'])):
#     tree_list[name]['t']['action_entropy'].append(get_entropy(dat[name]['joint_freq'][slice,:,:,:,:].sum((0,1,2))))
#     info_tree = get_info_tree(['fund', 'debt', 'power'])
#     info_tree['redundant_info'] = {}
#     info_tree['partial_info'] = {}
#     for node in info_tree['up_tree_keys']:
#       atts_redund = [[['fund', 'debt', 'power', 'action']], info_tree['node_sets'][node], [['action']]]
#       info_tree['redundant_info'][node] = get_redundant_info(dat, name, atts_slice, atts_redund, dat[name]['joint_freq'][slice,:,:,:,:])
#       info_tree['partial_info'][node] = info_tree['redundant_info'][node]
#       down_tree_info = 0
#       for down_node in info_tree['down_tree_full'][node]:
#         down_tree_info += info_tree['partial_info'][down_node]
#       info_tree['partial_info'][node] -= down_tree_info
#     if (slice == 0):
#       tree_list[name]['nodes'] = info_tree['up_tree_keys']
#       tree_list[name]['t']['partial'] = {}
#       tree_list[name]['t']['redundant'] = {}
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['t']['partial'][node] = [info_tree['partial_info'][node]]
#         tree_list[name]['t']['redundant'][node] = [info_tree['redundant_info'][node]]
#     else:
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['t']['partial'][node].append(info_tree['partial_info'][node])
#         tree_list[name]['t']['redundant'][node].append(info_tree['redundant_info'][node])
#
#   # slice by fund
#   tree_list[name]['fund'] = {'center': dat[name]['fund_bincenter'], 'action_entropy':[]}
#   atts_slice = ['t', 'debt', 'power', 'action']
#   for slice in range(len(dat[name]['fund_binfreq'])):
#     tree_list[name]['fund']['action_entropy'].append(get_entropy(dat[name]['joint_freq'][:,slice,:,:,:].sum((0,1,2))))
#     info_tree = get_info_tree(['t', 'debt', 'power'])
#     info_tree['redundant_info'] = {}
#     info_tree['partial_info'] = {}
#     for node in info_tree['up_tree_keys']:
#       atts_redund = [[['t', 'debt', 'power', 'action']], info_tree['node_sets'][node], [['action']]]
#       info_tree['redundant_info'][node] = get_redundant_info(dat, name, atts_slice, atts_redund, dat[name]['joint_freq'][:,slice,:,:,:])
#       info_tree['partial_info'][node] = info_tree['redundant_info'][node]
#       down_tree_info = 0
#       for down_node in info_tree['down_tree_full'][node]:
#         down_tree_info += info_tree['partial_info'][down_node]
#       info_tree['partial_info'][node] -= down_tree_info
#     if (slice == 0):
#       tree_list[name]['fund']['partial'] = {}
#       tree_list[name]['fund']['redundant'] = {}
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['fund']['partial'][node] = [info_tree['partial_info'][node]]
#         tree_list[name]['fund']['redundant'][node] = [info_tree['redundant_info'][node]]
#     else:
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['fund']['partial'][node].append(info_tree['partial_info'][node])
#         tree_list[name]['fund']['redundant'][node].append(info_tree['redundant_info'][node])
#
#   # slice by debt
#   tree_list[name]['debt'] = {'center': dat[name]['debt_bincenter'], 'action_entropy':[]}
#   atts_slice = ['t', 'fund', 'power', 'action']
#   for slice in range(len(dat[name]['debt_binfreq'])):
#     tree_list[name]['debt']['action_entropy'].append(get_entropy(dat[name]['joint_freq'][:,:,slice,:,:].sum((0,1,2))))
#     info_tree = get_info_tree(['t', 'fund', 'power'])
#     info_tree['redundant_info'] = {}
#     info_tree['partial_info'] = {}
#     for node in info_tree['up_tree_keys']:
#       atts_redund = [[['t', 'fund', 'power', 'action']], info_tree['node_sets'][node], [['action']]]
#       info_tree['redundant_info'][node] = get_redundant_info(dat, name, atts_slice, atts_redund, dat[name]['joint_freq'][:,:,slice,:,:])
#       info_tree['partial_info'][node] = info_tree['redundant_info'][node]
#       down_tree_info = 0
#       for down_node in info_tree['down_tree_full'][node]:
#         down_tree_info += info_tree['partial_info'][down_node]
#       info_tree['partial_info'][node] -= down_tree_info
#     if (slice == 0):
#       tree_list[name]['debt']['partial'] = {}
#       tree_list[name]['debt']['redundant'] = {}
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['debt']['partial'][node] = [info_tree['partial_info'][node]]
#         tree_list[name]['debt']['redundant'][node] = [info_tree['redundant_info'][node]]
#     else:
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['debt']['partial'][node].append(info_tree['partial_info'][node])
#         tree_list[name]['debt']['redundant'][node].append(info_tree['redundant_info'][node])
#
#   # slice by power
#   tree_list[name]['power'] = {'center': dat[name]['power_bincenter'], 'action_entropy': []}
#   atts_slice = ['t', 'fund', 'debt', 'action']
#   for slice in range(len(dat[name]['power_binfreq'])):
#     tree_list[name]['power']['action_entropy'].append(get_entropy(dat[name]['joint_freq'][:,:,:,slice,:].sum((0,1,2))))
#     info_tree = get_info_tree(['t', 'fund', 'debt'])
#     info_tree['redundant_info'] = {}
#     info_tree['partial_info'] = {}
#     for node in info_tree['up_tree_keys']:
#       atts_redund = [[['t', 'fund', 'debt', 'action']], info_tree['node_sets'][node], [['action']]]
#       info_tree['redundant_info'][node] = get_redundant_info(dat, name, atts_slice, atts_redund, dat[name]['joint_freq'][:,:,:,slice,:])
#       info_tree['partial_info'][node] = info_tree['redundant_info'][node]
#       down_tree_info = 0
#       for down_node in info_tree['down_tree_full'][node]:
#         down_tree_info += info_tree['partial_info'][down_node]
#       info_tree['partial_info'][node] -= down_tree_info
#     if (slice == 0):
#       tree_list[name]['power']['partial'] = {}
#       tree_list[name]['power']['redundant'] = {}
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['power']['partial'][node] = [info_tree['partial_info'][node]]
#         tree_list[name]['power']['redundant'][node] = [info_tree['redundant_info'][node]]
#     else:
#       for node in tree_list[name]['nodes']:
#         tree_list[name]['power']['partial'][node].append(info_tree['partial_info'][node])
#         tree_list[name]['power']['redundant'][node].append(info_tree['redundant_info'][node])
#
#
# # plot partial info
# fig = plt.figure()
# for n, name in enumerate(names):
#   for a,att in enumerate(atts[:-1]):
#     ax = plt.subplot2grid((3,4), (n,a))  # ,colspan=2)
#     for node in tree_list[name]['nodes']:
#       plt.plot(tree_list[name][att]['center'], tree_list[name][att]['partial'][node])
#     plt.plot(tree_list[name][att]['center'], tree_list[name][att]['action_entropy'], 'k', ls=':')
#
# # plot redundant info (more like mutual info I think?)
# fig = plt.figure()
# for n, name in enumerate(names):
#   for a,att in enumerate(atts[:-1]):
#     ax = plt.subplot2grid((3,4), (n,a))  # ,colspan=2)
#     for node in tree_list[name]['nodes']:
#       plt.plot(tree_list[name][att]['center'], tree_list[name][att]['redundant'][node])
#     plt.plot(tree_list[name][att]['center'], tree_list[name][att]['action_entropy'], 'k', ls=':')
