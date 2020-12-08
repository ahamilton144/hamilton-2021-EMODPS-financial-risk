##############################################################################################################
# make_moea_output_plots.py - python script to create plots for multi-objective optimization outputs
# Project started March 2018, last update May 2020
##############################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker, cm, colors
import seaborn as sns
import importlib
from datetime import datetime
import copy
import itertools

### Project functions ###
import functions_moea_output_plots


sns.set_style('white')
sns.set_context('paper', font_scale=1.55)

jpg = 1e-13
startTime = datetime.now()

dir_generated_inputs = './../../data/generated_inputs/'
dir_moea_output = './../../data/optimization_output/'
dir_figs = './../../figures/'

cmap_vir = cm.get_cmap('viridis')
col_vir = [cmap_vir(0.1),cmap_vir(0.4),cmap_vir(0.7),cmap_vir(0.85)]
cmap_blues = cm.get_cmap('Blues_r')
col_blues = [cmap_blues(0.1),cmap_blues(0.3),cmap_blues(0.5),cmap_blues(0.8)]
cmap_reds = cm.get_cmap('Reds_r')
col_reds = [cmap_reds(0.1),cmap_reds(0.3),cmap_reds(0.5),cmap_reds(0.8)]
cmap_purples = cm.get_cmap('Purples_r')
col_purples = [cmap_purples(0.1),cmap_purples(0.3),cmap_purples(0.5),cmap_purples(0.8)]
col_brewerQual4 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

##################################################################
#### Constants
##################################################################
fixed_cost = 0.914
mean_revenue = 127.80086602479503
mean_net_revenue = mean_revenue * (1 - fixed_cost)

# ##################################################################
#### Convergence, RBF analysis
##################################################################

# ### get runtime metrics for moea runs with different number of RBFs (nrbfs)
# print('Getting runtime metrics for moea runs with different number of RBFs..., ',
#       datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# metrics = {}
# nrbfs = (1, 2, 3, 4, 8, 12)
# for nrbf in nrbfs:
#     metrics[str(nrbf)+'rbf'] = []
#     for s in range(1, 11):
#         metrics[str(nrbf)+'rbf'].append(functions_moea_output_plots.get_metrics(dir_moea_output + '4obj_' + str(nrbf) +
#                                                                               'rbf/metrics/DPS_param150_seedS1_seedB' + str(s) + '.metrics',
#                                                                               '../../data/optimization_output/4obj_rbf_overall/DPS_4obj_rbf_overall_borg.hypervolume'))

# # ### plot hypervolume for test of number of radial basis functions (nrbfs)
# print('Plotting hypervolume (fig S1)..., ', datetime.now() - startTime)
# importlib.reload(functions_moea_output_plots)
# nfe = 150000
# fe_prints = 100
# fe_grid = np.arange(0, nfe+1, nfe/fe_prints)
# nseed = 10
# functions_moea_output_plots.plot_metrics(dir_figs, metrics, nrbfs, nseed, fe_grid)

###############################################################



##################################################################
#### Analysis of 2dv vs full DPS. both 2obj & 4obj problems
##################################################################

### read in ref sets
nobj = 4
ncon = 1
ref_dps_2obj_retest, ndv_dps_2obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '2obj_2rbf/DPS_2obj_2rbf_borg_retest.resultfile', nobj, ncon)
ref_2dv_2obj_retest, ndv_2dv_2obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '2obj_2dv/DPS_2obj_2dv_borg_retest.resultfile', nobj, ncon)
ref_dps_4obj_retest, ndv_dps_4obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '4obj_2rbf_moreSeeds/DPS_4obj_2rbf_moreSeeds_borg_retest.resultfile', nobj, ncon)
ref_2dv_4obj_retest, ndv_dps_4obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '4obj_2dv/DPS_4obj_2dv_borg_retest.resultfile', nobj, ncon)


# ### get distance from ideal point
dfs = [ref_dps_2obj_retest, ref_2dv_2obj_retest, ref_dps_4obj_retest, ref_2dv_4obj_retest]
#use log for reserve fund?
# for d in dfs:
#   d['maxFund'] = np.log10(d['maxFund']+0.01)
range_objs = {}
for n,d in enumerate(dfs):
  range_objs[n] = {}
  for o in ['annRev', 'maxDebt', 'maxComplex', 'maxFund']:
    range_objs[n][o] = [d[o].min(), d[o].max()]
for n, d in enumerate(dfs):
  for o in ['annRev', 'maxDebt', 'maxComplex', 'maxFund']:
    d[o + 'Norm'] = (d[o] - range_objs[n][o][0]) / (range_objs[n][o][1] - range_objs[n][o][0])
  d['annRevNorm'] = 1 - d['annRevNorm']
  d['totalDistance2obj'] = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2)
  d['totalDistance4obj'] = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2 + d['maxComplexNorm'] **2 + d['maxFundNorm'] **2)



### get limits for plots
lims2d = {'annRev':[9.4,11.09],'maxDebt':[0.,30.]}
lims3d = {'annRev':[9.4,11.13],'maxDebt':[0.,36.],'maxComplex':[0.,1.],'maxFund':[0.,125.]}
## undo axis padding (from https://stackoverflow.com/questions/30196503/2d-plots-are-not-sitting-flush-against-3d-axis-walls-in-python-mplot3d/41779162#41779162)
# def get_fixed_mins_maxs(lims):
#     mins, maxs = lims
#     deltas = (maxs - mins) / 12.
#     mins = mins + deltas / 4.
#     maxs = maxs - deltas / 4.
#     return [mins, maxs]
# for k in lims3d.keys():
#   lims3d[k] = get_fixed_mins_maxs(lims3d[k])
# padding = {'annRev': (lims3d['annRev'][1] - lims3d['annRev'][0])/50, 
#             'maxDebt': (lims3d['maxDebt'][1] - lims3d['maxDebt'][0])/50,
#            'maxComplex': (lims3d['maxComplex'][1] - lims3d['maxComplex'][0])/50}
# lims3d['annRev'][0] += padding['annRev']
# lims3d['annRev'][1] -= padding['annRev']
# lims3d['maxDebt'][0] += padding['maxDebt']
# lims3d['maxDebt'][1] -= padding['maxDebt']
# lims3d['maxComplex'][0] += padding['maxComplex']
# lims3d['maxComplex'][1] -= padding['maxComplex']



# ### Comparison of 2dv vs full dps, 2 objective version
# fig = plt.figure()
# ax = fig.add_subplot(111)
# min_dist = [np.where(ref_2dv_2obj_retest.totalDistance2obj == ref_2dv_2obj_retest.totalDistance2obj.min())[0][0],
#             np.where(ref_dps_2obj_retest.totalDistance2obj == ref_dps_2obj_retest.totalDistance2obj.min())[0][0]]
# x_min_dist = [ref_2dv_2obj_retest.maxDebt.iloc[min_dist[0]], ref_dps_2obj_retest.maxDebt.iloc[min_dist[1]]]
# y_min_dist = [ref_2dv_2obj_retest.annRev.iloc[min_dist[0]], ref_dps_2obj_retest.annRev.iloc[min_dist[1]]]
# ys = ref_2dv_2obj_retest.annRev
# xs = ref_2dv_2obj_retest.maxDebt
# p1 = ax.scatter(xs,ys, c=col_reds[2], marker='^', alpha=1, s=60)
# p1 = ax.scatter(x_min_dist[0], y_min_dist[0], c=col_reds[0], marker='^', alpha=1, s=60)
# ys = ref_dps_2obj_retest.annRev
# xs = ref_dps_2obj_retest.maxDebt
# p1 = ax.scatter(xs, ys, c=col_blues[2], marker='v', alpha=1, s=60)
# p1 = ax.scatter(x_min_dist[1], y_min_dist[1], c=col_blues[0], marker='v', alpha=1, s=60)
# plt.xticks([0,10,20,30])
# plt.yticks([9.5,10,10.5,11])
# plt.tick_params(length=3)
# plt.plot([1.2],[mean_net_revenue],marker='*',ms=20,c='k')
# plt.xlim(lims2d['maxDebt'])
# plt.ylim(lims2d['annRev'])
# plt.savefig(dir_figs + 'compare2dvDps_2objForm_2objView.jpg', bbox_inches='tight', dpi=500)




# ### Comparison of 4dv vs full dps, 4 objective version
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# min_dist = [np.where(ref_dps_4obj_retest.totalDistance4obj == ref_dps_4obj_retest.totalDistance4obj.min())[0][0],
#             np.where(ref_2dv_4obj_retest.totalDistance4obj == ref_2dv_4obj_retest.totalDistance4obj.min())[0][0]]
# z_min_dist = [ref_dps_4obj_retest.annRev.iloc[min_dist[0]], ref_2dv_4obj_retest.annRev.iloc[min_dist[1]]]
# y_min_dist = [ref_dps_4obj_retest.maxDebt.iloc[min_dist[0]], ref_2dv_4obj_retest.maxDebt.iloc[min_dist[1]]]
# x_min_dist = [ref_dps_4obj_retest.maxComplex.iloc[min_dist[0]], ref_2dv_4obj_retest.maxComplex.iloc[min_dist[1]]]
# s_min_dist = [ref_dps_4obj_retest.maxFund.iloc[min_dist[0]], ref_2dv_4obj_retest.maxFund.iloc[min_dist[1]]]
# zs = ref_dps_4obj_retest.annRev.drop(min_dist[0])
# ys = ref_dps_4obj_retest.maxDebt.drop(min_dist[0])
# xs = ref_dps_4obj_retest.maxComplex.drop(min_dist[0])
# ss = 20 + 1.3*ref_dps_4obj_retest.maxFund.drop(min_dist[0])
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.6, c=col_blues[2],zorder=2)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# p1 = ax.scatter(x_min_dist[0], y_min_dist[0], z_min_dist[0], s=s_min_dist[0], marker='v', alpha=1, c=col_blues[0],zorder=3)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# zs = ref_2dv_4obj_retest.annRev.drop(min_dist[1])
# ys = ref_2dv_4obj_retest.maxDebt.drop(min_dist[1])
# xs = ref_2dv_4obj_retest.maxComplex.drop(min_dist[1])
# ss = 20 + 1.3*ref_2dv_4obj_retest.maxFund.drop(min_dist[1])
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='^',alpha=0.6, c=col_reds[2],zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# p1 = ax.scatter(x_min_dist[1], y_min_dist[1], z_min_dist[1], s=s_min_dist[1], marker='^', alpha=1, c=col_reds[0], zorder=0)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([12, 24, 36])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[mean_net_revenue],marker='*',ms=25,c='k')
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# plt.savefig(dir_figs + 'compare2dvDps_4objForm_4objView.jpg', bbox_inches='tight', figsize=(4.5,8), dpi=500)




# ### get min/max size markers for legend
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# min_s = 20 + 1.3*min(ref_dps_4obj_retest.maxFund.min(),ref_2dv_4obj_retest.maxFund.max())
# max_s = 20 + 1.3*max(ref_dps_4obj_retest.maxFund.max(),ref_2dv_4obj_retest.maxFund.max())
# zs = ref_dps_4obj_retest.annRev.drop(min_dist[0])
# ys = ref_dps_4obj_retest.maxDebt.drop(min_dist[0])
# xs = ref_dps_4obj_retest.maxComplex.drop(min_dist[0])
# ss = 20 + 1.3*ref_dps_4obj_retest.maxFund.drop(min_dist[0])
# # p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.6, c=col_blues[2],zorder=2)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# zs = ref_2dv_4obj_retest.annRev.drop(min_dist[1])
# ys = ref_2dv_4obj_retest.maxDebt.drop(min_dist[1])
# xs = ref_2dv_4obj_retest.maxComplex.drop(min_dist[1])
# ss = 20 + 1.3*ref_2dv_4obj_retest.maxFund.drop(min_dist[1])
# # p1 = ax.scatter(xs, ys, zs, s=ss, marker='v',alpha=0.6, c=col_reds[2],zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# p1 = ax.scatter(xs[0], ys[0], zs[0], s=min_s, marker='v',alpha=0.6, c='0.5',zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# p1 = ax.scatter(xs[100], ys[100], zs[100], s=max_s, marker='v',alpha=0.6, c='0.5',zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([12, 24, 36])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[mean_net_revenue],marker='*',ms=25,c='k')
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# plt.savefig(dir_figs + 'compare2dvDps_4objForm_4objView_size.jpg', bbox_inches='tight', figsize=(4.5,8), dpi=500)







# ### get subproblem pareto fronts
# subproblems = ['1234','123','124','134','234','12','13','14','23','24','34']
# paretos = {}
# for s in subproblems:
#   paretos[s] = pd.read_csv(dir_moea_output + '4obj_2rbf_moreSeeds/DPS_4obj_2rbf_' + s + '_pareto.reference', sep=' ', names=['annRev','maxDebt','maxComplex','maxFund'],index_col=0)
#   paretos[s].index -= 1
#   paretos[s]['annRev'] *= -1

# subproblems_with_conflicts = ['1234','123','124','234','12','23','24']
# pareto_cols = {}
# pareto_cols['1234'] = ['annRev', 'maxDebt', 'maxComplex', 'maxFund']
# pareto_cols['123'] = ['annRev', 'maxDebt', 'maxComplex']
# pareto_cols['124'] = ['annRev', 'maxDebt', 'maxFund']
# pareto_cols['234'] = ['maxDebt', 'maxComplex', 'maxFund']
# pareto_cols['12'] = ['annRev', 'maxDebt']
# pareto_cols['23'] = ['maxDebt', 'maxComplex']
# pareto_cols['24'] = ['maxDebt', 'maxFund']


# range_objs = {}
# for n,k in enumerate(subproblems_with_conflicts):
#   d = paretos[k]
#   range_objs[k] = {}
#   for o in pareto_cols[k]:
#     range_objs[k][o] = [d[o].min(), d[o].max()]
#     d[o + 'Norm'] = (d[o] - range_objs[k][o][0]) / (range_objs[k][o][1] - range_objs[k][o][0])
#   if ('annRev' in pareto_cols[k]):
#     d['annRevNorm'] = 1 - d['annRevNorm']
#   squares = 0
#   if ('annRev' in pareto_cols[k]):
#     squares += d['annRevNorm'] **2
#   if ('maxDebt' in pareto_cols[k]):
#     squares += d['maxDebtNorm'] **2
#   if ('maxComplex' in pareto_cols[k]):
#     squares += d['maxComplexNorm'] **2
#   if ('maxFund' in pareto_cols[k]):
#     squares += d['maxFundNorm'] **2
#   d['totalDistance'] = np.sqrt(squares)






# ### plot pareto fronts for each subproblem
# fig = plt.figure()
# baseline = paretos['1234'].copy()
# ax = fig.add_subplot(1,1,1, projection='3d')
# subprob = paretos['12'].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
# zs = subprob.annRev.loc[ind]
# ys = subprob.maxDebt.loc[ind]
# xs = subprob.maxComplex.loc[ind]
# ss = 20 + 1.3 * subprob.maxFund.loc[ind]
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[0]], zorder=1)
# zs = subprob.annRev.drop(ind)
# ys = subprob.maxDebt.drop(ind)
# xs = subprob.maxComplex.drop(ind)
# ss = 20 + 1.3*subprob.maxFund.drop(ind)
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[2]], zorder=2)
# subprob = paretos['23'].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
# zs = subprob.annRev.loc[ind]
# ys = subprob.maxDebt.loc[ind]
# xs = subprob.maxComplex.loc[ind]
# ss = 20 + 1.3 * subprob.maxFund.loc[ind]
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='<', alpha=1, c=[col_reds[0]], zorder=1)
# zs = subprob.annRev.drop(ind)
# ys = subprob.maxDebt.drop(ind)
# xs = subprob.maxComplex.drop(ind)
# ss = 20 + 1.3*subprob.maxFund.drop(ind)
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='<', alpha=1, c=[col_reds[2]], zorder=2)
# subprob = paretos['24'].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
# zs = subprob.annRev.loc[ind]
# ys = subprob.maxDebt.loc[ind]
# xs = subprob.maxComplex.loc[ind]
# ss = 20 + 1.3 * subprob.maxFund.loc[ind]
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='>', alpha=1, c=[col_purples[0]], zorder=1)
# zs = subprob.annRev.drop(ind)
# ys = subprob.maxDebt.drop(ind)
# xs = subprob.maxComplex.drop(ind)
# ss = 20 + 1.3*subprob.maxFund.drop(ind)
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='>', alpha=1, c=[col_purples[2]], zorder=2)

# subprob = paretos['13'].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# zs = subprob.annRev
# ys = subprob.maxDebt
# xs = subprob.maxComplex
# ss = 20 + 1.3 * subprob.maxFund
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.4', zorder=1)
# subprob = paretos['14'].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# zs = subprob.annRev
# ys = subprob.maxDebt
# xs = subprob.maxComplex
# ss = 20 + 1.3 * subprob.maxFund
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.4', zorder=1)
# subprob = paretos['34'].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# zs = subprob.annRev
# ys = subprob.maxDebt
# xs = subprob.maxComplex
# ss = 20 + 1.3 * subprob.maxFund
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.4', zorder=1)
# zs = baseline.annRev
# ys = baseline.maxDebt
# xs = baseline.maxComplex
# ss = 20 + 1.3*baseline.maxFund
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.8', zorder=3)
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([12, 24, 36])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[mean_net_revenue],marker='*',ms=25,c='k')
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# plt.savefig(dir_figs + 'compareObjFormulations_2objSub.jpg', bbox_inches='tight', figsize=(4.5,8), dpi=500)


# for k in ['123','124','234']:
#   fig = plt.figure()
#   baseline = paretos['1234'].copy()
#   ax = fig.add_subplot(1,1,1, projection='3d')
#   subprob = paretos[k].copy()
#   baseline = baseline.drop(subprob.index, errors='ignore')
#   ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
#   zs = subprob.annRev.loc[ind]
#   ys = subprob.maxDebt.loc[ind]
#   xs = subprob.maxComplex.loc[ind]
#   ss = 20 + 1.3 * subprob.maxFund.loc[ind]
#   p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[0]])
#   zs = subprob.annRev.drop(ind)
#   ys = subprob.maxDebt.drop(ind)
#   xs = subprob.maxComplex.drop(ind)
#   ss = 20 + 1.3*subprob.maxFund.drop(ind)
#   p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[2]])
#   zs = baseline.annRev
#   ys = baseline.maxDebt
#   xs = baseline.maxComplex
#   ss = 20 + 1.3*baseline.maxFund
#   p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.8')
#   ax.set_xticks([0,0.25,0.5,0.75])
#   ax.set_yticks([12, 24, 36])
#   ax.set_zticks([9.5,10,10.5,11])
#   ax.view_init(elev=20, azim =-45)
#   ax.plot([0.01],[0.01],[mean_net_revenue],marker='*',ms=25,c='k')
#   ax.set_xlim(lims3d['maxComplex'])
#   ax.set_ylim(lims3d['maxDebt'])
#   ax.set_zlim(lims3d['annRev'])
#   plt.savefig(dir_figs + 'compareObjFormulations_' + k + '.jpg', bbox_inches='tight', figsize=(4.5,8), dpi=500)








# ### plot policies meeting brushing constraints
# k = '1234_constraint'
# paretos[k] = paretos['1234'].copy().iloc[:,0:4]
# pareto_cols[k] = pareto_cols['1234']
# min_annRev = mean_net_revenue * 0.975
# max_maxDebt = mean_net_revenue * 1.5
# max_maxFund = mean_net_revenue * 1.5
# brush_annRev = paretos[k].annRev >= min_annRev
# brush_maxDebt = paretos[k].maxDebt <= max_maxDebt
# brush_maxFund = paretos[k].maxFund <= max_maxFund
# paretos[k] = paretos[k].loc[(brush_annRev & brush_maxDebt & brush_maxFund), :]

# d = paretos[k]
# range_objs[k] = {}
# for o in pareto_cols[k]:
#   range_objs[k][o] = [d[o].min(), d[o].max()]
#   d[o + 'Norm'] = (d[o] - range_objs[k][o][0]) / (range_objs[k][o][1] - range_objs[k][o][0])
# if ('annRev' in pareto_cols[k]):
#   d['annRevNorm'] = 1 - d['annRevNorm']
# squares = 0
# if ('annRev' in pareto_cols[k]):
#   squares += d['annRevNorm'] ** 2
# if ('maxDebt' in pareto_cols[k]):
#   squares += d['maxDebtNorm'] ** 2
# if ('maxComplex' in pareto_cols[k]):
#   squares += d['maxComplexNorm'] ** 2
# if ('maxFund' in pareto_cols[k]):
#   squares += d['maxFundNorm'] ** 2
# d['totalDistance'] = np.sqrt(squares)


# fig = plt.figure()
# baseline = paretos['1234'].copy()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# subprob = paretos[k].copy()
# baseline = baseline.drop(subprob.index, errors='ignore')
# ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
# zs = subprob.annRev.loc[ind]
# ys = subprob.maxDebt.loc[ind]
# xs = subprob.maxComplex.loc[ind]
# ss = 20 + 1.3 * subprob.maxFund.loc[ind]
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[0]])
# zs = subprob.annRev.drop(ind)
# ys = subprob.maxDebt.drop(ind)
# xs = subprob.maxComplex.drop(ind)
# ss = 20 + 1.3 * subprob.maxFund.drop(ind)
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.4, c=[col_blues[2]])
# zs = baseline.annRev
# ys = baseline.maxDebt
# xs = baseline.maxComplex
# ss = 20 + 1.3 * baseline.maxFund
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=0.1, c='0.8')
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([12, 24, 36])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[mean_net_revenue],marker='*',ms=25,c='k')
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# plt.savefig(dir_figs + 'compareObjFormulations_' + k + '.jpg', bbox_inches='tight', figsize=(4.5, 8), dpi=500)








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

samp = pd.read_csv(dir_generated_inputs + '/synthetic_data.txt', delimiter=' ')

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





def simulate(revenue, payout, power, policy, dps_run_type):
  ny = len(revenue) - 1
  net_rev = revenue - MEAN_REVENUE * fixed_cost
  fund = np.zeros(ny + 1)
  debt = np.zeros(ny + 1)
  final_cashflow = np.zeros(ny)
  withdrawal = np.zeros(ny)
  value_snow_contract = np.zeros(ny)
  cash_in = np.zeros(ny)

  for i in range(ny):
    if dps_run_type == 0:
      max_fund = policy.dv1
      value_snow_contract[i] = policy.dv2
    else:
      value_snow_contract[i] = policySnowContractValue(fund[i], debt[i], power[i])
    net_payout_snow_contract = value_snow_contract[i] * payout[i+1]
    cash_in[i] = net_rev[i+1] + net_payout_snow_contract - debt[i] * interest_debt
    if dps_run_type == 0:
      # rule for withdrawal (or deposit), after growing fund at interestFund from last year
      final_cashflow[i] = get_cashflow_post_withdrawal_2dv(fund[i] * interest_fund, cash_in[i], 0, max_fund)
      withdrawal[i] = final_cashflow[i] - cash_in[i]
    else:
      withdrawal[i] = policyWithdrawal(fund[i]*interest_fund, debt[i]*interest_debt, power[i+1], cash_in[i])
      final_cashflow[i] = cash_in[i] + withdrawal[i]
    fund[i+1] = fund[i]*interest_fund - withdrawal[i]
    if (final_cashflow[i] < -EPS):
      debt[i+1] = -final_cashflow[i]
      final_cashflow[i] = 0

  # print('fund:', fund)
  # print('debt:', debt)
  # print('cfd:', value_snow_contract)
  # print('final_cashflow:', final_cashflow)

  return (fund[:-1], fund[:-1]*interest_fund, debt[:-1], debt[:-1]*interest_debt, power[:-1], power[1:], cash_in, value_snow_contract, withdrawal, final_cashflow)





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


##########################################################################
######### calculate cash flow after withdrawal (deposit) from (to) reserve fund ####
### returns scalar. ####
# ##########################################################################
def get_cashflow_post_withdrawal_2dv(fund_balance, cash_in, cashflow_target, maxFund):
  if (cash_in < cashflow_target):
    if (fund_balance < EPS):
      x = cash_in
    else:
      x = min(cash_in + fund_balance, cashflow_target)
  else:
    if (fund_balance > (maxFund - EPS)):
      x = cash_in + (fund_balance - maxFund)
    else:
      x = max(cash_in - (maxFund - fund_balance), cashflow_target)
  return(x)



### plot simulated state variables for 3 example policies over historical period (fig 8)
print('Plotting historical simulation for 3 policies, sfpuc baseline... (fig 8), ', datetime.now() - startTime)
# functions_moea_output_plots.plot_example_simulations(dir_figs, moea_solns_filtered, params_sfpuc, cases_sfpuc_index, historical_data, meanRevenue)


# #########################################################################
# ######### plot distribution of state variables over historical period for dynamic vs static ####
# ### outputs plot, no return. ####
# # ##########################################################################
# ind_2dv = np.where(ref_2dv_2obj_retest.totalDistance2obj == ref_2dv_2obj_retest.totalDistance2obj.min())[0][0]
# ind_dps = np.where(ref_dps_2obj_retest.totalDistance2obj == ref_dps_2obj_retest.totalDistance2obj.min())[0][0]
# # ind_2dv = np.where(ref_2dv_2obj_retest.maxDebt == ref_2dv_2obj_retest.maxDebt.min())[0][0]
# # ind_dps = np.where(ref_dps_2obj_retest.maxDebt == ref_dps_2obj_retest.maxDebt.min())[0][0]
# # ind_2dv = np.where(ref_2dv_2obj_retest.annRev == ref_2dv_2obj_retest.annRev.max())[0][0]
# # ind_dps = np.where(ref_dps_2obj_retest.annRev == ref_dps_2obj_retest.annRev.max())[0][0]
# soln_2dv = ref_2dv_2obj_retest.iloc[ind_2dv, :]
# soln_dps = ref_dps_2obj_retest.iloc[ind_dps, :]

# policies = [soln_dps, soln_2dv]
# dps_run_types = [1, 0]
# cols = [col_blues[2], col_reds[2], col_purples[2]]
# lss = ['-','-.','--']

# historical_data = pd.read_csv(dir_generated_inputs + '/historical_data.csv', sep=' ', index_col=0)

# fixed_cost = 0.914
# delta = 0.4
# Delta_fund = -1.73
# Delta_debt = 1
# lam = 0.25
# lam_prem_shift = 0

# samp = pd.read_csv(dir_generated_inputs + '/synthetic_data.txt', delimiter=' ')

# # get financial params
# discount_rate = 1 / (delta/100 + 1)
# interest_fund = (Delta_fund + delta)/100 + 1
# interest_debt = (Delta_debt + delta)/100 + 1
# discount_factor = discount_rate ** np.arange(1, NUM_YEARS+1)
# discount_normalization = 1 / np.sum(discount_factor)
# ny = historical_data.shape[0]
# # s = 1000 # random sample from synthetic


# fig = plt.figure(figsize=(6,10))
# gs1 = fig.add_gridspec(nrows=4, ncols=2, left=0, right=1, wspace=0.05, hspace=0.1)

# ax = fig.add_subplot(gs1[0,0])
# ax.annotate('a)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax.set_ylabel('SWE Index\n(inch)')
# ax.set_xlabel('Year')
# ax.set_yticks([10,25,40])
# # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax.tick_params(axis='y',which='both',labelleft=True,labelright=False)
# ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
# ax.xaxis.set_label_position('top')
# # ax.yaxis.set_label_position('right')
# l0, = ax.plot(historical_data['sweIndex'], c='k')

# ax0 = fig.add_subplot(gs1[0,1], sharex=ax)
# ax0.annotate('b)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Generation\n(TWh)', rotation=270, labelpad=35)
# ax0.set_xlabel('Year')
# ax0.set_yticks([1.2, 1.7, 2.2])
# ax0.tick_params(axis='y',which='both',labelleft=False,labelright=True)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
# ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
# ax0.plot(historical_data['gen'], c='k')

# ax0 = fig.add_subplot(gs1[1,0], sharex=ax)
# ax0.annotate('c)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Wholesale Price\n(\$/MWh)')
# # ax0.set_xlabel('Year')
# # ax0.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax0.tick_params(axis='y',which='both',labelleft=True,labelright=False)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# # ax0.xaxis.set_label_position('top')
# # ax0.yaxis.set_label_position('right')
# ax0.plot(historical_data['pow'], c='k')

# ax0 = fig.add_subplot(gs1[1,1], sharex=ax)
# ax0.annotate('d)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Net Revenue\n(\$)', rotation=270, labelpad=35)
# # ax0.set_xlabel('Year')
# # ax0.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax0.tick_params(axis='y',which='both',labelleft=False,labelright=True)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# # ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
# ax0.axhline(0, color='0.5', ls=':', zorder=1)
# ax0.plot(historical_data['rev'] - fixed_cost * MEAN_REVENUE, c='k')

# ax1 = fig.add_subplot(gs1[2,0], sharex=ax)
# ax2 = fig.add_subplot(gs1[2,1], sharex=ax)
# ax3 = fig.add_subplot(gs1[3,0], sharex=ax)
# ax4 = fig.add_subplot(gs1[3,1], sharex=ax)

# for i in range(2):
#   policy = policies[i]
#   dps_run_type = dps_run_types[i]
#   if dps_run_type == 1:
#     dv_d, dv_c, dv_b, dv_w, dv_a = getDV(policy)


#   fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, final_cashflow = simulate(historical_data['rev'].values, historical_data['cfd'].values, historical_data['powIndex'].values, policy, dps_run_type)
#   fund = np.append(fund_hedge, fund_withdrawal[-1])
#   debt = np.append(debt_hedge, debt_withdrawal[-1])


#   if i==0:
#     ax1.annotate('e)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax1.set_ylabel('CFD Slope\n(\$M/inch)')
#     # ax.set_xlabel('Year')
#     # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax1.tick_params(axis='y',which='both',labelleft=True,labelright=False)
#     ax1.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
#     # ax.xaxis.set_label_position('top')
#     # ax.yaxis.set_label_position('right')
#     ax1.set_yticks([0,0.4,0.8])
#     ax1.axhline(0, color='0.5', ls=':', zorder=1)
#     l1, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])
#   elif i == 1:
#     l2, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])

#   if i==0:
#     ax2.annotate('f)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax2.set_ylabel('Fund Balance\n(\$M)', rotation=270, labelpad=35)
#     # ax.set_xlabel('Year')
#     # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax2.tick_params(axis='y',which='both',labelleft=False,labelright=True)
#     ax2.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
#     # ax.xaxis.set_label_position('top')
#     ax2.yaxis.set_label_position('right')
#     ax2.axhline(0, color='0.5', ls=':', zorder=1)
#   ax2.plot(range(1987,2017), fund, c=cols[i], ls=lss[i])

#   if i==0:
#     ax3.annotate('g)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax3.set_ylabel('Debt\n(\$M)')
#     ax3.set_xlabel('Year')
#     # ax3.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax3.tick_params(axis='y',which='both',labelleft=True,labelright=False)
#     ax3.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
#     # ax3.xaxis.set_label_position('top')
#     # ax3.yaxis.set_label_position('right')
#     ax3.axhline(0, color='0.5', ls=':', zorder=1)
#   ax3.plot(range(1987,2017), debt, c=cols[i], ls=lss[i])

#   if i==0:
#     ax4.annotate('h)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax4.set_ylabel('Final Cashflow\n(\$M)', rotation=270, labelpad=35)
#     ax4.set_xlabel('Year')
#     # ax4.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax4.tick_params(axis='y',which='both',labelleft=False,labelright=True)
#     ax4.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
#     # ax4.xaxis.set_label_position('top')
#     ax4.yaxis.set_label_position('right')
#     ax4.axhline(0, color='0.5', ls=':', zorder=1)
#   ax4.plot(range(1988,2017), final_cashflow, c=cols[i], ls=lss[i])

# ax3.legend([l0, l1, l2],['Stochastic Driver','Dynamic','Static'], ncol=4, bbox_to_anchor=(2,-0.35))#, fontsize=12)

# plt.savefig(dir_figs + 'historicalSim_dynStat.jpg', bbox_inches='tight', dpi=500)




# #########################################################################
# ######### plot distribution of state variables over historical period for dynamic vs static ####
# ### outputs plot, no return. ####
# # ##########################################################################
# ind_2dv = np.where(ref_2dv_2obj_retest.totalDistance2obj == ref_2dv_2obj_retest.totalDistance2obj.min())[0][0]
# ind_dps = np.where(ref_dps_2obj_retest.totalDistance2obj == ref_dps_2obj_retest.totalDistance2obj.min())[0][0]
# # ind_2dv = np.where(ref_2dv_2obj_retest.maxDebt == ref_2dv_2obj_retest.maxDebt.min())[0][0]
# # ind_dps = np.where(ref_dps_2obj_retest.maxDebt == ref_dps_2obj_retest.maxDebt.min())[0][0]
# # ind_2dv = np.where(ref_2dv_2obj_retest.annRev == ref_2dv_2obj_retest.annRev.max())[0][0]
# # ind_dps = np.where(ref_dps_2obj_retest.annRev == ref_dps_2obj_retest.annRev.max())[0][0]
# soln_2dv = ref_2dv_2obj_retest.iloc[ind_2dv, :]
# soln_dps = ref_dps_2obj_retest.iloc[ind_dps, :]

# policies = [soln_dps, soln_2dv]
# dps_run_types = [1, 0]
# cols = [col_blues[2], col_reds[2], col_purples[2]]
# lss = ['-','-.','--']

# historical_data = pd.read_csv(dir_generated_inputs + '/historical_data.csv', sep=' ', index_col=0)

# fixed_cost = 0.914
# delta = 0.4
# Delta_fund = -1.73
# Delta_debt = 1
# lam = 0.25
# lam_prem_shift = 0

# samp = pd.read_csv(dir_generated_inputs + '/synthetic_data.txt', delimiter=' ')

# # get financial params
# discount_rate = 1 / (delta/100 + 1)
# interest_fund = (Delta_fund + delta)/100 + 1
# interest_debt = (Delta_debt + delta)/100 + 1
# discount_factor = discount_rate ** np.arange(1, NUM_YEARS+1)
# discount_normalization = 1 / np.sum(discount_factor)
# ny = historical_data.shape[0]
# # s = 1000 # random sample from synthetic


# fig = plt.figure(figsize=(6,10))
# gs1 = fig.add_gridspec(nrows=4, ncols=2, left=0, right=1, wspace=0.05, hspace=0.1)

# ax = fig.add_subplot(gs1[0,0])
# ax.annotate('a)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax.set_ylabel('SWE Index\n(inch)')
# ax.set_xlabel('Year')
# ax.set_yticks([10,25,40])
# # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax.tick_params(axis='y',which='both',labelleft=True,labelright=False)
# ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
# ax.xaxis.set_label_position('top')
# # ax.yaxis.set_label_position('right')
# l0, = ax.plot(historical_data['sweIndex'], c='k')

# ax0 = fig.add_subplot(gs1[0,1], sharex=ax)
# ax0.annotate('b)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Generation\n(TWh)', rotation=270, labelpad=35)
# ax0.set_xlabel('Year')
# ax0.set_yticks([1.2, 1.7, 2.2])
# ax0.tick_params(axis='y',which='both',labelleft=False,labelright=True)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
# ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
# ax0.plot(historical_data['gen'], c='k')

# ax0 = fig.add_subplot(gs1[1,0], sharex=ax)
# ax0.annotate('c)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Wholesale Price\n(\$/MWh)')
# # ax0.set_xlabel('Year')
# # ax0.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax0.tick_params(axis='y',which='both',labelleft=True,labelright=False)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# # ax0.xaxis.set_label_position('top')
# # ax0.yaxis.set_label_position('right')
# ax0.plot(historical_data['pow'], c='k')

# ax0 = fig.add_subplot(gs1[1,1], sharex=ax)
# ax0.annotate('d)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Net Revenue\n(\$)', rotation=270, labelpad=35)
# # ax0.set_xlabel('Year')
# # ax0.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax0.tick_params(axis='y',which='both',labelleft=False,labelright=True)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# # ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
# ax0.axhline(0, color='0.5', ls=':', zorder=1)
# ax0.plot(historical_data['rev'] - fixed_cost * MEAN_REVENUE, c='k')

# ax1 = fig.add_subplot(gs1[2,0], sharex=ax)
# ax2 = fig.add_subplot(gs1[2,1], sharex=ax)
# ax3 = fig.add_subplot(gs1[3,0], sharex=ax)
# ax4 = fig.add_subplot(gs1[3,1], sharex=ax)

# for i in range(2):
#   policy = policies[i]
#   dps_run_type = dps_run_types[i]
#   if dps_run_type == 1:
#     dv_d, dv_c, dv_b, dv_w, dv_a = getDV(policy)


#   fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, final_cashflow = simulate(historical_data['rev'].values, historical_data['cfd'].values, historical_data['powIndex'].values, policy, dps_run_type)
#   fund = np.append(fund_hedge, fund_withdrawal[-1])
#   debt = np.append(debt_hedge, debt_withdrawal[-1])


#   if i==0:
#     ax1.annotate('e)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax1.set_ylabel('CFD Slope\n(\$M/inch)')
#     # ax.set_xlabel('Year')
#     # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax1.tick_params(axis='y',which='both',labelleft=True,labelright=False)
#     ax1.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
#     # ax.xaxis.set_label_position('top')
#     # ax.yaxis.set_label_position('right')
#     ax1.set_yticks([0,0.4,0.8])
#     ax1.axhline(0, color='0.5', ls=':', zorder=1)
#     l1, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])
#   elif i == 1:
#     l2, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])

#   if i==0:
#     ax2.annotate('f)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax2.set_ylabel('Fund Balance\n(\$M)', rotation=270, labelpad=35)
#     # ax.set_xlabel('Year')
#     # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax2.tick_params(axis='y',which='both',labelleft=False,labelright=True)
#     ax2.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
#     # ax.xaxis.set_label_position('top')
#     ax2.yaxis.set_label_position('right')
#     ax2.axhline(0, color='0.5', ls=':', zorder=1)
#   ax2.plot(range(1987,2017), fund, c=cols[i], ls=lss[i])

#   if i==0:
#     ax3.annotate('g)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax3.set_ylabel('Debt\n(\$M)')
#     ax3.set_xlabel('Year')
#     # ax3.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax3.tick_params(axis='y',which='both',labelleft=True,labelright=False)
#     ax3.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
#     # ax3.xaxis.set_label_position('top')
#     # ax3.yaxis.set_label_position('right')
#     ax3.axhline(0, color='0.5', ls=':', zorder=1)
#   ax3.plot(range(1987,2017), debt, c=cols[i], ls=lss[i])

#   if i==0:
#     ax4.annotate('h)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax4.set_ylabel('Final Cashflow\n(\$M)', rotation=270, labelpad=35)
#     ax4.set_xlabel('Year')
#     # ax4.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax4.tick_params(axis='y',which='both',labelleft=False,labelright=True)
#     ax4.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
#     # ax4.xaxis.set_label_position('top')
#     ax4.yaxis.set_label_position('right')
#     ax4.axhline(0, color='0.5', ls=':', zorder=1)
#   ax4.plot(range(1988,2017), final_cashflow, c=cols[i], ls=lss[i])

# ax3.legend([l0, l1, l2],['Stochastic Driver','Dynamic','Static'], ncol=4, bbox_to_anchor=(2,-0.35))#, fontsize=12)

# plt.savefig(dir_figs + 'historicalSim_dynStat.jpg', bbox_inches='tight', dpi=500)




# #########################################################################
# ######### plot distribution of state variables over historical period for 3 example policies from MI ####
# ### outputs plot, no return. ####
# # ##########################################################################
# ## get example policies for each sensitivity index
# dat = pd.read_csv(dir_generated_inputs + '../policy_simulation/mi_combined.csv', index_col=0).sort_index()

# dat.fund_hedge_mi = dat.fund_hedge_mi.fillna(-1)
# dat.debt_hedge_mi = dat.debt_hedge_mi.fillna(-1)
# dat.power_hedge_mi = dat.power_hedge_mi.fillna(-1)

# fund_mi_max = [np.array(dat.fund_hedge_mi).argsort()[-2]]
# debt_mi_max = [np.array(dat.debt_hedge_mi).argsort()[-2]]
# power_mi_max = [np.array(dat.power_hedge_mi).argsort()[-4]]

# policies = [ref_dps_4obj_retest.iloc[fund_mi_max[0], :], ref_dps_4obj_retest.iloc[debt_mi_max[0], :], ref_dps_4obj_retest.iloc[power_mi_max[0], :]]
# dps_run_types = [1, 1, 1]
# cols = [col_blues[2], col_reds[2], col_purples[2]]
# lss = ['-','-.','--']

# historical_data = pd.read_csv(dir_generated_inputs + '/historical_data.csv', sep=' ', index_col=0)

# fixed_cost = 0.914
# delta = 0.4
# Delta_fund = -1.73
# Delta_debt = 1
# lam = 0.25
# lam_prem_shift = 0

# samp = pd.read_csv(dir_generated_inputs + '/synthetic_data.txt', delimiter=' ')

# # get financial params
# discount_rate = 1 / (delta/100 + 1)
# interest_fund = (Delta_fund + delta)/100 + 1
# interest_debt = (Delta_debt + delta)/100 + 1
# discount_factor = discount_rate ** np.arange(1, NUM_YEARS+1)
# discount_normalization = 1 / np.sum(discount_factor)
# ny = historical_data.shape[0]
# # s = 1000 # random sample from synthetic



# fig = plt.figure(figsize=(6,10))
# gs1 = fig.add_gridspec(nrows=4, ncols=2, left=0, right=1, wspace=0.05, hspace=0.1)

# ax = fig.add_subplot(gs1[0,0])
# ax.annotate('a)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax.set_ylabel('SWE Index\n(inch)')
# ax.set_xlabel('Year')
# ax.set_yticks([10,25,40])
# # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax.tick_params(axis='y',which='both',labelleft=True,labelright=False)
# ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
# ax.xaxis.set_label_position('top')
# # ax.yaxis.set_label_position('right')
# l0, = ax.plot(historical_data['sweIndex'], c='k')

# ax0 = fig.add_subplot(gs1[0,1], sharex=ax)
# ax0.annotate('b)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Generation\n(TWh)', rotation=270, labelpad=35)
# ax0.set_xlabel('Year')
# ax0.set_yticks([1.2, 1.7, 2.2])
# ax0.tick_params(axis='y',which='both',labelleft=False,labelright=True)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
# ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
# ax0.plot(historical_data['gen'], c='k')

# ax0 = fig.add_subplot(gs1[1,0], sharex=ax)
# ax0.annotate('c)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Wholesale Price\n(\$/MWh)')
# # ax0.set_xlabel('Year')
# # ax0.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax0.tick_params(axis='y',which='both',labelleft=True,labelright=False)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# # ax0.xaxis.set_label_position('top')
# # ax0.yaxis.set_label_position('right')
# ax0.plot(historical_data['pow'], c='k')

# ax0 = fig.add_subplot(gs1[1,1], sharex=ax)
# ax0.annotate('d)', xy=(0.01, 0.89), xycoords='axes fraction')
# ax0.set_ylabel('Net Revenue\n(\$)', rotation=270, labelpad=35)
# # ax0.set_xlabel('Year')
# # ax0.set_xticks(np.arange(0.85, 0.98, 0.04))
# ax0.tick_params(axis='y',which='both',labelleft=False,labelright=True)
# ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# # ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
# ax0.axhline(0, color='0.5', ls=':', zorder=1)
# ax0.plot(historical_data['rev'] - fixed_cost * MEAN_REVENUE, c='k')

# ax1 = fig.add_subplot(gs1[2,0], sharex=ax)
# ax2 = fig.add_subplot(gs1[2,1], sharex=ax)
# ax3 = fig.add_subplot(gs1[3,0], sharex=ax)
# ax4 = fig.add_subplot(gs1[3,1], sharex=ax)



# for i in range(len(policies)):
#   policy = policies[i]
#   dps_run_type = dps_run_types[i]
#   if dps_run_type == 1:
#     dv_d, dv_c, dv_b, dv_w, dv_a = getDV(policy)


#   fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, final_cashflow = simulate(historical_data['rev'].values, historical_data['cfd'].values, historical_data['powIndex'].values, policy, dps_run_type)
#   fund = np.append(fund_hedge, fund_withdrawal[-1])
#   debt = np.append(debt_hedge, debt_withdrawal[-1])


#   if i==0:
#     ax1.annotate('e)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax1.set_ylabel('CFD Slope\n(\$M/inch)')
#     # ax.set_xlabel('Year')
#     # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax1.tick_params(axis='y',which='both',labelleft=True,labelright=False)
#     ax1.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
#     # ax.xaxis.set_label_position('top')
#     # ax.yaxis.set_label_position('right')
#     ax1.set_yticks([0,0.4,0.8])
#     ax1.axhline(0, color='0.5', ls=':', zorder=1)
#     l1, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])
#   elif i == 1:
#     l2, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])
#   else:
#     l3, = ax1.plot(range(1988,2017), action_hedge, c=cols[i], ls=lss[i])

#   if i==0:
#     ax2.annotate('f)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax2.set_ylabel('Fund Balance\n(\$M)', rotation=270, labelpad=35)
#     # ax.set_xlabel('Year')
#     # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax2.tick_params(axis='y',which='both',labelleft=False,labelright=True)
#     ax2.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
#     # ax.xaxis.set_label_position('top')
#     ax2.yaxis.set_label_position('right')
#     ax2.axhline(0, color='0.5', ls=':', zorder=1)
#   ax2.plot(range(1987,2017), fund, c=cols[i], ls=lss[i])

#   if i==0:
#     ax3.annotate('g)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax3.set_ylabel('Debt\n(\$M)')
#     ax3.set_xlabel('Year')
#     # ax3.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax3.tick_params(axis='y',which='both',labelleft=True,labelright=False)
#     ax3.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
#     # ax3.xaxis.set_label_position('top')
#     # ax3.yaxis.set_label_position('right')
#     ax3.axhline(0, color='0.5', ls=':', zorder=1)
#   ax3.plot(range(1987,2017), debt, c=cols[i], ls=lss[i])

#   if i==0:
#     ax4.annotate('h)', xy=(0.01, 0.89), xycoords='axes fraction')
#     ax4.set_ylabel('Final Cashflow\n(\$M)', rotation=270, labelpad=35)
#     ax4.set_xlabel('Year')
#     # ax4.set_xticks(np.arange(0.85, 0.98, 0.04))
#     ax4.tick_params(axis='y',which='both',labelleft=False,labelright=True)
#     ax4.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
#     # ax4.xaxis.set_label_position('top')
#     ax4.yaxis.set_label_position('right')
#     ax4.axhline(0, color='0.5', ls=':', zorder=1)
#   ax4.plot(range(1988,2017), final_cashflow, c=cols[i], ls=lss[i])

# ax3.legend([l0, l1, l2],['Stochastic Driver','Fund-Sensitive','Debt-Sensitive','Power-Sensitive'], ncol=2, bbox_to_anchor=(2,-0.35))#, fontsize=12)

# plt.savefig(dir_figs + 'historicalSim_miPols.jpg', bbox_inches='tight', dpi=500)



#########################################################################
######### plot distribution of state variables over wet/avg/dry period for dynamic vs static ####
### outputs plot, no return. ####
# ##########################################################################
ind_2dv = np.where(ref_2dv_2obj_retest.totalDistance2obj == ref_2dv_2obj_retest.totalDistance2obj.min())[0][0]
ind_dps = np.where(ref_dps_2obj_retest.totalDistance2obj == ref_dps_2obj_retest.totalDistance2obj.min())[0][0]
# ind_2dv = np.where(ref_2dv_2obj_retest.maxDebt == ref_2dv_2obj_retest.maxDebt.min())[0][0]
# ind_dps = np.where(ref_dps_2obj_retest.maxDebt == ref_dps_2obj_retest.maxDebt.min())[0][0]
# ind_2dv = np.where(ref_2dv_2obj_retest.annRev == ref_2dv_2obj_retest.annRev.max())[0][0]
# ind_dps = np.where(ref_dps_2obj_retest.annRev == ref_dps_2obj_retest.annRev.max())[0][0]
soln_2dv = ref_2dv_2obj_retest.iloc[ind_2dv, :]
soln_dps = ref_dps_2obj_retest.iloc[ind_dps, :]

policies = [soln_2dv, soln_dps]
dps_run_types = [0, 1]
# cols = [col_blues[2], col_purples[2], col_reds[2]]
cmap = cm.get_cmap('viridis')
# cols = [cmap(0),cmap(0.15),cmap(0.6),cmap(0.85)]
cols = [cmap(0.15),cmap(0.6),cmap(0.85)]

lss = ['-','-.','--']
realization = ['_wet', '_avg', '_dry']

example_data = pd.read_csv(dir_generated_inputs + '/example_data.csv', sep=' ', index_col=0)

fixed_cost = 0.914
delta = 0.4
Delta_fund = -1.73
Delta_debt = 1
lam = 0.25
lam_prem_shift = 0


# get financial params
discount_rate = 1 / (delta/100 + 1)
interest_fund = (Delta_fund + delta)/100 + 1
interest_debt = (Delta_debt + delta)/100 + 1
discount_factor = discount_rate ** np.arange(1, NUM_YEARS+1)
discount_normalization = 1 / np.sum(discount_factor)
ny = example_data.shape[0] - 1
# s = 1000 # random sample from synthetic


fig = plt.figure(figsize=(6,10))
gs1 = fig.add_gridspec(nrows=4, ncols=3, left=0, right=1, wspace=0.05, hspace=0.1)

ax = fig.add_subplot(gs1[0,0])
ax.annotate('a)', xy=(0.01, 0.89), xycoords='axes fraction')
ax.set_ylabel('SWE Index\n(inch)')
ax.set_xlabel('Year')
ax.set_yticks([0,30,60])
ax.tick_params(axis='y',which='both',labelleft=True,labelright=False)
ax.tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
ax.xaxis.set_label_position('top')
# ax.yaxis.set_label_position('right')
ax.axhline(0, color='0.5', ls=':', zorder=1)
l0, = ax.plot(range(1, ny+1), example_data['sweIndex_wet'][1:], c=cols[0], ls=lss[0])
l1, = ax.plot(range(1, ny+1), example_data['sweIndex_avg'][1:], c=cols[1], ls=lss[1])
l2, = ax.plot(range(1, ny+1), example_data['sweIndex_dry'][1:], c=cols[2], ls=lss[2])


ax0 = fig.add_subplot(gs1[1,0], sharex=ax)
ax0.annotate('b)', xy=(0.01, 0.89), xycoords='axes fraction')
ax0.set_ylabel('Generation\n(TWh)')
# ax0.set_xlabel('Year')
ax0.set_yticks([1, 1.5, 2])
ax0.tick_params(axis='y',which='both',labelleft=True,labelright=False)
ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
ax0.plot(range(1, ny+1), example_data['gen_wet'][1:], c=cols[0], ls=lss[0])
ax0.plot(range(1, ny+1), example_data['gen_avg'][1:], c=cols[1], ls=lss[1])
ax0.plot(range(1, ny+1), example_data['gen_dry'][1:], c=cols[2], ls=lss[2])

ax0 = fig.add_subplot(gs1[2,0], sharex=ax)
ax0.annotate('c)', xy=(0.01, 0.89), xycoords='axes fraction')
ax0.set_ylabel('Wholesale Price\n(\$/MWh)')
# ax0.set_xlabel('Year')
ax0.set_yticks([30, 45, 60])
ax0.tick_params(axis='y',which='both',labelleft=True,labelright=False)
ax0.tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
# ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
ax0.plot(range(ny+1), example_data['powWt_wet'], c=cols[0], ls=lss[0])
ax0.plot(range(ny+1), example_data['powWt_avg'], c=cols[1], ls=lss[1])
ax0.plot(range(ny+1), example_data['powWt_dry'], c=cols[2], ls=lss[2])

ax0 = fig.add_subplot(gs1[3,0], sharex=ax)
ax0.annotate('d)', xy=(0.01, 0.89), xycoords='axes fraction')
ax0.set_ylabel('Net Revenue\n(\$)')
ax0.set_xlabel('Year')
ax0.set_yticks([-15, 0, 15, 30])
ax0.tick_params(axis='y',which='both',labelleft=True,labelright=False)
ax0.tick_params(axis='x',which='both',labelbottom=True,labeltop=False)
# ax0.xaxis.set_label_position('top')
# ax0.yaxis.set_label_position('right')
ax0.axhline(0, color='0.5', ls=':', zorder=1)
ax0.plot(range(1, ny+1), example_data['rev_wet'][1:] - fixed_cost * MEAN_REVENUE, c=cols[0], ls=lss[0])
ax0.plot(range(1, ny+1), example_data['rev_avg'][1:] - fixed_cost * MEAN_REVENUE, c=cols[1], ls=lss[1])
ax0.plot(range(1, ny+1), example_data['rev_dry'][1:] - fixed_cost * MEAN_REVENUE, c=cols[2], ls=lss[2])

ax01 = fig.add_subplot(gs1[0,1], sharex=ax)
ax11 = fig.add_subplot(gs1[1,1], sharex=ax)
ax21 = fig.add_subplot(gs1[2,1], sharex=ax)
ax31 = fig.add_subplot(gs1[3,1], sharex=ax)
ax02 = fig.add_subplot(gs1[0,2], sharex=ax, sharey=ax01)
ax12 = fig.add_subplot(gs1[1,2], sharex=ax, sharey=ax11)
ax22 = fig.add_subplot(gs1[2,2], sharex=ax, sharey=ax21)
ax32 = fig.add_subplot(gs1[3,2], sharex=ax, sharey=ax31)
axes = [[ax01, ax11, ax21, ax31], [ax02, ax12, ax22, ax32]]

### loop over policies
for i in range(2):
  policy = policies[i]
  dps_run_type = dps_run_types[i]
  if dps_run_type == 1:
    dv_d, dv_c, dv_b, dv_w, dv_a = getDV(policy)

  ### loop over realizations
  for j in range(3):
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, final_cashflow = \
       simulate(example_data['rev' + realization[j]].values, example_data['cfd' + realization[j]].values, example_data['powIndex' + realization[j]].values, policy, dps_run_type)
    fund = np.append(fund_hedge, fund_withdrawal[-1])
    debt = np.append(debt_hedge, debt_withdrawal[-1])

    print(i, realization[j])
    print('mean cf: ', final_cashflow.mean())
    print('max debt: ', debt.max())

    if (j==0):
      axes[i][0].set_xlabel('Year')
      axes[i][0].tick_params(axis='x',which='both',labelbottom=False,labeltop=True)
      axes[i][0].xaxis.set_label_position('top')
      axes[i][0].axhline(0, color='0.5', ls=':', zorder=1)
      if (i==0):
        axes[i][0].annotate('e)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][0].tick_params(axis='y',which='both',labelleft=False,labelright=False)
        axes[i][0].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
      else:
        axes[i][0].annotate('i)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][0].set_ylabel('CFD Slope\n(\$M/inch)', rotation=270, labelpad=35)
        axes[i][0].yaxis.set_label_position('right')
        axes[i][0].tick_params(axis='y',which='both',labelleft=False,labelright=True)
        axes[i][0].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
        axes[i][0].set_yticks([0,0.4,0.8])
    axes[i][0].plot(range(ny), action_hedge, c=cols[j], ls=lss[j])

    if (j==0):
      axes[i][1].axhline(0, color='0.5', ls=':', zorder=1)
      if (i==0):
        axes[i][1].annotate('f)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][1].tick_params(axis='y',which='both',labelleft=False,labelright=False)
        axes[i][1].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
      else:
        axes[i][1].annotate('j)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][1].set_ylabel('Fund Balance\n(\$M)', rotation=270, labelpad=35)
        axes[i][1].yaxis.set_label_position('right')
        axes[i][1].tick_params(axis='y',which='both',labelleft=False,labelright=True)
        axes[i][1].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
        axes[i][1].set_yticks([0,10,20])
    axes[i][1].plot(range(ny+1), fund, c=cols[j], ls=lss[j])

    if (j==0):
      axes[i][2].axhline(0, color='0.5', ls=':', zorder=1)
      if (i==0):
        axes[i][2].annotate('g)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][2].tick_params(axis='y',which='both',labelleft=False,labelright=False)
        axes[i][2].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
      else:
        axes[i][2].annotate('k)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][2].set_ylabel('Debt\n(\$M)', rotation=270, labelpad=35)
        axes[i][2].yaxis.set_label_position('right')
        axes[i][2].tick_params(axis='y',which='both',labelleft=False,labelright=True)
        axes[i][2].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
        axes[i][2].set_yticks([0,10,20])
    axes[i][2].plot(range(ny+1), debt, c=cols[j], ls=lss[j])

    if (j==0):
      axes[i][3].axhline(0, color='0.5', ls=':', zorder=1)
      axes[i][3].set_xlabel('Year') 
      if (i==0):
        axes[i][3].annotate('h)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][3].tick_params(axis='y',which='both',labelleft=False,labelright=False)
        axes[i][3].tick_params(axis='x',which='both',labelbottom=False,labeltop=False)
      else:
        axes[i][3].annotate('l)', xy=(0.01, 0.89), xycoords='axes fraction')
        axes[i][3].set_ylabel('Final Cashflow\n(\$M)', rotation=270, labelpad=35)
        axes[i][3].yaxis.set_label_position('right')
        axes[i][3].tick_params(axis='y',which='both',labelleft=False,labelright=True)
        axes[i][2].set_yticks([0,15,30])
    axes[i][3].plot(range(1, ny+1), final_cashflow, c=cols[j], ls=lss[j])


# ax3.legend([l0, l1, l2],['Stochastic Driver','Dynamic','Static'], ncol=4, bbox_to_anchor=(2,-0.35))#, fontsize=12)

plt.savefig(dir_figs + 'exampleSim_dynStat.jpg', bbox_inches='tight', dpi=500)




# print('Finished, ', datetime.now() - startTime)

