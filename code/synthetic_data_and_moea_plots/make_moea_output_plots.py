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

eps = 1e-13
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
nobj = 2
ncon = 1
ref_dps_2obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '2obj_2rbf/DPS_2obj_2rbf_borg_retest.resultfile', nobj, ncon)
ref_2dv_2obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '2obj_2dv/DPS_2obj_2dv_borg_retest.resultfile', nobj, ncon)

nobj = 4
ref_dps_4obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '4obj_2rbf/DPS_4obj_2rbf_borg_retest.resultfile', nobj, ncon)
ref_2dv_4obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '4obj_2dv/DPS_4obj_2dv_borg_retest.resultfile', nobj, ncon)



### get distance from ideal point
dfs = [ref_dps_2obj_retest, ref_2dv_2obj_retest, ref_dps_4obj_retest, ref_2dv_4obj_retest]
#use log for reserve fund?
# for d in dfs:
#   d['maxFund'] = np.log10(d['maxFund']+0.01)
range_objs = {}
for n,d in enumerate(dfs):
  range_objs[n] = {}
  for o in ['annRev', 'maxDebt', 'maxComplex', 'maxFund']:
    range_objs[n][o] = [d[o].min(), d[o].max()]
# range_objs[4] = {}
# for o in ['annRev', 'maxDebt', 'maxComplex', 'maxFund']:
#   range_objs[4][o] = range_objs[0][o]
#   for n, d in enumerate(dfs[1:]):
#     range_objs[4][o][0] = min(range_objs[4][o][0], range_objs[n+1][o][0])
#     range_objs[4][o][1] = max(range_objs[4][o][1], range_objs[n+1][o][1])
for n, d in enumerate(dfs):
  for o in ['annRev', 'maxDebt', 'maxComplex', 'maxFund']:
    d[o + 'Norm'] = (d[o] - range_objs[n][o][0]) / (range_objs[n][o][1] - range_objs[n][o][0])
  d['annRevNorm'] = 1 - d['annRevNorm']
  d['totalDistance2obj'] = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2)
  d['totalDistance4obj'] = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2 + d['maxComplexNorm'] **2 + d['maxFundNorm'] **2)



### Comparison of 2dv vs full dps, objective plots
### 3d plot
# lims = {'annRev':[1e6, -1e6],'maxDebt':[1e6, -1e6],'maxComplex':[1e6, -1e6],'maxFund':[1e6, -1e6]}
# for c,d in enumerate([ref_dps_4obj_retest, ref_2dv_4obj_retest, ref_dps_2obj_retest, ref_2dv_2obj_retest]):
#   lims['annRev'][0] = min(lims['annRev'][0], d.annRev.min())
#   lims['maxDebt'][0] = min(lims['maxDebt'][0], d.maxDebt.min())
#   lims['maxComplex'][0] = min(lims['maxComplex'][0], d.maxComplex.min())
#   lims['maxFund'][0] = min(lims['maxFund'][0], d.maxFund.min())
#   lims['annRev'][1] = max(lims['annRev'][1], d.annRev.max())
#   lims['maxDebt'][1] = max(lims['maxDebt'][1], d.maxDebt.max())
#   lims['maxComplex'][1] = max(lims['maxComplex'][1], d.maxComplex.max())
#   lims['maxFund'][1] = max(lims['maxFund'][1], d.maxFund.max())
lims = {'annRev':[9.4,11.13],'maxDebt':[0,40],'maxComplex':[0,1],'maxFund':[0,125]}
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



### Comparison of 2dv vs full dps, 2 objective version
fig = plt.figure()
ax = fig.add_subplot(111)
min_dist = [np.where(ref_2dv_2obj_retest.totalDistance2obj == ref_2dv_2obj_retest.totalDistance2obj.min())[0][0],
            np.where(ref_dps_2obj_retest.totalDistance2obj == ref_dps_2obj_retest.totalDistance2obj.min())[0][0]]
x_min_dist = [ref_2dv_2obj_retest.maxDebt.iloc[min_dist[0]], ref_dps_2obj_retest.maxDebt.iloc[min_dist[1]]]
y_min_dist = [ref_2dv_2obj_retest.annRev.iloc[min_dist[0]], ref_dps_2obj_retest.annRev.iloc[min_dist[1]]]
ys = ref_2dv_2obj_retest.annRev
xs = ref_2dv_2obj_retest.maxDebt
p1 = ax.scatter(xs,ys, c=col_reds[2], marker='^', alpha=1, s=60)
p1 = ax.scatter(x_min_dist[0], y_min_dist[0], c=col_reds[0], marker='^', alpha=1, s=60)
ys = ref_dps_2obj_retest.annRev
xs = ref_dps_2obj_retest.maxDebt
p1 = ax.scatter(xs, ys, c=col_blues[2], marker='v', alpha=1, s=60)
p1 = ax.scatter(x_min_dist[1], y_min_dist[1], c=col_blues[0], marker='v', alpha=1, s=60)
plt.xlim(lims['maxDebt'])
plt.ylim(lims['annRev'])
plt.xticks([0,10,20,30,40])
plt.yticks([9.5,10,10.5,11])
plt.tick_params(length=3)
plt.plot([1.2],[11.05],marker='*',ms=20,c='k')
plt.savefig(dir_figs + 'compare2dvDps_2objForm_2objView.eps', bbox_inches='tight', dpi=500)




### Comparison of 4dv vs full dps, 4 objective version
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
min_dist = [np.where(ref_dps_4obj_retest.totalDistance4obj == ref_dps_4obj_retest.totalDistance4obj.min())[0][0],
            np.where(ref_2dv_4obj_retest.totalDistance4obj == ref_2dv_4obj_retest.totalDistance4obj.min())[0][0]]
z_min_dist = [ref_dps_4obj_retest.annRev.iloc[min_dist[0]], ref_2dv_4obj_retest.annRev.iloc[min_dist[1]]]
y_min_dist = [ref_dps_4obj_retest.maxDebt.iloc[min_dist[0]], ref_2dv_4obj_retest.maxDebt.iloc[min_dist[1]]]
x_min_dist = [ref_dps_4obj_retest.maxComplex.iloc[min_dist[0]], ref_2dv_4obj_retest.maxComplex.iloc[min_dist[1]]]
s_min_dist = [ref_dps_4obj_retest.maxFund.iloc[min_dist[0]], ref_2dv_4obj_retest.maxFund.iloc[min_dist[1]]]
zs = ref_dps_4obj_retest.annRev.drop(min_dist[0])
ys = ref_dps_4obj_retest.maxDebt.drop(min_dist[0])
xs = ref_dps_4obj_retest.maxComplex.drop(min_dist[0])
ss = 20 + 1.3*ref_dps_4obj_retest.maxFund.drop(min_dist[0])
p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.6, c=col_blues[2],zorder=2)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
p1 = ax.scatter(x_min_dist[0], y_min_dist[0], z_min_dist[0], s=s_min_dist[0], marker='v', alpha=1, c=col_blues[0],zorder=3)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
zs = ref_2dv_4obj_retest.annRev.drop(min_dist[1])
ys = ref_2dv_4obj_retest.maxDebt.drop(min_dist[1])
xs = ref_2dv_4obj_retest.maxComplex.drop(min_dist[1])
ss = 20 + 1.3*ref_2dv_4obj_retest.maxFund.drop(min_dist[1])
p1 = ax.scatter(xs, ys, zs, s=ss, marker='^',alpha=0.6, c=col_reds[2],zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
p1 = ax.scatter(x_min_dist[1], y_min_dist[1], z_min_dist[1], s=s_min_dist[1], marker='^', alpha=1, c=col_reds[0], zorder=0)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'compare2dvDps_4objForm_4objView.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)




### get min/max size markers for legend
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
min_s = 20 + 1.3*min(ref_dps_4obj_retest.maxFund.min(),ref_2dv_4obj_retest.maxFund.max())
max_s = 20 + 1.3*max(ref_dps_4obj_retest.maxFund.max(),ref_2dv_4obj_retest.maxFund.max())
zs = ref_dps_4obj_retest.annRev.drop(min_dist[0])
ys = ref_dps_4obj_retest.maxDebt.drop(min_dist[0])
xs = ref_dps_4obj_retest.maxComplex.drop(min_dist[0])
ss = 20 + 1.3*ref_dps_4obj_retest.maxFund.drop(min_dist[0])
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.6, c=col_blues[2],zorder=2)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
zs = ref_2dv_4obj_retest.annRev.drop(min_dist[1])
ys = ref_2dv_4obj_retest.maxDebt.drop(min_dist[1])
xs = ref_2dv_4obj_retest.maxComplex.drop(min_dist[1])
ss = 20 + 1.3*ref_2dv_4obj_retest.maxFund.drop(min_dist[1])
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v',alpha=0.6, c=col_reds[2],zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
p1 = ax.scatter(xs[0], ys[0], zs[0], s=min_s, marker='v',alpha=0.6, c='0.5',zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
p1 = ax.scatter(xs[100], ys[100], zs[100], s=max_s, marker='v',alpha=0.6, c='0.5',zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'compare2dvDps_4objForm_4objView_size.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)







### get subproblem pareto fronts
subproblems = ['1234','123','124','134','234','12','13','14','23','24','34']
paretos = {}
for s in subproblems:
  paretos[s] = pd.read_csv(dir_moea_output + 'DPS_subproblems/DPS_' + s + '_pareto.reference', sep=' ', names=['annRev','maxDebt','maxComplex','maxFund'],index_col=0)
  paretos[s].index -= 1
  paretos[s]['annRev'] *= -1

subproblems_with_conflicts = ['1234','123','124','234','12','23','24']
pareto_cols = {}
pareto_cols['1234'] = ['annRev', 'maxDebt', 'maxComplex', 'maxFund']
pareto_cols['123'] = ['annRev', 'maxDebt', 'maxComplex']
pareto_cols['124'] = ['annRev', 'maxDebt', 'maxFund']
pareto_cols['234'] = ['maxDebt', 'maxComplex', 'maxFund']
pareto_cols['12'] = ['annRev', 'maxDebt']
pareto_cols['23'] = ['maxDebt', 'maxComplex']
pareto_cols['24'] = ['maxDebt', 'maxFund']


range_objs = {}
for n,k in enumerate(subproblems_with_conflicts):
  d = paretos[k]
  range_objs[k] = {}
  for o in pareto_cols[k]:
    range_objs[k][o] = [d[o].min(), d[o].max()]
    d[o + 'Norm'] = (d[o] - range_objs[k][o][0]) / (range_objs[k][o][1] - range_objs[k][o][0])
  if ('annRev' in pareto_cols[k]):
    d['annRevNorm'] = 1 - d['annRevNorm']
  squares = 0
  if ('annRev' in pareto_cols[k]):
    squares += d['annRevNorm'] **2
  if ('maxDebt' in pareto_cols[k]):
    squares += d['maxDebtNorm'] **2
  if ('maxComplex' in pareto_cols[k]):
    squares += d['maxComplexNorm'] **2
  if ('maxFund' in pareto_cols[k]):
    squares += d['maxFundNorm'] **2
  d['totalDistance'] = np.sqrt(squares)


#
#
# ### get "best" policy from each subproblem
# d = paretos['1234']
# best_pol = {}
# dist = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2 + d['maxComplexNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['1234'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2 + d['maxComplexNorm'] **2 )
# best_pol['123'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['124'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2 + d['maxComplexNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['134'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxDebtNorm'] **2 + d['maxComplexNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['234'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2 + d['maxDebtNorm'] **2)
# best_pol['12'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2+ d['maxComplexNorm'] **2 )
# best_pol['13'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['14'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxDebtNorm'] **2 + d['maxComplexNorm'] **2 )
# best_pol['23'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxDebtNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['24'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxComplexNorm'] **2 + d['maxFundNorm'] **2)
# best_pol['34'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['annRevNorm'] **2)
# best_pol['1'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxDebtNorm'] **2)
# best_pol['2'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxComplexNorm'] **2)
# best_pol['3'] = np.where(dist == dist.min())[0]
# dist = np.sqrt(d['maxFundNorm'] **2)
# best_pol['4'] = np.where(dist == dist.min())[0]
#
# best_pol_3o = []
# for k in ['123','124','134','234']:
#   best_pol_3o.append(best_pol[k][0])
# best_pol_2o = []
# for k in ['12','13','14','23','24','34']:
#   best_pol_2o.append(best_pol[k][0])
# best_pol_1o = []
# for k in ['1','2','3','4']:
#   best_pol_1o.append(best_pol[k][0])
#
# ### 4obj with subproblem bests
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# zs = ref_dps_4obj_retest.annRev
# ys = ref_dps_4obj_retest.maxDebt
# xs = ref_dps_4obj_retest.maxComplex
# ss = 20 + 1.3*ref_dps_4obj_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.1, c='0.3')#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# for i,ind in enumerate([best_pol['1234'], best_pol_3o, best_pol_2o, best_pol_1o]):
#   zs = ref_dps_4obj_retest.annRev.iloc[ind]
#   ys = ref_dps_4obj_retest.maxDebt.iloc[ind]
#   xs = ref_dps_4obj_retest.maxComplex.iloc[ind]
#   ss = 20 + 1.3*ref_dps_4obj_retest.maxFund.iloc[ind]
#   p1 = ax.scatter(xs,ys,zs, s=ss, marker='v', alpha=1, c=[col_vir[i]])#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([10,20,30,40])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
# plt.savefig(dir_figs + 'compareObjFormulations.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)





# ### plot subproblems
# fig = plt.figure()
# ax = fig.add_subplot(3,2,1)
# xs = -ref_dps_4obj_retest.annRev
# ys = ref_dps_4obj_retest.maxDebt
# cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs,ys, c=cs, cmap='viridis', marker='^', alpha=0.5, s=60)
# ax = fig.add_subplot(3,2,2)
# xs = -ref_dps_4obj_retest.annRev
# ys = ref_dps_4obj_retest.maxComplex
# cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs,ys, c=cs, cmap='viridis', marker='^', alpha=0.5, s=60)
# ax = fig.add_subplot(3,2,3)
# xs = -ref_dps_4obj_retest.annRev
# ys = ref_dps_4obj_retest.maxFund
# cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs,ys, c=cs, cmap='viridis', marker='^', alpha=0.5, s=60)
# ax = fig.add_subplot(3,2,4)
# xs = ref_dps_4obj_retest.maxDebt
# ys = ref_dps_4obj_retest.maxComplex
# cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs,ys, c=cs, cmap='viridis', marker='^', alpha=0.5, s=60)
# ax = fig.add_subplot(3,2,5)
# xs = ref_dps_4obj_retest.maxDebt
# ys = ref_dps_4obj_retest.maxFund
# cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs,ys, c=cs, cmap='viridis', marker='^', alpha=0.5, s=60)
# ax = fig.add_subplot(3,2,6)
# xs = ref_dps_4obj_retest.maxComplex
# ys = ref_dps_4obj_retest.maxFund
# cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs,ys, c=cs, cmap='viridis', marker='^', alpha=0.5, s=60)





### 4obj with subproblem bests
fig = plt.figure()
baseline = paretos['1234'].copy()
ax = fig.add_subplot(1,1,1, projection='3d')
subprob = paretos['13'].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
zs = subprob.annRev.loc[ind]
ys = subprob.maxDebt.loc[ind]
xs = subprob.maxComplex.loc[ind]
ss = 20 + 1.3 * subprob.maxFund.loc[ind]
p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[0]])
zs = subprob.annRev.drop(ind)
ys = subprob.maxDebt.drop(ind)
xs = subprob.maxComplex.drop(ind)
ss = 20 + 1.3*subprob.maxFund.drop(ind)
p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[2]])
subprob = paretos['23'].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
zs = subprob.annRev.loc[ind]
ys = subprob.maxDebt.loc[ind]
xs = subprob.maxComplex.loc[ind]
ss = 20 + 1.3 * subprob.maxFund.loc[ind]
p1 = ax.scatter(xs, ys, zs, s=ss, marker='<', alpha=1, c=[col_reds[0]])
zs = subprob.annRev.drop(ind)
ys = subprob.maxDebt.drop(ind)
xs = subprob.maxComplex.drop(ind)
ss = 20 + 1.3*subprob.maxFund.drop(ind)
p1 = ax.scatter(xs, ys, zs, s=ss, marker='<', alpha=1, c=[col_reds[2]])
subprob = paretos['24'].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
zs = subprob.annRev.loc[ind]
ys = subprob.maxDebt.loc[ind]
xs = subprob.maxComplex.loc[ind]
ss = 20 + 1.3 * subprob.maxFund.loc[ind]
p1 = ax.scatter(xs, ys, zs, s=ss, marker='>', alpha=1, c=[col_purples[0]])
zs = subprob.annRev.drop(ind)
ys = subprob.maxDebt.drop(ind)
xs = subprob.maxComplex.drop(ind)
ss = 20 + 1.3*subprob.maxFund.drop(ind)
p1 = ax.scatter(xs, ys, zs, s=ss, marker='>', alpha=1, c=[col_purples[2]])
subprob = paretos['13'].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
zs = subprob.annRev
ys = subprob.maxDebt
xs = subprob.maxComplex
ss = 20 + 1.3 * subprob.maxFund
p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.4')
subprob = paretos['14'].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
zs = subprob.annRev
ys = subprob.maxDebt
xs = subprob.maxComplex
ss = 20 + 1.3 * subprob.maxFund
p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.4')
subprob = paretos['34'].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
zs = subprob.annRev
ys = subprob.maxDebt
xs = subprob.maxComplex
ss = 20 + 1.3 * subprob.maxFund
p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.4')
zs = baseline.annRev
ys = baseline.maxDebt
xs = baseline.maxComplex
ss = 20 + 1.3*baseline.maxFund
p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.8')
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0,0.25,0.5,0.75])
ax.set_yticks([10,20,30,40])
ax.set_zticks([9.5,10,10.5,11])
ax.view_init(elev=20, azim =-45)
ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
plt.savefig(dir_figs + 'compareObjFormulations_2objSub.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)


for k in ['123','124','234']:
  fig = plt.figure()
  baseline = paretos['1234'].copy()
  ax = fig.add_subplot(1,1,1, projection='3d')
  subprob = paretos[k].copy()
  baseline = baseline.drop(subprob.index, errors='ignore')
  ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
  zs = subprob.annRev.loc[ind]
  ys = subprob.maxDebt.loc[ind]
  xs = subprob.maxComplex.loc[ind]
  ss = 20 + 1.3 * subprob.maxFund.loc[ind]
  p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[0]])
  zs = subprob.annRev.drop(ind)
  ys = subprob.maxDebt.drop(ind)
  xs = subprob.maxComplex.drop(ind)
  ss = 20 + 1.3*subprob.maxFund.drop(ind)
  p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[2]])
  zs = baseline.annRev
  ys = baseline.maxDebt
  xs = baseline.maxComplex
  ss = 20 + 1.3*baseline.maxFund
  p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.8')
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  ax.set_xticks([0,0.25,0.5,0.75])
  ax.set_yticks([10,20,30,40])
  ax.set_zticks([9.5,10,10.5,11])
  ax.view_init(elev=20, azim =-45)
  ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
  plt.savefig(dir_figs + 'compareObjFormulations_' + k + '.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)








### plot policies meeting brushing constraints
k = '1234_constraint'
paretos[k] = paretos['1234'].copy().iloc[:,0:4]
pareto_cols[k] = pareto_cols['1234']
fixed_cost = 0.914
MEAN_REVENUE = 128.48255822159567
mean_net_revenue = MEAN_REVENUE * (1 - fixed_cost)
min_annRev = mean_net_revenue * 0.95
max_maxDebt = mean_net_revenue * 1.5
max_maxFund = mean_net_revenue * 2
brush_annRev = paretos[k].annRev >= min_annRev
brush_maxDebt = paretos[k].maxDebt <= max_maxDebt
brush_maxFund = paretos[k].maxFund <= max_maxFund
paretos[k] = paretos[k].loc[(brush_annRev & brush_maxDebt & brush_maxFund), :]

d = paretos[k]
range_objs[k] = {}
for o in pareto_cols[k]:
  range_objs[k][o] = [d[o].min(), d[o].max()]
  d[o + 'Norm'] = (d[o] - range_objs[k][o][0]) / (range_objs[k][o][1] - range_objs[k][o][0])
if ('annRev' in pareto_cols[k]):
  d['annRevNorm'] = 1 - d['annRevNorm']
squares = 0
if ('annRev' in pareto_cols[k]):
  squares += d['annRevNorm'] ** 2
if ('maxDebt' in pareto_cols[k]):
  squares += d['maxDebtNorm'] ** 2
if ('maxComplex' in pareto_cols[k]):
  squares += d['maxComplexNorm'] ** 2
if ('maxFund' in pareto_cols[k]):
  squares += d['maxFundNorm'] ** 2
d['totalDistance'] = np.sqrt(squares)


fig = plt.figure()
baseline = paretos['1234'].copy()
ax = fig.add_subplot(1, 1, 1, projection='3d')
subprob = paretos[k].copy()
baseline = baseline.drop(subprob.index, errors='ignore')
ind = subprob.index[np.where(subprob.totalDistance == subprob.totalDistance.min())[0][0]]
zs = subprob.annRev.loc[ind]
ys = subprob.maxDebt.loc[ind]
xs = subprob.maxComplex.loc[ind]
ss = 20 + 1.3 * subprob.maxFund.loc[ind]
p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[0]])
zs = subprob.annRev.drop(ind)
ys = subprob.maxDebt.drop(ind)
xs = subprob.maxComplex.drop(ind)
ss = 20 + 1.3 * subprob.maxFund.drop(ind)
p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=1, c=[col_blues[2]])
zs = baseline.annRev
ys = baseline.maxDebt
xs = baseline.maxComplex
ss = 20 + 1.3 * baseline.maxFund
p1 = ax.scatter(xs, ys, zs, s=ss, marker='^', alpha=1, c='0.8')
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0, 0.25, 0.5, 0.75])
ax.set_yticks([10, 20, 30, 40])
ax.set_zticks([9.5, 10, 10.5, 11])
ax.view_init(elev=20, azim=-45)
ax.plot([0.01], [0.01], [11.09], marker='*', ms=25, c='k')
plt.savefig(dir_figs + 'compareObjFormulations_' + k + '.eps', bbox_inches='tight', figsize=(4.5, 8), dpi=500)

















print('Finished, ', datetime.now() - startTime)

