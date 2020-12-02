import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import matplotlib as mpl
from matplotlib import ticker, cm, colors
import copy
import itertools

sns.set_style('white')
# sns.set_context('paper', font_scale=3)

eps = 1e-10

cmap_vir = cm.get_cmap('viridis')
col_vir = [cmap_vir(0.1),cmap_vir(0.4),cmap_vir(0.7),cmap_vir(0.85)]
cmap_blues = cm.get_cmap('Blues_r')
col_blues = [cmap_blues(0.1),cmap_blues(0.3),cmap_blues(0.5),cmap_blues(0.8)]
cmap_reds = cm.get_cmap('Reds_r')
col_reds = [cmap_reds(0.1),cmap_reds(0.3),cmap_reds(0.5),cmap_reds(0.8)]
cmap_purples = cm.get_cmap('Purples_r')
col_purples = [cmap_purples(0.1),cmap_purples(0.3),cmap_purples(0.5),cmap_purples(0.8)]
col_brewerQual4 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

dir_data = '../../data/optimization_output/'
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



# read in ref sets
# ref_dps_2obj_borg, ndv_dps_2obj_borg = getSet(dir_data + 'DPS_2wd_2obj/DPS_2wd_2obj_borg_4obj.resultfile', 4)
ref_dps_2obj_retest, ndv_dps_2obj_retest = getSet(dir_data + 'DPS_2wd_2obj/DPS_2wd_2obj_borg_retest_4obj.resultfile', 4)
# ref_2dv_2obj_borg, ndv_2dv_2obj_borg = getSet(dir_data + 'DPS_2obj_2dv/DPS_2obj_2dv_borg_4obj.resultfile', 4)
ref_2dv_2obj_retest, ndv_2dv_2obj_retest = getSet(dir_data + 'DPS_2obj_2dv/DPS_2obj_2dv_borg_retest_4obj.resultfile', 4)

# ref_dps_4obj_borg, ndv_dps_4obj_borg = getSet(dir_data + 'DPS_2wd_F1D1P1/DPS_2wd_F1D1P1_borg.resultfile', 4)
ref_dps_4obj_retest, ndv_dps_4obj_retest = getSet(dir_data + 'DPS_2wd_F1D1P1/DPS_2wd_F1D1P1_borg_retest.resultfile', 4)
# ref_2dv_4obj_borg, ndv_2dv_4obj_borg = getSet(dir_data + 'DPS_4obj_2dv/DPS_4obj_2dv_borg.resultfile', 4)
ref_2dv_4obj_retest, ndv_2dv_4obj_retest = getSet(dir_data + 'DPS_4obj_2dv/DPS_4obj_2dv_borg_retest.resultfile', 4)



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
  paretos[s] = pd.read_csv(dir_data + 'DPS_subproblems/DPS_' + s + '_pareto.reference', sep=' ', names=['annRev','maxDebt','maxComplex','maxFund'],index_col=0)
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






#
# ### 4obj formulation, 4obj view
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# # vmin = min(ref_dps_4obj_retest.totalDistance4obj.min(), ref_2dv_4obj_retest.totalDistance4obj.min())*0.9
# # vmax = max(ref_dps_4obj_retest.totalDistance4obj.max(), ref_2dv_4obj_retest.totalDistance4obj.max())*1.1
# zs = ref_dps_4obj_retest.annRev
# ys = ref_dps_4obj_retest.maxDebt
# xs = ref_dps_4obj_retest.maxComplex
# ss = 10 + ref_dps_4obj_retest.maxFund
# # cs = ref_dps_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.6, c=col_blues[1])#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# # cb = fig.colorbar(p1, ax=ax, ticks=[0.1,1,10,100])
# # cb.set_label('MaxFund')
# zs = ref_2dv_4obj_retest.annRev
# ys = ref_2dv_4obj_retest.maxDebt
# xs = ref_2dv_4obj_retest.maxComplex
# ss = 10 + ref_2dv_4obj_retest.maxFund
# # cs = ref_2dv_4obj_retest.totalDistance4obj
# p1 = ax.scatter(xs, ys, zs, s=ss, marker='v',alpha=0.6, c=col_reds[1])#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
# # cb = fig.colorbar(p1, ax=ax, ticks=[0.1,1,10,100])
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([10,20,30,40])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
# # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# # ax.set_title('4obj space')
# plt.savefig(dir_figs + 'compare2dvDps_4objForm_4objView.png', bbox_inches='tight', figsize=(4.5,8), dpi=500)
#
#
#
# ### 2obj formulation, 4obj view
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# zs = ref_dps_2obj_retest.annRev
# ys = ref_dps_2obj_retest.maxDebt
# xs = ref_dps_2obj_retest.maxComplex
# cs = ref_dps_2obj_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues', vmin=0.1, vmax=lims3d['maxFund'][1], norm=colors.LogNorm() , marker='v', s=60)
# # cb = fig.colorbar(p1, ax=ax, ticks=[0.1,1,10,100])
# # cb.set_label('MaxFund')
# zs = ref_2dv_2obj_retest.annRev
# ys = ref_2dv_2obj_retest.maxDebt
# xs = ref_2dv_2obj_retest.maxComplex
# cs = ref_2dv_2obj_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Reds', vmin=0.1, vmax=lims3d['maxFund'][1], norm=colors.LogNorm() , marker='^', s=60)
# # cb = fig.colorbar(p1, ax=ax, ticks=[0.1,1,10,100])
# ax.set_xlim(lims3d['maxComplex'])
# ax.set_ylim(lims3d['maxDebt'])
# ax.set_zlim(lims3d['annRev'])
# ax.set_xticks([0,0.25,0.5,0.75])
# ax.set_yticks([10,20,30,40])
# ax.set_zticks([9.5,10,10.5,11])
# ax.view_init(elev=20, azim =-45)
# ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
# # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# # ax.set_title('4obj space')
# plt.savefig(dir_figs + 'compare2dvDps_2objForm_4objView.png', bbox_inches='tight', figsize=(4.5,8), dpi=500)
#
#
#
# ### 4 obj formulation, 2 obj view
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ys = ref_dps_4obj_retest.annRev
# xs = ref_dps_4obj_retest.maxDebt
# p1 = ax.scatter(xs, ys, c=col_blues[0], marker='v', alpha=0.6)
# ys = ref_2dv_4obj_retest.annRev
# xs = ref_2dv_4obj_retest.maxDebt
# p2 = ax.scatter(xs,ys, c=col_reds[1], marker='^', alpha=0.6)
# plt.xlim(lims['maxDebt'])
# plt.ylim(lims['annRev'])
# plt.xticks([0,10,20,30,40])
# plt.yticks([9.5,10,10.5,11])
# plt.tick_params(length=3)
# plt.plot([1.2],[11.05],marker='*',ms=25,c='k')
# plt.savefig(dir_figs + 'compare2dvDps_4objForm_2objView.png', bbox_inches='tight', dpi=500)
#
#
#
#
# ### Comparison of 2dv vs full dps, 2 objective version
# fig = plt.figure()
# ax = fig.add_subplot(111)
# vmin = min(ref_2dv_2obj_retest.totalDistance2obj.min(), ref_dps_2obj_retest.totalDistance2obj.min())*0.9
# vmax = max(ref_2dv_2obj_retest.totalDistance2obj.max(), ref_dps_2obj_retest.totalDistance2obj.max())*1.2
# ys = ref_2dv_2obj_retest.annRev
# xs = ref_2dv_2obj_retest.maxDebt
# cs = ref_2dv_2obj_retest.totalDistance2obj
# p1 = ax.scatter(xs,ys, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax, marker='^', alpha=1, s=60)
# ys = ref_dps_2obj_retest.annRev
# xs = ref_dps_2obj_retest.maxDebt
# cs = ref_dps_2obj_retest.totalDistance2obj
# p1 = ax.scatter(xs, ys, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax, marker='v', alpha=1, s=60)
# plt.xlim(lims['maxDebt'])
# plt.ylim(lims['annRev'])
# plt.xticks([0,10,20,30,40])
# plt.yticks([9.5,10,10.5,11])
# plt.tick_params(length=3)
# plt.plot([1.2],[11.05],marker='*',ms=25,c='k')
# plt.savefig(dir_figs + 'compare2dvDps_2objForm_2objView.png', bbox_inches='tight', dpi=500)
#




#
# ### Comparison of training vs test error
# # 2obj
# fig = plt.figure()
# ax = fig.add_subplot(311)
# xs = ref_full_2obj_retest.annRev - ref_full_2obj_borg.annRev
# ys = ref_full_2obj_retest.maxDebt - ref_full_2obj_borg.maxDebt
# p1 = ax.scatter(xs, ys)
# xs = ref_2dv_2obj_retest.annRev - ref_2dv_2obj_borg.annRev
# ys = ref_2dv_2obj_retest.maxDebt - ref_2dv_2obj_borg.maxDebt
# p1 = ax.scatter(xs,ys)
# ax.set_xlabel('AnnRev')
# ax.set_ylabel('MaxDebt')
# ax.set_title('2obj')
#
# # 4obj
# ax = fig.add_subplot(312, projection='3d')
# zs = ref_2dv_4obj_borg.annRev - ref_2dv_4obj_retest.annRev
# ys = ref_2dv_4obj_borg.maxDebt - ref_2dv_4obj_retest.maxDebt
# xs = ref_2dv_4obj_borg.maxComplex - ref_2dv_4obj_retest.maxComplex
# cs = ref_2dv_4obj_borg.maxFund - ref_2dv_4obj_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Reds_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# zs = ref_dps_4obj_borg.annRev - ref_full_4obj_retest.annRev
# ys = ref_dps_4obj_borg.maxDebt - ref_full_4obj_retest.maxDebt
# xs = ref_dps_4obj_borg.maxComplex - ref_full_4obj_retest.maxComplex
# cs = ref_dps_4obj_borg.maxFund - ref_full_4obj_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj')
#
#
# # 4obj
# ax = fig.add_subplot(313, projection='3d')
# plt.hist(ref_2dv_2obj_retest.constraint - ref_2dv_2obj_borg.constraint)
# plt.hist(ref_2dv_4obj_retest.constraint - ref_2dv_4obj_borg.constraint)
# plt.hist(ref_full_2obj_retest.constraint - ref_full_2obj_borg.constraint)
# plt.hist(ref_full_4obj_retest.constraint - ref_dps_4obj_borg.constraint)





#
# ### look at different seed results
# sets_2dv_2obj_borg = []
# sets_2dv_4obj_borg = []
# sets_full_2obj_borg = []
# sets_full_4obj_borg = []
# sets_2dv_2obj_retest = []
# sets_2dv_4obj_retest = []
# sets_full_2obj_retest = []
# sets_full_4obj_retest = []
#
# for s in range(1,21):
#   sets_2dv_2obj_borg.append(getSet(dir_data + 'DPS_2dv_2obj/sets/PortDPS_DPS_2dv_samp50000_seedS1_seedB'+str(s)+'.set', 2)[0])
#   sets_2dv_4obj_borg.append(getSet(dir_data + 'DPS_2dv_4obj/sets/PortDPS_DPS_2dv_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_full_2obj_borg.append(getSet(dir_data + 'DPS_full_2obj/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 2)[0])
#   sets_full_4obj_borg.append(getSet(dir_data + 'DPS_full_4obj/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_2dv_2obj_retest.append(getSet(dir_data + 'DPS_2dv_2obj/retest/PortDPS_DPS_2dv_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 2)[0])
#   sets_2dv_4obj_retest.append(getSet(dir_data + 'DPS_2dv_4obj/retest/PortDPS_DPS_2dv_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#   sets_full_2obj_retest.append(getSet(dir_data + 'DPS_full_2obj/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 2)[0])
#   sets_full_4obj_retest.append(getSet(dir_data + 'DPS_full_4obj/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#
#
# ### plot borg vs retest objectives
# fig = plt.figure()
# ax = fig.add_subplot(311)
# for s in range(20):
#   col = ['b', 'r']
#   for c,d in enumerate([sets_full_2obj_borg[s], sets_2dv_2obj_borg[s]]):
#     ax.scatter(d['annRev'], d['maxDebt'], c=col[c], alpha=0.5)
#   col = ['lightblue','pink']
#   for c,d in enumerate([sets_full_2obj_retest[s], sets_2dv_2obj_retest[s]]):
#     ax.scatter(d['annRev'], d['maxDebt'], c=col[c], alpha=0.5)
# plt.xlabel('annRev')
# plt.ylabel('maxDebt')
# plt.legend(['4rbf','2dv'])
# ax = fig.add_subplot(312)
# for s in range(20):
#   col = ['b', 'r']
#   for c,d in enumerate([sets_full_4obj_borg[s], sets_2dv_4obj_borg[s]]):
#     ax.scatter(d['annRev'], d['maxDebt'], c=col[c], alpha=0.5)
#   col = ['lightblue','pink']
#   for c,d in enumerate([sets_full_4obj_retest[s], sets_2dv_4obj_retest[s]]):
#     ax.scatter(d['annRev'], d['maxDebt'], c=col[c], alpha=0.5)
# plt.xlabel('annRev')
# plt.ylabel('maxDebt')
# plt.legend(['4rbf','2dv'])
# ax = fig.add_subplot(313)
# for s in range(20):
#   col = ['b', 'r']
#   for c,d in enumerate([sets_full_4obj_borg[s], sets_2dv_4obj_borg[s]]):
#     ax.scatter(d['maxComplex'], d['maxFund'], c=col[c], alpha=0.5)
#   col = ['lightblue','pink']
#   for c,d in enumerate([sets_full_4obj_retest[s], sets_2dv_4obj_retest[s]]):
#     ax.scatter(d['maxComplex'], d['maxFund'], c=col[c], alpha=0.5)
# plt.xlabel('maxComplex')
# plt.ylabel('maxFund')
# plt.legend(['4rbf','2dv'])







#
# ### look at different sample retests
# samples = []
# samples.append(getSet(dir_data + 'DPS_full_4obj/DPS_full_4obj_borg.resultfile', 4)[0])
#
# for s in range(2,12):
#   samples.append(getSet(dir_data + 'DPS_full_4obj/DPS_full_4obj_borg_retestS'+str(s)+'.set', 4)[0])
# cm = plt.get_cmap('viridis')
# cs = [ cm(x) for x in np.arange(1,11)/11]
#
# ### plot borg vs retest objectives
# fig = plt.figure()
# ax = fig.add_subplot(211)
# for s in range(1,11):
#   plt.scatter(samples[s]['annRev'], samples[s]['maxDebt'], c=cs[s-1], cmap='viridis', alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.scatter(samples[0]['annRev'], samples[0]['maxDebt'], c='k')  # , cmap='viridis', alpha=0.5)
# plt.xlabel('annRev')
# plt.ylabel('maxDebt')
# ax = fig.add_subplot(212)
# for s in range(1,11):
#   plt.scatter(samples[s]['maxComplex'], samples[s]['maxFund'], c=cs[s-1], cmap='viridis', alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.scatter(samples[0]['maxComplex'], samples[0]['maxFund'], c='k')  # , cmap='viridis', alpha=0.5)
# plt.xlabel('maxComplex')
# plt.ylabel('maxFund')
#
#
#
# ### histogram of error for each objective
# fig = plt.figure()
# ax = fig.add_subplot(221)
# for s in range(1,11):
#   plt.hist(samples[s]['annRev'] - samples[0]['annRev'], alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.axvline(0.075, c='k')
# plt.axvline(-0.075, c='k')
# plt.xlabel('annRev')
# ax = fig.add_subplot(222)
# for s in range(1,11):
#   plt.hist(samples[s]['maxDebt'] - samples[0]['maxDebt'], alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.axvline(0.225, c='k')
# plt.axvline(-0.225, c='k')
# plt.xlabel('maxDebt')
# ax = fig.add_subplot(223)
# for s in range(1,11):
#   plt.hist(samples[s]['maxComplex'] - samples[0]['maxComplex'], alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.axvline(0.05, c='k')
# plt.axvline(-0.05, c='k')
# plt.xlabel('maxComplex')
# ax = fig.add_subplot(224)
# for s in range(1,11):
#   plt.hist(samples[s]['maxFund'] - samples[0]['maxFund'], alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.axvline(0.225, c='k')
# plt.axvline(-0.225, c='k')
# plt.xlabel('maxFund')






#fn for cleaning runtime metrics
def getMetrics(metricfile, hvfile):
  # read data
  df = pd.read_csv(metricfile, sep=' ')
  names = list(df.columns)
  names[0] = names[0].replace('#','')
  df.columns = names

  hv = pd.read_csv(hvfile, sep=' ', header=None)
  df['Hypervolume'] /= hv.iloc[0,0]
  return df

# metrics_2dv_2obj_borg = []
metrics_2dv_4obj_borg = []
# metrics_dps_2obj_borg = []
metrics_dps_4obj_borg = []

for s in range(1,11):
  # metrics_2dv_2obj_borg.append(getMetrics(dir_data + 'DPS_2obj_2dv/metrics/PortDPS_DPS_2dv_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
  #                                         dir_data + 'DPS_2obj_combined_borg/overall.hypervolume'))
  metrics_2dv_4obj_borg.append(getMetrics(dir_data + 'DPS_4obj_2dv/metrics/PortDPS_DPS_2dv_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  # metrics_dps_2obj_borg.append(getMetrics(dir_data + 'DPS_2wd_2obj/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB' + str(s) + '_borg.metrics',
  #                                         dir_data + 'DPS_2obj_combined_borg/overall.hypervolume'))
  metrics_dps_4obj_borg.append(getMetrics(dir_data + 'DPS_2wd_F1D1P1/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB' + str(s) + '_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))


### plot nfe vs hv
fig = plt.figure()
# ax = fig.add_subplot(211)
# for s in range(10):
#   col = [col_reds[1], col_blues[1]]
#   nfe = np.arange(0, 201, 1)
#   for c,d in enumerate([metrics_2dv_2obj_borg[s], metrics_dps_2obj_borg[s]]):
#     hv = d['Hypervolume'].values
#     hv = np.insert(hv, 0, 0)
#     plt.plot(nfe, hv, c=col[c])
# plt.legend(['Static', 'Dynamic',])
# ax = fig.add_subplot(212)
for s in range(10):
  col = [col_reds[1], col_blues[1]]
  nfe = np.arange(0, 120.1, 120/200)
  for c,d in enumerate([metrics_2dv_4obj_borg[s], metrics_dps_4obj_borg[s]]):
    hv = d['Hypervolume'].values
    hv = np.insert(hv, 0, 0)
    plt.plot(nfe, hv, c=col[c])
plt.legend(['Static', 'Dynamic',])
# plt.xlabel('Number of Function Evaluations')
# plt.ylabel('Hypervolume')
plt.savefig(dir_figs + 'compare2dvDps_hv.png', bbox_inches='tight', dpi=500)





### analysis of num rbf
metrics_1rbf = []
metrics_2rbf = []
metrics_3rbf = []
metrics_4rbf = []

for s in range(1,11):
  metrics_1rbf.append(getMetrics(dir_data + 'DPS_2wd_1rbf/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_2rbf.append(getMetrics(dir_data + 'DPS_2wd_2rbf/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_3rbf.append(getMetrics(dir_data + 'DPS_2wd_3rbf/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_4rbf.append(getMetrics(dir_data + 'DPS_2wd_4rbf/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))

### plot nfe vs hv for num rbfs
fig = plt.figure()
# col = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
# ax = fig.add_subplot(321)
for s in range(10):
  for c, d in enumerate(
          [metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    hv = d['Hypervolume'].values
    hv = np.insert(hv, 0, 0)
    plt.plot(nfe, hv, c=col_vir[c], alpha=0.7)
plt.legend(['1 RBF', '2 RBF', '3 RBF', '4 RBF'])
plt.savefig(dir_figs + 'compareRbfs_hv.png', bbox_inches='tight', dpi=500)

fig = plt.figure()
for s in range(10):
  for c, d in enumerate(
          [metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    hv = d['Hypervolume'].values
    hv = np.insert(hv, 0, 0)
    plt.plot(nfe, hv, c=col_vir[c], alpha=0.7)
    plt.xlim([100,122])
    plt.ylim([0.96,1])
    plt.xticks([100,105,110,115,120])
    plt.yticks([0.96,0.97,0.98,0.99,1.0])
# plt.legend(['1 RBF', '2 RBF', '3 RBF', '4 RBF'])
plt.savefig(dir_figs + 'compareRbfs_hv_zoom.png', bbox_inches='tight', dpi=500)




### plot nfe vs all metrics
fig = plt.figure()
col = ['red','orange','green','blue']
ax = fig.add_subplot(321)
for s in range(10):
  for c,d in enumerate([metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    plt.plot(d['Hypervolume'], c=col[c])
  plt.legend(['1','2','3','4'])
plt.xlabel('Number of function evaluations')
plt.ylabel('Hypervolume')
ax = fig.add_subplot(322)
for s in range(10):
  for c,d in enumerate([metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    plt.plot(d['GenerationalDistance'], c=col[c])
  # plt.legend(['1','2','3','4','5','6'])
ax = fig.add_subplot(323)
for s in range(10):
  for c,d in enumerate([metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    plt.plot(d['InvertedGenerationalDistance'], c=col[c])
  # plt.legend(['1','2','3','4','5','6'])
ax = fig.add_subplot(324)
for s in range(10):
  for c,d in enumerate([metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    plt.plot(d['Spacing'], c=col[c])
  # plt.legend(['1','2','3','4','5','6'])
ax = fig.add_subplot(325)
for s in range(10):
  for c,d in enumerate([metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    plt.plot(d['EpsilonIndicator'], c=col[c])
  # plt.legend(['1','2','3','4','5','6'])
ax = fig.add_subplot(326)
for s in range(10):
  for c,d in enumerate([metrics_1rbf[s], metrics_2rbf[s], metrics_3rbf[s], metrics_4rbf[s]]):
    plt.plot(d['MaximumParetoFrontError'], c=col[c])
  # plt.legend(['1','2','3','4','5','6'])






### analysis of rbf inputs
metrics_F1D1P1 = []
metrics_F1D1P0 = []
metrics_F1D0P1 = []
metrics_F0D1P1 = []
metrics_F0D0P1 = []
metrics_F0D1P0 = []
metrics_F1D0P0 = []
metrics_F0D0P0 = []
metrics_sepRbf = []
metrics_shareRbf = []

for s in range(1,11):
  metrics_F1D1P1.append(getMetrics(dir_data + 'DPS_2wd_F1D1P1/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F1D1P0.append(getMetrics(dir_data + 'DPS_2wd_F1D1P0/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F1D0P1.append(getMetrics(dir_data + 'DPS_2wd_F1D0P1/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F0D1P1.append(getMetrics(dir_data + 'DPS_2wd_F0D1P1/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F0D0P1.append(getMetrics(dir_data + 'DPS_2wd_F0D0P1/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F0D1P0.append(getMetrics(dir_data + 'DPS_2wd_F0D1P0/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F1D0P0.append(getMetrics(dir_data + 'DPS_2wd_F1D0P0/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_F0D0P0.append(getMetrics(dir_data + 'DPS_2wd_F0D0P0/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_sepRbf.append(getMetrics(dir_data + 'DPS_sepRbf/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))
  metrics_shareRbf.append(getMetrics(dir_data + 'DPS_shareRbf/metrics/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_borg.metrics',
                                          dir_data + 'DPS_2wd_combined_borg/overall.hypervolume'))


### plot nfe vs hv for diff formulations
fig = plt.figure()
nfe = np.arange(0, 120.1, 120 / 200)
col_scal = [0.2,0.5,0.8]
for s in range(10):
  for c, d in enumerate(
          [metrics_shareRbf[s], metrics_sepRbf[s], metrics_F1D1P1[s]]):
    hv = d['Hypervolume'].values
    hv = np.insert(hv, 0, 0)
    plt.plot(nfe, hv, c=cmap_vir(col_scal[c]), alpha=0.7)
plt.legend(['F1','F2','F3'])
plt.savefig(dir_figs + 'compareFormulation_hv.png', bbox_inches='tight', dpi=500)

fig = plt.figure()
for s in range(10):
  for c, d in enumerate(
          [metrics_shareRbf[s], metrics_sepRbf[s], metrics_F1D1P1[s]]):
    hv = d['Hypervolume'].values
    hv = np.insert(hv, 0, 0)
    plt.plot(nfe, hv, c=cmap_vir(col_scal[c]), alpha=0.7)
    plt.xlim([100,122])
    plt.ylim([0.94,1])
    plt.xticks([100,105,110,115,120])
    plt.yticks([0.94,0.96,0.98,1.0])
# plt.legend(['1 RBF', '2 RBF', '3 RBF', '4 RBF'])
plt.savefig(dir_figs + 'compareFormulation_hv_zoom.png', bbox_inches='tight', dpi=500)



### plot distribution of HV for diff inputs
titles = ['F0D0P0', 'F0D0P1', 'F1D0P0', 'F0D1P0', 'F1D0P1', 'F0D1P1', 'F1D1P0', 'F1D1P1', 'sepRbf','shareRbf']
hvs = np.zeros((10*10, 3))
for s in range(10):
  for c, d in enumerate(
          [metrics_F0D0P0[s], metrics_F0D0P1[s], metrics_F1D0P0[s], metrics_F0D1P0[s],  metrics_F1D0P1[s], metrics_F0D1P1[s], metrics_F1D1P0[s],  metrics_F1D1P1[s], metrics_sepRbf[s], metrics_shareRbf[s]]):
    hvs[(s*10)+c, 0] = d['Hypervolume'].iloc[-1]
    hvs[(s*10)+c, 1] = c
    hvs[(s*10)+c, 2] = (1+s)/11
plt.scatter(hvs[:,1], hvs[:,0], c=hvs[:,2], cmap='viridis', alpha=0.7)
plt.xticks(range(10), titles, rotation=320, ha='left')
plt.savefig(dir_figs + 'compareInputs_hv_pts.png', bbox_inches='tight', dpi=500)





### plot objs for diff inputs
ref_F0D0P0, dum = getSet(dir_data + 'DPS_2wd_F0D0P0/DPS_2wd_F0D0P0_retest.resultfile', 4)
ref_F0D0P1, dum = getSet(dir_data + 'DPS_2wd_F0D0P1/DPS_2wd_F0D0P1_retest.resultfile', 4)
ref_F1D0P0, dum = getSet(dir_data + 'DPS_2wd_F1D0P0/DPS_2wd_F1D0P0_retest.resultfile', 4)
ref_F0D1P0, dum = getSet(dir_data + 'DPS_2wd_F0D1P0/DPS_2wd_F0D1P0_retest.resultfile', 4)
ref_F1D0P1, dum = getSet(dir_data + 'DPS_2wd_F1D0P1/DPS_2wd_F1D0P1_retest.resultfile', 4)
ref_F0D1P1, dum = getSet(dir_data + 'DPS_2wd_F0D1P1/DPS_2wd_F0D1P1_retest.resultfile', 4)
ref_F1D1P0, dum = getSet(dir_data + 'DPS_2wd_F1D1P0/DPS_2wd_F1D1P0_retest.resultfile', 4)
ref_F1D1P1, dum = getSet(dir_data + 'DPS_2wd_F1D1P1/DPS_2wd_F1D1P1_retest.resultfile', 4)
ref_sepRbf, dum = getSet(dir_data + 'DPS_sepRbf/DPS_sepRbf_retest.resultfile', 4)
ref_shareRbf, dum = getSet(dir_data + 'DPS_shareRbf/DPS_shareRbf_retest.resultfile', 4)


# 3d plot
titles = ['F0D0P0', 'F0D0P1', 'F1D0P0', 'F0D1P0', 'F1D0P1', 'F0D1P1', 'F1D1P0', 'F1D1P1', 'sepRbf','shareRbf']
# lims = {'annRev':[1e6, -1e6],'maxDebt':[1e6, -1e6],'maxComplex':[1e6, -1e6],'maxFund':[1e6, -1e6]}
# for c,d in enumerate([ref_F0D0P0, ref_F0D0P1, ref_F1D0P0, ref_F0D1P0, ref_F1D0P1, ref_F0D1P1, ref_F1D1P0, ref_F1D1P1]):
#   lims['annRev'][0] = min(lims['annRev'][0], d.annRev.min())
#   lims['maxDebt'][0] = min(lims['maxDebt'][0], d.maxDebt.min())
#   lims['maxComplex'][0] = min(lims['maxComplex'][0], d.maxComplex.min())
#   lims['maxFund'][0] = min(lims['maxFund'][0], d.maxFund.min())
#   lims['annRev'][1] = max(lims['annRev'][1], d.annRev.max())
#   lims['maxDebt'][1] = max(lims['maxDebt'][1], d.maxDebt.max())
#   lims['maxComplex'][1] = max(lims['maxComplex'][1], d.maxComplex.max())
#   lims['maxFund'][1] = max(lims['maxFund'][1], d.maxFund.max())
lims = {'annRev':[9.4,11.1],'maxDebt':[0,40],'maxComplex':[0,1],'maxFund':[0,125]}
padding = {'annRev':(lims['annRev'][1] - lims['annRev'][0])/50, 'maxDebt':(lims['maxDebt'][1] - lims['maxDebt'][0])/50,
           'maxComplex':(lims['maxComplex'][1] - lims['maxComplex'][0])/50, 'maxFund':(lims['maxFund'][1] - lims['maxFund'][0])/50}
lims3d = copy.deepcopy(lims)
lims3d['annRev'][0] += padding['annRev']
lims3d['annRev'][1] -= padding['annRev']
lims3d['maxDebt'][0] += padding['maxDebt']
lims3d['maxDebt'][1] -= padding['maxDebt']
lims3d['maxComplex'][0] += padding['maxComplex']
lims3d['maxComplex'][1] -= padding['maxComplex']

for c,d in enumerate([ref_F0D0P0, ref_F0D0P1, ref_F1D0P0, ref_F0D1P0, ref_F1D0P1, ref_F0D1P1, ref_F1D1P0, ref_F1D1P1, ref_sepRbf, ref_shareRbf]):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  zs = d.annRev
  ys = d.maxDebt
  xs = d.maxComplex
  cs = d.maxFund
  p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', vmin=0.1, vmax=lims['maxFund'][1], norm=colors.LogNorm())
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  ax.set_xticks([0, 0.25, 0.5, 0.75])
  ax.set_yticks([10, 20, 30, 40])
  ax.set_zticks([ 9.5, 10, 10.5, 11])
  ax.set_xticklabels(['','','','','',''])
  ax.set_yticklabels(['','','','','',''])
  ax.set_zticklabels(['','','','','',''])
  ax.view_init(elev=20, azim=-45)
  ax.plot([0.01], [0.01], [11.09], marker='*', ms=25, c='k')
  plt.savefig(dir_figs + 'compareInputs_objs' + titles[c] + '.png', bbox_inches='tight', figsize=(1.125,2), dpi=500)
for c, d in enumerate([ref_sepRbf]):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  zs = d.annRev
  ys = d.maxDebt
  xs = d.maxComplex
  cs = d.maxFund
  p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', vmin=0.1, vmax=lims['maxFund'][1], norm=colors.LogNorm())
  cb = fig.colorbar(p1, ax=ax, ticks=[0.1,1,10,100])
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  ax.set_xticks([0, 0.25, 0.5, 0.75])
  ax.set_yticks([10, 20, 30, 40])
  ax.set_zticks([ 9.5, 10, 10.5, 11])
  ax.set_xticklabels(['','','','','',''])
  ax.set_yticklabels(['','','','','',''])
  ax.set_zticklabels(['','','','','',''])
  ax.view_init(elev=20, azim=-45)
  ax.plot([0.01], [0.01], [11.09], marker='*', ms=25, c='k')
  plt.savefig(dir_figs + 'compareInputs_objs_cb' + titles[c] + '.png', bbox_inches='tight', figsize=(1.125,2), dpi=500)







### plot objs for diff inputs, with brushing
fixed_cost = 0.914
MEAN_REVENUE = 128.48255822159567
mean_net_revenue = MEAN_REVENUE * (1 - fixed_cost)
min_annRev = mean_net_revenue * 0.95
max_maxDebt = mean_net_revenue * 1.25
max_maxFund = mean_net_revenue * 3

for c,d in enumerate([ref_F0D0P0, ref_F0D0P1, ref_F1D0P0, ref_F0D1P0, ref_F1D0P1, ref_F0D1P1, ref_F1D1P0, ref_F1D1P1, ref_sepRbf, ref_shareRbf]):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  brush_annRev = d.annRev >= min_annRev
  brush_maxDebt = d.maxDebt <= max_maxDebt
  brush_maxFund = d.maxFund <= max_maxFund
  ref_full_yes = d.loc[(brush_annRev & brush_maxDebt & brush_maxFund), :]
  ref_full_no = d.loc[~(brush_annRev & brush_maxDebt & brush_maxFund), :]
  zs = ref_full_no.annRev
  ys = ref_full_no.maxDebt
  xs = ref_full_no.maxComplex
  cs = ref_full_no.maxFund
  p1 = ax.scatter(xs, ys, zs, c='0.7', alpha=0.3)  # , marker='v')
  zs = ref_full_yes.annRev
  ys = ref_full_yes.maxDebt
  xs = ref_full_yes.maxComplex
  cs = ref_full_yes.maxFund
  p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', vmin=0.1, vmax=lims['maxFund'][1], norm=colors.LogNorm())
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  ax.set_xticks([0, 0.25, 0.5, 0.75])
  ax.set_yticks([10, 20, 30, 40])
  ax.set_zticks([ 9.5, 10, 10.5, 11])
  ax.set_xticklabels(['','','','','',''])
  ax.set_yticklabels(['','','','','',''])
  ax.set_zticklabels(['','','','','',''])
  ax.view_init(elev=20, azim=-45)
  ax.plot([0.01], [0.01], [11.09], marker='*', ms=25, c='k')
  plt.savefig(dir_figs + 'compareInputs_objs_brush' + titles[c] + '.png', bbox_inches='tight', figsize=(1.125,2), dpi=500)











### plot objs for diff rbfs
ref_1rbf, dum = getSet(dir_data + 'DPS_2wd_1rbf/DPS_2wd_1rbf_borg_retest.resultfile', 4)
ref_2rbf, dum = getSet(dir_data + 'DPS_2wd_2rbf/DPS_2wd_2rbf_borg_retest.resultfile', 4)
ref_3rbf, dum = getSet(dir_data + 'DPS_2wd_3rbf/DPS_2wd_3rbf_borg_retest.resultfile', 4)
ref_4rbf, dum = getSet(dir_data + 'DPS_2wd_4rbf/DPS_2wd_4rbf_borg_retest.resultfile', 4)
for c,d in enumerate([ref_1rbf, ref_2rbf, ref_3rbf, ref_4rbf]):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  zs = d.annRev
  ys = d.maxDebt
  xs = d.maxComplex
  cs = d.maxFund
  p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', vmin=0.1, vmax=lims['maxFund'][1], norm=colors.LogNorm())
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  ax.set_xticks([0, 0.25, 0.5, 0.75])
  ax.set_yticks([10, 20, 30, 40])
  ax.set_zticks([ 9.5, 10, 10.5, 11])
  ax.set_xticklabels(['','','','','',''])
  ax.set_yticklabels(['','','','','',''])
  ax.set_zticklabels(['','','','','',''])
  ax.view_init(elev=20, azim=-45)
  ax.plot([0.01], [0.01], [11.09], marker='*', ms=25, c='k')
  plt.savefig(dir_figs + 'compareRbfs_objs' + titles[c] + '.png', bbox_inches='tight', figsize=(1.125,2), dpi=500)














#
# def wd(value, fund, cash, maxfund):
#   if (value > 0):
#     value = min(value, fund)
#   if (value < 0):
#     if (-value > cash):
#       value = -max(cash, 0)
#   if ((fund - value) > maxfund):
#     value = fund - maxfund
#   return value
#
#
#
#
# ### look at different seed results
# sets_1rbf_borg = []
# sets_2rbf_borg = []
# sets_3rbf_borg = []
# sets_4rbf_borg = []
# sets_5rbf_borg = []
# sets_6rbf_borg = []
# sets_1rbf_retest = []
# sets_2rbf_retest = []
# sets_3rbf_retest = []
# sets_4rbf_retest = []
# sets_5rbf_retest = []
# sets_6rbf_retest = []
#
#
# for s in range(1,11):
#   sets_1rbf_borg.append(getSet(dir_data + 'DPS_full_4obj_1rbf/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_2rbf_borg.append(getSet(dir_data + 'DPS_full_4obj_2rbf/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_3rbf_borg.append(getSet(dir_data + 'DPS_full_4obj_3rbf/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_4rbf_borg.append(getSet(dir_data + 'DPS_full_4obj/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_5rbf_borg.append(getSet(dir_data + 'DPS_full_4obj_5rbf/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_6rbf_borg.append(getSet(dir_data + 'DPS_full_4obj_6rbf/sets/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'.set', 4)[0])
#   sets_1rbf_retest.append(getSet(dir_data + 'DPS_full_4obj_1rbf/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#   sets_2rbf_retest.append(getSet(dir_data + 'DPS_full_4obj_2rbf/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#   sets_3rbf_retest.append(getSet(dir_data + 'DPS_full_4obj_3rbf/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#   sets_4rbf_retest.append(getSet(dir_data + 'DPS_full_4obj/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#   sets_5rbf_retest.append(getSet(dir_data + 'DPS_full_4obj_5rbf/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#   sets_6rbf_retest.append(getSet(dir_data + 'DPS_full_4obj_6rbf/retest/PortDPS_DPS_maxDebt_samp50000_seedS1_seedB'+str(s)+'_retestS2.set', 4)[0])
#
#
# ### plot borg vs retest objectives
# fig = plt.figure()
# ax = fig.add_subplot(211)
# for s in range(10):
#   col = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
#   for c,d in enumerate([sets_1rbf_borg[s], sets_2rbf_borg[s], sets_3rbf_borg[s], sets_4rbf_borg[s],
#                         sets_5rbf_borg[s], sets_6rbf_borg[s]]):
#     ax.scatter(d['annRev'], d['maxDebt'], c=col[c], alpha=0.5)
# ax = fig.add_subplot(212)
# for s in range(10):
#   col = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
#   for c, d in enumerate([sets_1rbf_borg[s], sets_2rbf_borg[s], sets_3rbf_borg[s], sets_4rbf_borg[s],
#                          sets_5rbf_borg[s], sets_6rbf_borg[s]]):
#     ax.scatter(d['maxComplex'], d['maxFund'], c=col[c], alpha=0.5)
#
#
#
# # read in ref sets
# ref_1rbf_borg = getSet(dir_data + 'DPS_ref_rbf_borg/DPS_full_4obj_1rbf_borg.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_2rbf_borg = getSet(dir_data + 'DPS_ref_rbf_borg/DPS_full_4obj_2rbf_borg.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_3rbf_borg = getSet(dir_data + 'DPS_ref_rbf_borg/DPS_full_4obj_3rbf_borg.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_4rbf_borg = getSet(dir_data + 'DPS_ref_rbf_borg/DPS_full_4obj_4rbf_borg.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_5rbf_borg = getSet(dir_data + 'DPS_ref_rbf_borg/DPS_full_4obj_5rbf_borg.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_6rbf_borg = getSet(dir_data + 'DPS_ref_rbf_borg/DPS_full_4obj_6rbf_borg.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_1rbf_retest = getSet(dir_data + 'DPS_ref_rbf_retest/DPS_full_4obj_1rbf_retest.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_2rbf_retest = getSet(dir_data + 'DPS_ref_rbf_retest/DPS_full_4obj_2rbf_retest.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_3rbf_retest = getSet(dir_data + 'DPS_ref_rbf_retest/DPS_full_4obj_3rbf_retest.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_4rbf_retest = getSet(dir_data + 'DPS_ref_rbf_retest/DPS_full_4obj_4rbf_retest.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_5rbf_retest = getSet(dir_data + 'DPS_ref_rbf_retest/DPS_full_4obj_5rbf_retest.reference', 4, has_dv=False, has_constraint=False)[0]
# ref_6rbf_retest = getSet(dir_data + 'DPS_ref_rbf_retest/DPS_full_4obj_6rbf_retest.reference', 4, has_dv=False, has_constraint=False)[0]
#
#
#
#
#
#
#
# ### Comparison of ref set with diff rbf
# # 3d plot
# fig = plt.figure()
# ax = fig.add_subplot(321, projection='3d')
# zs = ref_1rbf_borg.annRev
# ys = ref_1rbf_borg.maxDebt
# xs = ref_1rbf_borg.maxComplex
# cs = ref_1rbf_borg.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# # zs = ref_2dv_4obj_borg.annRev
# # ys = ref_2dv_4obj_borg.maxDebt
# # xs = ref_2dv_4obj_borg.maxComplex
# # cs = ref_2dv_4obj_borg.maxFund
# # p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Reds_r')#, norm=colors.LogNorm())
# # cb = fig.colorbar(p1, ax=ax)
# # cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj space')
#
# ax = fig.add_subplot(322, projection='3d')
# zs = ref_2rbf_borg.annRev
# ys = ref_2rbf_borg.maxDebt
# xs = ref_2rbf_borg.maxComplex
# cs = ref_2rbf_borg.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj space')
# ax = fig.add_subplot(323, projection='3d')
# zs = ref_3rbf_borg.annRev
# ys = ref_3rbf_borg.maxDebt
# xs = ref_3rbf_borg.maxComplex
# cs = ref_3rbf_borg.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj space')
# ax = fig.add_subplot(324, projection='3d')
# zs = ref_4rbf_borg.annRev
# ys = ref_4rbf_borg.maxDebt
# xs = ref_4rbf_borg.maxComplex
# cs = ref_4rbf_borg.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj space')
# ax = fig.add_subplot(325, projection='3d')
# zs = ref_5rbf_borg.annRev
# ys = ref_5rbf_borg.maxDebt
# xs = ref_5rbf_borg.maxComplex
# cs = ref_5rbf_borg.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj space')
# ax = fig.add_subplot(326, projection='3d')
# zs = ref_6rbf_borg.annRev
# ys = ref_6rbf_borg.maxDebt
# xs = ref_6rbf_borg.maxComplex
# cs = ref_6rbf_borg.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('4obj space')
#
#
#
#
#
#
#
#
# ### Comparison of ref set with diff rbf
# # 3d plot
# fig = plt.figure()
# ax = fig.add_subplot(321, projection='3d')
# zs = ref_1rbf_retest.annRev
# ys = ref_1rbf_retest.maxDebt
# xs = ref_1rbf_retest.maxComplex
# cs = ref_1rbf_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# ax.set_title('1rbf')
# ax = fig.add_subplot(322, projection='3d')
# zs = ref_2rbf_retest.annRev
# ys = ref_2rbf_retest.maxDebt
# xs = ref_2rbf_retest.maxComplex
# cs = ref_2rbf_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# ax.set_title('2rbf')
# ax = fig.add_subplot(323, projection='3d')
# zs = ref_3rbf_retest.annRev
# ys = ref_3rbf_retest.maxDebt
# xs = ref_3rbf_retest.maxComplex
# cs = ref_3rbf_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# ax.set_title('3rbf')
# ax = fig.add_subplot(324, projection='3d')
# zs = ref_4rbf_retest.annRev
# ys = ref_4rbf_retest.maxDebt
# xs = ref_4rbf_retest.maxComplex
# cs = ref_4rbf_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# ax.set_title('4rbf')
# ax = fig.add_subplot(325, projection='3d')
# zs = ref_5rbf_retest.annRev
# ys = ref_5rbf_retest.maxDebt
# xs = ref_5rbf_retest.maxComplex
# cs = ref_5rbf_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# cb.set_label('MaxFund')
# ax.set_xlabel('MaxComplexity')
# ax.set_ylabel('MaxDebt')
# ax.set_zlabel('AnnRev')
# ax.set_title('5rbf')
# ax = fig.add_subplot(326, projection='3d')
# zs = ref_6rbf_retest.annRev
# ys = ref_6rbf_retest.maxDebt
# xs = ref_6rbf_retest.maxComplex
# cs = ref_6rbf_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='Blues_r')#, norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax)
# # cb.set_label('MaxFund')
# # ax.set_xlabel('MaxComplexity')
# # ax.set_ylabel('MaxDebt')
# # ax.set_zlabel('AnnRev')
# ax.set_title('6rbf')
#








### test/plot policies from DPS
dps = getSet(dir_data + 'DPS_2wd_F1D1P1/DPS_2wd_F1D1P1_borg_retest.resultfile', 4, sort=True)[0]

### plot specific draws
# brush_annRev = dps.annRev >= min_annRev
# brush_maxDebt = dps.maxDebt <= max_maxDebt
# brush_maxFund = dps.maxFund <= max_maxFund
# ref_full_yes = dps.loc[(brush_annRev & brush_maxDebt & brush_maxFund), :]
# ref_full_no = dps.loc[~(brush_annRev & brush_maxDebt & brush_maxFund), :]
policy_ranks = [9,498,2059]
dps_choice = dps.iloc[policy_ranks,:]
# dps_choice.iloc[:,-5:]
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
zs = dps.annRev
ys = dps.maxDebt
xs = dps.maxComplex
cs = dps.maxFund
p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', vmin=0.1, vmax=lims['maxFund'][1], norm=colors.LogNorm())
# cb = fig.colorbar(p1, ax=ax, ticks=[0.1,1,10,100])
zs = dps_choice.annRev.values
ys = dps_choice.maxDebt.values
xs = dps_choice.maxComplex.values
cs = dps_choice.maxFund.values
labels = ['A','B','C']
for i in range(dps_choice.shape[0]):
  pw = ax.text(xs[i], ys[i], zs[i], labels[i], color='red', horizontalalignment='center', verticalalignment='center', size=18, weight='bold')
ax.set_xlim(lims3d['maxComplex'])
ax.set_ylim(lims3d['maxDebt'])
ax.set_zlim(lims3d['annRev'])
ax.set_xticks([0, 0.25, 0.5, 0.75])
ax.set_yticks([10, 20, 30, 40])
ax.set_zticks([9.5, 10, 10.5, 11])
ax.set_xticklabels(['', '', '', '', '', ''])
ax.set_yticklabels(['', '', '', '', '', ''])
ax.set_zticklabels(['', '', '', '', '', ''])
ax.view_init(elev=20, azim=-45)
ax.plot([0.01], [0.01], [11.09], marker='*', ms=25, c='k')
plt.savefig(dir_figs + 'policies_objs.png', bbox_inches='tight', dpi=500)












######################################################################
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
MEAN_REVENUE = 128.48255822159567     # mean revenue in absense of any financial risk mgmt. Make sure this is consistent with current input HHSamp revenue column.
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
NUM_RBF = 4       # number of radial basis functions
SHARED_RBFS = 2     # 0 = 1 rbf shared between hedge and withdrawal policies. 1 = separate rbf for each. 2 = rbf for hedge, and 2dv formulation for withdrawal.
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

samp = pd.read_csv('../../Documents/DataFilesNonGit/HHsamp10132019.txt', delimiter=' ')

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




# dps_choice = dps.iloc[0, :]
# dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
# policyWithdrawal(0.000000,  0.0, 37.136353 , 1.105171)


### plot snow contract policies (2d)
# policy_ranks = [6, 15, 800, 2143]
policy_ranks = [9,498,2059]
# policy_ranks = [2059,2058,2057]
policies = ['LowDebt','MedDebt','HighDebt']
power = [samp.powIndex.mean()] #[30, 42.5, 55] #
ny = 20
ns = 100
zmax = -100
zmin = 100
dict = {}
samples = np.arange(ns)*1000
for p in range(len(power)):
  for m in range(len(policy_ranks)):
    name = 'p'+str(p)+'_m'+str(m)
    dict[name] = {}
    # get policy params
    dps_choice = dps.iloc[policy_ranks[m],:]
    dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
    # get grid of values for policy heatmap
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[:3000].values,
                                                                samp.sswp.iloc[:3000].values,
                                                                samp.powIndex.iloc[:3000].values)
    maxFundDebt = fund_hedge.max() + 1
    minFundDebt = -(debt_hedge.max() + 1)
    dFundDebt = (maxFundDebt - minFundDebt) / 500
    maxPow = power_hedge.max() + 1
    minPow = power_hedge.min() - 1
    dPow = (maxPow - minPow) / 500
    xt, w = np.mgrid[slice(minFundDebt, maxFundDebt + dFundDebt, dFundDebt), slice(minPow, maxPow + dPow, dPow)]
    x = np.maximum(xt, 0)
    y = -np.minimum(xt, 0)
    z = policySnowContractValue(x, y, w)
    z = z[:-1, :-1]
    zmax = max(z.max(), zmax)
    zmin = min(z.min(), zmin)
    # get trajectories through state space
    dict[name]['maxDebt'] = dps_choice['maxDebt']
    dict[name]['fund_hedge'], dict[name]['debt_hedge'], dict[name]['power_hedge'], dict[name]['power_withdrawal'], dict[name]['cash_in'] = {}, {}, {}, {}, {}
    for s in range(ns):
      fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
      dict[name]['fund_hedge'][s], dict[name]['debt_hedge'][s], dict[name]['power_hedge'][s], dict[name]['power_withdrawal'][s], dict[name]['cash_in'][s] = fund_hedge, debt_hedge, power_hedge, power_withdrawal, cash_in
levels = MaxNLocator(nbins=40).tick_values(zmin, zmax)
cmap = plt.get_cmap('RdYlBu')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig = plt.figure()
for p in range(len(power)):
  for m in range(len(policy_ranks)):
    dps_choice = dps.iloc[policy_ranks[m],:]
    dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
    ax1 = fig.add_subplot(len(power),3,p*3+m+1)
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[:3000].values, samp.sswp.iloc[:3000].values, samp.powIndex.iloc[:3000].values)
    maxFundDebt = fund_hedge.max() + 1
    minFundDebt = -(debt_hedge.max() + 1)
    dFundDebt = (maxFundDebt - minFundDebt) / 500
    maxPow = power_hedge.max() + 1
    minPow = power_hedge.min() - 1
    dPow = (maxPow - minPow) / 500
    xt, w = np.mgrid[slice(minFundDebt, maxFundDebt + dFundDebt, dFundDebt), slice(minPow, maxPow + dPow, dPow)]
    x = np.maximum(xt, 0)
    y = -np.minimum(xt, 0)
    # plot policy
    z = policySnowContractValue(x, y, w)
    z = z[:-1, :-1]
    im = ax1.pcolormesh(xt, w, z, cmap=cmap, norm=norm)
    if (m==len(policy_ranks)-1):
      fig.colorbar(im, ax=ax1)
    # plot state space
    name = 'p'+str(p)+'_m'+str(m)
    for s in range(ns):
      ax1.plot(dict[name]['fund_hedge'][s] - dict[name]['debt_hedge'][s], dict[name]['power_hedge'][s], c='0.3', alpha=0.2)
    ax1.set_xlabel('Balance')
    ax1.set_ylabel('Power')
    ax1.set_title(policies[m] + ' ' + str(round(dict[name]['maxDebt'], 1)))







### plot input/policy scatter plots
# policy_ranks = [6, 15, 800, 2143]
policy_ranks = [9,498,2059]
# policy_ranks = [2059,2058,2057]
policies = ['LowDebt','MedDebt','HighDebt']
power = [samp.powIndex.mean()] #[30, 42.5, 55] #
ny = 20
ns = 100
zmax = -100
zmin = 100
dict = {}
samples = np.arange(ns)*1000
for p in range(1):#len(power)):
  for m in range(1):#len(policy_ranks)):
    name = 'p'+str(p)+'_m'+str(m)
    dict[name] = {}
    # get policy params
    dps_choice = dps.iloc[policy_ranks[m],:]
    dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
    # get grid of values for policy heatmap
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[:3000].values,
                                                                samp.sswp.iloc[:3000].values,
                                                                samp.powIndex.iloc[:3000].values)





# policy_ranks = [9,498,2059]
policy_ranks = range(192,len(dps))
# policies = ['LowDebt','MedDebt','HighDebt']
ny = 20
ns = 5000
zmax = -100
zmin = 100
# dict = {}
samples = np.random.choice([int(x) for x in np.arange(1e6 - 21)], size=ns, replace=True)
for m in policy_ranks:
  name = 'm'+str(m)
  dict = {}
  # get policy params
  dps_choice = dps.iloc[policy_ranks[m],:]
  dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
  # get trajectories through state space
  dict['annRev'] = dps_choice['annRev']
  dict['maxDebt'] = dps_choice['maxDebt']
  dict['maxComplex'] = dps_choice['maxComplex']
  dict['maxFund'] = dps_choice['maxFund']
  dict['fund_hedge'] = np.array([])
  # dict['fund_withdrawal'] = np.array([])
  dict['debt_hedge'] = np.array([])
  # dict['debt_withdrawal'] = np.array([])
  dict['power_hedge'] = np.array([])
  # dict['power_withdrawal'] = np.array([])
  # dict['cash_in'] = np.array([])
  dict['action_hedge'] = np.array([])
  # dict['action_withdrawal'] = np.array([])
  for s in range(ns):
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
    dict['fund_hedge'] = np.append(dict['fund_hedge'], fund_hedge)
    # dict['fund_withdrawal'] = np.append(dict['fund_withdrawal'], fund_withdrawal)
    dict['debt_hedge'] = np.append(dict['debt_hedge'], debt_hedge)
    # dict['debt_withdrawal'] = np.append(dict['debt_withdrawal'], debt_withdrawal)
    dict['power_hedge'] = np.append(dict['power_hedge'], power_hedge)
    # dict['power_withdrawal'] = np.append(dict['power_withdrawal'], power_withdrawal)
    # dict['cash_in'] = np.append(dict['cash_in'], cash_in)
    dict['action_hedge'] = np.append(dict['action_hedge'], action_hedge)
    # dict['action_withdrawal'] = np.append(dict['action_withdrawal'], action_withdrawal)
  pd.to_pickle(dict, 'data/save_dps_samples/'+str(m)+'.pkl')
  print(m)


ranges = {'fund_hedge':[100,0], 'debt_hedge':[100,0], 'power_hedge':[100,0], 'action_hedge':[4,0]}
for name in names:
  for att in list(ranges.keys()):
    ranges[att][0] = min(ranges[att][0], dict[name][att].min())
    ranges[att][1] = max(ranges[att][1], dict[name][att].max())
ranges = {'fund_hedge':[0,35], 'debt_hedge':[0,55], 'power_hedge':[30,60], 'action_hedge':[0,2]}

def getProbGrid(xdat, ydat, xdomain, ydomain):
  xmin = xdomain[0]
  xmax = xdomain[1]
  ymin = ydomain[0]
  ymax = ydomain[1]
  yrange = ymax - ymin
  xrange = xmax - xmin

  nstep = 40
  dy = yrange / nstep
  dx = xrange / nstep
  freqMat = np.zeros([nstep, nstep])
  for i in range(len(ydat)):
    row = int(np.floor((ydat[i] - ymin) / dy))
    col = int(np.floor((xdat[i] - xmin) / dx))
    freqMat[row, col] += 1
  freqMat /= len(ydat)
  freqMat = np.log10(freqMat)
  freqMat[freqMat == -np.inf] = np.nan
  return freqMat

names = ['m0','m1','m2']
freqMat = {}
attPairs = ['fund_fund','fund_debt','fund_power','fund_action','debt_fund','debt_debt','debt_power','debt_action','power_fund','power_debt','power_power','power_action','action_fund','action_debt','action_power','action_action']
for name in names:
  freqMat[name] = {}
  for attPair in attPairs:
    [att1, att2] = attPair.split(sep='_')
    freqMat[name][attPair + '_hedge'] = getProbGrid(dict[name][att1 + '_hedge'], dict[name][att2 + '_hedge'], ranges[att1 + '_hedge'],ranges[att2 + '_hedge'])

ranges['freq_hedge'] = [0,-1000]
for name in names:
  for attPair in attPairs:
    ranges['freq_hedge'][0] = min(ranges['freq_hedge'][0], np.nanmin(freqMat[name][attPair + '_hedge']))
    ranges['freq_hedge'][1] = max(ranges['freq_hedge'][1], np.nanmax(freqMat[name][attPair + '_hedge']))

for name in names:
  fig = plt.figure()
  col = 0
  row = -1
  for attPair in attPairs:
    row += 1
    if row == 4:
      row = 0
      col += 1
    ax = plt.subplot2grid((4, 4), (row, col))  # ,colspan=2)
    plt.imshow(freqMat[name][attPair + '_hedge'], cmap='RdYlBu_r', origin='lower',
               norm=mpl.colors.Normalize(vmin=ranges['freq_hedge'][0], vmax=ranges['freq_hedge'][1]))



def getDuDx(dat, covariate):
  if (covariate=='fund'):
    xplus = dat['fund'] + dat['fund'].mean()*0.005
    xmin = np.maximum(dat['fund'] - dat['fund'].mean()*0.005, 0)
    uplus = policySnowContractValue(xplus, dat['debt'], dat['power'])
    umin = policySnowContractValue(xmin, dat['debt'], dat['power'])
  elif (covariate=='debt'):
    xplus = dat['debt'] + dat['debt'].mean()*0.005
    xmin = np.maximum(dat['debt'] - dat['debt'].mean()*0.005, 0)
    uplus = policySnowContractValue(dat['fund'], xplus, dat['power'])
    umin = policySnowContractValue(dat['fund'], xmin, dat['power'])
  elif (covariate=='power'):
    xplus = dat['power'] + dat['power'].mean()*0.005
    xmin = np.maximum(dat['power'] - dat['power'].mean()*0.005, 0)
    uplus = policySnowContractValue(dat['fund'], dat['debt'], xplus)
    umin = policySnowContractValue(dat['fund'], dat['debt'], xmin)
  dudx = (uplus - umin) / (xplus - xmin)
  return dudx



### state-dependent sensitivity analysis (use simple R^2 method first)
name = 'm0'
predictors = ['t','fund','debt','power']
dat = pd.DataFrame({'action':dict[name]['action_hedge'], 'fund':dict[name]['fund_hedge'], 'debt':dict[name]['debt_hedge'], 'power':dict[name]['power_hedge'], 't':np.tile(np.arange(20), int(len(dict[name]['power_hedge'])/20))})
sensitivity = {}
ngrid = 10
eps = 1e-6
for predictor in predictors:
  sensitivity[predictor] = {}
  for covariate in predictors:
    sensitivity[predictor][covariate + '_R2'] = np.zeros(ngrid)
    sensitivity[predictor][covariate + '_var'] = np.zeros(ngrid)
    sensitivity[predictor]['var_component_' + covariate] = []
    sensitivity[predictor]['var_component_' + covariate + '_mean'] = np.zeros(ngrid)
  for cov1, cov2 in itertools.combinations(predictors, r=2):
    sensitivity[predictor][cov1 + '_' + cov2 + '_cov'] = np.zeros(ngrid)
    sensitivity[predictor]['var_component_' + cov1 + '_' + cov2] = []
    sensitivity[predictor]['var_component_' + cov1 + '_' + cov2 + '_mean'] = np.zeros(ngrid)

for predictor in predictors:
  xmin = dat[predictor].min()
  xmax = dat[predictor].max()
  dx = (xmax - xmin) / (ngrid - 1)
  xgrid = np.arange(xmin - dx / 2, xmax + dx / 2 + eps, dx)
  sensitivity[predictor]['grid'] = xgrid
  sensitivity[predictor]['action_var'] = np.zeros(len(xgrid) - 1)
  sensitivity[predictor]['total_R2'] = np.zeros(len(xgrid) - 1)
  for i in range(ngrid):
    filter = (dat[predictor] > xgrid[i]) & (dat[predictor] < xgrid[i+1])
    if filter.sum() > 2:
      sensitivity[predictor]['action_var'][i] = dat['action'].loc[filter].var()
      sensitivity[predictor]['total_R2'][i] = sm.ols(formula='action ~ t*fund*debt*power', data=dat.loc[filter]).fit().rsquared
      if (np.isnan(sensitivity[predictor]['action_var'][i]) == False) & (dat['action'].loc[filter].max() - dat['action'].loc[filter].min() > eps):
        for covariate in predictors:
          sensitivity[predictor][covariate + '_R2'][i] = sm.ols(formula = 'action ~ ' + covariate, data = dat.loc[filter]).fit().rsquared
          sensitivity[predictor][covariate + '_var'][i] = dat[covariate].loc[filter].var()
        for cov1, cov2 in itertools.combinations(predictors, r=2):
          sensitivity[predictor][cov1 + '_' + cov2 + '_cov'][i] = np.cov(dat[cov1].loc[filter], dat[cov2].loc[filter])[0,1]
        # now get dudx for variance decomposition
        dudx = {'fund': getDuDx(dat.loc[filter], 'fund'),
                'debt': getDuDx(dat.loc[filter], 'debt'),
                'power': getDuDx(dat.loc[filter], 'power')}
        sensitivity[predictor]['var_component_fund'].append((dudx['fund'].values)**2 * sensitivity[predictor]['fund_var'][i])
        sensitivity[predictor]['var_component_fund_mean'][i] = sensitivity[predictor]['var_component_fund'][-1].mean()
        sensitivity[predictor]['var_component_debt'].append((dudx['debt'].values)**2 * sensitivity[predictor]['debt_var'][i])
        sensitivity[predictor]['var_component_debt_mean'][i] = sensitivity[predictor]['var_component_debt'][-1].mean()
        sensitivity[predictor]['var_component_power'].append((dudx['power'].values)**2 * sensitivity[predictor]['power_var'][i])
        sensitivity[predictor]['var_component_power_mean'][i] = sensitivity[predictor]['var_component_power'][-1].mean()
        sensitivity[predictor]['var_component_fund_debt'].append((dudx['fund'].values) * (dudx['debt'].values) * sensitivity[predictor]['fund_debt_cov'][i])
        sensitivity[predictor]['var_component_fund_debt_mean'][i] = sensitivity[predictor]['var_component_fund_debt'][-1].mean()
        sensitivity[predictor]['var_component_fund_power'].append((dudx['fund'].values) * (dudx['power'].values) * sensitivity[predictor]['fund_power_cov'][i])
        sensitivity[predictor]['var_component_fund_power_mean'][i] = sensitivity[predictor]['var_component_fund_power'][-1].mean()
        sensitivity[predictor]['var_component_debt_power'].append((dudx['debt'].values) * (dudx['power'].values) * sensitivity[predictor]['debt_power_cov'][i])
        sensitivity[predictor]['var_component_debt_power_mean'][i] = sensitivity[predictor]['var_component_debt_power'][-1].mean()
    else:
      sensitivity[predictor]['action_var'][i] = np.nan
      sensitivity[predictor]['total_R2'][i] = np.nan
      for covariate in predictors:
        sensitivity[predictor][covariate + '_R2'][i] = np.nan
        sensitivity[predictor][covariate + '_var'][i] = np.nan







### mutual-information based analysis
x = dat[predictor]['action']








### plot results of state-dep SA
for predictor in ['fund','debt','power']:
  fig = plt.figure()
  stack_r2 = np.zeros(ngrid)
  for component in ['fund','debt','power','fund_debt','fund_power','debt_power']:
    stack_r2 += sensitivity[predictor]['var_component_' + component + '_mean']
    plt.plot(sensitivity[predictor]['grid'][:-1], stack_r2)
  plt.plot(sensitivity[predictor]['grid'][:-1], sensitivity[predictor]['action_var'], c='k', ls=':')
  plt.title(predictor)







#
# ### plot snow contract policies (scatter with color of policy)
# policy_ranks = [6,640,2144]
# policies = ['LowDebt','MedDebt','HighDebt']
# ny = 20
# ns = 100
# zmax = -100
# zmin = 100
# dict = {}
# samples = np.arange(ns)*1000
# for m in range(len(policy_ranks)):
#   name = 'm'+str(m)
#   dict[name] = {}
#   # get policy params
#   dps_choice = dps.iloc[policy_ranks[m],:]
#   dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
#   # get trajectories through state space
#   dict[name]['maxDebt'] = dps_choice['maxDebt']
#   dict[name]['fund'], dict[name]['debt'], dict[name]['power_hedge'], dict[name]['power_withdrawal'], dict[name]['cash_in'], dict[name]['policy'] = {}, {}, {}, {}, {}, {}
#   for s in range(ns):
#     fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
#     zmax = max(value_snow_contract.max(), zmax)
#     zmin = min(value_snow_contract.min(), zmin)
#     dict[name]['fund'][s], dict[name]['debt'][s], dict[name]['power_hedge'][s], dict[name]['power_withdrawal'][s], dict[name]['cash_in'][s], dict[name]['policy'][s] = fund, debt, power_hedge, power_withdrawal, cash_in, z
# levels = MaxNLocator(nbins=40).tick_values(zmin, zmax)
# cmap = plt.get_cmap('RdYlBu')
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# fig = plt.figure()
# for m in range(len(policy_ranks)):
#   # plot state space & policy
#   ax1 = fig.add_subplot(1, 3, m + 1)
#   name = 'm'+str(m)
#   for s in range(ns):
#     p1 = ax1.scatter(dict[name]['fund'][s] - dict[name]['debt'][s], dict[name]['power_hedge'][s], c=dict[name]['policy'][s], cmap=cmap, norm=norm, alpha=0.9)
#   if (m == len(policy_ranks) - 1):
#     fig.colorbar(p1, ax=ax1)
#   ax1.set_xlabel('Balance')
#   ax1.set_ylabel('Power')
#   ax1.set_title(policies[m]+' '+str(round(dict[name]['maxDebt'], 1)))











### plot withdrawal policies (2d)
policy_ranks = [9,498,2059]
policies = ['LowDebt','MedDebt','HighDebt']
power = [samp.powIndex.mean()] #[30, 42.5, 55] #
ny = 20
ns = 100
zmax = -100
zmin = 100
dict = {}
samples = np.arange(ns)*1000
for p in range(len(power)):
  for m in range(len(policy_ranks)):
    name = 'p'+str(p)+'_m'+str(m)
    dict[name] = {}
    # get policy params
    dps_choice = dps.iloc[policy_ranks[m],:]
    dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
    # get grid of values for policy heatmap
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[:3000].values,
                                                                samp.sswp.iloc[:3000].values,
                                                                samp.powIndex.iloc[:3000].values)
    maxFundDebt = fund_withdrawal.max() + 1
    minFundDebt = -(debt_withdrawal.max() + 1)
    dFundDebt = (maxFundDebt - minFundDebt) / 500
    maxCash = cash_in.max() + 1
    minCash = cash_in.min() - 1
    dCash = (maxCash - minCash) / 500
    xt, w = np.mgrid[slice(minFundDebt, maxFundDebt + dFundDebt, dFundDebt), slice(minCash, maxCash + dCash, dCash)]
    x = np.maximum(xt, 0)
    y = -np.minimum(xt, 0)
    # plot policy
    z = policyWithdrawal(x, y, power[p], w)
    z = z[:-1, :-1]
    zmax = max(z.max(), zmax)
    zmin = min(z.min(), zmin)
    # get trajectories through state space
    dict[name]['maxDebt'] = dps_choice['maxDebt']
    dict[name]['fund_withdrawal'], dict[name]['debt_withdrawal'], dict[name]['power_hedge'], dict[name]['power_withdrawal'], dict[name]['cash_in'] = {}, {}, {}, {}, {}
    for s in range(ns):
      fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
      dict[name]['fund_withdrawal'][s], dict[name]['debt_withdrawal'][s], dict[name]['power_hedge'][s], dict[name]['power_withdrawal'][s], dict[name]['cash_in'][s] = fund_withdrawal, debt_withdrawal, power_hedge, power_withdrawal, cash_in
levels = MaxNLocator(nbins=40).tick_values(zmin, zmax)
cmap = plt.get_cmap('RdYlBu')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig = plt.figure()
for p in range(len(power)):
  for m in range(len(policy_ranks)):
    dps_choice = dps.iloc[policy_ranks[m],:]
    dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
    ax1 = fig.add_subplot(len(power),3,p*3+m+1)
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[:3000].values, samp.sswp.iloc[:3000].values, samp.powIndex.iloc[:3000].values)
    maxFundDebt = fund_withdrawal.max()+1
    minFundDebt = -(debt_withdrawal.max()+1)
    dFundDebt = (maxFundDebt - minFundDebt)/500
    maxCash = cash_in.max()+1
    minCash = cash_in.min()-1
    dCash = (maxCash - minCash)/500
    xt, w = np.mgrid[slice(minFundDebt, maxFundDebt+dFundDebt, dFundDebt), slice(minCash, maxCash+dCash, dCash)]
    x = np.maximum(xt, 0)
    y = -np.minimum(xt, 0)
    # plot policy
    z = policyWithdrawal(x,y,power[p],w)
    z = z[:-1, :-1]
    im = ax1.pcolormesh(xt, w, z, cmap=cmap, norm=norm)
    if (m==len(policy_ranks)-1):
      fig.colorbar(im, ax=ax1)
    # plot state space
    name = 'p'+str(p)+'_m'+str(m)
    for s in range(ns):
      ax1.plot(dict[name]['fund_withdrawal'][s] - dict[name]['debt_withdrawal'][s], dict[name]['cash_in'][s], c='0.3', alpha=0.2)
    ax1.set_xlabel('Balance')
    ax1.set_ylabel('CashIn')
    ax1.set_title(policies[m] + ' ' + str(round(dict[name]['maxDebt'], 1)))







### plot withdrawal policies (scatter with color of policy)
policy_ranks = [9,498,2059]
policies = ['LowDebt','MedDebt','HighDebt']
power = [samp.powIndex.mean()] #[30, 42.5, 55] #
ny = 20
ns = 100
zmax = -100
zmin = 100
dict = {}
samples = np.arange(ns)*1000
for m in range(len(policy_ranks)):
  name = 'm'+str(m)
  dict[name] = {}
  # get policy params
  dps_choice = dps.iloc[policy_ranks[m],:]
  dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
  # get trajectories through state space
  dict[name]['maxDebt'] = dps_choice['maxDebt']
  dict[name]['fund'], dict[name]['debt'], dict[name]['power_hedge'], dict[name]['power_withdrawal'], dict[name]['cash_in'], dict[name]['policy'] = {}, {}, {}, {}, {}, {}
  for s in range(ns):
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
    zmax = max(withdrawal.max(), zmax)
    zmin = min(withdrawal.min(), zmin)
    dict[name]['fund'][s], dict[name]['debt'][s], dict[name]['power_hedge'][s], dict[name]['power_withdrawal'][s], dict[name]['cash_in'][s], dict[name]['policy'][s] = fund, debt, power_hedge, power_withdrawal, cash_in, z
levels = MaxNLocator(nbins=40).tick_values(zmin, zmax)
cmap = plt.get_cmap('RdYlBu')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig = plt.figure()
for m in range(len(policy_ranks)):
  # plot state space & policy
  ax1 = fig.add_subplot(1, 3, m + 1)
  name = 'm'+str(m)
  for s in range(ns):
    p1 = ax1.scatter(dict[name]['fund'][s] - dict[name]['debt'][s], dict[name]['cash_in'][s], c=dict[name]['policy'][s], cmap=cmap, norm=norm, alpha=0.9)
  if (m == len(policy_ranks) - 1):
    fig.colorbar(p1, ax=ax1)
  ax1.set_xlabel('Balance')
  ax1.set_ylabel('CashIn')
  ax1.set_title(policies[m]+' '+str(round(dict[name]['maxDebt'], 1)))








### output dataframe of simulation output for parallel coords in R
# policy_ranks = [303, 959, 43]
# for j, k in enumerate(policy_ranks):
#   for i in range(dps.shape[0]):
#     if ((dps.annRev.iloc[i] == annRev[k]) & (dps.maxDebt.iloc[i] == maxDebt[k]) & (dps.maxFund.iloc[i] == maxFund[k]) & (dps.maxComplex.iloc[i] == maxComplex[k])):
#       policy_ranks[j] = i
policy_ranks = [304, 967, 43]
# dps.iloc[policy_ranks,-5:]
policies = ['LowDebt','MedDebt','HighDebt']
ny = 20
ns = 20
zmax = -100
zmin = 100
dict = {}
samples = np.arange(ns)*1000
for m in range(len(policy_ranks)):
  name = 'm'+str(m)
  dict[name] = {}
  # get policy params
  dps_choice = dps.iloc[policy_ranks[m],:]
  dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
  # get trajectories through state space
  dict[name]['maxDebt'] = dps_choice['maxDebt']
  dict[name]['fund_hedge'] = np.array([])
  dict[name]['fund_withdrawal'] = np.array([])
  dict[name]['debt_hedge'] = np.array([])
  dict[name]['debt_withdrawal'] = np.array([])
  dict[name]['power_hedge'] = np.array([])
  dict[name]['power_withdrawal'] = np.array([])
  dict[name]['cash_in'] = np.array([])
  dict[name]['action_hedge'] = np.array([])
  dict[name]['action_withdrawal'] = np.array([])
  for s in range(ns):
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
    dict[name]['fund_hedge'] = np.append(dict[name]['fund_hedge'], fund_hedge)
    dict[name]['fund_withdrawal'] = np.append(dict[name]['fund_withdrawal'], fund_withdrawal)
    dict[name]['debt_hedge'] = np.append(dict[name]['debt_hedge'], debt_hedge)
    dict[name]['debt_withdrawal'] = np.append(dict[name]['debt_withdrawal'], debt_withdrawal)
    dict[name]['power_hedge'] = np.append(dict[name]['power_hedge'], power_hedge)
    dict[name]['power_withdrawal'] = np.append(dict[name]['power_withdrawal'], power_withdrawal)
    dict[name]['cash_in'] = np.append(dict[name]['cash_in'], cash_in)
    dict[name]['action_hedge'] = np.append(dict[name]['action_hedge'], action_hedge)
    dict[name]['action_withdrawal'] = np.append(dict[name]['action_withdrawal'], action_withdrawal)
df = pd.DataFrame({'fund_hedge':[], 'fund_withdrawal':[], 'debt_hedge':[], 'debt_withdrawal':[], 'power_hedge':[],
                   'power_withdrawal':[], 'cash_in':[], 'action_hedge':[], 'action_withdrawal':[], 'policy':[]})
for m in range(len(policy_ranks)):
  # plot state space & policy
  # ax1 = fig.add_subplot(1, 3, m + 1)
  name = 'm'+str(m)
  df = df.append(pd.DataFrame({'fund_hedge':dict[name]['fund_hedge'], 'fund_withdrawal':dict[name]['fund_withdrawal'], 'debt_hedge':dict[name]['debt_hedge'], 'debt_withdrawal':dict[name]['debt_withdrawal'],
                               'power_hedge':dict[name]['power_hedge'], 'power_withdrawal':dict[name]['power_withdrawal'], 'cash_in':dict[name]['cash_in'], 'action_hedge':dict[name]['action_hedge'],
                               'action_withdrawal':dict[name]['action_withdrawal'], 'policy':m}))
df.to_csv(dir_figs + 'policySim.csv', index=False)










def simulate_objectives(revenue, payout, power):
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
  #get objectives
  ann_adj_rev = discount_normalization * (np.sum(adj_rev * discount_factor) + fund[ny] * interest_fund * discount_factor[0] - debt[ny] * interest_debt * discount_factor[0])
  max_debt = np.max(debt)
  if (np.max(value_snow_contract) > EPS):
    max_complexity = 1
  else:
    max_complexity = 0
  max_fund = np.max(fund)

  return (ann_adj_rev, max_debt, max_complexity, max_fund)





### plot distributions of simulation objectives (e.g. annualized adj rev)
policy_ranks = [30,622,1420]
policies = ['LowDebt','MedDebt','HighDebt']
ny = 20
ns = 100000
dict = {}
samples = np.random.choice(range(1, dps.shape[0] - ny), size=ns)
for m in range(len(policy_ranks)):
  name = 'm'+str(m)
  dict[name] = {}
  # get policy params
  dps_choice = dps.iloc[policy_ranks[m],:]
  dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
  # get obejctives for each sim
  dict[name]['avgAnnRev_borg'] = dps_choice['annRev']
  dict[name]['q95MaxDebt_borg'] = dps_choice['maxDebt']
  dict[name]['avgMaxComplex_borg'] = dps_choice['maxComplex']
  dict[name]['avgMaxFund_borg'] = dps_choice['maxFund']
  dict[name]['annRev_sim'] = np.zeros(ns)
  dict[name]['maxDebt_sim'] = np.zeros(ns)
  dict[name]['maxComplex_sim'] = np.zeros(ns)
  dict[name]['maxFund_sim'] = np.zeros(ns)
  for s in range(ns):
    dict[name]['annRev_sim'][s], dict[name]['maxDebt_sim'][s], dict[name]['maxComplex_sim'][s], dict[name]['maxFund_sim'][s] = simulate_objectives(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)

fig = plt.figure()
ax = plt.subplot(221)
plt.hist(dict['m0']['annRev_sim'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 22.1, 22/40))
plt.hist(dict['m1']['annRev_sim'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 22.1, 22/40))
plt.hist(dict['m2']['annRev_sim'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 22.1, 22/40))
# plt.axvline(x=np.mean(dict['m0']['annRev_sim']), color=col[0], linewidth=2, linestyle='--')
# plt.axvline(x=np.mean(dict['m1']['annRev_sim']), color=col[2], linewidth=2, linestyle='--')
# plt.axvline(x=np.mean(dict['m2']['annRev_sim']), color=col[3], linewidth=2, linestyle='--')
plt.axvline(x=(dict['m0']['avgAnnRev_borg']), color=col[0], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m1']['avgAnnRev_borg']), color=col[2], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m2']['avgAnnRev_borg']), color=col[3], linewidth=2, linestyle=':')
ax.set_yticks([],[])
plt.xlabel('Annualized Revenue ($mm/yr)')
ax = plt.subplot(222)
plt.hist(dict['m0']['maxDebt_sim'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 36.1, 36/40))
plt.hist(dict['m1']['maxDebt_sim'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 36.1, 36/40))
plt.hist(dict['m2']['maxDebt_sim'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 36.1, 36/40))
# plt.axvline(x=np.quantile(dict['m0']['maxDebt_sim'], 0.95), color=col[0], linewidth=2, linestyle='--')
# plt.axvline(x=np.quantile(dict['m1']['maxDebt_sim'], 0.95), color=col[2], linewidth=2, linestyle='--')
# plt.axvline(x=np.quantile(dict['m2']['maxDebt_sim'], 0.95), color=col[3], linewidth=2, linestyle='--')
plt.axvline(x=(dict['m0']['q95MaxDebt_borg']), color=col[0], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m1']['q95MaxDebt_borg']), color=col[2], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m2']['q95MaxDebt_borg']), color=col[3], linewidth=2, linestyle=':')
plt.legend(['Low debt','Medium debt','High debt'])
ax.set_yticks([],[])
plt.xlabel('Max Debt ($mm/yr)')
ax = plt.subplot(223)
plt.hist(dict['m0']['maxComplex_sim'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 1.001, 1/40))
plt.hist(dict['m1']['maxComplex_sim'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 1.001, 1/40))
plt.hist(dict['m2']['maxComplex_sim'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 1.001, 1/40))
plt.axvline(x=(dict['m0']['avgMaxComplex_borg']), color=col[0], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m1']['avgMaxComplex_borg']), color=col[2], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m2']['avgMaxComplex_borg']), color=col[3], linewidth=2, linestyle=':')
ax.set_yticks([],[])
plt.xlabel('Max Complexity')
ax = plt.subplot(224)
plt.hist(dict['m0']['maxFund_sim'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 16.1, 16/40))
plt.hist(dict['m1']['maxFund_sim'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 16.1, 16/40))
plt.hist(dict['m2']['maxFund_sim'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 16.1, 16/40))
plt.axvline(x=(dict['m0']['avgMaxFund_borg']), color=col[0], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m1']['avgMaxFund_borg']), color=col[2], linewidth=2, linestyle=':')
plt.axvline(x=(dict['m2']['avgMaxFund_borg']), color=col[3], linewidth=2, linestyle=':')
ax.set_yticks([],[])
plt.xlabel('Max Fund ($mm)')











### plot distribution of unaggregated objectives (e.g. adjusted revenues)
policy_ranks = [30,622,1420]
policies = ['LowDebt','MedDebt','HighDebt']
ny = 20
ns = 1000
zmax = -100
zmin = 100
dict = {}
samples = np.arange(ns)*1000
for m in range(len(policy_ranks)):
  name = 'm'+str(m)
  dict[name] = {}
  # get policy params
  dps_choice = dps.iloc[policy_ranks[m],:]
  dv_d, dv_c, dv_b, dv_w, dv_a = getDV(dps_choice)
  # get trajectories through state space
  dict[name]['fund_hedge'] = np.array([])
  dict[name]['fund_withdrawal'] = np.array([])
  dict[name]['debt_hedge'] = np.array([])
  dict[name]['debt_withdrawal'] = np.array([])
  dict[name]['power_hedge'] = np.array([])
  dict[name]['power_withdrawal'] = np.array([])
  dict[name]['cash_in'] = np.array([])
  dict[name]['action_hedge'] = np.array([])
  dict[name]['action_withdrawal'] = np.array([])
  dict[name]['adj_rev'] = np.array([])
  for s in range(ns):
    fund_hedge, fund_withdrawal, debt_hedge, debt_withdrawal, power_hedge, power_withdrawal, cash_in, action_hedge, action_withdrawal, adj_rev = simulate(samp.revRetail.iloc[samples[s]:(samples[s]+ny+1)].values, samp.sswp.iloc[samples[s]:(samples[s]+ny+1)].values, samp.powIndex.iloc[samples[s]:(samples[s]+ny+1)].values)
    dict[name]['fund_hedge'] = np.append(dict[name]['fund_hedge'], fund_hedge)
    dict[name]['fund_withdrawal'] = np.append(dict[name]['fund_withdrawal'], fund_withdrawal)
    dict[name]['debt_hedge'] = np.append(dict[name]['debt_hedge'], debt_hedge)
    dict[name]['debt_withdrawal'] = np.append(dict[name]['debt_withdrawal'], debt_withdrawal)
    dict[name]['power_hedge'] = np.append(dict[name]['power_hedge'], power_hedge)
    dict[name]['power_withdrawal'] = np.append(dict[name]['power_withdrawal'], power_withdrawal)
    dict[name]['cash_in'] = np.append(dict[name]['cash_in'], cash_in)
    dict[name]['action_hedge'] = np.append(dict[name]['action_hedge'], action_hedge)
    dict[name]['action_withdrawal'] = np.append(dict[name]['action_withdrawal'], action_withdrawal)
    dict[name]['adj_rev'] = np.append(dict[name]['adj_rev'], adj_rev)


fig = plt.figure()
ax = plt.subplot2grid((3,2),(0,0),colspan=2)
plt.hist(dict['m0']['adj_rev'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 52.1, 52/40))
plt.hist(dict['m1']['adj_rev'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 52.1, 52/40))
plt.hist(dict['m2']['adj_rev'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 52.1, 52/40))
ax.set_yticks([],[])
plt.legend(['Low debt','Medium debt','High debt'])
plt.xlabel('Adjusted Revenue ($mm)')
ax = plt.subplot2grid((3,2),(1,0),rowspan=1)
plt.hist(dict['m0']['fund_hedge'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 16.1, 16/40))
plt.hist(dict['m1']['fund_hedge'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 16.1, 16/40))
plt.hist(dict['m2']['fund_hedge'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 16.1, 16/40))
ax.set_yticks([],[])
plt.xlabel('Fund balance ($mm)')
ax = plt.subplot2grid((3,2),(1,1),rowspan=1)
plt.hist(dict['m0']['debt_hedge'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 32.1, 32/40))
plt.hist(dict['m1']['debt_hedge'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 32.1, 32/40))
plt.hist(dict['m2']['debt_hedge'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 32.1, 32/40))
ax.set_yticks([],[])
plt.xlabel('Debt balance ($mm)')
ax = plt.subplot2grid((3,2),(2,0),rowspan=1)
plt.hist(dict['m0']['action_hedge'], density=True, alpha=0.6, color=col[0], bins=np.arange(0, 1.101, 1.1/40))
plt.hist(dict['m1']['action_hedge'], density=True, alpha=0.6, color=col[2], bins=np.arange(0, 1.101, 1.1/40))
plt.hist(dict['m2']['action_hedge'], density=True, alpha=0.6, color=col[3], bins=np.arange(0, 1.101, 1.1/40))
ax.set_yticks([],[])
plt.xlabel('Action: hedge ($mm/inch)')
ax = plt.subplot2grid((3,2),(2,1),rowspan=1)
plt.hist(dict['m0']['action_withdrawal'], density=True, alpha=0.6, color=col[0], bins=np.arange(-20,20, 40/40))
plt.hist(dict['m1']['action_withdrawal'], density=True, alpha=0.6, color=col[2], bins=np.arange(-20,20, 40/40))
plt.hist(dict['m2']['action_withdrawal'], density=True, alpha=0.6, color=col[3], bins=np.arange(-20,20, 40/40))
ax.set_yticks([],[])
plt.xlabel('Action: Withdrawal/deposit ($mm)')











### plot policies meeting brushing constraints
ref_full_4obj_retest, ndv_ref_full_4obj_retest = getSet(dir_data + 'DPS_4input_3/DPS_4input_3_retest.resultfile', 4)
fixed_cost = 0.914
MEAN_REVENUE = 128.48255822159567
mean_net_revenue = MEAN_REVENUE * (1 - fixed_cost)
min_annRev = mean_net_revenue * 0.95
max_maxDebt = mean_net_revenue * 1.5
max_maxFund = mean_net_revenue * 3
brush_annRev = ref_full_4obj_retest.annRev >= min_annRev
brush_maxDebt = ref_full_4obj_retest.maxDebt <= max_maxDebt
brush_maxFund = ref_full_4obj_retest.maxFund <= max_maxFund
ref_full_4obj_retest_yes = ref_full_4obj_retest.loc[(brush_annRev & brush_maxDebt & brush_maxFund),:]
ref_full_4obj_retest_no = ref_full_4obj_retest.loc[~(brush_annRev & brush_maxDebt & brush_maxFund),:]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
# zs = ref_full_4obj_retest.annRev
# ys = ref_full_4obj_retest.maxDebt
# xs = ref_full_4obj_retest.maxComplex
# cs = ref_full_4obj_retest.maxFund
# p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis_r')
zs = ref_full_4obj_retest_no.annRev
ys = ref_full_4obj_retest_no.maxDebt
xs = ref_full_4obj_retest_no.maxComplex
cs = ref_full_4obj_retest_no.maxFund
p1 = ax.scatter(xs, ys, zs, c='0.7')
zs = ref_full_4obj_retest_yes.annRev
ys = ref_full_4obj_retest_yes.maxDebt
xs = ref_full_4obj_retest_yes.maxComplex
cs = ref_full_4obj_retest_yes.maxFund
p1 = ax.scatter(xs, ys, zs, c=cs, cmap='viridis_r', alpha=1)
cb = fig.colorbar(p1, ax=ax)
cb.set_label('MaxFund')
ax.set_xlabel('MaxComplexity')
ax.set_ylabel('MaxDebt')
ax.set_zlabel('AnnRev')















