##############################################################################################################
### functions_moea_output_plots.py - python functions used in analyzing and plotting outputs from multi-objective optimization
###     multi-objective optimization
### Project started May 2017, last update Jan 2020
##############################################################################################################
import numpy as np
import pandas as pd
# import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator
import seaborn as sns
# import matplotlib as mpl
from matplotlib import ticker, cm, colors
# from mpl_toolkits.mplot3d import Axes3D
import copy
# import itertools

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.55)

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







################################################################

#fn for cleaning runtime metrics
def get_metrics(metricfile, hvfile):
  # read data
  df = pd.read_csv(metricfile, sep=' ')
  names = list(df.columns)
  names[0] = names[0].replace('#','')
  df.columns = names

  hv = pd.read_csv(hvfile, sep=' ', header=None)
  df['Hypervolume'] /= hv.iloc[0,0]
  return df






#########################################################################
###### plot nfe vs hv for num rbfs
### outputs plot, no return. ####
# ##########################################################################
def plot_metrics(dir_figs, metrics, nrbfs, nseed, fe_grid):
  ### plot moea convergence metrics for different number RBFs
  fig = plt.figure(figsize=(6,8))
  gs1 = fig.add_gridspec(nrows=3, ncols=1, left=0, right=1, wspace=0.1, hspace=0.1)

  ### hypervolume
  ax = fig.add_subplot(gs1[0,0])
  ax.annotate('a)', xy=(0.01, 0.89), xycoords='axes fraction')
  # col = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
  col = ['0.8', '#d73027', '0.65', '0.5','0.35',  '0.2']
  # cmap_vir = cm.get_cmap('viridis')
  # col = [cmap_vir(0.1),cmap_vir(0.25),cmap_vir(0.4),cmap_vir(0.55),cmap_vir(0.7),cmap_vir(0.85)]

  for c, d in enumerate(nrbfs):
    for s in range(nseed):
      hv = metrics[str(d) + 'rbf'][s]['Hypervolume'].values
      hv = np.insert(hv, 0, 0)
      if s < nseed-1:
        ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
      else:
        if c==0:
          l0, =  ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
        elif c==1:
          l1, =  ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
        elif c==2:
          l2, =  ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
        elif c==3:
          l3, =  ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
        elif c==4:
          l4, =  ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
        elif c==5:
          l5, =  ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
  # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  ax.set_ylim([0.8,1])
  ax.set_xticklabels([])
  # ax.set_xlabel('Function Evaluations')
  ax.set_ylabel('Hypervolume\n' + r'$\rightarrow$')
  # plt.savefig(dir_figs + 'compareRbfs_hv.eps', bbox_inches='tight', dpi=500)

  # ### zoomed in version
  # ax = fig.add_subplot(gs1[0,1])
  # ax.annotate('b)', xy=(0.01, 0.89), xycoords='axes fraction')
  # for c, d in enumerate(nrbfs):
  # # for c, d in enumerate((1,2,4)):
  #   for s in range(nseed):
  #     hv = metrics[str(d) + 'rbf'][s]['Hypervolume'].values
  #     hv = np.insert(hv, 0, 0)
  #     ax.plot(fe_grid/1000, hv, c=col[c], alpha=0.5)
  # # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  # ax.set_xlim([100,150])
  # ax.set_ylim([0.95,0.992])
  # ax.set_xlabel('Function Evaluations')
  # ax.set_ylabel('Hypervolume')
  # # plt.savefig(dir_figs + 'compareRbfs_hv_zoom.eps', bbox_inches='tight', dpi=500)

  ### Additive Epsilon Indicator
  ax = fig.add_subplot(gs1[1,0])
  ax.annotate('b)', xy=(0.01, 0.89), xycoords='axes fraction')
  for c, d in enumerate(nrbfs):
    for s in range(nseed):
      hv = metrics[str(d) + 'rbf'][s]['EpsilonIndicator'].values
      hv = np.insert(hv, 0, 0)
      ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
  # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  ax.set_ylim([0,0.3])
  ax.set_xticklabels([])
  # ax.set_xlabel('Function Evaluations')
  ax.set_ylabel('Epsilon\nIndicator\n' + r'$\leftarrow$')
  # plt.savefig(dir_figs + 'compareRbfs_hv.eps', bbox_inches='tight', dpi=500)

  # ### zoomed in version
  # ax = fig.add_subplot(gs1[1,1])
  # ax.annotate('d)', xy=(0.01, 0.89), xycoords='axes fraction')
  # for c, d in enumerate(nrbfs):
  # # for c, d in enumerate((1,2,4)):
  #   for s in range(nseed):
  #     hv = metrics[str(d) + 'rbf'][s]['EpsilonIndicator'].values
  #     hv = np.insert(hv, 0, 0)
  #     ax.plot(fe_grid/1000, hv, c=col[c], alpha=0.5)
  # # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  # ax.set_xlim([100,150])
  # ax.set_ylim([0.05,0.12])
  # ax.set_xlabel('Function Evaluations')
  # ax.set_ylabel('Epsilon Indicator')

  ### Generational Distance
  ax = fig.add_subplot(gs1[2,0])
  ax.annotate('c)', xy=(0.01, 0.89), xycoords='axes fraction')
  for c, d in enumerate(nrbfs):
    for s in range(nseed):
      hv = metrics[str(d) + 'rbf'][s]['GenerationalDistance'].values
      hv = np.insert(hv, 0, 0)
      ax.plot(fe_grid/1000, hv, c=col[c], alpha=1, zorder=9 - abs(d - 2))
  # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  ax.set_ylim([0,0.007])
  ax.set_xlabel('Function Evaluations')
  ax.set_ylabel('Generational\nDistance\n' + r'$\leftarrow$')
  # plt.savefig(dir_figs + 'compareRbfs_hv.eps', bbox_inches='tight', dpi=500)

  # ### zoomed in version
  # ax = fig.add_subplot(gs1[2,1])
  # ax.annotate('f)', xy=(0.01, 0.89), xycoords='axes fraction')
  # for c, d in enumerate(nrbfs):
  # # for c, d in enumerate((1,2,4)):
  #   for s in range(nseed):
  #     hv = metrics[str(d) + 'rbf'][s]['GenerationalDistance'].values
  #     hv = np.insert(hv, 0, 0)
  #     ax.plot(fe_grid/1000, hv, c=col[c], alpha=0.5)
  # # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  # ax.set_xlim([100,150])
  # # ax.set_ylim([0.95,0.992])
  # ax.set_xlabel('Function Evaluations')
  # ax.set_ylabel('Generational Distance')


  ax.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'], title='Number of RBFs', ncol=6, bbox_to_anchor=(1.02,-0.35), title_fontsize=14)
  plt.savefig(dir_figs + 'compareRbfs.jpg', bbox_inches='tight', dpi=500)





  # fig = plt.figure()
  # # ax = fig.add_subplot(321)
  # # nrbfs = [12,2]
  # for c, d in enumerate(nrbfs):
  #   for s in range(nseed):
  #     hv = metrics[str(d) + 'rbf'][s]['EpsilonIndicator'].values
  #     hv = np.insert(hv, 0, 0)
  #     if s < nseed-1:
  #       plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #     else:
  #       if c==0:
  #         l0, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==1:
  #         l1, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==2:
  #         l2, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==3:
  #         l3, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==4:
  #         l4, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==5:
  #         l5, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  # # plt.ylim([0,1])
  # plt.savefig(dir_figs + 'compareRbfs_ei.jpg', bbox_inches='tight', dpi=500)

  # fig = plt.figure()
  # # ax = fig.add_subplot(321)
  # for c, d in enumerate(nrbfs):
  #   for s in range(nseed):
  #     hv = metrics[str(d) + 'rbf'][s]['GenerationalDistance'].values
  #     hv = np.insert(hv, 0, 0)
  #     if s < nseed-1:
  #       plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #     else:
  #       if c==0:
  #         l0, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==1:
  #         l1, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==2:
  #         l2, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==3:
  #         l3, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==4:
  #         l4, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  #       elif c==5:
  #         l5, =  plt.plot(fe_grid/1000, hv, c=col[c], alpha=0.7)
  # plt.legend([l0, l1, l2, l3, l4, l5], ['1','2','3','4','8','12'])
  # # plt.ylim([0,1])
  # plt.savefig(dir_figs + 'compareRbfs_gd.jpg', bbox_inches='tight', dpi=500)

  return







#fn for cleaning set data
def get_set(file, nobj, ncon, has_dv = True, has_constraint = True, sort = False):
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




### fn to get limits for 3d plots, accounting for automatic padding
def get_plot_limits(df_dps, df_2dv):
  # lims = {'annRev':[1e6, -1e6],'maxDebt':[1e6, -1e6],'maxComplex':[1e6, -1e6],'maxFund':[1e6, -1e6]}
  # for c,d in enumerate([df_dps, df_2dv]):
  #   lims['annRev'][0] = min(lims['annRev'][0], d.annRev.min())
  #   lims['maxDebt'][0] = min(lims['maxDebt'][0], d.maxDebt.min())
  #   lims['maxComplex'][0] = min(lims['maxComplex'][0], d.maxComplex.min())
  #   lims['maxFund'][0] = min(lims['maxFund'][0], d.maxFund.min())
  #   lims['annRev'][1] = max(lims['annRev'][1], d.annRev.max())
  #   lims['maxDebt'][1] = max(lims['maxDebt'][1], d.maxDebt.max())
  #   lims['maxComplex'][1] = max(lims['maxComplex'][1], d.maxComplex.max())
  #   lims['maxFund'][1] = max(lims['maxFund'][1], d.maxFund.max())
  lims = {'annRev':[9.4,11.13],'maxDebt':[0,40],'maxComplex':[0,1],'maxFund':[0,125]}
  #Find the amount of padding for 3d plot
  padding = {'annRev':(lims['annRev'][1] - lims['annRev'][0])/50, 'maxDebt':(lims['maxDebt'][1] - lims['maxDebt'][0])/50,
            'maxComplex':(lims['maxComplex'][1] - lims['maxComplex'][0])/50, 'maxFund':(lims['maxFund'][1] - lims['maxFund'][0])/50}
  lims3d = copy.deepcopy(lims)
  lims3d['annRev'][0] += padding['annRev']
  lims3d['annRev'][1] -= padding['annRev']
  lims3d['maxDebt'][0] += padding['maxDebt']
  lims3d['maxDebt'][1] -= padding['maxDebt']
  lims3d['maxComplex'][0] += padding['maxComplex']
  lims3d['maxComplex'][1] -= padding['maxComplex']

  return lims, lims3d


### fn to 3d-plot DPS vs 2dv reference sets (4 objectives)
def plot_formulations_4obj(df_dps, df_2dv, lims3d, dir_figs):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection='3d')
  min_dist = [np.where(df_dps.totalDistance4obj == df_dps.totalDistance4obj.min())[0][0],
              np.where(df_2dv.totalDistance4obj == df_2dv.totalDistance4obj.min())[0][0]]
  z_min_dist = [df_dps.annRev.iloc[min_dist[0]], df_2dv.annRev.iloc[min_dist[1]]]
  y_min_dist = [df_dps.maxDebt.iloc[min_dist[0]], df_2dv.maxDebt.iloc[min_dist[1]]]
  x_min_dist = [df_dps.maxComplex.iloc[min_dist[0]], df_2dv.maxComplex.iloc[min_dist[1]]]
  s_min_dist = [df_dps.maxFund.iloc[min_dist[0]], df_2dv.maxFund.iloc[min_dist[1]]]
  zs = df_dps.annRev.drop(min_dist[0])
  ys = df_dps.maxDebt.drop(min_dist[0])
  xs = df_dps.maxComplex.drop(min_dist[0])
  ss = 20 + 1.3*df_dps.maxFund.drop(min_dist[0])
  p1 = ax.scatter(xs, ys, zs, s=ss, marker='v', alpha=0.6, c=col_blues[2],zorder=2)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
  p1 = ax.scatter(x_min_dist[0], y_min_dist[0], z_min_dist[0], s=s_min_dist[0], marker='v', alpha=1, c=col_blues[0],zorder=3)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
  # zs = df_2dv.annRev.drop(min_dist[1])
  # ys = df_2dv.maxDebt.drop(min_dist[1])
  # xs = df_2dv.maxComplex.drop(min_dist[1])
  # ss = 20 + 1.3*df_2dv.maxFund.drop(min_dist[1])
  # p1 = ax.scatter(xs, ys, zs, s=ss, marker='^',alpha=0.6, c=col_reds[2],zorder=3)#, c=cs, cmap=cmap_reds, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
  # p1 = ax.scatter(x_min_dist[1], y_min_dist[1], z_min_dist[1], s=s_min_dist[1], marker='^', alpha=1, c=col_reds[0], zorder=0)#, c=cs, cmap=cmap_blues, vmin=vmin, vmax=vmax) #, norm=colors.LogNorm() , )
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  ax.set_xticks([0,0.25,0.5,0.75])
  ax.set_yticks([10,20,30,40])
  ax.set_zticks([9.5,10,10.5,11])
  ax.view_init(elev=20, azim =-45)
  ax.plot([0.01],[0.01],[11.09],marker='*',ms=25,c='k')
  plt.savefig(dir_figs + 'compare2dvDps_4obj.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)

  return



### fn to 3d-plot min/max marker size for use in legend (combined in illustrator)
def plot_marker_size_4obj(df_dps, df_2dv, lims3d, dir_figs):
  fig = plt.figure()
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection='3d')
  min_s = 20 + 1.3*min(df_dps.maxFund.min(),df_2dv.maxFund.max())
  max_s = 20 + 1.3*max(df_dps.maxFund.max(),df_2dv.maxFund.max())
  zs = df_dps.annRev
  ys = df_dps.maxDebt
  xs = df_dps.maxComplex
  ss = 20 + 1.3*df_dps.maxFund
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
  plt.savefig(dir_figs + 'compare2dvDps_4obj_marker.eps', bbox_inches='tight', figsize=(4.5,8), dpi=500)

  return





def plot_subproblems(df_dps, lims3d, dir_moea_output, dir_figs):
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


  ### plot 4obj with subproblem bests
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

  return




