######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker, cm, colors
import seaborn as sns
import copy
import itertools
from datetime import datetime

### Project functions ###
import functions_moea_output_plots

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.55)

startTime = datetime.now()

EPS = 1e-10
ny = 20
ns = 500
nbins_entropy = 50


dir_data = '../../data/'
dir_figs = '../../figures/'

cmap_vir_mi = cm.get_cmap('viridis')
cmap_vir_mi.set_under('grey')

#########################################################################



def get_probability_grid(xdat, ydat, xdomain, ydomain):
  xmin = xdomain[0]
  xmax = xdomain[1]
  ymin = ydomain[0]
  ymax = ydomain[1]
  yspan = ymax - ymin
  xspan = xmax - xmin

  nstep = 10
  dy = yspan / nstep
  dx = xspan / nstep
  freqMat = np.zeros([nstep, nstep])
  for i in range(len(ydat)):
    row = int(np.floor((ydat[i] - ymin) / dy))
    col = int(np.floor((xdat[i] - xmin) / dx))
    freqMat[row, col] += 1
  freqMat /= len(ydat)
  freqMat = np.log10(freqMat)
  freqMat[freqMat == -np.inf] = np.nan
  return freqMat



def sort_bins(d, n_bins, try_sep_bins):
  separate_min = False
  separate_max = False
  if (try_sep_bins == True):
    if (np.mean(np.abs(d - d.min()) < EPS) > 0.05):
      separate_min = True
    if (np.mean(np.abs(d - d.max()) < EPS) > 0.05):
      separate_max = True
  if (separate_min):
    d_min_val = d.min()
    d_min_n = np.sum(np.abs(d - d_min_val) < EPS)
    d_interior = d[np.abs(d - d_min_val) > EPS]
    n_bins -= 1
  if (separate_max):
    d_max_val = d.max()
    d_max_n = np.sum(np.abs(d - d_max_val) < EPS)
    if (separate_min):
      d_interior = d_interior[np.abs(d_interior - d_max_val) > EPS]
    else:
      d_interior = d[np.abs(d - d_max_val) > EPS]
    n_bins -= 1
  if (separate_min | separate_max):
    (d_bins, bins, dum) = plt.hist(d_interior, bins=n_bins)
  else:
    (d_bins, bins, dum) = plt.hist(d, bins=n_bins)
  if (separate_min):
    bins = np.insert(bins, 0, d_min_val + EPS)
    bins = np.insert(bins, 0, d_min_val - EPS)
    d_bins = np.insert(d_bins, 0, 0)
    d_bins = np.insert(d_bins, 0, d_min_n)
    n_bins += 2
  else:
    bins[0] -= EPS
  if (separate_max):
    bins = np.append(bins, d_max_val - EPS)
    bins = np.append(bins, d_max_val + EPS)
    d_bins = np.append(d_bins, 0)
    d_bins = np.append(d_bins, d_max_n)
    n_bins += 2
  else:
    bins[-1] += EPS
  bincenter = (bins[:-1] + bins[1:]) / 2
  while (np.sum(d_bins < EPS) > 0):
    ind = np.where(d_bins < EPS)[0]
    d_bins = np.delete(d_bins, ind)
    bins = np.delete(bins, ind + 1)
    bincenter = np.delete(bincenter, ind)
    n_bins -= 1
  binpoint = np.digitize(d, bins=bins) - 1
  binfreq = d_bins / len(d)
  return binfreq, bincenter, binpoint





def get_joint_probability(dat, name, atts):
  nbintot = ns * ny
  ndim = len(atts)
  nbin = []
  d = []
  p = []
  for i in range(ndim):
    d.append(dat[name][atts[i] + '_binpoint'])
    p.append(np.unique(dat[name][atts[i] + '_binpoint']))
    nbin.append(len(p[i]))
  joint_prob = np.histogramdd(np.transpose(d), bins=nbin)[0] / nbintot
  dat[name]['joint_freq'] = joint_prob
  return(dat)



@np.vectorize
def prob_log_prob(numerator_prob, denominator_prob = -1):
  if numerator_prob < EPS:
    z = 0
  else:
    if (denominator_prob < EPS):
      z = numerator_prob * np.log2(numerator_prob)
    else:
      z = numerator_prob * (np.log2(1 / denominator_prob) - np.log2(1 / numerator_prob))
  return z


def get_entropy(probs):
  entropy = - np.sum(prob_log_prob(probs))
  return entropy


def get_mutual_info(dat, name, atts_full, atts_mi):
  mutual_info = 0
  probs = dat[name]['joint_freq']
  sets_mi = {}
  for i in range(len(atts_mi)):
    sets_mi[i] = {}
    sets_mi[i]['n_att'] = len(atts_mi[i])
    sets_mi[i]['n_grid'] = {}
    sets_mi[i]['idx_base'] = {}
    for att in atts_mi[i]:
      sets_mi[i]['n_grid'][att] = len(dat[name][att + '_binfreq'])
      sets_mi[i]['idx_base'][att] = [idx for idx, x in enumerate(list(sets_mi[0]['n_grid'].keys())) if x == att]
  for j in range(len(sets_mi)):
    probs_temp = probs.copy()
    for i in range(len(atts_full)-1, -1, -1):
      if ((atts_full[i] in atts_mi[j]) == False):
        probs_temp = probs_temp.sum(i)
    sets_mi[j]['probs_integrated'] = probs_temp
  if (len(sets_mi[0]['n_grid'].keys()) == 2):
    # two-pt mutual info (e.g. I(X1;Y))
    try:
      for i in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[0]]):
        for j in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[1]]):
            numerator_prob = sets_mi[0]['probs_integrated'][i,j]
            denominator_prob = sets_mi[1]['probs_integrated'][i] * sets_mi[2]['probs_integrated'][j]
            mutual_info += prob_log_prob(numerator_prob, denominator_prob)
    except:
      ### if above fails, it is because variable only takes 1 value and MI undefined (e.g., reserve always zero). Let's just say it's zero.
      mutual_info = np.nan

  ### Note: only set up for 2-part MI at present, haven't validated that code below works
  # elif (len(sets_mi[0]['n_grid'].keys()) == 4):
  #   if (len(sets_mi.keys())-1 == 4):
  #     # 4-pt interaction (e.g. I(X1;X2;X3;Y))
  #     for i in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[0]]):
  #       for j in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[1]]):
  #         for k in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[2]]):
  #           for l in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[3]]):
  #               numerator_prob = sets_mi[0]['probs_integrated'][i, j, k, l]
  #               denominator_prob = sets_mi[1]['probs_integrated'][i] * sets_mi[2]['probs_integrated'][j] * sets_mi[3]['probs_integrated'][k] * sets_mi[4]['probs_integrated'][l]
  #               mutual_info += prob_log_prob(numerator_prob, denominator_prob)
  #   else:
  #     # 3+1 mutual interaction (e.g. I(X1,X2,X3;Y))
  #     for i in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[0]]):
  #       for j in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[1]]):
  #         for k in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[2]]):
  #           for l in range(sets_mi[0]['n_grid'][list(sets_mi[0]['n_grid'].keys())[3]]):
  #               numerator_prob = sets_mi[0]['probs_integrated'][i, j, k, l]
  #               denominator_prob = sets_mi[1]['probs_integrated'][i, j, k] * sets_mi[2]['probs_integrated'][l]
  #               mutual_info += prob_log_prob(numerator_prob, denominator_prob)
  return mutual_info







### plot mutual information for policies in 4-objective space
def plot_MI(mi, mi_name, example_pol, name, lims, lims3d, mean_net_revenue, fig_format):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection='3d')
  zs = mi.annRev
  ys = mi.maxDebt
  xs = mi.maxComplex
  ss = [20 + 1.3 * x for x in mi.maxFund]
  cs = mi[mi_name]
  p1 = ax.scatter(xs, ys, zs, s=ss, c=cs, cmap=cmap_vir_mi, vmin=lims['mi']['min'], vmax=lims['mi']['max'], marker='v', alpha=0.4)
  if len(example_pol) > 0:
    zs = [mi.annRev[x] for x in example_pol]
    ys = [mi.maxDebt[x] for x in example_pol]
    xs = [mi.maxComplex[x] for x in example_pol]
    ss = [20 + 1.3 * x for x in [mi.maxFund[x] for x in example_pol]]
    cs = [mi[mi_name][x] for x in example_pol]
    p1 = ax.scatter(xs[0], ys[0], zs[0], s=ss, c=cs, cmap=cmap_vir_mi, vmin=lims['mi']['min'], vmax=lims['mi']['max'], marker='v', edgecolors='k', lw=1.5)
  ax.set_xticks([0,0.25,0.5,0.75])
  ax.set_yticks([12, 24, 36])
  ax.set_zticks([9.5,10,10.5,11])
  ax.view_init(elev=20, azim =-45)
  ax.plot([0.01],[0.01],[mean_net_revenue+0.05],marker='*',ms=15,c='k')
  ax.set_xlim(lims3d['maxComplex'])
  ax.set_ylim(lims3d['maxDebt'])
  ax.set_zlim(lims3d['annRev'])
  plt.savefig(dir_figs + name + '.' + fig_format, bbox_inches='tight', figsize=(4.5,8), dpi=500)

  return



### get dataframe of simulation results, output for parallel coords in R
def add_mi_table2(mi, policy_ranks):
  ### table 2 output
  table2 = pd.read_csv('../../figures/table2.csv', index_col=0)
  mi_2obj = pd.read_csv(dir_data + 'policy_simulation/2obj/mi_combined.csv', index_col=0).sort_index()

  table2['fund_hedge_mi'] = np.nan
  table2['debt_hedge_mi'] = np.nan
  table2['power_hedge_mi'] = np.nan

  for i in range(mi_2obj.shape[0]):
    if (abs(mi_2obj.annRev[i] - table2.annRev[1]) < 1e-6) and (abs(mi_2obj.maxDebt[i] - table2.maxDebt[1]) < 1e-6) and (abs(mi_2obj.maxComplex[i] - table2.maxComplex[1]) < 1e-6) and (abs(mi_2obj.maxFund[i] - table2.maxFund[1]) < 1e-6):
      table2['fund_hedge_mi'][1] = mi_2obj.fund_hedge_mi[i]
      table2['debt_hedge_mi'][1] = mi_2obj.debt_hedge_mi[i]
      table2['power_hedge_mi'][1] = mi_2obj.power_hedge_mi[i]
  for i in range(mi.shape[0]):
    if (abs(mi.annRev[i] - table2.annRev[2]) < 1e-6) and (abs(mi.maxDebt[i] - table2.maxDebt[2]) < 1e-6) and (abs(mi.maxComplex[i] - table2.maxComplex[2]) < 1e-6) and (abs(mi.maxFund[i] - table2.maxFund[2]) < 1e-6):
      table2['fund_hedge_mi'][2] = mi.fund_hedge_mi[i]
      table2['debt_hedge_mi'][2] = mi.debt_hedge_mi[i]
      table2['power_hedge_mi'][2] = mi.power_hedge_mi[i]

  for i in policy_ranks:
    table2 = table2.append(mi[['annRev','maxDebt','maxComplex','maxFund','fund_hedge_mi','debt_hedge_mi','power_hedge_mi']].iloc[i, :])
  table2.reset_index(inplace=True, drop=True)
  table2.formulation.iloc[-3:] = '4obj_dynamic_MI'
  table2.to_csv('../../figures/table2_mi.csv')

  return




### get dataframe of simulation results, output for parallel coords in R
def get_parallel_coord_data(samp, dps, policy_ranks, ny, ns, fig_format):
  np.random.seed(203)

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
    dv_d, dv_c, dv_b, dv_w, dv_a = functions_moea_output_plots.get_dvs(dps_choice)
  
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

    df = df.append(pd.DataFrame({'fund_hedge':results[:, 0], 'fund_withdrawal':results[:, 1], 'debt_hedge':results[:, 2], 'debt_withdrawal':results[:, 3],
                                'power_hedge':results[:, 4], 'power_withdrawal':results[:, 5], 'cash_in':results[:, 6], 'action_hedge':results[:, 7],
                                'action_withdrawal':results[:, 8], 'policy':m}))
  df.to_csv(dir_data + 'policy_simulation/4obj/mi_examples_simulation.csv', index=False)

  return
