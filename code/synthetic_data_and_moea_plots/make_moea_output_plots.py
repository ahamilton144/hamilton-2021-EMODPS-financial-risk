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
from matplotlib.lines import Line2D
import seaborn as sns
import importlib
from datetime import datetime
import copy
import itertools

### Project functions ###
import functions_moea_output_plots


sns.set_style('ticks')
sns.set_context('paper', font_scale=1.55)

startTime = datetime.now()

dir_generated_inputs = './../../data/generated_inputs/'
dir_moea_output = './../../data/optimization_output/'
dir_figs = './../../figures/'
fig_format = 'jpg'

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

### get runtime metrics for moea runs with different number of RBFs (nrbfs)
# print('Getting runtime metrics for moea runs with different number of RBFs..., ',
#       datetime.now() - startTime)
importlib.reload(functions_moea_output_plots)
metrics = {}
nrbfs = (1, 2, 3, 4, 8, 12)
for nrbf in nrbfs:
    metrics[str(nrbf)+'rbf'] = []
    for s in range(1, 11):
        metrics[str(nrbf)+'rbf'].append(functions_moea_output_plots.get_metrics(dir_moea_output + '4obj_' + str(nrbf) +
                                                                              'rbf/metrics/DPS_param150_seedS1_seedB' + str(s) + '.metrics',
                                                                              '../../data/optimization_output/4obj_rbf_overall/DPS_4obj_rbf_overall_borg.hypervolume'))

# ### plot hypervolume for test of number of radial basis functions (nrbfs)
# print('Plotting hypervolume (fig S1)..., ', datetime.now() - startTime)
importlib.reload(functions_moea_output_plots)
nfe = 150000
fe_prints = 100
fe_grid = np.arange(0, nfe+1, nfe/fe_prints)
nseed = 10
functions_moea_output_plots.plot_metrics(dir_figs, metrics, nrbfs, nseed, fe_grid, fig_format)



# ##################################################################
# #### Analysis of 2dv vs full DPS. both 2obj & 4obj problems
# ##################################################################

### read in reference sets
nobj = 4
ncon = 1
ref_dps_2obj_retest, ndv_dps_2obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '2obj_2rbf/DPS_2obj_2rbf_borg_retest.resultfile', nobj, ncon)
ref_2dv_2obj_retest, ndv_2dv_2obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '2obj_2dv/DPS_2obj_2dv_borg_retest.resultfile', nobj, ncon)
ref_dps_4obj_retest, ndv_dps_4obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '4obj_2rbf_moreSeeds/DPS_4obj_2rbf_moreSeeds_borg_retest.resultfile', nobj, ncon)
ref_2dv_4obj_retest, ndv_dps_4obj_retest = functions_moea_output_plots.get_set(dir_moea_output + '4obj_2dv/DPS_4obj_2dv_borg_retest.resultfile', nobj, ncon)

# ### get best compromise solution for each set using topsis method
dfs = [ref_dps_2obj_retest, ref_2dv_2obj_retest, ref_dps_4obj_retest, ref_2dv_4obj_retest]
dfs = functions_moea_output_plots.topsis_dynstat(dfs)

### get limits for plots
lims2d = {'annRev':[9.4,11.06],'maxDebt':[-1,30.]}
lims3d = {'annRev':[9.4,11.13],'maxDebt':[0.,36.],'maxComplex':[0.,1.]}

### Comparison of 2dv vs full dps, 2 objective version. output policy performance values for table2
table2 = functions_moea_output_plots.plot_2obj_dynstat(ref_2dv_2obj_retest, ref_dps_2obj_retest, lims2d, fig_format)

### Comparison of 2dv vs full dps, 4 objective version
functions_moea_output_plots.plot_4obj_dynstat(ref_2dv_4obj_retest, ref_dps_4obj_retest, lims3d, fig_format)

### plot static & dynamic policy trajectories over wet & dry periods
functions_moea_output_plots.plot_example_simulation(ref_2dv_2obj_retest, ref_dps_2obj_retest, fig_format)



# ##################################################################
# #### Analysis of 4-objective subproblems & brushing
# ##################################################################

### plot min & max marker sizes for 3d 4-objective plot legend
functions_moea_output_plots.plot_4obj_markersize(ref_2dv_4obj_retest, ref_dps_4obj_retest, lims3d, fig_format)

### plot performance of solution sets from all sub-problems
paretos, pareto_cols = functions_moea_output_plots.plot_subproblems(ref_dps_4obj_retest, lims3d, ref_dps_2obj_retest, lims2d, fig_format)

# ### plot solutions meeting brushing constraints. output policy performance values for table2
constraints = {'annRev': 0.975, 'maxDebt': 1.5, 'maxComplex': np.inf, 'maxFund': 1.5}
functions_moea_output_plots.plot_4obj_brush(paretos, pareto_cols, constraints, lims3d, fig_format, table2)








