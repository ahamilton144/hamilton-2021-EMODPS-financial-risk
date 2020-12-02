##############################################################################################################
### functions_revenues_contracts.py - python functions used in creating simulated synthetic revenues and
###     index contract payouts, plus related plots
### Project started May 2017, last update Jan 2020
##############################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statsmodels.formula.api as sm
import seaborn as sbn
from scipy import stats as st
from scipy.optimize import minimize


sbn.set_style('white')
sbn.set_context('paper', font_scale=1.55)

cmap = cm.get_cmap('viridis')
col = [cmap(0.1),cmap(0.3),cmap(0.6),cmap(0.8)]


N_SAMPLES = 1000000
eps = 1e-13



##########################################################################
######### Simulate revenue, matching SFPUC 2016 rates and demands ###########
############## Returns dataframe of monthly revenues ($M/mnth) #########################################
##########################################################################

def simulate_revenue(dir_generated_inputs, gen, hp_GWh, hp_dolPerKwh, genSynth, powSynth, powHistSampleStart, redo = False, save = False):
  if (redo):
    # nYr = int(len(powSynth) / 12)
    # yrSim = np.full((1, nYr * 12), 0)
    # for i in range(1, nYr):
    #     yrSim[0, (12 * i):(12 * (i + 1))] = i

    ### rel b/w swe/gen and mtid
    # plt.scatter(hp_GWh.mtid.loc[2010:2016], swe.danWtAvg.loc[2010:2016])
    # np.corrcoef(hp_GWh.mtid.loc[2010:2016], swe.danWtAvg.loc[2010:2016])
    # get total amt above muni demand in each yr
    gen['aboveMuni'] = gen.tot - hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12
    gen['mtid'] = gen.aboveMuni.apply(lambda x: max(x, 0))
    # gen.mtid.loc[(gen.wmnth < 7) ] = 0     # assume mtid only buys power Apr-Sept
    hp_GWh['estMtid'] = np.nan
    hp_GWh.estMtid.loc[2010:2016] = gen.loc[gen.wyear > 2009, :].groupby('wyear').sum().mtid
    # plt.scatter(hp_GWh.estMtid.loc[2010:2016], hp_GWh.mtid.loc[2010:2016])
    # np.corrcoef(hp_GWh.estMtid.loc[2010:2016], hp_GWh.mtid.loc[2010:2016])
    # reg to get percentage estimated for mtid
    lmMtid = sm.ols(formula='mtid ~ est-1',
                    data=pd.DataFrame({'mtid': hp_GWh.mtid.loc[2010:2016], 'est': hp_GWh.estMtid.loc[2010:2016]}))
    lmMtid = lmMtid.fit()
    # print(lmMtid.summary())

    # plt.scatter(hp_GWh.estMtid.loc[2010:2016], lmMtid.predict())
    mtidGrowFrac = lmMtid.params[0]
    # gen.mtid = gen.mtid * mtidGrowFrac
    # gen['aboveMuniMtid'] = np.where(gen.aboveMuni > 0, gen.aboveMuni - gen.mtid, gen.aboveMuni)
    # plt.plot(gen.wmnth.loc[gen.wyear==2012],gen.aboveMuni.loc[gen.wyear==2012])
    # plt.plot(gen.aboveMuni)
    # plt.plot(gen.aboveMuniMtid)
    # plt.plot(gen.mtid)


    # revenue model: monthly gen & price, assume const demand to muni, 48% surplus (from regression above) to mtid throughout year (only if mtid rate < wholesale).
    # Rest to Wholesale. Also must buy power to meet unmet muni.
    def revenue_model_milDollars(sampGen_GWh, sampPow_DolPerkWh, dem_M_GWh, mtidFrac, rate_DolPerkWh_M,
                           rate_DolPerkWh_mtid):
      dem_mtid_GWh = np.maximum((sampGen_GWh - dem_M_GWh) * mtidFrac, 0)
      dem_mtid_GWh.loc[(sampPow_DolPerkWh < rate_DolPerkWh_mtid).values] = 0
      # dem_mtid_GWh.loc[(dem_mtid_GWh.index % 12 < 6)] = 0  # assume mtid only buys power Apr-Sept
      rev = (dem_M_GWh * rate_DolPerkWh_M + dem_mtid_GWh * rate_DolPerkWh_mtid + \
             (sampGen_GWh - dem_M_GWh - dem_mtid_GWh) * sampPow_DolPerkWh)
      return (rev)  # returns revenues in $Mil

    # simulated revs for synthetic time series
    revSim = revenue_model_milDollars(genSynth.gen,
                                powSynth.powPrice/1000,
                                hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12,
                                mtidGrowFrac,
                                hp_dolPerKwh['M'].iloc[hp_dolPerKwh.shape[0] - 1] ,
                                hp_dolPerKwh['mtid'].iloc[hp_dolPerKwh.shape[0] - 1] )
    # choose power price series to use with historical data
    powHistSample = powSynth.powPrice.iloc[powHistSampleStart:(powHistSampleStart+len(gen.tot))].reset_index(drop=True)
    # simulated revs for historical generation w/ random synth power price & current fixed muni/mtid rates
    revHist = pd.DataFrame({'rev': revenue_model_milDollars(gen.tot.reset_index(drop=True),
                                                      powSynth.powPrice.iloc[3600:(3600+len(gen.tot))].reset_index(drop=True)/1000,
                                                      hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12,
                                                      mtidGrowFrac,
                                                      hp_dolPerKwh['M'].iloc[hp_dolPerKwh.shape[0] - 1],
                                                      hp_dolPerKwh['mtid'].iloc[hp_dolPerKwh.shape[0] - 1]),
                            'wmnth': gen.wmnth,
                            'wyear': gen.wyear})

    if (save):
      revSim.to_pickle(dir_generated_inputs + 'revSim.pkl')
      revHist.to_pickle(dir_generated_inputs + 'revHist.pkl')
      powHistSample.to_pickle(dir_generated_inputs + 'powHistSample.pkl')


  else:
    revSim = pd.read_pickle(dir_generated_inputs + 'revSim.pkl')
    revHist = pd.read_pickle(dir_generated_inputs + 'revHist.pkl')
    powHistSample = pd.read_pickle(dir_generated_inputs + 'powHistSample.pkl')

  return (revHist, powHistSample, revSim)





##########################################################################
######### wang transform function ###########
############## Returns dataframe with net payout #########################################
##########################################################################

def wang(df, contractType, lam, k, cap=-1., premOnly=False, lastYrTrig=-1., count=0):  # df should be dataframe with columns 'asset' and 'prob'; contractType is 'put' or 'call'
  # print(count)
  if contractType == 'put':
    df['payout'] = df['asset'].apply(lambda x: max(k - x, 0))
    lam = -abs(lam)
  elif contractType == 'call':
    df['payout'] = df['asset'].apply(lambda x: max(x - k, 0))
    lam = abs(lam)
  elif contractType == 'shortcall':
    df['payout'] = df['asset'].apply(lambda x: max(-max(x - k, 0), -(cap - k)))
    lam = -abs(lam)
  elif contractType == 'putWithLastYrTrig':
    df['payout'] = df['asset'].apply(lambda x: max(k - x, 0))
    gttrig = (df['asset'].iloc[:-1] > lastYrTrig).values
    df['payout'].iloc[1:].loc[gttrig] = 0
    df['payout'].iloc[0] = 0
    lam = -abs(lam)
  else:
    df['payout'] = np.nan
  df.sort_values(inplace=True, by='payout')
  dum = df['prob'].cumsum()  # asset cdf
  dum = st.norm.cdf(st.norm.ppf(dum) + lam)  # risk transformed payout cdf
  dum = np.append(dum[0], np.diff(dum))  # risk transformed asset pdf
  prem = (dum * df['payout']).sum()
  if premOnly == False:
    df.sort_index(inplace=True)
    return ((df['payout'] - prem))
  else:
    return prem




##########################################################################
######### snow index contract net payouts ###########
############## Returns dataframe with net payout #########################################
##########################################################################

def snow_contract_payout(dir_generated_inputs, sweWtSynth, contractType = 'put', lambdaRisk = 0.25, strikeQuantile = 0.6,
                       capQuantile = 0.95, redo = False, save = False):

  if (redo):
    if (contractType == 'put'):
      snowPayoutSim = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1/sweWtSynth.shape[0]}), contractType='put',
                           lam=lambdaRisk, k=sweWtSynth.quantile(strikeQuantile), premOnly=False)
      if (save):
        save_location = dir_generated_inputs + 'payoutPut%sSim.pkl' % int(strikeQuantile*100)
        snowPayoutSim.to_pickle(save_location)

    if (contractType == 'shortcall'):
      snowPayoutSim = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1/sweWtSynth.shape[0]}), contractType='shortcall',
                           lam=lambdaRisk, k=sweWtSynth.quantile(strikeQuantile),
                           cap=sweWtSynth.quantile(capQuantile), premOnly=False)
      if (save):
        save_location = dir_generated_inputs + 'payoutShortCall%sSim.pkl' % int(strikeQuantile*100)
        snowPayoutSim.to_pickle(save_location)

    elif (contractType == 'cfd'):
      snowPayoutSim = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1/sweWtSynth.shape[0]}), contractType='put',
                           lam=lambdaRisk, k=sweWtSynth.quantile(strikeQuantile), premOnly=False)
      snowPayoutSim = snowPayoutSim + wang(pd.DataFrame({'asset': sweWtSynth, 'prob': 1 / sweWtSynth.shape[0]}),
                                           contractType='shortcall', lam=0, k=sweWtSynth.quantile(strikeQuantile),
                                           cap=sweWtSynth.quantile(capQuantile), premOnly=False)
      if (save):
        snowPayoutSim.to_pickle(dir_generated_inputs + 'payoutCfdSim.pkl')

  else:
    if (contractType == 'put'):
      save_location = dir_generated_inputs + 'payoutPut%sSim.pkl' % int(strikeQuantile * 100)
      snowPayoutSim = pd.read_pickle(save_location)
    if (contractType == 'shortcall'):
      save_location = dir_generated_inputs + 'payoutShortCall%sSim.pkl' % int(strikeQuantile * 100)
      snowPayoutSim = pd.read_pickle(save_location)
    elif (contractType == 'cfd'):
      snowPayoutSim = pd.read_pickle(dir_generated_inputs + 'payoutCfdSim.pkl')

  return (snowPayoutSim)





##########################################################################
######### plot snow contract  (fig 4) ###########
############## Returns figure #########################################
##########################################################################
def plot_contract(dir_figs, sweWtSynth, payoutCfdSim, lambda_shifts):

  strike = sweWtSynth.quantile(0.5)
  prob = 1 / sweWtSynth.shape[0]
  # first get prem for base case
  prem_base = wang(pd.DataFrame({'asset': sweWtSynth, 'prob': prob}), contractType='put', lam=0.25, k=strike, premOnly=True)
  for i in range(len(lambda_shifts)):
    lambda_shifts[i] = prem_base - wang(pd.DataFrame({'asset': sweWtSynth, 'prob': prob}), contractType='put',
                                        lam=lambda_shifts[i], k=strike, premOnly=True)

  ### plot regime as function of debt and uncertain params
  plt.figure()
  ax = plt.subplot2grid((3,1), (0, 0))
  # ax.set_xlabel('SWE Index (inch)')
  ax.set_ylabel('Density')
  # ax.set_xticks(np.arange(0.85, 0.98, 0.04))
  ax.set_xlim([0,60])
  ax.set_ylim([0,0.04])
  # ax.set_yticks(np.arange(0, 6))
  ax.tick_params(axis='x', which='both', labelbottom=False,labeltop=True)
  ax.tick_params(axis='y', which='both', labelleft=False,labelright=False)
  # ax.xaxis.set_label_position('top')

  sbn.kdeplot(sweWtSynth, ax=ax, c='k', lw=2)

  ax = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
  ax.set_xlabel('SWE Index (inch)')
  ax.set_ylabel('Net Payout ($M)')
  ax.set_xlim([0, 60])
  ax.tick_params(axis='y', which='both', labelleft=False,labelright=False)
  ax.axhline(0, color='grey', linestyle=':')
  kinkY = np.min(payoutCfdSim)
  kinkX = np.min(sweWtSynth.loc[payoutCfdSim < kinkY + eps])
  line3, = ax.plot([0, kinkX, 60], [kinkX + kinkY, kinkY, kinkY], color=col[0], linewidth=2)
  plot_name = dir_figs + 'contract.png'
  plt.savefig(plot_name, dpi=500)

  return





##########################################################################
######### power index contract net payouts ###########
############## Returns dataframe with net payout #########################################
##########################################################################

def power_price_index(powSynth, genSynth, revSim, hp_GWh):
  nYr = int(len(powSynth) / 12)
  yrSim = np.full((1, nYr * 12), 0)
  mnthSim = np.full((1, nYr * 12), 0)
  for i in range(1, nYr):
    yrSim[0, (12 * i):(12 * (i + 1))] = i
  for i in range(12):
    mnthSim[0, i::12] = i

  muniDemand = hp_GWh['M'].iloc[hp_GWh.shape[0] - 1] / 12

  genExcess = genSynth.gen - muniDemand
  genExcessAvg = genExcess.groupby(mnthSim[0]).mean()
  genWeights = genExcessAvg / genExcessAvg.sum()
  powSynth['powPriceWt'] = (np.array(powSynth.powPrice).reshape(
      [nYr, 12]) * np.array(genWeights).reshape([1, 12])).reshape(nYr * 12)
  pwyr = powSynth.powPriceWt.groupby(yrSim[0, :]).sum()

  # plt.scatter(pwyr.iloc[:1000], revSim.groupby(yrSim[0, :]).sum().iloc[:1000])
  # np.corrcoef(pwyr.iloc[:1000], revSim.groupby(yrSim[0, :]).sum().iloc[:1000])
  # plt.scatter(pwyr.iloc[:1000], revSim.groupby(yrSim[0, :]).sum().iloc[:1000] + 0.83*snowCont.iloc[:1000])
  # np.corrcoef(pwyr.iloc[:1000], revSim.groupby(yrSim[0, :]).sum().iloc[:1000] + 0.83*snowCont.iloc[:1000])

  psep = powSynth.powPrice[11::12].values

  # plt.scatter(psep[:1000],pwyr[1:1001])
  # np.corrcoef(psep[:-1],pwyr[1:])

  ### regress ln(pwyr_t) = a*ln(pwyr_{t-1}) + b*ln(psep_{t-1}) + c + eps, eps ~ N(0, sig2)
  ### then by moment generating function proof,
  ### E[pwyr_t] = pwyr_{t-1}^a * psep_{t-1}^b * exp(sig2 / 2)
  lmPswp = sm.ols(formula='ln_pwyr0 ~ ln_pwyr1 + ln_psep1',
                  data=pd.DataFrame({'ln_pwyr0': np.log(pwyr[1:].values), 'ln_pwyr1': np.log(pwyr[:-1].values),
                                      'ln_psep1': np.log(psep[:-1])}))
  lmPswp = lmPswp.fit()
  # print(lmPswp.summary())

  E_pwyr = np.full(nYr, np.nan)
  E_pwyr[1:] = (pwyr[:-1].values ** lmPswp.params[1]) * (psep[:-1] ** lmPswp.params[2]) * \
                np.exp(lmPswp.params[0]) * np.exp(np.var(lmPswp.resid) / 2)

  # plt.scatter(psep[:999], pwyr[1:1000])
  # plt.scatter(st.rankdata(psep[:999]), st.rankdata(pwyr[1:1000]))
  # np.corrcoef(psep[:-1], netpay[1:])
  # st.spearmanr(psep[:-1], netpay[1:])

  return E_pwyr







##########################################################################
######### save synthetic data needed for moea ###########
############## Saves csv, no return #########################################
##########################################################################
def save_synthetic_data_moea(dir_generated_inputs, revSimWyr, payoutCfdSim, powerIndex):
  synthetic_data = pd.DataFrame({'revenue': revSimWyr.values, 'payoutCfd': payoutCfdSim.values,
                                 'power': powerIndex}).iloc[1:, :].reset_index(drop=True)[['revenue', 'payoutCfd', 'power']]
  synthetic_data.to_csv(dir_generated_inputs + 'synthetic_data.txt',sep=' ', index=False)




