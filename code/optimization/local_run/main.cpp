
/* main.cpp

  Adapted by Andrew Hamilton from Lake Problem DPS. Started Jul 2018, last update May 2020.
  University of North Carolina at Chapel Hill
  andrew.hamilton@unc.edu

  Lake Problem DPS {
    Riddhi Singh, May, 2014
    The Pennsylvania State University
    rus197@psu.edu

    Adapted by Tori Ward, July 2014
    Cornell University
    vlw27@cornell.edu

    Adapted by Jonathan Herman and David Hadka, Sept-Dec 2014
    Cornell University and The Pennsylvania State University

    Adapted by Julianne Quinn, July 2015 as DPS problem
    Cornell University
    jdq8@cornell.edu
  }


*/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sstream>
#include <ctime>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include "./../../misc/boostutil.h"
#include "./../../misc/borg/borg.h"
#include "./../../misc/borg/moeaframework.h"
//#include "mpi.h"

#define DPS_RUN_TYPE 1          // 0: 2dv version; 1: full DPS with RBFs, maxDebt formulation; 2: full DPS with RBFs, minRev formulation
#define BORG_RUN_TYPE 1       // 0: single run no borg; 1: borg run, serial; 2: borg parallel for cluster;
#define NUM_YEARS 20                   //20yr sims
#define NUM_SAMPLES 500
#define NUM_DECISIONS_TOTAL 2           // each year, have to choose value snow contract + withdrawal
#define NUM_LINES_STOCHASTIC_INPUT 999999    //Input file samp.txt has 1M rows, 6 cols.
#define NUM_VARIABLES_STOCHASTIC_INPUT 3            //6 cols in input: swe,powIndex,revRetail,revWholesale,sswp,pswp, sweCorr0.25,sweCorr0.50,sweCorr0.75
#define INDEX_STOCHASTIC_REVENUE 0   // revenue in first column
#define INDEX_STOCHASTIC_SNOW_PAYOUT 1    // snow contract payout in 2nd column
#define INDEX_STOCHASTIC_POWER_INDEX 2  // power price index in 3rd column
#define MEAN_REVENUE 127.80086602479503     // mean revenue in absense of any financial risk mgmt
#define NORMALIZE_SNOW_CONTRACT_SIZE 4.0
#define NORMALIZE_REVENUE 250.0
#define NORMALIZE_FUND 150.0
#define NORMALIZE_POWER_PRICE 350.0
#define NORMALIZE_SWE 150.0
#define BUFFER_MAX_SIZE 5000
#define EPS 0.0000000000001
#define NUM_OBJECTIVES 4
#define EPS_ANNREV 0.075
#define EPS_MAXDEBT 0.225
#define EPS_MINREV 0.225
#define EPS_MAXCOMPLEXITY 0.05001
#define EPS_MAXFUND 0.225
#if DPS_RUN_TYPE<2
#define NUM_CONSTRAINTS 1
#else
#define NUM_CONSTRAINTS 0
#endif
#define EPS_CONS1 0.05
#define NUM_RBF 4      // number of radial basis functions, used if DPS_RUN_TYPE>0
#define SHARED_RBFS 1     // 1 = 1 rbf shared between hedge and withdrawal policies. 0 = separate rbf for each. 2 = rbf for hedge, and 2dv formulation for withdrawal.
#if SHARED_RBFS==2
#define NUM_INPUTS_RBF 3        // inputs: fund balance, debt balance, power index.  If NUM_INPUTS_RBF==4, 4th is swe correlate
#else
#define NUM_INPUTS_RBF 4    // inputs: fund balance, debt balance, power index, rev+hedge cash flow. If NUM_INPUTS_RBF==5, 5th is swe correlate
#endif
#if DPS_RUN_TYPE>0
#if SHARED_RBFS==0
#define NUM_DV (2 * NUM_DECISIONS_TOTAL * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * (NUM_RBF + 2))
#elif SHARED_RBFS==1
#define NUM_DV (2 * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * (NUM_RBF + 2))
#elif SHARED_RBFS==2
#define NUM_DV (2 * NUM_RBF * NUM_INPUTS_RBF) + (NUM_DECISIONS_TOTAL * 2) + NUM_RBF
#endif
#else
#define NUM_DV 2
#endif
#define MIN_SNOW_CONTRACT 0.05          // DPS_RUN_TYPE==0 only: if contract slope dv < $0.05M/inch, act as if 0.
#define MIN_MAX_FUND 0.05               // DPS_RUN_TYPE==0 only: if max fund dv < $0.05M, act as if 0.
#define NUM_PARAM 6         // cost_fraction, discount_rate, delta_interest_fund, delta_interest_debt, lambda, lambda_prem_shift
#define NUM_PARAM_SAMPLES 1  // number of LHC samples in param file. Last line is values for SFPUC, Oct 2016.
#define USEINRBF_FUND_HEDGE 1.   // for input leave-one-out experiment. If 1, input will be used in rbf, else not used.
#define USEINRBF_DEBT_HEDGE 1.
#define USEINRBF_POWER_HEDGE 1.
#define USEINRBF_FUND_WITHDRAWAL 1.
#define USEINRBF_DEBT_WITHDRAWAL 1.
#define USEINRBF_POWER_WITHDRAWAL 1.
#define USEINRBF_CASHIN_WITHDRAWAL 1.

namespace ublas = boost::numeric::ublas;
namespace tools = boost::math::tools;
namespace accumulator = boost::accumulators;
using namespace std;

void normalizeWeights(ublas::vector<double> & f_dv_w);
double policyWithdrawal(const double f_fund_balance, const double f_debt, const double f_power_price_index,
                             const double f_cash_in, const double f_swe_correlate, const ublas::vector<double> & f_dv_d,
                             const ublas::vector<double> & f_dv_c, const ublas::vector<double> & f_dv_b,
                             const ublas::vector<double> & f_dv_w, const ublas::vector<double> & f_dv_a);
double policySnowContractValue(const double f_fund_balance, const double f_debt, const double f_power_price_index,
                               const double f_swe_correlate, const ublas::vector<double> & f_dv_d,
                               const ublas::vector<double> & f_dv_c, const ublas::vector<double> & f_dv_b,
                               const ublas::vector<double> & f_dv_w, const ublas::vector<double> & f_dv_a) ;
double policyWithdrawal_2dv(const double f_fund_balance, const double f_cash_in, const double f_max_fund_size);
double policySnowContractValue_2dv(const double f_value);
double policyMaxFund_2dv(const double f_value);

double stochastic_input[NUM_LINES_STOCHASTIC_INPUT][NUM_VARIABLES_STOCHASTIC_INPUT];                   // Stochastic variables
double param_LHC_sample[NUM_PARAM][NUM_PARAM_SAMPLES];              // financial parameters
#if (BORG_RUN_TYPE == 0)
double problem_dv[NUM_DV];
double problem_objs[NUM_OBJECTIVES];
double problem_constraints[NUM_CONSTRAINTS];
double pareto[NUM_DV + 2*(NUM_OBJECTIVES + NUM_CONSTRAINTS)][10000];
double N_pareto = 0;
#endif

ublas::vector<double> annualized_adjusted_revenue(NUM_SAMPLES);     // objectives
ublas::vector<double> max_hedge_complexity(NUM_SAMPLES);
ublas::vector<double> max_fund_balance(NUM_SAMPLES);
#if DPS_RUN_TYPE<2
ublas::vector<double> debt_steal(NUM_SAMPLES);
#else
ublas::vector<double> min_adjusted_revenue(NUM_SAMPLES);
#endif

ublas::vector<double> revenue(NUM_YEARS);                          // state variables
ublas::vector<double> payout_snow_contract(NUM_YEARS);      // snow contract (sswp or sput)
ublas::vector<double> swe_correlate(NUM_YEARS);      // snow contract (sswp or sput)
ublas::vector<double> power_price_index(NUM_YEARS+1);             // power price index
ublas::vector<double> discount_factor(NUM_YEARS);

ublas::vector<double> adjusted_revenue(NUM_YEARS);                     // decisions
ublas::vector<double> fund_balance(NUM_YEARS + 1);
ublas::vector<double> fund_withdrawal(NUM_YEARS);
ublas::vector<double> debt(NUM_YEARS + 1);
ublas::vector<double> value_snow_contract(NUM_YEARS);           // value of contract 1
#if SHARED_RBFS==2
ublas::vector<double> dv_w(NUM_RBF);      // decision variables, dps params
#else
ublas::vector<double> dv_w(NUM_RBF * NUM_DECISIONS_TOTAL);      // decision variables, dps params
#endif
ublas::vector<double> dv_d(NUM_DECISIONS_TOTAL);
ublas::vector<double> dv_a(NUM_DECISIONS_TOTAL);
#if SHARED_RBFS==0
ublas::vector<double> dv_c(NUM_DECISIONS_TOTAL * NUM_RBF * NUM_INPUTS_RBF);
ublas::vector<double> dv_b(NUM_DECISIONS_TOTAL * NUM_RBF * NUM_INPUTS_RBF);
#else
ublas::vector<double> dv_c(NUM_RBF * NUM_INPUTS_RBF);
ublas::vector<double> dv_b(NUM_RBF * NUM_INPUTS_RBF);
#endif


string read_directory;          // directory where input files are stored
#if BORG_RUN_TYPE>0
string write_directory;         // directory where "sets" and "runtime" folders are located, for writing output
#else
string set_file;            // set file to read in dv's and objectives
string retest_file;         // file to write reevaluated objectives to
#endif
double cost_fraction;   // params to be looped over with LHC sample, doing borg each time. Declare globally.
double avg_surplus_revenue;
double discount_rate;
double interest_fund;
double interest_debt;
double lambda_prem_shift;
int LHC_set;
int NFE;
int NFE_counter = 0;
unsigned int seed_borg;
unsigned int seed_sample;      // use same seed for each function evaluation, so always comparing same simulations. should be less noisy.
int lines_to_use[NUM_SAMPLES];

#if DPS_RUN_TYPE<2
typedef accumulator::accumulator_set<double, accumulator::stats<accumulator::tag::tail_quantile<accumulator::right> > > accumulator_t;
#endif

// problem for borg search
void portfolioProblem(double *problem_dv, double *problem_objs, double *problem_constraints) {
    NFE_counter += 1;
//    if ((NFE_counter % 100) == 0){printf("%d\n", NFE_counter);}
//    printf("%d\n", NFE_counter);

    // initialize variables
    zero(annualized_adjusted_revenue);
    zero(max_hedge_complexity);
    zero(max_fund_balance);
    zero(revenue);
    zero(swe_correlate);
    zero(power_price_index);
    zero(payout_snow_contract);
    zero(discount_factor);
    zero(dv_d);
    zero(dv_c);
    zero(dv_b);
    zero(dv_w);
    zero(dv_a);
#if DPS_RUN_TYPE<2
    zero(debt_steal);
#else
    zero(min_adjusted_revenue);
#endif

#if DPS_RUN_TYPE>0
    // get dvs
//    printf("\ndv_d:\t");
    for (int i = 0; i < dv_d.size(); i++){
        dv_d(i) = problem_dv[i];                      // cutoffs: dv_d = [CFMAX, XMIN1, XMIN2]
//        printf("%f  ", dv_d(i));
    }
//    printf("\n dv_c:\t");
    for (int i = 0; i < dv_c.size(); i++){
        dv_c(i) = problem_dv[i + dv_d.size()];        // centers for RBFs
//        printf("%f  ", dv_c(i));
    }
//    printf("\n dv_b:\t");
    for (int i = 0; i < dv_b.size(); i++){
        dv_b(i) = max(EPS, problem_dv[i + dv_d.size() + dv_c.size()]);          // radii for RBFs
//        printf("%f  ", dv_b(i));
    }
    for (int i = 0; i < dv_w.size(); i++){
        dv_w(i) = problem_dv[i + dv_d.size() + dv_c.size() + dv_b.size()];        // weights for RBFs
    }
//    printf("\n dv_a:\t");
    for (int i = 0; i < dv_a.size(); i++){
        dv_a(i) = problem_dv[i + dv_d.size() + dv_c.size() + dv_b.size() + dv_w.size()];                      // const addition
//        printf("%f  ", dv_a(i));
    }
    // normalize weights
    normalizeWeights(dv_w);
//    printf("\n dv_a:\t");
    for (int i = 0; i < dv_w.size(); i++){
//        printf("%f  ", dv_w(i));
    }
//    printf("\n");
#else
    double fixed_max_fund = policyMaxFund_2dv(problem_dv[0]);
    double fixed_value_snow_contract = policySnowContractValue_2dv(problem_dv[1]);
#endif

    // create discounting factor
    double discount_normalization = 0.0;      // discounting normalization, 1/sum_(discount_factor)
    for (int i = 0; i < NUM_YEARS; i++){
        discount_factor(i) = pow(discount_rate, i+1);
        discount_normalization += discount_factor(i);
    }
    discount_normalization = 1.0 / discount_normalization;

#if DPS_RUN_TYPE<2
    accumulator_t debt_q95(accumulator::tag::tail<accumulator::right>::cache_size = NUM_SAMPLES);    // accumulator object for calculating upper 5th quantile of debt
#endif

    double net_payout_snow_contract = 0.;
    double cash_in = 0.;

    // run revenue model simulation
    for (int s = 0; s < NUM_SAMPLES; s++) {
        // randomly generated revenues
#if NUM_SAMPLES > 1
        int index = lines_to_use[s];
#else
        int index = 1;
#endif
//        printf("\n\nSample %d  %d\n", s, index);

        // get the random revenue from the States of the world file
        //each line of SOW file covers 20 years of revenue
        power_price_index(0) = stochastic_input[index - 1][INDEX_STOCHASTIC_POWER_INDEX];
        for (int i = 0; i < NUM_YEARS; i++) {
            revenue(i) = (stochastic_input[index + i][INDEX_STOCHASTIC_REVENUE] - MEAN_REVENUE * cost_fraction);
            payout_snow_contract(i) = stochastic_input[index + i][INDEX_STOCHASTIC_SNOW_PAYOUT];
            power_price_index(i+1) = stochastic_input[index + i][INDEX_STOCHASTIC_POWER_INDEX];
//            printf("%f  %f  %f\n", revenue(i), payout_snow_contract(i), payout_power_contract(i));
        }

        // initial simulation variables
        zero(value_snow_contract);
        zero(fund_balance);
        zero(fund_withdrawal);
        zero(adjusted_revenue);
        zero(debt);

        max_hedge_complexity(s) = 0;
#if DPS_RUN_TYPE==0
        if (fixed_value_snow_contract > EPS) {
            max_hedge_complexity(s) = 1;
        }
#endif

        //calculate new revenues, contingency fund balance, objectives
        for (int i = 0; i < NUM_YEARS; i++) {

            // find next policy-derived index insurance and CF withdrawal

#if  DPS_RUN_TYPE>0
            value_snow_contract(i) = policySnowContractValue(fund_balance(i), debt(i), power_price_index(i),
                                                             swe_correlate(i),dv_d, dv_c, dv_b, dv_w, dv_a);
            if (abs(value_snow_contract(i)) > EPS) {
                max_hedge_complexity(s) = 1;
            }
            net_payout_snow_contract = value_snow_contract(i) * (payout_snow_contract(i) - lambda_prem_shift);
            cash_in = revenue(i) + net_payout_snow_contract - (debt(i) * interest_debt);
            fund_withdrawal(i) = policyWithdrawal(fund_balance(i) * interest_fund, debt(i) * interest_debt,
                                                        power_price_index(i + 1), cash_in, swe_correlate(i),
                                                        dv_d, dv_c, dv_b, dv_w, dv_a);
            adjusted_revenue(i) = cash_in + fund_withdrawal(i);

#else
            net_payout_snow_contract = fixed_value_snow_contract * (payout_snow_contract(i) - lambda_prem_shift);
            cash_in = revenue(i) + net_payout_snow_contract - (debt(i) * interest_debt);
            fund_withdrawal(i) = policyWithdrawal_2dv(fund_balance(i) * interest_fund, cash_in, fixed_max_fund);
            adjusted_revenue(i) = cash_in + fund_withdrawal(i);
#endif

#if DPS_RUN_TYPE<2
            if (adjusted_revenue(i) < -EPS){
                debt(i + 1) = -adjusted_revenue(i);
                adjusted_revenue(i) = 0;
            }
#endif
            fund_balance(i + 1) = fund_balance(i) * interest_fund - fund_withdrawal(i);

            annualized_adjusted_revenue(s) += adjusted_revenue(i) * discount_factor(i);

        }
        annualized_adjusted_revenue(s) = discount_normalization *
                                         (annualized_adjusted_revenue(s) +
                                          ((fund_balance(NUM_YEARS) * interest_fund * discount_factor(0)) -
                                           (debt(NUM_YEARS) * interest_debt * discount_factor(0))) *
                                          discount_factor(NUM_YEARS - 1));
#if DPS_RUN_TYPE<2
        debt_q95(vmax(debt));                                                               //q95(max(debt)) constraint
        debt_steal(s) = debt(NUM_YEARS) - debt(NUM_YEARS - 1);
#else
        min_adjusted_revenue(s) = vmin(adjusted_revenue);
#endif

        max_fund_balance(s) = vmax(fund_balance);

    }

    // aggregate objectives
    problem_objs[0] = -1 * vsum(annualized_adjusted_revenue) / NUM_SAMPLES; // max: average annualized adjusted_revenue, across samp
#if DPS_RUN_TYPE<2
    problem_objs[1] = accumulator::quantile(debt_q95, accumulator::quantile_probability = 0.95);    //minimize 95th percentile of max debt
#else
    problem_objs[1] = -1 * vsum(min_adjusted_revenue) / NUM_SAMPLES;
#endif
#if NUM_OBJECTIVES > 2
    problem_objs[2] = 1 * vsum(max_hedge_complexity) / NUM_SAMPLES; // min: avg_avg_hedging complexity
    problem_objs[3] = 1 * vsum(max_fund_balance) / NUM_SAMPLES; // min: max_fund_balance
#endif

#if DPS_RUN_TYPE<2
    // check constraint
    problem_constraints[0] = max(0.0, (vsum(debt_steal) / NUM_SAMPLES) - EPS_CONS1);
#endif

//    printf("\n\n\n%f   %f   %f   %f   %f\n\n\n", problem_objs[0], problem_objs[1], problem_objs[2], problem_objs[3], problem_constraints[0]);

    annualized_adjusted_revenue.clear();
    max_hedge_complexity.clear();
    max_fund_balance.clear();
    fund_balance.clear();
    revenue.clear();
    swe_correlate.clear();
    power_price_index.clear();
    payout_snow_contract.clear();
    discount_factor.clear();
    value_snow_contract.clear();
    fund_withdrawal.clear();
    adjusted_revenue.clear();
    dv_d.clear();
    dv_c.clear();
    dv_b.clear();
    dv_w.clear();
    dv_a.clear();
#if DPS_RUN_TYPE<2
    debt_steal.clear();
    debt.clear();
#else
    min_adjusted_revenue.clear();
#endif

}

// normalize RBF weights. for each decision, weights should sum to 1.
void normalizeWeights(ublas::vector<double> & f_dv_w){
    int num_decisions_rbf = f_dv_w.size() / NUM_RBF;
    for (int j = 0; j < num_decisions_rbf; j++){
        double total = 0.0;
        for(int i = j * NUM_RBF; i < (j + 1) * NUM_RBF; i++) {
            total += f_dv_w(i);
        }
        if (total != 0){
            for (int i = j * NUM_RBF; i < (j + 1) * NUM_RBF; i++){
                f_dv_w(i) = f_dv_w(i) / total;
//                printf("%f   ",dv_w(i));
            }
        }
    }

}



// calculate withdrawal (+)/deposit (-) from reserve fund at end of year, using fund balance, power price, and cash flow, and power price index as inputs. This version uses adjusted rev for RBF, then backcalculates withdrawal.
double policyWithdrawal(const double f_fund_balance, const double f_debt, const double f_power_price_index,
                             const double f_cash_in, const double f_swe_correlate, const ublas::vector<double> & f_dv_d,
                             const ublas::vector<double> & f_dv_c, const ublas::vector<double> & f_dv_b,
                             const ublas::vector<double> & f_dv_w, const ublas::vector<double> & f_dv_a) {
    int decision_order = 1;
    double cash_out = 0;
    double withdrawal = 0.;
    double arg_exp = 0.;
    // decision between 0 and 1 for cash_out (aka f_cash_in + withdrawal) based on RBF and state variable inputs
    for (int i = 0; i < NUM_RBF; i++){
#if SHARED_RBFS==0
        arg_exp =  - pow(f_fund_balance * USEINRBF_FUND_WITHDRAWAL / NORMALIZE_FUND - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i), 2)
                        / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i), 2)
                   - pow(f_debt * USEINRBF_DEBT_WITHDRAWAL / NORMALIZE_FUND - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1), 2)
                     / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1), 2)
                   - pow(f_power_price_index * USEINRBF_POWER_WITHDRAWAL / NORMALIZE_POWER_PRICE - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2), 2)
                         / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2), 2)
                    - pow((f_cash_in * USEINRBF_CASHIN_WITHDRAWAL + NORMALIZE_REVENUE)/(2 * NORMALIZE_REVENUE) - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 3), 2)
                         / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 3), 2);
        cash_out += f_dv_w(decision_order * NUM_RBF + i) * exp(arg_exp) ;

#elif SHARED_RBFS==1
        arg_exp =  - pow(f_fund_balance * USEINRBF_FUND_WITHDRAWAL / NORMALIZE_FUND - f_dv_c(NUM_INPUTS_RBF * i), 2)
                   / pow(f_dv_b(NUM_INPUTS_RBF * i), 2)
                   - pow(f_debt * USEINRBF_DEBT_WITHDRAWAL / NORMALIZE_FUND - f_dv_c(NUM_INPUTS_RBF * i + 1), 2)
                     / pow(f_dv_b(NUM_INPUTS_RBF * i + 1), 2)
                   - pow(f_power_price_index * USEINRBF_POWER_WITHDRAWAL / NORMALIZE_POWER_PRICE - f_dv_c(NUM_INPUTS_RBF * i + 2), 2)
                     / pow(f_dv_b(NUM_INPUTS_RBF * i + 2), 2)
                   - pow((f_cash_in * USEINRBF_CASHIN_WITHDRAWAL + NORMALIZE_REVENUE)/(2 * NORMALIZE_REVENUE) - f_dv_c(NUM_INPUTS_RBF * i + 3), 2)
                     / pow(f_dv_b(NUM_INPUTS_RBF * i + 3), 2);
        cash_out += f_dv_w(decision_order * NUM_RBF + i) * exp(arg_exp) ;

#elif SHARED_RBFS==2
        cash_out +=  0.0;
#endif
//        printf("%f  ", cash_out);
    }
    cash_out += f_dv_a(decision_order);
    // now scale back to [-NORMALIZE_REVENUE,NORMALIZE_REVENUE]
    cash_out = max(min((cash_out * 2 * NORMALIZE_REVENUE) - NORMALIZE_REVENUE, NORMALIZE_REVENUE), -NORMALIZE_REVENUE);
    // now write as withdrawal for policy return
    withdrawal = cash_out - f_cash_in;
    // ensure that cant withdraw more than fund balance
    if (withdrawal > EPS){
        withdrawal = min(withdrawal , f_fund_balance);
    }else if (withdrawal < -EPS){
        // ensure that cant deposit more than available cash flow
        withdrawal = max(withdrawal, -max(f_cash_in, 0.));
    }
    // ensure that (fund balance - withdrawal (+ deposit)) isnt larger than max fund size
    if ((f_fund_balance - withdrawal) > f_dv_d(decision_order) * NORMALIZE_FUND){
        withdrawal = (f_fund_balance - (f_dv_d(decision_order) * NORMALIZE_FUND));
    }

//    printf("%f  %f  %f  %f  %f  %f  %f  %f\n", f_fund_balance, f_debt, f_power_price_index, f_cash_in, f_dv_d(decision_order) * NORMALIZE_FUND, arg_exp, cash_out, withdrawal);
    return withdrawal;
}





// calculate value multiplier for snow contract this year. input = fund balance.
double policySnowContractValue(const double f_fund_balance, const double f_debt,  const double f_power_price_index,
                               const double f_swe_correlate, const ublas::vector<double> & f_dv_d,
                               const ublas::vector<double> & f_dv_c, const ublas::vector<double> & f_dv_b,
                               const ublas::vector<double> & f_dv_w, const ublas::vector<double> & f_dv_a)  {
    int decision_order = 0;
    double value = 0;
    double arg_exp;
    // decision between 0 and 1 based on RBF and state var inputs
    for (int i = 0; i < NUM_RBF; i++){
#if SHARED_RBFS==0
        arg_exp = -pow(f_fund_balance * USEINRBF_FUND_HEDGE / NORMALIZE_FUND - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i), 2)
                    / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i), 2)
                  - pow(f_debt * USEINRBF_DEBT_HEDGE / NORMALIZE_FUND - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1), 2)
                   / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1), 2)
                  - pow(f_power_price_index * USEINRBF_POWER_HEDGE / NORMALIZE_POWER_PRICE - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2), 2)
                         / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2), 2);
#elif SHARED_RBFS==1
        arg_exp = -pow(f_fund_balance * USEINRBF_FUND_HEDGE / NORMALIZE_FUND - f_dv_c(NUM_INPUTS_RBF * i), 2)
                  / pow(f_dv_b(NUM_INPUTS_RBF * i), 2)
                  - pow(f_debt * USEINRBF_DEBT_HEDGE / NORMALIZE_FUND - f_dv_c(NUM_INPUTS_RBF * i + 1), 2)
                    / pow(f_dv_b(NUM_INPUTS_RBF * i + 1), 2)
                  - pow(f_power_price_index * USEINRBF_POWER_HEDGE / NORMALIZE_POWER_PRICE - f_dv_c(NUM_INPUTS_RBF * i + 2), 2)
                    / pow(f_dv_b(NUM_INPUTS_RBF * i + 2), 2);
#elif SHARED_RBFS==2
        arg_exp = -pow(f_fund_balance * USEINRBF_FUND_HEDGE / NORMALIZE_FUND - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i), 2)
                    / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i), 2)
                  - pow(f_debt * USEINRBF_DEBT_HEDGE / NORMALIZE_FUND - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1), 2)
                   / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 1), 2)
                  - pow(f_power_price_index * USEINRBF_POWER_HEDGE / NORMALIZE_POWER_PRICE - f_dv_c((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2), 2)
                         / pow(f_dv_b((decision_order * NUM_INPUTS_RBF * NUM_RBF) + NUM_INPUTS_RBF * i + 2), 2);
#endif
        value += f_dv_w(decision_order * NUM_RBF + i) * exp(arg_exp);

    }
    // scale back to [0, NORMALIZE_SNOW_CONTRACT_SIZE]
    value = max(min((value + f_dv_a(decision_order)) * NORMALIZE_SNOW_CONTRACT_SIZE, NORMALIZE_SNOW_CONTRACT_SIZE), 0.);

    // enforce minimum contract size
    if (value < f_dv_d(decision_order) * NORMALIZE_SNOW_CONTRACT_SIZE){
        value = 0.0;
    }

//    printf("%f  %f  %f  %f  %f\n", f_fund_balance, f_debt, f_power_price_index, arg_exp, value);
    return value;
}



// calculate 2dv version (DPS_RUN_TYPE==0) withdrawal at end of year, using fund balance, cash flow, and max fund size as inputs.
double policyWithdrawal_2dv(const double f_fund_balance, const double f_cash_in, const double f_max_fund_size) {
    double withdrawal = 0.;
    if (f_cash_in < -EPS){  // negative cash_in, want to make a withdrawal
        if (f_fund_balance < EPS){  // is there non-zero fund for withdrawal?
            withdrawal = 0.;
        }else{
            withdrawal = min(-f_cash_in, f_fund_balance);   // if so, withdraw amount (-cash_in), or whole fund, whichever is smaller
        }
    }else{      // positive cash_in, want to deposit
        if (f_fund_balance > (f_max_fund_size - EPS)){  // is fund balance larger than allowed max?
            withdrawal = f_fund_balance - f_max_fund_size;  // if so, withdraw surplus
        }else{
            withdrawal = max(-(f_max_fund_size - f_fund_balance), -f_cash_in);   // else deposit up to max, or whole positive cash flow, whichever is smaller
        }
    }
    return withdrawal;
}



// calculate 2dv version (DPS_RUN_TYPE==0) snow contract value, ensuring >= MIN_SNOW_CONTRACT
double policySnowContractValue_2dv(const double f_value) {
    double value_edit;
    if (f_value < MIN_SNOW_CONTRACT){
        value_edit = 0.;
    }else{
        value_edit = f_value;
    }
    return value_edit;
}



// calculate 2dv version (DPS_RUN_TYPE==0) snow contract value, ensuring >= MIN_SNOW_CONTRACT
double policyMaxFund_2dv(const double f_value) {
    double value_edit;
    if (f_value < MIN_MAX_FUND){
        value_edit = 0.;
    }else{
        value_edit = f_value;
    }
    return value_edit;
}



int main(int argc, char* argv[]) {

    clock_t begin = clock();

#if BORG_RUN_TYPE > 0
    // setting random seeds
    seed_borg = atoi(argv[1]);
    seed_sample = atoi(argv[2]);      // use same seed for sample each time, so always comparing same simulations
    NFE = atoi(argv[3]);
    read_directory = argv[4];       // directory where input files are stored
    write_directory = argv[5];      // directory where "sets" and "runtime" folders are located, for writing output
#else
    seed_sample = atoi(argv[1]);
    LHC_set = atoi(argv[2]);
    read_directory = argv[3];       // directory where input files are stored
    set_file = argv[4];      // directory where "sets" and "runtime" folders are located, for writing output
    retest_file = argv[5];      // file "retest" files are writen
#endif

    // get stochastic inputs
    for (int i = 0; i < NUM_LINES_STOCHASTIC_INPUT; i++) {
        for (int j = 0; j < NUM_VARIABLES_STOCHASTIC_INPUT; j++) {
            stochastic_input[i][j] = 0.0;
        }
    }

    FILE *myfile;
    string dir_HHsamp = read_directory;
    myfile = fopen(dir_HHsamp.append("synthetic_data.txt").c_str(), "r");
    printf("%s", dir_HHsamp.c_str());

    int linenum = 0;
    char testbuffer[BUFFER_MAX_SIZE];

    if (myfile == NULL) {
        perror("Error opening HHsamp \n");
    } else {
        char buffer[BUFFER_MAX_SIZE];
        fgets(buffer, BUFFER_MAX_SIZE, myfile);       // eat header line
        while (fgets(buffer, BUFFER_MAX_SIZE, myfile) != NULL) {
            linenum++;
            if (buffer[0] != '#') {
                char *pStart = testbuffer;
                char *pEnd;
                for (int i = 0; i < BUFFER_MAX_SIZE; i++) {
                    testbuffer[i] = buffer[i];
                }
                for (int cols = 0; cols < NUM_VARIABLES_STOCHASTIC_INPUT; cols++) {
                    stochastic_input[linenum - 1][cols] = strtod(pStart, &pEnd);
                    pStart = pEnd;
//                    printf("%f ",stochastic_input[linenum-1][cols]);
                }
//                printf("\n");
            }
        }

    }
    fclose(myfile);

    // read in LHC dv
    FILE *myfile2;
    string dir_param = read_directory;
    myfile2 = fopen(dir_param.append("param_SFPUC_withLamPremShift.txt").c_str(), "r");
    char testbuffer2[BUFFER_MAX_SIZE];
    linenum = 0;

    if (myfile2 == NULL) {
        perror("Error opening param \n");
    } else {
        char buffer2[BUFFER_MAX_SIZE];
        fgets(buffer2, BUFFER_MAX_SIZE, myfile2);       // eat header line
        while (fgets(buffer2, BUFFER_MAX_SIZE, myfile2) != NULL) {
            linenum++;
            if (buffer2[0] != '#') {
                char *pStart2 = testbuffer2;
                char *pEnd2;
                for (int i = 0; i < BUFFER_MAX_SIZE; i++) {
                    testbuffer2[i] = buffer2[i];
                }
                for (int i = 0; i < NUM_PARAM; i++) {
                    param_LHC_sample[i][linenum - 1] = strtod(pStart2, &pEnd2);
                    pStart2 = pEnd2;
//                    if (linenum == 1){
//                        printf("%f  %d\n",param_LHC_sample[i][linenum-1], i);
//                    }
                }
            }
        }
    }
//    printf("\n");
    fclose(myfile2);

    // get samples from stochastic input file
    srand(seed_sample);
    for (int s = 0; s < NUM_SAMPLES; s++) {
        //choose NUM_SAMPLES number of samples (starting year out of NUM_YEARS). Cant be 0, since need power_price_index[t-1], and cant be less than NUM_YEARS from end
        lines_to_use[s] = rand() % (NUM_LINES_STOCHASTIC_INPUT - NUM_YEARS - 1) + 1;
//        printf("%d\n", lines_to_use[s]);
    }
#if BORG_RUN_TYPE == 2
    // interface with Borg-MS
    BORG_Algorithm_ms_startup(&argc, &argv);
    BORG_Algorithm_ms_max_evaluations(NFE);
    BORG_Algorithm_output_frequency(NFE / 200);
#endif
#if BORG_RUN_TYPE == 0      // get dv and params from argv, not borg, and just run once

    // read snow contract put strike premiums
    FILE *myfile4;
    string dir_set_read = set_file;
    myfile4 = fopen(dir_set_read.c_str(), "r");
    char testbuffer4[BUFFER_MAX_SIZE];
    linenum = 0;

    if (myfile4 == NULL) {
        perror("Error opening set file");
    } else {
        char buffer4[BUFFER_MAX_SIZE];
//        fgets(buffer4, BUFFER_MAX_SIZE, myfile4);       // eat header line
        while (fgets(buffer4, BUFFER_MAX_SIZE, myfile4) != NULL) {
            linenum++;
            if (buffer4[0] != '#') {
                char *pStart4 = testbuffer4;
                char *pEnd4;
                for (int i = 0; i < BUFFER_MAX_SIZE; i++) {
                    testbuffer4[i] = buffer4[i];
                }
                for (int i = 0; i < (NUM_DV + NUM_OBJECTIVES + NUM_CONSTRAINTS); i++) {
                    pareto[i][linenum - 1] = strtod(pStart4, &pEnd4);
                    pStart4 = pEnd4;
                }
            }
        }
    }
//    printf("\n");
    fclose(myfile4);
    N_pareto = linenum;

    // read snow contract put strike premiums
    ofstream retest_write;
    retest_write.open(retest_file.c_str(), ios::out | ios::trunc);
    retest_write << std::setprecision(10);

//    for (int i = 0; i < N_pareto; i++){
    for (int i = 2; i < 5; i++){
        // decision variables
        for (int j = 0; j < NUM_DV; j++) {
            problem_dv[j] = pareto[j][i];
        }

        // params from LHC sample
        cost_fraction = param_LHC_sample[0][LHC_set];             // fraction of MEAN_REVENUE that is must-meet costs
        double delta = param_LHC_sample[1][LHC_set];              // discount rate, as %/yr
        double Delta_interest_fund = param_LHC_sample[2][LHC_set];    // interest rate on reserve funds, as %/yr, markdown below delta (all negative)
        double Delta_interest_debt = param_LHC_sample[3][LHC_set];    // interest rate charged on debt, as %/yr, markup above delta (all positive)
        lambda_prem_shift = param_LHC_sample[5][LHC_set];         // shift in snow contract premium to apply, based on lambda parameter (relative to based premiums from lambda=0.25)

        // calculated params for LHC sensitivity analysis, used in portfolioProblem
        avg_surplus_revenue = MEAN_REVENUE * (1. - cost_fraction);
        discount_rate = 1. / (delta / 100. + 1.);
        interest_fund = (Delta_interest_fund + delta) / 100. + 1.;
        interest_debt = (Delta_interest_debt + delta) / 100. + 1.;

        // run dps with given dv
        portfolioProblem(problem_dv, problem_objs, problem_constraints);

        for (int j = 0; j < (NUM_DV); ++j) {
            retest_write << pareto[j][i] << " ";
        }
        for (int j = 0; j < NUM_OBJECTIVES; ++j) {
            retest_write << problem_objs[j] << " ";
        }
        for (int j = 0; j < NUM_CONSTRAINTS; ++j) {
            retest_write << problem_constraints[j] << "\n";
        }

    }
    retest_write.close();


#elif BORG_RUN_TYPE > 0
    // loop over uncertain parameters in LHC sample, borg each time

    for (int p = 0; p < NUM_PARAM_SAMPLES; ++p){
        LHC_set = p;
        // params from LHC sample
        cost_fraction = param_LHC_sample[0][p];             // fraction of MEAN_REVENUE that is must-meet costs
        double delta = param_LHC_sample[1][p];              // discount rate, as %/yr
        double Delta_interest_fund = param_LHC_sample[2][p];    // interest rate on reserve funds, as %/yr, markdown below delta (all negative)
        double Delta_interest_debt = param_LHC_sample[3][p];    // interest rate charged on debt, as %/yr, markup above delta (all positive)
        lambda_prem_shift = param_LHC_sample[5][p];         // shift in snow contract premium to apply, based on lambda parameter (relative to based premiums from lambda=0.25)

        // calculated params for LHC sensitivity analysis, used in portfolioProblem
        avg_surplus_revenue = MEAN_REVENUE * (1. - cost_fraction);
        discount_rate = 1. / (delta / 100. + 1.);
        interest_fund = (Delta_interest_fund + delta) / 100. + 1.;
        interest_debt = (Delta_interest_debt + delta) / 100. + 1.;

        // Define the problem with decisions, objectives, constraints and the evaluation function
        BORG_Problem problem = BORG_Problem_create(NUM_DV, NUM_OBJECTIVES, NUM_CONSTRAINTS, portfolioProblem);

#if DPS_RUN_TYPE>0
        // Set all the parameter bounds and epsilons
        for (int i = 0; i < dv_d.size(); i++){
            BORG_Problem_set_bounds(problem, i, 0, 1);            //threshold params  (dv_d)
        }
        for (int i = 0; i < dv_c.size(); i++){
            BORG_Problem_set_bounds(problem, dv_d.size() + i, -1, 1);             // dv_c
        }
        for (int i = 0; i < dv_b.size(); i++){
            BORG_Problem_set_bounds(problem, dv_d.size() + dv_c.size() + i, EPS, 1);             // dv_b (needs to be > 0 or numerical issues)
        }
        for (int i = 0; i < dv_w.size(); i++){
            BORG_Problem_set_bounds(problem, dv_d.size() + dv_c.size() + dv_b.size() + i, 0, 1);             // dv_w
        }
        for (int i = 0; i < dv_a.size(); i++){
            BORG_Problem_set_bounds(problem, dv_d.size() + dv_c.size() + dv_b.size() + dv_w.size() + i, -1, 1);            //const addition (dv_a)
        }
#else
        BORG_Problem_set_bounds(problem, 0, 0, NORMALIZE_FUND);            //bounds for contingency max size
        BORG_Problem_set_bounds(problem, 1, 0, NORMALIZE_SNOW_CONTRACT_SIZE);            //bounds for swap contract slope
#endif

        BORG_Problem_set_epsilon(problem, 0, EPS_ANNREV); // avg_annualized_adjusted_revenue (units $M, so 0.01=$10,000)
#if DPS_RUN_TYPE<2
        BORG_Problem_set_epsilon(problem, 1, EPS_MAXDEBT); // q95_max_debt (units fraction of avg_surplus_revenue, so 0.02=2%)
#else
        BORG_Problem_set_epsilon(problem, 1, EPS_MINREV); // q95_max_debt (units fraction of avg_surplus_revenue, so 0.02=2%)
#endif
#if NUM_OBJECTIVES > 2
        BORG_Problem_set_epsilon(problem, 2, EPS_MAXCOMPLEXITY); // max_hedge_complexity (unitless)
        BORG_Problem_set_epsilon(problem, 3, EPS_MAXFUND); // max_fund_balance (units $M)
#endif

#if BORG_RUN_TYPE == 1

        //This is set up to run only one seed at a time
        char output_filename[256];
        FILE *output_file = NULL;
        char *write_directory_cstr = new char [write_directory.length()+1];
        strcpy(write_directory_cstr, write_directory.c_str());
        delete[] write_directory_cstr;
        sprintf(output_filename, "%ssets/DPS_seedS%d_seedB%d.set", write_directory_cstr, seed_sample, seed_borg);

        BORG_Random_seed(seed_borg);
        BORG_Archive result = BORG_Algorithm_run(problem, NFE); // this actually runs the optimization

        //If this is the master node, print out the final archive
        if (result != NULL) {
            output_file = fopen(output_filename, "w");
            if (!output_file) {
                BORG_Debug("Unable to open final output file\n");
            }
            BORG_Archive_print(result, output_file);
            BORG_Archive_destroy(result);
            fclose(output_file);
        }

        BORG_Problem_destroy(problem);

#elif BORG_RUN_TYPE == 2
        //This is set up to run only one seed at a time
        char output_filename[256];
        char runtime[256];
        FILE *output_file = NULL;
        char *write_directory_cstr = new char [write_directory.length()+1];
        strcpy(write_directory_cstr, write_directory.c_str());
        sprintf(output_filename, "%ssets/DPS_seedS%d_seedB%d.set", write_directory_cstr, seed_sample, seed_borg);
        sprintf(runtime, "%sruntime/DPS_seedS%d_seedB%d.runtime", write_directory_cstr, seed_sample, seed_borg);
        delete[] write_directory_cstr;

        BORG_Algorithm_output_runtime(runtime);

        BORG_Random_seed(seed_borg);
        BORG_Archive result = BORG_Algorithm_ms_run(problem); // this actually runs the optimization

        //If this is the master node, print out the final archive
        if (result != NULL) {
            output_file = fopen(output_filename, "w");
            if (!output_file) {
                BORG_Debug("Unable to open final output file\n");
            }
            BORG_Archive_print(result, output_file);
            BORG_Archive_destroy(result);
            fclose(output_file);
        }

        BORG_Problem_destroy(problem);

#endif

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC / 60.0;
//        printf("%d\t%f\n", p, elapsed_secs);
    }
#endif

#if (BORG_RUN_TYPE==2)
    BORG_Algorithm_ms_shutdown();
#endif

    return EXIT_SUCCESS;

}


