import pandas      as pd
import numpy       as np
from   frankie     import DayForecast, optimize, baseline, cross_val_tpr
from   datetime    import date, timedelta
from   scipy.stats import t
import yfinance    as yf
import matplotlib.pyplot as plt
import gc; gc.enable()

import math 
import random

import warnings
warnings.filterwarnings("ignore")

# Constants
SIMULATIONS    = 1000

# Helper function: random Student T value (leverages t.rvs() from scipy)
# nu equals number of degrees of freedom
def student_t(nu, n=1): 
    if n==1:
        return t.rvs(nu, size=1)[0]
    return t.rvs(nu, size=n)


def forecast_future(ticker, day, period, lags, split_date, cut_off_date, verbose):

    # Create an DayForecast instance for MSFT stock and extract the fitted 
    # parameters into dedicated lists
    myForecast = DayForecast(ticker=ticker, period=period, 
                    split_date=split_date, cut_off_date=cut_off_date,
                    days=[day], lags=lags, verbose=False)
    params = myForecast.res_dict[str(day)+'d'].params

    mu_names = [ 'Const']
    mu_coefs = [ float(params[params.index == 'Const']) ]
    mu_lags = [0]

    for arg_name in params.index:
        if 'Close' in arg_name:
            mu_names.append(arg_name)
            index = int(arg_name[6:-1])
            mu_lags.append(index)
            mu_coefs.append( float(params[params.index == arg_name]) )

    max_lag = max(mu_lags)

    sigma_names = ['omega', 'alpha[1]', 'gamma[1]', 'beta[1]']
    sigma_coefs = [ float(params[params.index == name]) for name in sigma_names ]

    nu = float(params[params.index == "nu"])

    if verbose == True:
        print("mu_names    =", mu_names)
        print("mu_coefs    =", mu_coefs)
        print("mu_lags     =", mu_lags)
        print("max_lag     =", max_lag)
        print("sigma_names =", sigma_names)
        print("sigma_coefs =", sigma_coefs)
        print("nu          =", nu)
        print('')

    # Forecast the percentage change for the test period starting from split_date (above)
    tp_rate, forecast_df = myForecast.forecast(horizon=1, verbose=False)
    if verbose == True:
        print('tp_rate=', tp_rate)
        print("forecast_df after call to forecast():\n")
        print(forecast_df.tail(10))
        print('')

    # Set up column names so we don't have to recompute them over again and again
    target_col   = "target"    +str(day)
    mu_col       = "mu_t_"     +str(day)+"d"
    e_col        = "e_t_"      +str(day)+"d"
    eps_col      = "eps_t_"    +str(day)+"d"
    forecast_col = "forecast_" +str(day)+"d"
    pred_col     = "pred_"     +str(day)+"d"
    resid_col    = "resid_"    +str(day)+"d"

    # Make a local copy of hist and add columns that are needed for the prediction 
    # calculations
    hist              = myForecast.hist.loc[myForecast.hist.index < myForecast.split_date]
    hist[target_col]  = 100 * hist.Close.pct_change(day)
    hist              = hist.dropna()

    if verbose == True:
        print('hist data frame (hist_df after creating target3 column:\n')
        print(hist.tail(max_lag))
        print('')

    # Create my forecast data frame containing the last `max_lag` rows of `hist`
    # data frame and the predicted values from forecast() (i.e. `forecast_df`)
    hist[pred_col] = 0
    mf_df = pd.concat([hist[[target_col, pred_col]].iloc[-max_lag:], 
                    forecast_df[[target_col, pred_col]].loc[forecast_df.index >= myForecast.split_date]], 
                    axis=0)

    mf_df               = mf_df.reset_index()
    mf_df[mu_col]       = mf_df[target_col]
    mf_df[forecast_col] = 0

    # Set initial values
    random.seed(42)
    start_index      = max_lag - 1
    mf_df[resid_col] = mf_df[target_col] - mf_df[pred_col]
    sigma_0          = np.std(mf_df[resid_col].iloc[start_index:])
    e_0              = student_t(nu)

    sigma_t_minus_1  = sigma_0
    eps_t_minus_1    = sigma_0 * e_0
    indicator_t      = np.sign(eps_t_minus_1)

    # The model below implements:
    # 
    # r_t         = mu_t + eps_t
    # mu_t        = const + close_1 * mu_{t-1} + close_3 * mu_{t-3} + 
    #               close_5 * mu_{t-5} + close_10 * mu_{t-10}
    # e_t ~ T(nu)
    # indicator_t = 1 if eps_{t-1} < 0 else 0
    # sigma_t     = omega + alpha * abs(eps_{t-1}) 
    #             + gamma * abs(eps_{t-1}) * indicator_t + beta * sigma{t-1}
    # eps_t       = sigma_t * e_t 
    #
    # source: https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html) 
    #
    # NB: in the code below forecast_t is used instead of r_t
    #

    for index in range(max_lag, len(mf_df)+1):
        # Calculate mu_t
        mu_t_minus = [ float(mf_df[mu_col].iloc[index-lag]) for lag in mu_lags[1:] ]
        mu_t = mu_coefs[0] + np.dot(mu_t_minus, mu_coefs[1:])

        # Calculate sigma_t
        values          = [1, abs(eps_t_minus_1), abs(eps_t_minus_1) * indicator_t, 
                        sigma_t_minus_1]
        sigma_t         = np.dot(sigma_coefs, values )

        # Calculate forecast_t SIMULATIONS times
        e_t        = student_t(nu, SIMULATIONS)
        eps_t      = e_t * sigma_t
        forecast_t = np.array([mu_t] * SIMULATIONS) + eps_t

        # Save the results. The forecast needs to be saved at index - 1
        # as the prediction is for today needs to be made the day before.
        sigma_t_minus_1                     = sigma_t
        eps_t_minus_1                       = np.mean(eps_t)
        indicator_t                         = np.sign(eps_t_minus_1) 
        if index - day -1 >= 0:
            mf_df[forecast_col].iloc[index - day - 1] = np.mean(forecast_t)
        
        if verbose == True:
            print("index=", index, 
                  "forecast_t (mean)=", np.mean(forecast_t), 
                  "saved index=", index-day)

    mf_df.index = mf_df['Date']

    if verbose == True:
        print('\nmf_df.tail(20):\n')
        print(mf_df.tail(20))
        print('')

    # Shift the target, prediction, and mu columns up n days
    mf_df[target_col] = mf_df[target_col].shift(-day)
    mf_df[pred_col] = mf_df[pred_col].shift(-day)
    mf_df[mu_col] = mf_df[mu_col].shift(-day)

    # Now do the forward calculation for n days in the future
    # Note that this saves the mu_t value as that is needed for the next
    # day calculation
    for index in range(-day, 0):
        # Calculate mu_t
        mu_t_minus = [ float(mf_df[mu_col].iloc[index-lag]) for lag in mu_lags[1:] ]
        mu_t = mu_coefs[0] + np.dot(mu_t_minus, mu_coefs[1:])

        # Calculate sigma_t
        values          = [1, abs(eps_t_minus_1), abs(eps_t_minus_1) * indicator_t, 
                        sigma_t_minus_1]
        sigma_t         = np.dot(sigma_coefs, values )

        # Calculate forecast_t SIMULATIONS times
        e_t        = student_t(nu, SIMULATIONS)
        eps_t      = e_t * sigma_t
        forecast_t = np.array([mu_t] * SIMULATIONS) + eps_t

        # Save the results. The forecast needs to be saved at index - 1
        # as the prediction is for today needs to be made the day before.
        sigma_t_minus_1                 = sigma_t
        eps_t_minus_1                   = np.mean(eps_t)
        indicator_t                     = np.sign(eps_t_minus_1) 
        mf_df[forecast_col].iloc[index] = np.mean(forecast_t)
        mf_df[mu_col].iloc[index]       = mu_t

        if verbose == True:
            print("index=", index, 
                  "mu_t_minus=", mu_t_minus, 
                  "forecast_t (mean)=", np.mean(forecast_t), 
                  "saved index=", index)

    mf_df.index = mf_df['Date']
    if verbose == True:
        print("len(mf_df)=", len(mf_df))

        print('\nmf_df.tail(20):\n')
        print(mf_df.tail(20))
        print('')

        mf_df[[target_col, forecast_col, pred_col]].tail(20).plot(figsize=(20,8))
        plt.show()

    return tp_rate, mf_df

if __name__ == '__main__':
    DATAPATH = '/Users/frkornet/Flatiron/Stock-Market-Final-Project/data/'
    stocks_df = pd.read_csv(f'{DATAPATH}optimal_params.csv')
    print(stocks_df)
    print(stocks_df.columns)

    # Determine split_date (30 days earlier than today)
    today = date.today()
    split_date = str(today - timedelta(30*6))
    print("split_date=", split_date)

    ticker = 'SPPI'
    tp_rate, mf_df = forecast_future(ticker=ticker, day=5, period='3y', lags=[1, 3], 
                                    split_date=split_date, cut_off_date=None, verbose=False)
    print("tp_rate=", tp_rate)
    print("mf_df.tail(20)=\n", mf_df.tail(20))
    
    mf_df[['target5', 'forecast_5d', 'pred_5d']].tail(20).plot(figsize=(20,8))
    plt.title(ticker)
    plt.show()

    ticker = 'MBI'
    tp_rate, mf_df = forecast_future(ticker=ticker, day=8, period='8y', lags=[1, 3], 
                                    split_date=split_date, cut_off_date=None, verbose=False)
    print("tp_rate=", tp_rate)
    print("mf_df.tail(20)=\n", mf_df.tail(20))
    
    mf_df[['target8', 'forecast_8d', 'pred_8d']].tail(20).plot(figsize=(20,8))
    plt.title(ticker)
    plt.show()