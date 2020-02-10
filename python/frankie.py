#
# Initial version of forecasting class. It allows you to forecast the 
# percentage change 'n' days ahead. It is used to help in building
# a set of stock market "up" models (a notebook)
#
# Author:   Frank Kornet
# Date:     10 February 2020
# Version:  0.1
#


from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import pandas as pd
import numpy as np
import yfinance as yf

from pmdarima.datasets import load_msft
from pmdarima.arima import ADFTest
from pmdarima import tsdisplay
from pmdarima import plot_pacf
from pmdarima import auto_arima

from arch import arch_model
from arch.univariate import ARX, GARCH, StudentsT, Normal

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix

import gc; gc.enable()

class DayForecast(object):
    """
    The DayForecast class builds an ARX model and uses that to forecast a
    stock's percentage change in `Close` price. The class uses yfinance to
    retrieve stock data.

    The class consists of an initialize method and a method for forecasting.
    The initialize method builds the model that is then used subsequently 
    by the forecasting method. 
    """
    def __init__(self, ticker, period, split_date, cut_off_date, days, 
                 lags, verbose):
        """
        Initialize the 'n' Day Forecasting class. 

        Input arguments:

        - ticker:       The ticker symbol of the stock that will allow us 
                        to retrieve stock data using yfinance.
        - period:       The period for which we want to retrieve data. This
                        is typically specified in number of years, e.g. 
                        "10y" for ten years worth opf data.
        - split_date:   The date from where the test data starts. The
                        date does not need to be a "true" date on which the 
                        stock market is open. The code will find the closest
                        date when the market is open and use that as the 
                        split_date for determining where the test data starts.
        - cut_off_date: the date until how far we want to keep the data from
                        yfinance. This is used to cut off the data, so that we 
                        can test different periods for forecasting 
        - days:         a list of days. For each day in the list a separate
                        ARX model is built, and then used when `forecasting()`
                        is called.
        - lags:         a list with the lags used by the ARX mean model
        - verbose:      boolean whether we want to get summary output or
                        not. If True the res.summary() is printed. This allows
                        you to see what the coeeficients are for the different
                        ARX models.

        ReturnsL:       None 

        """

        self.ticker = ticker
        self.asset = yf.Ticker(self.ticker)
        self.hist  = self.asset.history(period=period)

        if cut_off_date != None:
            # print("cut_off_date: len(self.hist)=", len(self.hist))
            self.hist = self.hist.loc[self.hist.index < cut_off_date]
            # print("cut_off_date: len(self.hist)=", len(self.hist))
        
        self.hist['Test'] = (self.hist.index >= split_date).astype(np.int)
        self.split_date = self.hist[self.hist.Test==1].head(1).index[0]

        # Code block below does not do anything useful at this point, but 
        # helps me with debugging to determine the range of the stocks 
        # retrieved. So, I leave it in for now.
        min_date = min(self.hist.index)
        max_date = max(self.hist.index)
        min_year = min_date.year
        max_year = max_date.year

        # ARX uses a data column to build an ARCH model. We fit the ARCH 
        # model to build a model that can be used by `forecasting()`
        # to forecast percentage change from `split_date` onwards until
        # `cut_off_date`. To preserve these they are stored in separate
        # python dictionaries.
        self.days=days
        self.lags = lags
        self.data_dict = {}
        self.ar_dict = {}
        self.res_dict = {}
 
        for day in days:
            col = str(day)+"d"
            self.data_dict[col] = 100*self.hist.Close.pct_change(day).dropna()

            data = self.data_dict[col]
            ar = ARX(data, lags=self.lags)
            ar.distribution = StudentsT()
            ar.volatility = GARCH(o=1, power=1)
            self.ar_dict[col] = ar

            res = ar.fit(disp='off', last_obs=split_date, show_warning=False)
            self.res_dict[col] = res

            if verbose == True:
                print(self.res_dict[col].summary())

            gc.collect()
        
    def forecast(self, horizon=1, verbose=False):
        """
        Based upon the ARX models built in the initialize phase, forecast
        the percentage change. It does this for each day specified in the
        initialize step.

        Input arguments:

        - horizon:  arch.univariate argument `horizon`. According to ARCH
                    documentation this is "Number of steps to forecast".
                    For now code works with horizons=1. Other values
                    needs to be tested, so be careful setting it to a
                    different value at this stage.
        - verbose:  boolean whether or not to print a summary of the 
                    forecast. Note this is done for each day. It contains
                    a confusion matrix, as well as charts comparing 
                    forecasted data with target data.

        Returns:

        - tpr_list:     a list of TPRs. For each day it will include the 
                        calculated TPR.
        - forecast_df:  Pandas data frame that contains the forecasted data.
                        For each day it will provide four columns:
                        preds_#d:   the percentage change predicted
                        target#:    the target percentage change
                        y_actual#:  the target boolean indicator. 
                                    1=market went up ; 0=otherwise
                        y_pred#:    the forecasted boolean indicator
                                    1=market went up ; 0=otherwise

        """

        tpr_list = []
        forecast_df = pd.DataFrame()

        for day in self.days:
            gc.collect()
            col=str(day)+"d"
            self.forecasts = self.res_dict[col].forecast(horizon=horizon, 
                                            start=self.split_date, 
                                            method='simulation')

            pred_col      = "pred_"+str(day)+"d"
            target_col    = "target"+str(day)
            y_actuals_col = "y_actual"+str(day)
            y_preds_col   = "y_pred"+str(day)

            self.horizons = self.forecasts.mean
            self.horizons[pred_col] = self.horizons['h.1']
            self.horizons[target_col] = self.data_dict[col]

            preds = self.horizons.loc[self.horizons.index >= self.split_date]
            y_preds = preds[pred_col].apply(lambda x: 1 if x > 0 else 0)
            self.horizons[y_preds_col] = y_preds
            y_actuals = preds[target_col].apply(lambda x: 1 if x > 0 else 0)
            self.horizons[y_actuals_col] = y_actuals

            del self.horizons['h.1']
            if len(forecast_df) == 0:
                forecast_df = self.horizons
            else:
                forecast_df = pd.concat([forecast_df, self.horizons], axis=1)

            conf_mat = confusion_matrix(y_actuals, y_preds)
            tn, fn, fp, tp = conf_mat.ravel()
            tpr_list.append(tp/(fn + tp))

            if verbose == True:
                print(f'{day}-day TPR:')
                print('==========')
                print('')
                print("tn=", tn, "fn=", fn, "tp=", tp, "fp=", fp)
                print('')
                print(conf_mat)
                print('')
                print("TP/(FN+TP)=", tp/(fn + tp), "FN+TP=", fn+tp)

                pred_cols = []
                y_cols = []
                for col in self.horizons.columns:
                    if 'pred_' in col or 'target' in col:
                        pred_cols.append(col)
                    if 'y_' in col:
                        y_cols.append(col)
                
                self.horizons[pred_cols].loc[self.split_date:].plot(figsize=(20,8))
                plt.show()
                self.horizons[y_cols].loc[self.split_date:].plot(figsize=(20,8))
                plt.show()
                self.horizons[pred_cols].tail(20).plot(figsize=(20,8))
                plt.show()
                print(self.horizons.tail(20))

        return tpr_list, forecast_df;

#
# Constants used by optimize(), baseline(), and cross_val_tpr()
#
DAY_LIST    = [3, 5, 8, 10]
LAGS_LIST   = [ [1, 3],
                [1, 3, 5],
                [1, 3, 5, 10],
                [1, 3, 5, 10, 20],
                [1, 3, 5, 10, 20, 30],
                [1, 3, 5, 10, 20, 30, 45],
                [1, 3, 5, 10, 20, 30, 45, 60]]

PERIOD_LIST = [ "3y", "5y", "8y", "10y", "15y"]
MAX_PERIOD = "15y"

DATE_LIST   = [ ('2019-01-01', '2020-02-09'), ('2018-01-01', '2019-02-09'), 
                  ('2017-01-01', '2018-02-09'), ('2016-01-01', '2017-02-09'),
                  ('2015-01-01', '2016-02-09'), ('2014-01-01', '2015-02-09'),
                  ('2013-01-01', '2014-02-09'), ('2012-01-01', '2013-02-09'),
                  ('2011-01-01', '2012-02-09'), ('2010-01-01', '2011-02-09'), 
                  ('2009-01-01', '2010-02-09'), ('2008-01-01', '2009-02-09'), 
                  ('2007-01-01', '2008-02-09'), ('2006-01-01', '2007-02-09') 
                ]

def optimize(sdf, idx):
    """
    For each stock in stock data frame determine the optimal parameters.
    It does this as follows:
    - For each day specified in DAY_LIST determine the optimal parameters 
      by passing the days to DayForecast class initialization method. 
    - For each period in PERIOD_LIST it tries all of the lags combinations
      in LAG_LIST. As it tries the different combinations, `forecast()` keeps
      track of the best scores so far for each day.

    At the end it prints out for each day the optimal parameters. It also 
    updates the stock data frame `sdf` with the optimal parameters and the 
    best achieved TPR. This is stored for each day:

    - up#_period:   the optimal period for day #
    - up#_lags:     the optimal lags for day #
    - up#_tpr:      the TPR for period and lags (this is the best TPR 
                    achieved)

    Input arguments:

    - sdf:  the stock data frame, that contains the stocks we need to
            optimize parameters for. 
    - idx:  the stocks to optimize are identified by the Pandas index. 
            So, sdf.loc[idx] determines which stocks will be optimized.

    Returns: None

    """

    # Create columns to store results of `optimize()`` if needed
    for day in DAY_LIST:
        period_col = 'up'+str(day)+'_period'
        lags_col   = 'up'+str(day)+'_lags'
        tpr_col    = 'up'+str(day)+'_tpr'
        if period_col not in sdf.columns:
            sdf[period_col] = ''
        if lags_col not in sdf.columns:
            sdf[lags_col] = ''
        if tpr_col not in sdf.columns:
            sdf[tpr_col] = 0

    for i, ticker in zip(idx, sdf['TICKER'].loc[idx]):
        name_of_issuer = sdf['NAME_OF_ISSUER'].loc[i]
        print(f"{i} {ticker} ({name_of_issuer}):")
        print("="*(7+len(ticker)+len(name_of_issuer)))
        print('')

        best_tp_rate = {}
        best_lags    = {}
        best_period  = {}

        for day in DAY_LIST:
            col=str(day)+"d"
            best_lags[col] = None
            best_period[col] = None
            best_tp_rate[col] = -1

        # The main loop where we try different period and lags 
        # combinations to find the optimal params.
        for period in PERIOD_LIST:
            for lags in LAGS_LIST: 
                try:
                    gc.collect()  
                    myForecast = DayForecast(ticker=ticker, period=period, 
                                    split_date="2019-07-01", cut_off_date=None,
                                    days=DAY_LIST, lags=lags, verbose=False)
                    tp_rate, _ = myForecast.forecast(horizon=1, verbose=False)

                    for day, tpr in zip(DAY_LIST, tp_rate):
                        col = str(day) + "d"
                        if tpr > best_tp_rate[col]:
                           best_period[col]  = period
                           best_lags[col]    = lags
                           best_tp_rate[col] = round(tpr,4)
                except:
                    continue
        
        for day in DAY_LIST:
            dcol = str(day) + "d"
            print(f'{day}-day forecast:')
            print('best_period=', best_period[dcol])
            print("best_lags=", best_lags[dcol])
            print('best_tp_rate=', best_tp_rate[dcol])
            print("")

            ucol = 'up'+str(day)
            tidx = sdf['TICKER'] == ticker
            sdf.loc[tidx, ucol+'_period'] = best_period[dcol]
            sdf.loc[tidx, ucol+'_lags']   = ','.join(map(str, best_lags[dcol]))
            sdf.loc[tidx, ucol+'_tpr']    = round(best_tp_rate[dcol], 4)

    print('')
    print('Stocks with optimal settings:')
    print('=============================')
    print('')
    print(sdf.loc[idx])

def baseline(sdf, idx, from_date, end_date):
    """
    Determines for each stock in stock data frame (`sdf`) specified by `idx`
    the naive TPR over the period from `from_date` to `end_date`. This allows
    you to put the optimal parameters in perspective. The best TPR should at 
    least exceed naive TPR. This is calculated for each day.

    Input arguments:

    - sdf:          the stock data frame
    - idx:          index of the stocks for which we need to calculate
                    naive TPRs
    - from_date:    the start date over which the naive TPR needs to be
                    calculated. Format is 'yyyy-mm-dd' (string).
    - end_date:     the end date over which the naive TPR needs to be
                    calculated. Format is 'yyyy-mm-dd' (string). 

    Returns: None

    """

    day_list = DAY_LIST

    sdf['naive1_tpr'] = 0
    sdf['naive3_tpr'] = 0
    sdf['naive5_tpr'] = 0
    sdf['naive10_tpr'] = 0

    for i, ticker in zip(idx, sdf['TICKER'].loc[idx]):
        name_of_issuer = sdf['NAME_OF_ISSUER'].loc[i]
        print(f"{i} {ticker} ({name_of_issuer}):") 
        print("="*(7+len(ticker)+len(name_of_issuer)))
        print('')
        
        asset = yf.Ticker(ticker)
        hist  = asset.history(period="max")        
        print('')
        print('period:', from_date, '-', end_date)
        hist = hist.loc[(hist.index >= from_date) & (hist.index <= end_date)]
        for day in day_list:
            col = "target_"+str(day)+"d"
            hist[col] = 100*hist.Close.pct_change(day).dropna()
            hist[col] = hist[col].apply(lambda x: 1 if x > 0 else 0)
            som, lengte = hist[col].sum(), len(hist)
            
            naive_tpr = round(som / lengte, 4)
            sdf.loc[i, 'naive'+str(day)+"_tpr"] = naive_tpr
            
            print(f"{day}-day forecast naive TPR: {naive_tpr} (={som}/{lengte})")
    
        print('')

def cross_val_tpr(sdf, idx):
    """
    Calculate the TPR for the selected stocks over different periods to
    provide a range of TPR of the optimal models under different market 
    conditions. The periods to be tested are specified by PERIOD_LIST.
    It goes back until 2005 to ensure that we see how the model behaves
    under the stock crash of 2007  2008. 

    Input arguments:

    - sdf:          the stock data frame
    - idx:          index of the stocks for which we need to calculate
                    naive TPRs
    
    Returns: None
    """
    date_list = DATE_LIST
    day_list  = DAY_LIST

    for i, ticker in zip(idx, sdf['TICKER'].loc[idx]):
        name_of_issuer = sdf['NAME_OF_ISSUER'].loc[i]
        print(f"{i} {ticker} ({name_of_issuer}):") 
        print("="*(7+len(ticker)+len(name_of_issuer)))
        print('')

        for day in day_list:
            ucol = 'up'+str(day)
    
            period = sdf[ucol+'_period'].loc[i]
            lags   = [ int(x) for x in sdf[ucol+'_lags'].loc[i].split(',') ]
            tpr    = sdf[ucol+'_tpr'].loc[i]

            tpr_list = []
            for split_date, cut_off_date in date_list:
                try:
                    gc.collect()
                    myForecast = DayForecast(ticker=ticker, 
                                            period=MAX_PERIOD, 
                                            split_date=split_date, 
                                            cut_off_date=cut_off_date,
                                            days=[day], 
                                            lags=lags, verbose=False)

                    tp_rate, _ = myForecast.forecast(horizon=1, verbose=False)
                    tpr_list.append(round(tp_rate[0],4))

                except:
                    continue

            print(f'{day}-day forecast: {tpr_list}')

            # Exclude -1s from the mean and std calculations
            mlist = []
            for tpr in tpr_list:
                if tpr == -1:
                    continue
                mlist.append(tpr)
            
            if len(mlist) > 0:
                print(f"   min={min(tpr_list)} max={max(tpr_list)}")
                print(f"   mean={round(sum(mlist)/len(mlist), 4)}  std={np.std(mlist)}")
            else:
                print(f'   no valid data found...')
            
            print('')

def main():
    DATAPATH = '/Users/frkornet/Flatiron/Stock-Market-Final-Project/data/'
    sdf = pd.read_csv(f'{DATAPATH}stocks.csv')
    # sdf = sdf.loc[sdf.TICKER > ''].reset_index()
    sdf = sdf.loc[sdf.TICKER == 'SPPI'].reset_index()
    if 'index' in sdf.columns:
        del sdf['index']

    # running out of battery - need stronger charger (in the office)
    idx = sdf.index 
    #idx = sdf.sample(2, random_state=42).index
    print(sdf.loc[idx])
   
    print('')
    print('OPTIMIZE PARAMS:')
    print('================')
    print('')
    optimize(sdf, idx)
    print('')

    print('')
    print('BASELINE TPR:')
    print('=============')
    print('')
    baseline(sdf, idx, '2019-07-01', '2020-02-09')
    print('')

    cols = ['TICKER', 'up3_tpr', 'naive3_tpr', 'up5_tpr', 'naive5_tpr', 
            'up8_tpr', 'naive8_tpr', 'up10_tpr', 'naive10_tpr']
    print(sdf[cols].loc[idx])
    # sdf.loc[idx].to_csv(f'{PATH}optimal_params.csv')

    print('')
    print('CROSS VALIDATION TPR:')
    print('=====================')
    print('')
    cross_val_tpr(sdf, idx)
    print('')

if __name__ == "__main__":
    # Execute only if run as a script
    main()
