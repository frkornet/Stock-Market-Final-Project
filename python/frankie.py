#
# First attempt at writing a class for forecasting stocks five days in the future.
# It will help me in understanding the code and making sure that I know how to
# put the key code in a class structure.
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
    def __init__(self, ticker, period, split_date, days, lags, verbose):
        self.ticker = ticker
        self.asset = yf.Ticker(self.ticker)
        self.hist  = self.asset.history(period=period)

        self.hist['Year']  = self.hist.index.year
        self.hist['Month'] = self.hist.index.month
        self.hist['DoW']   = self.hist.index.dayofweek

        self.hist['Test'] = (self.hist.index >= split_date).astype(np.int)
        self.split_date = self.hist[self.hist.Test==1].head(1).index[0]

        self.days=days
        self.data = 100*self.hist.Close.pct_change(self.days).dropna()

        self.lags = lags

        self.ar = ARX(self.data, lags=self.lags)
        self.ar.distribution = StudentsT()
        self.ar.volatility = GARCH(o=1, power=1)

        self.res = self.ar.fit(disp='off', last_obs=split_date, show_warning=False)
        if verbose == True:
            print(self.res.summary())

    def forecast(self, horizon=1, verbose=False):
        self.forecasts = self.res.forecast(horizon=horizon, start=self.split_date, method='simulation')

        self.horizons = self.forecasts.mean
        self.horizons['target'] = self.data

        preds = self.horizons.loc[self.horizons.index >= self.split_date]
        y_preds = preds['h.'+str(horizon)].apply(lambda x: 1 if x > 0 else 0)
        y_actuals = preds['target'].apply(lambda x: 1 if x > 0 else 0)

        conf_mat = confusion_matrix(y_actuals, y_preds)
        tn, fn, fp, tp = conf_mat.ravel()

        # horizons[split_date:].plot(figsize=(20,8))
        if verbose == True:
            print("tn=", tn, "fn=", fn, "tp=", tp, "fp=", fp)
            print('')
            print(conf_mat)
            print('')
            print("TP/(FN+TP)=", tp/(fn + tp), "FN+TP=", fn+tp)

            self.horizons[self.split_date:].plot(figsize=(20,8))
            plt.show()
            self.horizons.tail(20).plot(figsize=(20,8))
            plt.show()
            print(self.horizons.tail(20))

        return tp/(fn + tp)

def optimize(sdf, idx):
    day_list = [1, 3, 5, 10]

    for day in day_list:
        sdf['up'+str(day)+'_period'] = ''
        sdf['up'+str(day)+'_lags'] = ''
        sdf['up'+str(day)+'_tpr'] = 0

    lags_list = [ [1, 3],
                [1, 3, 5],
                [1, 3, 5, 10],
                [1, 3, 5, 10, 20],
                [1, 3, 5, 10, 20, 30],
                [1, 3, 5, 10, 20, 30, 45],
                [1, 3, 5, 10, 20, 30, 45],
                [1, 3, 5, 10, 20, 30, 45, 60]]

    period_list = [ "3y", "5y", "8y", "10y", "15y", "20y"]


    for i, ticker in zip(idx, sdf['TICKER'].loc[idx]):
        name_of_issuer = sdf['NAME_OF_ISSUER'].loc[i]
        print(f"{i} {ticker} ({name_of_issuer}):"); print("="*(7+len(ticker)+len(name_of_issuer)))
        print('')

        for day in day_list:
            print(f'{day}-day forecast:')

            best_lags = None
            best_period = None
            best_tp_rate = -1

            for period in period_list:
                # print('')
                # print("period:", period)
                # print('')
                for lags in lags_list: 
                    try:
                        gc.collect()  
                        myForecast = DayForecast(ticker=ticker, period=period, split_date="2019-07-01", 
                                                days=day, lags=lags, verbose=False)
                        tp_rate = myForecast.forecast(horizon=1, verbose=False)
                        # print('lags=', lags, 'tp_rate=', tp_rate)
                        if tp_rate > best_tp_rate:
                            best_period, best_lags, best_tp_rate = period, lags, tp_rate
                    except:
                        continue

            print('best_period=', best_period)
            print('best_tp_rate=', best_tp_rate)
            print("best_lags=", best_lags)
            print("")

            sdf.loc[sdf['TICKER'] == ticker, 'up'+str(day)+'_period'] = best_period
            sdf.loc[sdf['TICKER'] == ticker, 'up'+str(day)+'_lags']   = ','.join(map(str, best_lags))
            sdf.loc[sdf['TICKER'] == ticker, 'up'+str(day)+'_tpr']    = best_tp_rate

    print('')
    print('Stocks with optimal settings:')
    print('=============================')
    print('')
    print(sdf.loc[idx])

def baseline(sdf, idx):
    day_list = [1, 3, 5, 10]

    sdf['naive1_tpr'] = 0
    sdf['naive3_tpr'] = 0
    sdf['naive5_tpr'] = 0
    sdf['naive10_tpr'] = 0

    for i, ticker in zip(idx, sdf['TICKER'].loc[idx]):
        name_of_issuer = sdf['NAME_OF_ISSUER'].loc[i]
        print(f"{i} {ticker} ({name_of_issuer}):"); print("="*(7+len(ticker)+len(name_of_issuer)))
        print('')
        
        asset = yf.Ticker(ticker)
        hist  = asset.history(period="20y")
        print('period: 20y')
        for day in day_list:
            col = "target_"+str(day)+"d"
            hist[col] = 100*hist.Close.pct_change(day)
            hist[col] = hist[col].apply(lambda x: 1 if x > 0 else 0)
            
            som, lengte = hist[col].sum(), len(hist)
            naive_tpr = round(som / lengte, 4)
            
            print(f"{day}-day forecast naive TPR: {naive_tpr} (={som}/{lengte})")
        
        print('')
        print('period: >= 2019-07-01')
        hist = hist.loc[hist.index >= '2019-07-01']
        for day in day_list:
            col = "target_"+str(day)+"d"
            som, lengte = hist[col].sum(), len(hist)
            
            naive_tpr = round(som / lengte, 4)
            sdf['naive'+str(day)+"_tpr"].loc[i] = naive_tpr
            
            print(f"{day}-day forecast naive TPR: {naive_tpr} (={som}/{lengte})")
    
        print('')

def main():
    PATH = '/Users/frkornet/Flatiron/Stock-Market-Final-Project/'
    sdf = pd.read_csv(f'{PATH}data/stocks.csv')
    sdf = sdf.loc[sdf.TICKER > ''].reset_index()
    if 'index' in sdf.columns:
        del sdf['index']

    #idx = sdf.sample(10, random_state=42).index
    idx = sdf.index
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
    baseline(sdf, idx)
    print('')

    cols = ['TICKER', 'up1_tpr', 'naive1_tpr', 'up3_tpr', 'naive3_tpr', 
            'up5_tpr', 'naive5_tpr', 'up10_tpr', 'naive10_tpr']
    print(sdf[cols].loc[idx])
    sdf.loc[idx].to_csv(f'{PATH}data/optimal_params.csv')

if __name__ == "__main__":
    # execute only if run as a script
    main()


# FRK: old code for testing...
# myForecast = DayForecast(ticker="MSFT", period="10y", split_date="2019-07-01",
#                          days=5, lags=[1, 3, 5, 10, 20, 30],
#                          verbose=False)
# tp_rate = myForecast.forecast(horizon=1, verbose=False)
# print("tp_rate=", tp_rate)