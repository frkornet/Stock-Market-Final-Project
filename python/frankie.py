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


def main():
    myForecast = DayForecast(ticker="MSFT", period="10y", split_date="2019-07-01",
                             days=5, lags=[1, 3, 5, 10, 20, 30, 45, 60, 90, 120],
                             verbose=False)
    tp_rate = myForecast.forecast(horizon=1, verbose=False)
    print("tp_rate=", tp_rate)

if __name__ == "__main__":
    # execute only if run as a script
    main()