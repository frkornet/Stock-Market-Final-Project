import pandas      as pd
import numpy       as np
from   frankie     import DayForecast, optimize, baseline, cross_val_tpr
from   datetime    import date, timedelta
from   scipy.stats import t
import yfinance    as yf
import matplotlib.pyplot as plt
import gc; gc.enable()

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer, load_iris, make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from category_encoders import WOEEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

from scipy.signal import savgol_filter, argrelmin, argrelmax

import math 
import random

import talib

import warnings
warnings.filterwarnings("ignore")

def ticker_stats(ticker, day, verbose):
    period = "10y" # "max"
    asset  = yf.Ticker(ticker)
    hist   = asset.history(period=period)
    col = f'perc_change{day}'
    hist[col] = 100*hist.Close.pct_change(day).shift(-day)
    
    idx = hist[col] <= 0
    neg_count = hist.loc[idx].count()[0]
    pos_count = hist.loc[~idx].count()[0]
    tot_count = len(hist)
    
    if verbose == True:
        print("TICKER:", ticker)
        print("="*(8+len(ticker)))
        print('')
        print('neg_count=', neg_count, f'neg_count %={neg_count}/{tot_count} = {neg_count/tot_count}')
        print('pos_count=', pos_count, f'pos_count %={pos_count}/{tot_count} = {pos_count/tot_count}')
        print('')
        print(hist[col].describe())
        print('')
        print(hist[col].tail(10))
        print('')
        hist[col].plot(figsize=(20,8))
        plt.title(ticker+f" ({day} day % change)")
        plt.show()
        print('')
    
    return hist

def smooth(hist):
    window = 15
    hist['smooth'] = savgol_filter(hist.Close, 2*window+1, polyorder=3)
    hist['smooth'] = savgol_filter(hist.smooth, 2*window+1, polyorder=1)
    hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=3)
    hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=1)
    return hist

def features(data, hist, target):
    
    windows = [3, 5, 10, 15, 20, 30, 45, 60]

    for i in windows:
        ma = data.Close.rolling(i).mean()
        # Moving Average Convergence Divergence (MACD)
        data['MACD_'+str(i)] = ma - data.Close
        data['PctDiff_'+str(i)] = data.Close.diff(i)
        data['StdDev_'+str(i)] = data.Close.rolling(i).std()

    factor = data.Close.copy()
    for c in data.columns.tolist():
        data[c] = data[c] / factor

    data[target] = hist[target]
    data = data.dropna()
    del data['Close']
    
    return data

def stringify(data):
    df = pd.DataFrame(data)
    for c in df.columns.tolist():
        df[c] = df[c].astype(str)
    return df

def print_ticker_heading(ticker):
    print("*******************")
    print("***", ticker, " "*6, "***" )
    print("*******************")
    print('')

def balanced_scorecard(data, target, verbose):
    # Pipeline for weighted evidence balanced scorecard
    encoder   = WOEEncoder()
    binner    = KBinsDiscretizer(n_bins=5, encode='ordinal')
    objectify = FunctionTransformer(func=stringify, check_inverse=False, validate=False)
    imputer   = SimpleImputer(strategy='constant', fill_value=0.0)
    clf       = LogisticRegression(class_weight='balanced', random_state=42)

    pipe = make_pipeline(binner, objectify, encoder, imputer, clf)

    # cross validate after initializing X and y
    used_cols = [c for c in data.columns.tolist() if c not in [target]]
    data[target] = data[target].apply(lambda x: 1 if x == 1 else 0)
    X, y = data[used_cols].values, data[target].values
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')

    if verbose == True:
        # print results of cross validation
        print('')
        print("mean cross_val_score=", scores.mean(), "std cross_val_score=", scores.std())
        print('\n')

    return pipe, X, y

def determine_minima_n_maxima(tickers, verbose):
    
    print("tickers=", tickers)
    min_indexes = []
    max_indexes = []

    for ticker in tickers:

        # free up memory
        gc.collect()

        if verbose == True:
            print_ticker_heading(ticker)

        # get stock data and smooth the Close curve
        hist = ticker_stats(ticker, 3, False)
        hist = smooth(hist)

        # set target and calculate mean Close
        target = 'target'
        hist[target] = 0
        mean_close = hist.Close.mean()

        # identify the minima of the curve (save a copy of the data for later)
        min_ids = argrelmin(hist.smooth.values)[0].tolist()
        if verbose == True:
            print("min_ids=", min_ids)
        min_indexes.append(min_ids)

        # identify the maxima and save a copy of the data for later processing
        max_ids = argrelmax(hist.smooth.values)[0].tolist()
        if verbose == True:
            print("max_ids=", max_ids)
        max_indexes.append(max_ids)

        # set the local minima to onbe and local maxima to minus one
        hist[target].iloc[min_ids] = 1
        hist[target].iloc[max_ids] = -1

        # plot the Close and smooth curve and the buy and sell signals 
        if verbose == True:
            plt.figure(figsize=(20,8))
            hist.Close.plot()
            hist.smooth.plot()
            (hist[target]*(mean_close/4)).plot()
            plt.title(ticker)
            plt.show()

        # NB: we do not include smooth in data!
        data = hist[['Close', 'Open', 'Low', 'High']]
        data = features(data, hist, target)

        if verbose == True:
            balanced_scorecard(data, target, verbose)
        
    return min_indexes, max_indexes


def align_minima_n_maxima(tickers, min_indices, max_indices, verbose):
    
    for i, ticker in enumerate(tickers):
        
        if len(min_indices[i]) == 0 or len(max_indices[i]) == 0:
            continue

        min_id = min_indices[i][0]
        max_id = max_indices[i][0]
        
        if verbose == True:
            print('Ticker:', ticker, "min_id=", min_id, "max_id=", max_id)

        modified=""
        while min_id > max_id:
            modified=" (*)"
            max_indices[i].pop(0)
            if len(max_indices) == 0:
                break
            else:
                max_id = max_indices[i][0]

        if verbose == True:
            print('     ', ticker, "min_id=", min_id, "max_id=", max_id, modified)
            
    return min_indices, max_indices


def plot_trades(tickers, min_indices, max_indices):
    
    for i, ticker in enumerate(tickers):
    
        gc.collect()

        print_ticker_heading(ticker)

        hist   = ticker_stats(ticker, 3, False)
        hist   = smooth(hist)
        slopes = hist.smooth.diff(1).copy()

        for j, buy_id in enumerate(min_indices[i]):

            buy_close = hist.Close.iloc[buy_id]
            buy_date  = hist.index[buy_id]
            print('   ', "buy_close=", buy_close, "buy_date=", buy_date, "buy_id=", buy_id)
            plt.plot(buy_date, buy_close, marker='o', markersize=8, color="red")

            if j < len(max_indices[i]):
                sell_id = max_indices[i][j]

                sell_close = hist.Close.iloc[sell_id]
                sell_date  = hist.index[sell_id]
                print('   ', "sell_close=", sell_close, "sell_date=", sell_date, "sell_id=", sell_id)

                gain       = sell_close - buy_close
                gain_pct   = round((gain / buy_close)*100, 1) 
                print('   ', "gain=", gain, f"gain_pct={gain_pct}%")

                plt.plot(sell_date, sell_close, marker='o', markersize=8, color="red") 

                subset_slopes = slopes[buy_id:sell_id]
                print('   ', "subset_slopes:")
                print('   ', "min=",  np.min(subset_slopes), 
                             "max=",  np.max(subset_slopes), 
                             "mean=", np.mean(subset_slopes),
                             "std=",  np.std(subset_slopes) )

                days_in_trade = sell_id - buy_id
                print('   ', "days in trade=", days_in_trade)
                daily_return = (1+gain_pct/100) ** (1/days_in_trade) - 1
                daily_return = round(daily_return * 100, 2)
                print('   ', f"daily compounded return={daily_return}%")

            else:
                sell_id = buy_id + 10

            sell_id = sell_id + 5
            if sell_id + 5 >= len(hist):
                sell_id = len(hist)

            buy_id = buy_id - 5
            if buy_id < 0:
                buy_id = 0

            print('   ', "buy_id=", buy_id, "sell_id=", sell_id)
            hist.Close.iloc[buy_id:sell_id].plot()
            plt.title(f"index: {buy_id}")
            plt.show()
    
        print('')

def split_data(stock_df, used_cols, target, train_pct):
    # set how many rows to discard (from start) and where test starts
    sacrifice = 0 # 50
    test_starts_at = int(len(stock_df)*train_pct)
    
    X = stock_df[used_cols].iloc[sacrifice:]
    y = stock_df[target].iloc[sacrifice:]

    X_train = stock_df[used_cols].iloc[sacrifice:test_starts_at]
    X_test  = stock_df[used_cols].iloc[test_starts_at:]
    y_train = stock_df[target].iloc[sacrifice:test_starts_at]
    y_test  = stock_df[target].iloc[test_starts_at:]
    
    return X, y, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print('Hello world')