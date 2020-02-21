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


# Constants
BUY  = 1
SELL = 2
TOLERANCE = 1e-6
STOP_LOSS = -10 # max loss: -10%

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

def smooth(hist, ticker):
    window = 15
    try:
        hist['smooth'] = savgol_filter(hist.Close, 2*window+1, polyorder=3)
    except:
        print(f"first savgol did converge for {ticker}!")
    
    try:
        hist['smooth'] = savgol_filter(hist.smooth, 2*window+1, polyorder=1)
    except:
        print(f"second savgol did converge for {ticker}!")
    
    try:
        hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=3)
    except:
        print(f"third savgol did converge for {ticker}!")

    try:    
        hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=1)
    except:
        print(f"fourth savgol did converge for {ticker}!")
    
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
        hist = smooth(hist, ticker)

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


def get_signals(hist, target, threshold):
    # NB: we do not include smooth in data!
    data = hist[['Close', 'Open', 'Low', 'High']]
    data = features(data, hist, target)

    used_cols = [c for c in data.columns.tolist() if c not in [target]]
    X, y, X_train, X_test, y_train, y_test = split_data(data, used_cols, target, 0.7)

    encoder   = WOEEncoder()
    binner    = KBinsDiscretizer(n_bins=5, encode='ordinal')
    objectify = FunctionTransformer(func=stringify, check_inverse=False, validate=False)
    imputer   = SimpleImputer(strategy='constant', fill_value=0.0)
    clf       = LogisticRegression(class_weight='balanced', random_state=42)

    pipe = make_pipeline(binner, objectify, encoder, imputer, clf)
    pipe.fit(X_train, y_train.values)

    signals = (pipe.predict_proba(X_test)  > threshold).astype(int)[:,1]
    return signals


def merge_buy_n_sell_signals(buy_signals, sell_signals):
    
    assert len(buy_signals) == len(sell_signals), "buy_signal and sell_signal lengths different!"
    
    buy_n_sell = [0] * len(buy_signals)
    length     = len(buy_n_sell)
    i          = 0
    state      = SELL
    
    while i < length:
        if state == SELL and buy_signals[i] == 1:
            state = BUY
            buy_n_sell[i] = 1
        
        elif state == BUY and sell_signals[i] == 1:
            state = SELL
            buy_n_sell[i] = 2
            continue
        
        i = i + 1
    
    return buy_n_sell

def extract_trades(hist, buy_n_sell, ticker, verbose):
    test_start_at = len(hist) - len(buy_n_sell)
    
    state       = SELL
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)
    
    for i, b_or_s in enumerate(buy_n_sell):
        
        if b_or_s == BUY:
            buy_id    = test_start_at + i
            buy_close = hist.Close.iloc[buy_id]
            buy_date  = hist.index[buy_id]
            state = SELL
            
        if b_or_s == SELL:
            sell_id    = test_start_at + i
            sell_close = hist.Close.iloc[sell_id]
            sell_date  = hist.index[sell_id] 
            
            gain = sell_close - buy_close
            gain_pct = round( (gain / buy_close)*100, 2)
            
            trading_days = sell_id - buy_id
            
            daily_return = (1+gain_pct/100) ** (1/trading_days) - 1
            daily_return = round(daily_return * 100, 2)
            
            trade_dict = {'buy_date'    : [buy_date],  'buy_close'    : [buy_close],
                         'sell_date'    : [sell_date], 'sell_close'   : [sell_close],
                         'gain_pct'     : [gain_pct],  'trading_days' : [trading_days],
                         'daily_return' : [daily_return], 'ticker' : ticker }
            possible_trades_df = pd.concat([possible_trades_df, 
                                           pd.DataFrame(trade_dict)])
            
            #$print("buy_id=",  buy_id,  "buy_close=",  buy_close,  "buy_date=", buy_date)
            #print("sell_id=", sell_id, "sell_close=", sell_close, "sell_date=", sell_date)
            #print("gain=", gain, f"gain_pct={gain_pct}%")
            #print("trading_days=", trading_days)
            #print(f"daily compounded return={daily_return}%")
            #print('')
    
    if verbose == True:
        print("****EXTRACT_TRADES****")
        print(possible_trades_df)
    
    return possible_trades_df

def get_possible_trades(tickers, threshold, verbose):
    
    print("tickers=", tickers)
    target = 'target'
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
        'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)
    
    for ticker in tickers:

        # free up memory
        gc.collect()

        if verbose == True:
            print_ticker_heading(ticker)

        # get stock data and smooth the Close curve
        hist = ticker_stats(ticker, 3, False)
        hist = smooth(hist, ticker)

        # get the buy signals
        hist[target] = 0
        min_ids = argrelmin(hist.smooth.values)[0].tolist()
        hist[target].iloc[min_ids] = 1        
        buy_signals = get_signals(hist, target, threshold)
        #print("buy_signals=", buy_signals, '\n')

        # get the sell signals
        hist[target] = 0
        max_ids = argrelmax(hist.smooth.values)[0].tolist()
        hist[target].iloc[max_ids] = 1
        sell_signals = get_signals(hist, target, threshold)
        #print("sell_signals=", sell_signals, '\n')
        
        # merge the buy and sell signals
        buy_n_sell = merge_buy_n_sell_signals(buy_signals, sell_signals)
        # print("buy_n_sell=", buy_n_sell, '\n')
        
        # extract trades
        ticker_df = extract_trades(hist, buy_n_sell, ticker, verbose)
        possible_trades_df = pd.concat([possible_trades_df, ticker_df])
    
    possible_trades_df.trading_days = possible_trades_df.trading_days.astype(int)
    return possible_trades_df

class Capital(object):
    def __init__(self):
        cols = [ 'date', 'capital', 'in_use', 'free']
        self.df = pd.DataFrame(columns=cols)
        
    def day_close (self, close_date, capital, in_use, free):

        assert capital >= 0, "capital needs to be zero or greater"
        assert in_use  >= 0, "in_use needs to be zero or greater"
        assert free    >= 0, "free needs to be zero or greater"
        
        assert abs(capital - in_use - free) < TOLERANCE, "capital and in_use + free deviating too much!"
        
        close_dict = { 'date'    : [close_date], 
                       'capital' : [capital],
                       'in_use'  : [in_use],
                       'free'    : [free]
                     }
        
        self.df = pd.concat( [self.df, pd.DataFrame(close_dict)] )

class PnL(object):
    def __init__(self, start_date, end_date, capital, in_use, free, max_stocks):
        
        cols = [ 'date', 'ticker', 'action', 'orig_amount', 'close_amount',
                 'no_shares', 'daily_gain', 'daily_return', 'invested']
        self.df = pd.DataFrame(columns=cols)

        self.myCapital  = Capital()   
        self.invested   = {}
        self.start      = start_date
        self.end        = end_date
        self.capital    = capital
        self.in_use     = in_use
        self.free       = free
        self.max_stocks = max_stocks
        
    def buy_stock (self, ticker, buy_date, sell_date, amount):
        
        assert amount > 0,                      f"amount ({amount}) needs to be greater than zero!"
        assert ticker not in self.invested,     f"already own shares in {ticker}!"
        assert len(self.invested) < self.max_stocks, f"already own maximum # stocks ({self.max_stocks})!"
        #assert self.free >= amount,             f"you do not have enough free cash to buy ({self.free})!"
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, "capital and in_use + free deviating too much!"
        
        # Make sure we have the money to buy stock
        if amount > self.free:
            if self.free >0:
                amount=self.free
                print(f"you do not have {amount} and setting amount to {self.free}")
            else:
                assert 0 == 1, "no money to buy anything!"

        # Retrieve the historical data for stock ticker and save it while we're invested
        asset  = yf.Ticker(ticker)
        hist   = asset.history(start=self.start, end=self.end)
        self.invested[ticker] = hist.copy()
        
        # Get share price and calculate how many shares we can buy
        idx = self.invested[ticker].index == buy_date
        share_price = float(self.invested[ticker].Close.loc[idx])
        no_shares = amount / share_price
        
        # Reduce free and increase in_use by amount
        self.free   = self.free - amount
        self.in_use = self.in_use + amount
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, "capital and in_use + free deviating too much!"
        
        # Store the buy action in self.df data frame
        buy_dict = {'date'         : [buy_date],
                    'ticker'       : [ticker],
                    'action'       : ['BUY'],
                    'orig_amount'  : [amount],
                    'close_amount' : [amount],
                    'no_shares'    : [no_shares],
                    'daily_gain'   : [0.0],
                    'daily_return' : [0.0],
                    'invested'     : 1
                   }
        
        buy_df = pd.DataFrame(buy_dict)
        self.df = pd.concat([self.df, buy_df])
     
    def sell_stock (self, ticker, sell_date):
        
        assert self.capital >= 0, "capital needs to be zero or greater"
        assert self.in_use  >= 0, "in_use needs to be zero or greater"
        assert self.free    >= 0, "free needs to be zero or greater"        
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, "capital and in_use + free deviating too much!"

        # Return if we do not own the stock (may be due to a forced stop-loss sales)
        if ticker not in self.invested:
            return 
        
        # Get the latest close_amount for ticker and no_shares owned
        idx           = (self.df.ticker == ticker) & (self.df.invested==1)
        no_shares     = float(self.df['no_shares'].loc[idx])
        close_amount  = float(self.df['close_amount'].loc[idx])
        orig_amount   = float(self.df['orig_amount'].loc[idx])
        self.df.loc[idx, 'invested'] = 0
        
        # Calculate how much the sell will earn
        idx           = self.invested[ticker].index == sell_date
        share_price   = float(self.invested[ticker].Close.loc[idx])
        today_amount  = no_shares * share_price
        delta_amount  = today_amount - close_amount
        delta_pct     = (delta_amount / close_amount) * 100

        # print the profit/loss of the trade
        print(f"profit of selling {ticker} on {sell_date}: ",
              f"{today_amount - orig_amount}", 
              f"{round(((today_amount - orig_amount)/orig_amount)*100,2)}%")
        
        # Correct in_use and capital for delta_amount
        self.capital  = self.capital + delta_amount
        self.in_use   = self.in_use  + delta_amount
        
        # Shift today's amount (in_use -> free)
        # We do not allow in_use to become negative, even if it is by
        # a small amount...
        self.in_use   = self.in_use - today_amount
        if self.in_use < 0:
            self.in_use = 0 
        self.free     = self.free   + today_amount

        if abs(self.capital - self.in_use - self.free) > TOLERANCE:
            print("self.capital=", self.capital)
            print("self.in_use=", self.in_use)
            print("self.free=", self.free)
            print("diff=", abs(self.capital - self.in_use - self.free))
            assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
                   "capital and in_use + free deviating too much!"
        
        # Save the stock sell
        sell_dict = {'date'        : [sell_date],
                    'ticker'       : [ticker],
                    'action'       : ['SELL'],
                    'orig_amount'  : [orig_amount],
                    'close_amount' : [today_amount],
                    'no_shares'    : [no_shares],
                    'daily_gain'   : [delta_amount],
                    'daily_return' : [delta_pct],
                    'invested'     : 0
                   }
        
        sell_df = pd.DataFrame(sell_dict)
        self.df = pd.concat([self.df, sell_df])

        # Remove stock from invested dictionary
        del self.invested[ticker]
        
    def day_close(self, close_date):
        
        tickers = list(self.invested.keys())
        for ticker in tickers:
            
            # Get the latest close_amount for ticker and no_shares owned
            df_idx        = (self.df.ticker == ticker) & (self.df.invested==1)
            # print("day close:", self.df.loc[df_idx])
            no_shares     = float(self.df['no_shares'].loc[df_idx])
            close_amount  = float(self.df['close_amount'].loc[df_idx])
            orig_amount   = float(self.df['orig_amount'].loc[df_idx])
            self.df.loc[df_idx, 'invested'] = 0

            # Calculate how much the sell will earn
            hist_idx      = self.invested[ticker].index == close_date
            share_price   = float(self.invested[ticker].Close.loc[hist_idx])
            today_amount  = no_shares * share_price
            delta_amount  = today_amount - close_amount
            delta_pct     = (delta_amount / close_amount) * 100

            # check if we reached a stop loss condition
            gain_pct = ((today_amount - orig_amount) / orig_amount) * 100
            if gain_pct < STOP_LOSS:
                print(f"breached stop-loss and selling {ticker}...")
                self.df.loc[df_idx, 'invested'] = 1
                self.sell_stock(ticker, close_date)
                continue

            # Correct in_use and capital for delta_amount
            self.capital  = self.capital + delta_amount
            self.in_use   = self.in_use  + delta_amount
            assert abs(self.capital - self.in_use - self.free) < TOLERANCE, "capital and in_use + free deviating too much!"


            close_dict = {'date'         : [close_date],
                         'ticker'       : [ticker],
                         'action'       : ['CLOSE'],
                         'orig_amount'  : [orig_amount],
                         'close_amount' : [today_amount],
                         'no_shares'    : [no_shares],
                         'daily_gain'   : [delta_amount],
                         'daily_return' : [delta_pct],
                         'invested'     : 1
                   }
        
            close_df = pd.DataFrame(close_dict)
            self.df = pd.concat([self.df, close_df])
            
        # Store overall end day result in myCapital
        self.myCapital.day_close(close_date, self.capital, self.in_use, self.free)

   

def backtester():

    # Read the data
    DATAPATH = '/Users/frkornet/Flatiron/Stock-Market-Final-Project/data/'
    sdf = pd.read_csv(f'{DATAPATH}optimal_params.csv')
    idx = (sdf.TICKER > '') & (sdf.TICKER != 'FNMB')
    sdf = sdf.loc[idx].reset_index()
    if 'index' in sdf.columns:
        del sdf['index']


    # Select the stocks we want to backtest
    exclude_list = ['FNWB', 'AIZ']
    tickers = []
    for ticker in sdf.TICKER.to_list():
        if ticker in exclude_list:
            continue
        tickers.append(ticker)
    print(f"Simulating {len(tickers)} stocks")

    # Determine for the selected stocks all possible trades
    min_indices, max_indices = determine_minima_n_maxima(tickers, False)
    min_indices, max_indices = align_minima_n_maxima(tickers, min_indices, max_indices, True)
    possible_trades_df = get_possible_trades(tickers, 0.5, False)

    # Create a dictionary that stores the mean gain_pct per ticker.
    # This controls whether backtester is willing to invest in the stock
    cols = ['ticker', 'trading_days', 'gain_pct', 'daily_return']
    mean_dict = possible_trades_df[cols].groupby('ticker').agg(['mean']).to_dict()

    # Determine start and end date for backtesting period.
    start_date, end_date = min(possible_trades_df.buy_date), max(possible_trades_df.sell_date)
    start_date = start_date - timedelta(5)
    end_date = end_date + timedelta(5)

    # Pull down MSFT stock for period and use that as basis for deteriming
    # the stock market trading days
    asset = yf.Ticker('MSFT')
    hist  = asset.history(period="max")
    idx = (hist.index >= start_date) & (hist.index <= end_date)
    backtest_trading_dates = hist.loc[idx].index.to_list()
    #print(backtest_trading_dates[0], backtest_trading_dates[-1])

    # Initialize the key variable
    capital           = 10000
    free              = 10000
    in_use            = 0
    max_stocks        = 5
    myCapital         = Capital()
    myPnL             = PnL(start_date, end_date, capital, in_use, free, max_stocks)

    # Sort the possible trades so they are processed in order
    i_possible_trades = 0
    possible_trades   = possible_trades_df.sort_values(by=['buy_date', 'gain_pct'], ascending=[True, False])
    possible_trades   = possible_trades.reset_index()

    sell_dates        = {}
    stocks_owned      = 0

    print("Days to simulate:", len(possible_trades))

    for trading_day, trading_date in enumerate(backtest_trading_dates):
        stocks_owned = len(myPnL.invested)

        #
        # Sell stocks if we have reached the sell_date
        #
        if trading_date in sell_dates:
            to_sell = sell_dates.pop(trading_date, [])
            for ticker in to_sell:
                if ticker in myPnL.invested:
                    print(f"*** selling {ticker} on {trading_date}")
                    myPnL.sell_stock(ticker, trading_date)
                    stocks_owned = len(myPnL.invested)

        #
        # Buy stocks if we have reached the buy_date
        #
        if i_possible_trades < len(possible_trades):
            buy_date  = possible_trades.buy_date.iloc[i_possible_trades]
            sell_date = possible_trades.sell_date.iloc[i_possible_trades]
            ticker    = possible_trades.ticker.iloc[i_possible_trades]
            
            while trading_date == buy_date:
                #
                # Determine what to do with the possible buy trade
                #
                if mean_dict[('gain_pct', 'mean')][ticker] > 0:
                    print(f"*** buying {ticker} on {buy_date} with target sell date of {sell_date}")
                    amount = myPnL.capital / max_stocks

                    # If we reached max_stocks, check if this stock is expected to
                    # perform better then lowest performing invested stock. If that is the case, 
                    # sell lowest expected performing stock, so that we can buy stock
                    if stocks_owned >= max_stocks:
                        expected_gain = mean_dict[('gain_pct', 'mean')][ticker]

                        lowest_expected_gain = None 
                        lowest_ticker        = None
                        for t in myPnL.invested.keys():
                            t_gain = mean_dict[('gain_pct', 'mean')][t]
                            if lowest_expected_gain is None or t_gain < lowest_expected_gain:
                                lowest_expected_gain = t_gain
                                lowest_ticker        = t
                        
                        if lowest_expected_gain is not None and expected_gain > lowest_expected_gain:
                            print(f"*** selling {lowest_ticker} on {trading_date} to free up money for {ticker}")
                            myPnL.sell_stock(lowest_ticker, trading_date)
                            stocks_owned = len(myPnL.invested)

                    if stocks_owned < max_stocks:
                        myPnL.buy_stock(ticker, buy_date, sell_date, amount)
                        stocks_owned = len(myPnL.invested)

                        # save the sell date for future processing
                        if sell_date in sell_dates:
                            sell_dates[sell_date].append(ticker)
                        else:
                            sell_dates[sell_date] = [ ticker ]
                    else:
                        print(f"maxed out: {ticker} is not expected to perform better than stocks already invested in")
                        print(f"invested in: {myPnL.invested.keys()} ({stocks_owned})")
                        print('')

                # move to next possible trading opportunity
                i_possible_trades = i_possible_trades + 1
                if i_possible_trades >= len(possible_trades):
                    break
                
                buy_date  = possible_trades.buy_date.iloc[i_possible_trades]
                sell_date = possible_trades.sell_date.iloc[i_possible_trades]
                ticker    = possible_trades.ticker.iloc[i_possible_trades]

        #
        # Post closing of the day
        # 
        # print(trading_date, stocks_owned, len(myPnL.invested), stocks_owned == len(myPnL.invested))
        #      myPnL.capital, myPnL.in_use, myPnL.free,
        #      abs(myPnL.capital - myPnL.in_use - myPnL.free),
        #      abs(myPnL.capital - myPnL.in_use - myPnL.free) < TOLERANCE)
        myPnL.day_close(trading_date)
        
    print(i_possible_trades, stocks_owned)
    return myPnL.df, myPnL.myCapital.df


if __name__ == "__main__":
    myPnL_df, myCapital_df = backtester()

    print(myPnL_df)
    print('')
    myCapital_df.index = myCapital_df.date

    print(myCapital_df)
    print('')

    to_plot_cols = ['capital', 'in_use']
    myCapital_df[to_plot_cols].plot(figsize=(18,10))
    plt.show()
    print('')