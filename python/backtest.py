#
# Stockie - Stock Trading System
#
# Initial version of backtesting module. It allows you to backtest as well as 
# to get buy and sell recommendations (still to be implemented as part of final
# project). The backtest_xxx.ipynb notebooks depend on the functionality provided
# by this module.
#
# Author:   Frank Kornet
# Date:     24 February 2020
# Version:  0.1
#

import pandas                as pd
import numpy                 as np
from   datetime              import date, timedelta
from   scipy.stats           import t
import yfinance              as yf
import matplotlib.pyplot     as plt
import gc; gc.enable()
from   tqdm                  import tqdm
import time
import logging

from sklearn.linear_model    import LogisticRegression
from category_encoders       import WOEEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline        import make_pipeline, Pipeline
from sklearn.preprocessing   import KBinsDiscretizer, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.impute          import SimpleImputer

from scipy.signal            import savgol_filter, argrelmin, argrelmax
from datetime                import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

# Constants
BUY       = 1
SELL      = 2
TOLERANCE = 1e-6
STOP_LOSS = -10 # max loss: -10%
DATAPATH  = '/Users/frkornet/Flatiron/Stock-Market-Final-Project/data/'
LOGPATH   = '/Users/frkornet/Flatiron/Stock-Market-Final-Project/log/'

def log(msg, both=False):
    logging.info(msg)
    if both == True:
        print(msg)

def get_hist(ticker, period):
    """
    Retrieve historical stock date from yfinance and return data to caller.
    """
    asset  = yf.Ticker(ticker)
    hist   = asset.history(period=period)
    
    return hist

def smooth(hist, ticker):
    """
    Smooth the Close price curve of hist data frame returned by yfinance. Two
    values are returned. The first is whether or not the smooth was successful 
    (True is successful and False is unsuccessful). The second value is the
    hist dataframe with an extra column smooth containing the smoothed Close
    curve.
    """
    window = 15

    try:
        #print(ticker)
        hist['smooth'] = savgol_filter(hist.Close, 2*window+1, polyorder=3)
        hist['smooth'] = savgol_filter(hist.smooth, 2*window+1, polyorder=1)
        hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=3)   
        hist['smooth'] = savgol_filter(hist.smooth, window, polyorder=1)
        return True, hist
    except:
        #print(f"Failed to smooth prices for {ticker}!")
        return False, hist

def features(data, hist, target):
    """
    Given a standard yfinance data dataframe, add features that will help
    the balanced scorecard to recognize buy and sell signals in the data.
    The features are added as columns in the data dataframe. 
    
    The original hist dataframe from yfinance is provided, so we can copy
    the target to the data dataframe. The data dataframe with the extra 
    features is returned. The target argument contains the name of the 
    column that contains the the target.
    """
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
    """
    Convert a Pandas dataframe with numeric columns into a dataframe with only
    columns of the data type string (in Pandas terminology an object). The 
    modified dataframe is returned. Note that the original dataframe is lost.
    """
    df = pd.DataFrame(data)
    for c in df.columns.tolist():
        df[c] = df[c].astype(str)
    return df

def print_ticker_heading(ticker):
    """
    Print a heading for the ticker on the console. Nothing is returned. Is
    a helper function only to avoid having to duplicate the same code over
    and over again. 
    """
    print("*******************")
    print("***", ticker, " "*6, "***" )
    print("*******************")
    print('')

def balanced_scorecard(data, target, verbose):
    """
    Create a balanced weight of evidence scorecard. The scorecard is put into
    a sklearn pipeline, and then cross validated. The scores are printed if
    verbose is set to True. Otherwise the scores are not printed.

    The balanced scorecard returns the pipe, as well as the X (features) and
    y (target) values. Note that these are no longer Pandas dataframe columns.
    Instead they are numpy series.
    """

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

def check_min_index(min_index, hist):
    """
    """
    assert min_index >= 0 and min_index < len(hist), "index out of bound!"

    while min_index < len(hist)-1:
        next_close    = hist.Close.iloc[min_index + 1]
        current_close = hist.Close.iloc[min_index]
        up_tomorrow = next_close > current_close
        if up_tomorrow == True:
            return min_index
        min_index = min_index + 1
    
    return min_index

def check_max_index(max_index, hist):
    """
    """
    assert max_index >= 0 and max_index < len(hist), "index out of bound!"

    while max_index < len(hist)-1 and max_index > 0:
        next_close    = hist.Close.iloc[max_index + 1]
        current_close = hist.Close.iloc[max_index]
        down_tomorrow = next_close < current_close
        if down_tomorrow == True:
            return max_index
        max_index = max_index + 1
    
    return max_index

def determine_minima_n_maxima(tickers, period, verbose):
    """
    For a list of tickers determine their local minima and local maxima. 
    The function loops through the tickers and does the following:
    (1) retrieve the stock data for the ticker,
    (2) smooth the Close price curve,
    (3) use argrelmin and argrelmax to determine the local minima and ]
        maxima of the smoothed curve, and
    (4) if verbose is True, a target column is created with 1 for local 
        minima and -1 for local maxima and then the Close price curve,
        the smoothed curve and the target are plotted.

    The function returns the local minima indices, the local maxima 
    indices, and a list of tickers for which we are unable to smooth
    the Close price curve.
    """
    
    #print("tickers=", tickers)
    min_indexes    = []
    max_indexes    = []
    failed_tickers = []

    #print('Determining local minima and maxima...\n')
    for ticker in tqdm(tickers, desc="local minima and maxima: "):

        # free up memory
        gc.collect()

        if verbose == True:
            print_ticker_heading(ticker)

        # get stock data and smooth the Close curve
        hist = get_hist(ticker, period)
        success, hist = smooth(hist, ticker)
        if success == False:
            failed_tickers.append(ticker)
            continue

        # set target and calculate mean Close
        target = 'target'
        hist[target] = 0
        mean_close = hist.Close.mean()

        # identify the minima of the curve (save a copy of the data for later)
        min_ids = argrelmin(hist.smooth.values)[0].tolist()
        if verbose == True:
            print("min_ids=", min_ids)

        # Make sure that the next day stock is not going down further
        checked_min_ids = []
        for min_index in min_ids:
            updated_index = check_min_index(min_index, hist)
            checked_min_ids.append(updated_index)

        min_indexes.append(checked_min_ids)

        # identify the maxima and save a copy of the data for later processing
        max_ids = argrelmax(hist.smooth.values)[0].tolist()
        if verbose == True:
            print("max_ids=", max_ids)

        # Make sure that the next day stock is not going up further
        checked_max_ids = []
        for max_index in max_ids:
            updated_index = check_max_index(max_index, hist)
            checked_max_ids.append(updated_index)

        max_indexes.append(checked_max_ids)

        # plot the Close and smooth curve and the buy and sell signals 
        if verbose == True:

            hist[target].iloc[min_ids] = 1
            hist[target].iloc[max_ids] = -1

            plt.figure(figsize=(20,8))
            hist.Close.plot()
            hist.smooth.plot()
            (hist[target]*(mean_close/4)).plot()
            plt.title(ticker)
            plt.show()
        
    return min_indexes, max_indexes, failed_tickers


def align_minima_n_maxima(tickers, min_indices, max_indices, verbose):
    """
    For each ticker ensure that the index of the first buy (in min_indices) 
    is larger than the first sell (in max_indices), since the backtester 
    cannot sell what it has not yet bought. 

    The sell actions at the beginning (i.e. the first entries 
    from max_indices) are dropped until the first buy comes before any 
    sell action.

    Note that min_indices[i] contains the list of local minima indices
    for ticker i, and max_indices[i] contains the list of local maxima
    indices for ticker i.

    The function returns min_indices, max_indices where max_indices has 
    potentially been changed.
    """
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


def plot_trades(tickers, min_indices, max_indices, period):
    """
    For a list of tickers, local minima, and local maxima determine the 
    gain/loss of each individual trade. Note that the local minima
    and local maxima are based upon the smoothed curve. The profit/gain
    is based upon the actual Close price. So, there may be some
    discrepencies. The function is mainly used to visualize the trades 
    and is mostly used in testing settings.

    The function draws for each trade a plot, with a red dot indicating 
    the starting point of the trade on the Close price curve, and a
    second red dot indicating the sell of the stock ticker.

    Note this function is only practical for small number of trades. 
    Even a similation of 100 stocks easily gets a few hundred 
    executed trades over a 3 year period.
    """

    for i, ticker in enumerate(tickers):
    
        gc.collect()
        print_ticker_heading(ticker)

        hist   = get_hist(ticker, period)
        success, hist   = smooth(hist)
        if success == False:
            continue
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
    """
    Split data set into a training and test data set:
    - X contains the features for training and predicting. 
    - y contains the target for training and evaluating the performance.

    Used_cols contain the features (i.e. columns) that yiu want to use
    for training and prediction. Target contains the name of the column
    that is the target.

    Function returns X and y for cross validation, X and y for training, 
    and X and y for testing.
    """

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
    """
    Used to predict buy and sell signals. The function itself has no awareness
    what it is predicting. It is just a helper function used by 
    get_possible_trades().

    Target is the column that contains the target. The other columns are
    considered to be features to be used for training and prection.

    The function uses a balanced weight of evidence scorecard to predict the 
    signals. It returns the signals array.

    Note that the function uses 70% for training and 30% for testing. The 
    date where the split happens is dependent on how much data the hist
    dataframe contains. So, the caller will not see a single split date for
    all tickers. 
    """
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
    """
    The functions will take two lists and produce a single containing the 
    buy and sell signals. The merged list will start with always buy signal.
    This is achieved by setting the state to SELL. That ensures that all sell 
    signals are quietly dropped until we get to the first buy signal.

    Note: this function does not enforce that each buy signal is matched with  
    a sell signal.

    The function implements a simple deterministic state machine that flips 
    from SELL to BUY and back from BUY to SELL.

    A buy in the merged list is 1 and a sell is 2. The merged list is
    returned to the caller at the end.
    """

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
    """
    Given a merged buy and sell list, extract all complete buy and sell pairs 
    and store each pair as a trade in a dataframe (i.e. possible_trades_df). 
    
    The possible trades dataframe contains the ticker, buy date, sell date, 
    the close price at buy date, the cloe price at sell data, the gain 
    percentage, and the daily compounded return.

    Note that hist, contains the data from yfinance for the ticker, so we
    can calculate the above values to be stored in the possible trades
    dataframe.

    The function returns the possible trades dataframe to the caller.
    The possible trades for a single ticker.

    The function assumes that the buy_n_sell list is well formed and does 
    not carry out any checks. Since the list is typiclly created by 
    merge_buy_n_sell_signals(), this should be the case.

    TODO: extend the functionality so that the buy at the end without 
    a matching signal is storted in an open position dataframe. The caller
    is then responsible for merging all open position of all tickers into 
    a single dataframe.  
    """

    test_start_at = len(hist) - len(buy_n_sell)
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)
    
    buy_id = -1

    for i, b_or_s in enumerate(buy_n_sell):
        
        if b_or_s == BUY:
            buy_id    = test_start_at + i
            buy_close = hist.Close.iloc[buy_id]
            buy_date  = hist.index[buy_id]
            
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
                         'daily_return' : [daily_return], 'ticker'    : [ticker] }
            possible_trades_df = pd.concat([possible_trades_df, 
                                           pd.DataFrame(trade_dict)])
    
    if verbose == True:
        log("****EXTRACT_TRADES****")
        log(possible_trades_df)
    
    if buy_id > 0:
        buy_opportunity_df = {'ticker'    : [ticker] , 
                              'buy_date'  : [buy_date],  
                              'buy_close' : [buy_close],
                             }
        buy_opportunity_df = pd.DataFrame(buy_opportunity_df)
    else:
        cols=['ticker', 'buy_date', 'buy_close']
        buy_opportunity_df = pd.DataFrame(columns=cols)

    return possible_trades_df, buy_opportunity_df


def get_possible_trades(tickers, threshold, period, verbose):
    """
    The main driver that calls other functions to do the work with the aim
    of extracting all possible trades for all tickers. For each ticker it
    performs the following steps:

    1) retrieve the historical ticker information (using yfinance),
    2) smooth the Close price curve,
    3) get the buy signals (using balanced scorecard),
    4) get the sell signals (using balanced scorecard),
    5) merge the buy and sell signals, and
    6) extract the possible trades and add that to the overall dataframe
       containing all possible trades for all tickers.

    The dataframe with all possible trades is then returned to the caller 
    at the end.
    """
    # print("tickers=", tickers)
    target = 'target'
    
    cols = ['buy_date', 'buy_close', 'sell_date', 'sell_close', 'gain_pct',
            'trading_days', 'daily_return', 'ticker' ]
    possible_trades_df = pd.DataFrame(columns=cols)

    cols=['ticker', 'buy_date', 'buy_close']
    buy_opportunities_df = pd.DataFrame(columns=cols)
    
    #print('Determining possible trades...\n')
    for ticker in tqdm(tickers, desc="possible trades: "):

        try:
            # free up memory
            gc.collect()

            if verbose == True:
                print_ticker_heading(ticker)

            # get stock data and smooth the Close curve
            hist = get_hist(ticker, period)
            success, hist = smooth(hist, ticker)
            if success == False:
                continue

            # get the buy signals
            hist[target] = 0
            min_ids = argrelmin(hist.smooth.values)[0].tolist()
            hist[target].iloc[min_ids] = 1        
            buy_signals = get_signals(hist, target, threshold)

            # get the sell signals
            hist[target] = 0
            max_ids = argrelmax(hist.smooth.values)[0].tolist()
            hist[target].iloc[max_ids] = 1
            sell_signals = get_signals(hist, target, threshold)
            
            # merge the buy and sell signals
            buy_n_sell = merge_buy_n_sell_signals(buy_signals, sell_signals)
            
            # extract trades
            ticker_df, buy_df = extract_trades(hist, buy_n_sell, ticker, verbose)
            possible_trades_df = pd.concat([possible_trades_df, ticker_df])
            buy_opportunities_df = pd.concat([buy_opportunities_df, buy_df])

        except:
            print(f"Failed to get possible trades for {ticker}")
            continue
    
    possible_trades_df.trading_days = possible_trades_df.trading_days.astype(int)
    return possible_trades_df, buy_opportunities_df


class Capital(object):
    """
    A simple class to record the growth of decline of capital, in_use, and 
    free after each trading day. The data is stored in a dataframe with
    four columns: date, capital, in_use, and free. The method day_close()
    is used to store the capital, in_use, and free after closing the day.

    This class is used by PnL class whgich is the real workhorse. 
    """
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
    """
    The PnL class keeps track of all the buys, sells, and day closes. 
    The class keeps track of things via a dataframe. The dataframe consists 
    of the following columns: date, ticker, action (BUY, SELL, or CLOSE), 
    original amount invested in the stock, close amount at the end of the 
    trading day, number of shares owned, the stop loss price undert which we 
    will sell the stock (typically 90% of the starting share price),
    the daily gain (positive is gain; negative is loss), daily compounded
    return percentage, and a flag invested.

    The flag is used to retrieve the last record for a particular active 
    trade. The class ensures that there is always one record for which 
    invested is 1. It is important to ensure this is the case as otherwise 
    the code will not work.

    The class consists of three methods:

    - buy_stock() : buy a specified amount of a particular stock on buy date
                    with a planned sell date
    - sell_stock(): sell a stock on a specific sell date
    - day_close() : for each open ticker investment update the value of the
                    position. If the share price droipped below stop loss,
                    carry out a forced sell of stock. After that
                    record the day end capital, in_use, and free.
    
    The class enforces that no more than max_stocks are owned. max_stocks is 
    set at initialization. It also enforces, that capital = in_use + free. 
    Since these are floats, the code use the following trick to ensure they 
    are basically the same: abs(capital - in_use - free) < TOLERANCE where
    TOLERANCE is 1E-6 (i.e. close to zero).

    """

    def __init__(self, start_date, end_date, capital, in_use, free, max_stocks):
        
        cols = [ 'date', 'ticker', 'action', 'orig_amount', 'close_amount',
                 'no_shares', 'stop_loss', 'daily_gain', 'daily_pct', 
                 'days_in_trade', 'invested']
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
        """
        Buy a stock on a specifid day and store it in the actual trades dataframe (i.e. df).
        
        The active investments are recorded in the invested dictionary. The historical
        stock data is stored in invested[ticker].

        Returns nothing.
        """
        
        assert amount > 0,                      f"amount ({amount}) needs to be greater than zero!"
        assert ticker not in self.invested,     f"already own shares in {ticker}!"
        assert len(self.invested) < self.max_stocks, f"already own maximum # stocks ({self.max_stocks})!"
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, "capital and in_use + free deviating too much!"
        
        # Make sure we have the money to buy stock
        if amount > self.free:
            if self.free > 0:
                amount=self.free
                log(f"you do not have {amount} and setting amount to {self.free}")
            else:
                log(f"you do not have any money left to buy ({self.free})! Not buying...")
                return

        # Retrieve the historical data for stock ticker and save it while we're invested
        asset  = yf.Ticker(ticker)
        hist   = asset.history(start=self.start, end=self.end)
        self.invested[ticker] = hist.copy()
        
        # Get share price and calculate how many shares we can buy
        # Also, set stop loss share price at 10 %
        idx = self.invested[ticker].index == buy_date
        share_price = float(self.invested[ticker].Close.loc[idx])
        no_shares = amount / share_price
        stop_loss = share_price * 0.9
        
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
                    'stop_loss'    : [stop_loss],
                    'daily_gain'   : [0.0],
                    'daily_pct'    : [0.0],
                    'days_in_trade': [0],
                    'invested'     : [1]
                   }
        
        buy_df = pd.DataFrame(buy_dict)
        self.df = pd.concat([self.df, buy_df])
     
    def sell_stock (self, ticker, sell_date):
        """
        Sell stock on specified date. Also, remove the ticker from invested after
        the position has been closed. 

        Returns nothing.
        """

        assert self.capital >= 0, "capital needs to be zero or greater"
        assert self.in_use  >= 0, "in_use needs to be zero or greater"
        assert self.free    >= 0, "free needs to be zero or greater"        
        assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
               "capital and in_use + free deviating too much!"

        # Return if we do not own the stock (may be due to a forced stop-loss sales)
        if ticker not in self.invested:
            return 
        
        # Get the latest close_amount for ticker and no_shares owned
        idx           = (self.df.ticker == ticker) & (self.df.invested==1)
        no_shares     = float(self.df['no_shares'].loc[idx])
        close_amount  = float(self.df['close_amount'].loc[idx])
        orig_amount   = float(self.df['orig_amount'].loc[idx])
        stop_loss     = float(self.df['stop_loss'].loc[idx])
        days_in_trade = int(self.df['days_in_trade'].loc[idx])
        self.df.loc[idx, 'invested'] = 0
        
        # Calculate how much the sell will earn
        idx           = self.invested[ticker].index == sell_date
        share_price   = float(self.invested[ticker].Close.loc[idx])
        today_amount  = no_shares * share_price
        delta_amount  = today_amount - close_amount
        delta_pct     = (delta_amount / close_amount) * 100

        # print the profit/loss of the trade
        log(f"profit of selling {ticker} on {sell_date}: "
              f"{today_amount - orig_amount}"
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
            log("self.capital=", self.capital)
            log("self.in_use=", self.in_use)
            log("self.free=", self.free)
            log("diff=", abs(self.capital - self.in_use - self.free))
            assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
                   "capital and in_use + free deviating too much!"
        
        # Save the stock sell
        sell_dict = {'date'        : [sell_date],
                    'ticker'       : [ticker],
                    'action'       : ['SELL'],
                    'orig_amount'  : [orig_amount],
                    'close_amount' : [today_amount],
                    'no_shares'    : [no_shares],
                    'stop_loss'    : [stop_loss],
                    'daily_gain'   : [delta_amount],
                    'daily_pct'    : [delta_pct],
                    'days_in_trade': [days_in_trade + 1],
                    'invested'     : 0
                   }
        
        sell_df = pd.DataFrame(sell_dict)
        self.df = pd.concat([self.df, sell_df])

        # Remove stock from invested dictionary
        del self.invested[ticker]
        
    def day_close(self, close_date):
        """
        Day end close. Updates the value of the stocks actively invested in and
        updates the variables capital, in_use, and free. After the value of each
        position has been updated, store capital, in_use, and free.

        It also has a safety net to print a warning if the value of an open 
        position changes by more than ten percent. This is put in place as 
        occasionally the data returned by yfinance is incorrect. This will alert
        us that this may be happening. 

        Returns nothing.
        """
        # print("day_close:")
        tickers = list(self.invested.keys())
        for ticker in tickers:
            
            # Get the latest close_amount for ticker and no_shares owned
            df_idx        = (self.df.ticker == ticker) & (self.df.invested==1)
            log(f"{ticker}:\n {self.df.loc[df_idx]}")
            no_shares     = float(self.df['no_shares'].loc[df_idx])
            close_amount  = float(self.df['close_amount'].loc[df_idx])
            orig_amount   = float(self.df['orig_amount'].loc[df_idx])
            stop_loss     = float(self.df['stop_loss'].loc[df_idx])
            days_in_trade = int(self.df['days_in_trade'].loc[df_idx])
            self.df.loc[df_idx, 'invested'] = 0

            # Calculate how much the sell will earn
            hist_idx      = self.invested[ticker].index == close_date
            share_price   = float(self.invested[ticker].Close.loc[hist_idx])
            today_amount  = no_shares * share_price
            delta_amount  = today_amount - close_amount
            delta_pct     = (delta_amount / close_amount) * 100

            # check if we reached a stop loss condition
            gain_pct = ((today_amount - orig_amount) / orig_amount) * 100
            if share_price < stop_loss:
                log(f"breached stop-loss and selling {ticker}...")
                self.df.loc[df_idx, 'invested'] = 1
                self.sell_stock(ticker, close_date)
                continue

            # Report a suspicious high change per stock/day. Threshold for now set at 10%
            # Allows us to see what other stocks may have issues than just SBT...
            if abs(delta_amount / self.capital) > 0.1:
                log('', True)
                log('********************', True)
                log(f'*** WARNING      *** capital changed by more than 10% for {ticker} on {close_date}!', True)
                log(f'***              *** no_shares={no_shares} share_price={share_price} today_amount={today_amount}', True)
                log(f'***              *** orig_amount={orig_amount} close_amount={close_amount} delta_amount={delta_amount}', True)
                log('********************', True)
                log('', True)
            
            # Correct in_use and capital for delta_amount
            self.capital  = self.capital + delta_amount
            self.in_use   = self.in_use  + delta_amount
            assert abs(self.capital - self.in_use - self.free) < TOLERANCE, \
                   "capital and in_use + free deviating too much!"

            close_dict = {'date'         : [close_date],
                         'ticker'       : [ticker],
                         'action'       : ['CLOSE'],
                         'orig_amount'  : [orig_amount],
                         'close_amount' : [today_amount],
                         'no_shares'    : [no_shares],
                         'stop_loss'    : [stop_loss],
                         'daily_gain'   : [delta_amount],
                         'daily_pct'    : [delta_pct],
                         'days_in_trade': [days_in_trade + 1],
                         'invested'     : 1
                   }
        
            close_df = pd.DataFrame(close_dict)
            self.df = pd.concat([self.df, close_df])
            
        # Store overall end day result in myCapital
        self.myCapital.day_close(close_date, self.capital, self.in_use, self.free)

   

def backtester(log_fnm, requested_tickers, period, capital, max_stocks):
    """
    The backtester() that will determine for the requested tickers:
    1) build a list of remaining tickers that exclude the tickers
       on the exclude list
    2) for the remaining tickers, determine for the specified period 
       the possible trades. Note that the possible trades only cover
       the period of the test data
    3) determine the average expected gain per ticker (this will be used 
       to determine whether the stock we want to buy is a better 
       investment opportunity than any of the stocks currently invested in)
    4) determine the start date and end date of backtesting period. Note 
       that we subtract 5 days from the start date and add 5 days to the 
       end date to allow things to settle
    5) create a PnL instance and initialize it with a capital of $10k 
       and set free to $10k and in_use to $0k. The max_stocks are currently 
       set to five stocks.
    6) sort the possible trades by buy_date and gain_pct. That way we ensure 
       that the possible trades are processed in the correct order. It also 
       makes the main loop simpler
    7) Now loop through all possible trades and carry out the buy and sells.
       The loop is complicated. The outer loop goes through all the trading 
       days and does the following:

       7.1) check if there are any stocks that need to be sold on this 
            trading day. The dictionary sell_dates contains the planned 
            sell dates for tickers. Sell all the stocks that need to be sold
            on this day. Note that sell_dates contains a list of tickers to
            be sold on a specific day.

       7.2) then check if there are any stocks to buy on this day. If there are
            do the following:
            - make sure that the stock to buy does have a positive expected
              gain
            - if we have reached the maximum number of stocks (which is the 
              vast majority of the time), check if the expected gain of the 
              "worst" performing stock is less than the expected gain of the
              stock we are intending to buy. If this is the case, then sell
              the "worst" performing stock to create space for the new to 
              buy stock
            - check that we have not reached the maximum number of stocks 
              and that we have enough cash (variable free) to buy at least
              25% of the amount we want to buy. Note amount is calculated as 
              capital divided by maximum number of stock. If we meet the
              above criteria then we will buy the stock and record the 
              planned sell date in sell_dates dictionary.

            Note the above is done for all possible trades on a particular 
            trading day.

       7.3) call day_close() method to value all open positions and to
            record results at the end of the trading day (i.e. capital, 
            in_use, and free).

    The function returns the PnL dataframe, Capital dataframe, and 
    possible trades dataframe. These can then be used for further analysis.
    """

    # create log file and write call information out
    logging.basicConfig(filename=f'{LOGPATH}{log_fnm}', 
                        format='%(asctime)s %(message)s', 
                        level=logging.DEBUG)
    log(f"backtester() started: log_fnm={log_fnm}"
        f" period={period} capital={capital} max_stocks={max_stocks}")
    log(f"Requested_tickers={requested_tickers}\n\n")

    # Read exclude list (bad Close price curves and not able to smooth)
    exclude_df = pd.read_csv(f'{DATAPATH}exclude.csv')
    exclude_list = exclude_df.ticker.to_list()
    log(f"exclude_list={exclude_list}\n")

    tickers = []
    for ticker in requested_tickers:
        if ticker in exclude_list:
            continue
        tickers.append(ticker)
    log(f"After aplying exclude list {len(tickers)} stocks left\n", True)

    # Read ticker_stats list and apply "good" filter
    ticker_stats_df = pd.read_csv(f'{DATAPATH}ticker_stats.csv')
    good_tickers = ticker_stats_df.loc[ticker_stats_df.good == 1].ticker.to_list()
    log(f"good_tickers={good_tickers}\n")

    possible_tickers = tickers
    tickers = []
    for t in possible_tickers:
        if t in good_tickers:
            tickers.append(t)

    log(f"After applying good tickers filter {len(tickers)} stocks left for simulation\n", True)
    log(f'tickers={tickers} for simulation')
    time.sleep(1)

    # Determine for the selected stocks all possible trades
    min_indices, max_indices, failed_tickers = determine_minima_n_maxima(tickers, period, False)
    log('', True)
    log("Unable to determine local minima and maxima for the following tickers:", True)
    log(failed_tickers, True)
    remaining_tickers = []
    for ticker in tickers:
        if ticker in failed_tickers:
            continue
        remaining_tickers.append(ticker)
    log(f"Simulating with remaining {len(remaining_tickers)} stocks\n", True)
    time.sleep(1)

    min_indices, max_indices = align_minima_n_maxima(remaining_tickers, min_indices, max_indices, False)
    possible_trades_df, buy_opportunities_df = get_possible_trades(remaining_tickers, 0.5, period, False)

    # Create a dictionary that stores the mean gain_pct per ticker.
    # This controls whether backtester is willing to invest in the stock
    cols = ['ticker', 'trading_days', 'gain_pct', 'daily_return']
    mean_dict = possible_trades_df[cols].groupby('ticker').agg(['mean']).to_dict()
    mean_df = possible_trades_df[cols].groupby('ticker').agg(['mean']).reset_index()
    mean_df.columns=['ticker', 'trading_days', 'gain_pct', 'daily_return']
    buy_opportunities_df = pd.merge(buy_opportunities_df, mean_df, how='inner')

    # Determine start and end date for backtesting period.
    start_date, end_date = min(possible_trades_df.buy_date), max(possible_trades_df.sell_date)
    start_date = start_date - timedelta(5)
    end_date = end_date + timedelta(5)

    # Pull down MSFT stock for period and use that as basis for determining
    # the stock market trading days
    asset = yf.Ticker('MSFT')
    hist  = asset.history(period="max")
    idx = (hist.index >= start_date) & (hist.index <= end_date)
    backtest_trading_dates = hist.loc[idx].index.to_list()

    # Initialize the key variable
    free              = capital
    in_use            = 0
    myPnL             = PnL(start_date, end_date, capital, in_use, free, max_stocks)

    # Sort the possible trades so they are processed in order
    i_possible_trades = 0
    possible_trades   = possible_trades_df.sort_values(by=['buy_date', 'gain_pct'], ascending=[True, False])
    possible_trades   = possible_trades.reset_index()

    sell_dates        = {}
    stocks_owned      = 0

    log('', True)
    log(f"Possible trades to simulate: {len(possible_trades)}", True)
    log(f"Trading days to simulate   : {len(backtest_trading_dates)}\n", True)
    time.sleep(1)

    for trading_day, trading_date in enumerate(tqdm(backtest_trading_dates, desc="simulate trades: ")):

        #
        # Sell stocks if we have reached the sell_date
        #
        if trading_date in sell_dates:
            to_sell = sell_dates.pop(trading_date, [])
            for ticker in to_sell:
                if ticker in myPnL.invested:
                    log(f"invested in: {list(myPnL.invested.keys())} ({len(myPnL.invested)})")
                    log(f"capital={myPnL.capital} in_use={myPnL.in_use} free={myPnL.free}")
                    log(f"*** selling {ticker} on {trading_date}")
                    myPnL.sell_stock(ticker, trading_date)
                    log(f"after selling invested in: {list(myPnL.invested.keys())} ({len(myPnL.invested)})")
                    log(f"capital={myPnL.capital} in_use={myPnL.in_use} free={myPnL.free}")

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
                    amount = myPnL.capital / max_stocks
                    log(f"*** buying {amount} in {ticker} on {buy_date} with target sell date of {sell_date}")

                    # If we reached max_stocks, check if this stock is expected to
                    # perform better then lowest performing invested stock. If that is the case, 
                    # sell lowest expected performing stock, so that we can buy stock
                    if len(myPnL.invested) >= max_stocks:
                        expected_gain = mean_dict[('gain_pct', 'mean')][ticker]

                        lowest_expected_gain = None 
                        lowest_ticker        = None
                        for t in myPnL.invested.keys():
                            t_gain = mean_dict[('gain_pct', 'mean')][t]
                            if lowest_expected_gain is None or t_gain < lowest_expected_gain:
                                lowest_expected_gain = t_gain
                                lowest_ticker        = t
                        
                        if lowest_expected_gain is not None and expected_gain > lowest_expected_gain:
                            log(f"*** selling {lowest_ticker} on {trading_date} to free up money for {ticker}")
                            myPnL.sell_stock(lowest_ticker, trading_date)
                        else:
                            log(f"maxed out: {ticker} is not expected to perform better than stocks already invested in")
                            log(f"invested in: {list(myPnL.invested.keys())} ({len(myPnL.invested)})")
                            log('')                   

                    # Only attempt to buy a stock when below max # stocks and 
                    # we have enough free money to buy at least 25% of stock 
                    if len(myPnL.invested) < max_stocks and myPnL.free >= amount*0.25:
                        log(f"enough money ({myPnL.free}) to buy {ticker} (capital={myPnL.capital}")
                        log(f"invested in: {list(myPnL.invested.keys())} ({len(myPnL.invested)})")
                        myPnL.buy_stock(ticker, buy_date, sell_date, amount)
                        log(f"after buy: invested in {list(myPnL.invested.keys())} ({len(myPnL.invested)}")
                        log(f"capital={myPnL.capital} in_use={myPnL.in_use} free={myPnL.free}")

                        # save the sell date for future processing
                        if sell_date in sell_dates:
                            sell_dates[sell_date].append(ticker)
                        else:
                            sell_dates[sell_date] = [ ticker ]
                    else:
                        log(f"not enough money to buy 25% of stock; not buying")
                        log(f"invested in: {list(myPnL.invested.keys())} ({len(myPnL.invested)})")
                        log(f"capital={myPnL.capital} in_use={myPnL.in_use} free={myPnL.free}")

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

        cap_before = myPnL.capital
        log(f"before day_close: {trading_date} {len(myPnL.invested)} "
            f"{round(myPnL.capital,2)} {round(myPnL.in_use,2)} {round(myPnL.free,2)} "
            f"{round(abs(myPnL.capital - myPnL.in_use - myPnL.free),6)} "
            f"{abs(myPnL.capital - myPnL.in_use - myPnL.free) < TOLERANCE}")
        myPnL.day_close(trading_date)
        log(f"after day_close: {trading_date} {len(myPnL.invested)} "
            f"{round(myPnL.capital,2)} {round(myPnL.in_use,2)} {round(myPnL.free,2)} "
            f"{round(abs(myPnL.capital - myPnL.in_use - myPnL.free),6)} "
            f"{abs(myPnL.capital - myPnL.in_use - myPnL.free) < TOLERANCE}")

    log(f"i_possible_trades={i_possible_trades} stocks_owned={stocks_owned}")
    myPnL.df.days_in_trade = myPnL.df.days_in_trade.astype(int)

    # Get today's and yesterday's date
    today = datetime.today()
    yesterday = today - timedelta(1)
    today, yesterday = today.strftime('%Y-%m-%d'), yesterday.strftime('%Y-%m-%d')


    log('', True)
    log("Today's buying recommendations:\n", True)
    idx = (buy_opportunities_df.buy_date == today) & (buy_opportunities_df.gain_pct > 0)
    df = buy_opportunities_df.loc[idx].sort_values(by='daily_return', ascending=False)[0:max_stocks]
    log(df, True)
    log('', True)

    log('', True)
    log("Yesterday's buying recommendations:\n", True)
    idx = (buy_opportunities_df.buy_date == yesterday) & (buy_opportunities_df.gain_pct > 0)
    df = buy_opportunities_df.loc[idx].sort_values(by='daily_return', ascending=False)[0:max_stocks]
    log(df, True)
    log('', True)

    return myPnL.df, myPnL.myCapital.df, possible_trades_df, buy_opportunities_df


if __name__ == "__main__":

    sdf = pd.read_csv(f'{DATAPATH}stocks_1000.csv')
    idx = (sdf.TICKER > '')
    sdf = sdf.loc[idx].reset_index()
    tickers = sdf.TICKER.to_list()

    log_fnm = 'backtest_2000.log'
    myPnL_df, myCapital_df, possible_trades_df, buy_opportunities_df = backtester(log_fnm, tickers, "10y", 10000, 5)

    print(myPnL_df)
    print('')
    myCapital_df.index = myCapital_df.date

    print(myCapital_df)
    print('')

    to_plot_cols = ['capital', 'in_use']
    myCapital_df[to_plot_cols].plot(figsize=(18,10))
    plt.show()
    print('')