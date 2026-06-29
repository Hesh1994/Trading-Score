"""
Scoring Module
Core logic for computing technical indicators, evaluating criteria, and generating stock scores.
Uses only numpy and pandas (no external indicator libraries required).
"""

import pandas as pd
import numpy as np
import warnings
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG, classify_signal

warnings.filterwarnings('ignore')


# ============================================================================
# INDICATOR CALCULATION FUNCTIONS (ONE PER INDICATOR)
# ============================================================================

def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index)"""
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(df, period_short=15, period_long=45):
    """Calculate SMA short and long"""
    sma_short = df['close'].rolling(window=period_short).mean()
    sma_long = df['close'].rolling(window=period_long).mean()
    return sma_short, sma_long


def calculate_ema(df, period=15):
    """Calculate SMA (replaces EMA)"""
    return df['close'].rolling(window=period).mean()


def calculate_mfi(df, period=14):
    """Calculate Money Flow Index"""
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    price_change = typical_price.diff()
    positive_flow = money_flow.where(price_change > 0, 0)
    negative_flow = money_flow.where(price_change < 0, 0)
    
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    mfi_ratio = positive_sum / (negative_sum + 1e-10)
    mfi = 100 - (100 / (1 + mfi_ratio))
    
    return mfi


def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    """Calculate Stochastic %K and %D"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / ((highest_high - lowest_low) + 1e-10)
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k.fillna(0), stoch_d.fillna(0)


def calculate_aroon(df, period=25):
    """Calculate Aroon Up and Down"""
    high = df['high']
    low = df['low']
    
    aroon_up = []
    aroon_down = []
    
    for i in range(len(df)):
        if i < period - 1:
            aroon_up.append(0)
            aroon_down.append(0)
        else:
            window_high = high.iloc[i-period+1:i+1]
            window_low = low.iloc[i-period+1:i+1]
            
            periods_since_high = period - (window_high.idxmax() - window_high.index[0]).days - 1
            periods_since_low = period - (window_low.idxmin() - window_low.index[0]).days - 1
            
            aroon_up.append(((period - max(periods_since_high, 0)) / period) * 100)
            aroon_down.append(((period - max(periods_since_low, 0)) / period) * 100)
    
    return pd.Series(aroon_up, index=df.index), pd.Series(aroon_down, index=df.index)


def calculate_bollinger(df, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    bb_upper = sma + (std * std_dev)
    bb_lower = sma - (std * std_dev)
    
    return bb_upper, sma, bb_lower


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, macd_signal


# ============================================================================
# CRITERIA EVALUATION FUNCTIONS
# ============================================================================

def evaluate_rsi_criteria(rsi_current, buy_criteria, sell_criteria, parameters=None):
    """Evaluate RSI buy and sell criteria with configurable thresholds"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(rsi_current):
        # Get thresholds from parameters if using parameter-based thresholds
        if parameters and buy_criteria.get('threshold_type') == 'parameter':
            buy_threshold = parameters.get('buy_threshold', 50.0)
            sell_threshold = parameters.get('sell_threshold', 50.0)
        else:
            buy_threshold = buy_criteria.get('threshold', 30)
            sell_threshold = sell_criteria.get('threshold', 70)
        
        # Evaluate buy criteria
        if buy_criteria['operator'] in ['<', 'less_than']:
            buy_triggered = rsi_current < buy_threshold
        elif buy_criteria['operator'] in ['>', 'greater_than']:
            buy_triggered = rsi_current > buy_threshold
        
        # Evaluate sell criteria
        if sell_criteria['operator'] in ['>', 'greater_than']:
            sell_triggered = rsi_current > sell_threshold
        elif sell_criteria['operator'] in ['<', 'less_than']:
            sell_triggered = rsi_current < sell_threshold
    
    return buy_triggered, sell_triggered


def evaluate_sma_criteria(sma_short, sma_long, buy_criteria, sell_criteria):
    """Evaluate SMA crossover criteria"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(sma_short) and pd.notna(sma_long):
        if buy_criteria.get('direction') == 'above':
            buy_triggered = sma_short > sma_long
        
        if sell_criteria.get('direction') == 'below':
            sell_triggered = sma_short < sma_long
    
    return buy_triggered, sell_triggered


def evaluate_ema_criteria(price, ema, buy_criteria, sell_criteria):
    """Evaluate EMA criteria (price vs EMA)"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(price) and pd.notna(ema):
        if buy_criteria['operator'] in ['>', 'greater_than']:
            buy_triggered = price > ema
        
        if sell_criteria['operator'] in ['<', 'less_than']:
            sell_triggered = price < ema
    
    return buy_triggered, sell_triggered


def evaluate_mfi_criteria(mfi, buy_criteria, sell_criteria):
    """Evaluate MFI criteria"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(mfi):
        if buy_criteria['operator'] in ['<', 'less_than']:
            buy_triggered = mfi < buy_criteria['threshold']
        
        if sell_criteria['operator'] in ['>', 'greater_than']:
            sell_triggered = mfi > sell_criteria['threshold']
    
    return buy_triggered, sell_triggered


def evaluate_stochastic_criteria(stoch_k, stoch_d, buy_criteria, sell_criteria):
    """Evaluate Stochastic crossover criteria"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(stoch_k) and pd.notna(stoch_d):
        if buy_criteria.get('direction') == 'above' and stoch_k > stoch_d:
            buy_triggered = stoch_k < buy_criteria.get('threshold', 20)
        
        if sell_criteria.get('direction') == 'below' and stoch_k < stoch_d:
            sell_triggered = stoch_k > sell_criteria.get('threshold', 80)
    
    return buy_triggered, sell_triggered


def evaluate_aroon_criteria(aroon_up, aroon_down, buy_criteria, sell_criteria):
    """Evaluate Aroon criteria"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(aroon_up) and pd.notna(aroon_down):
        buy_up_threshold = buy_criteria.get('aroon_up_threshold', 70)
        buy_down_threshold = buy_criteria.get('aroon_down_threshold', 30)
        buy_triggered = (aroon_up > buy_up_threshold) and (aroon_down < buy_down_threshold)
        
        sell_down_threshold = sell_criteria.get('aroon_down_threshold', 70)
        sell_up_threshold = sell_criteria.get('aroon_up_threshold', 30)
        sell_triggered = (aroon_down > sell_down_threshold) and (aroon_up < sell_up_threshold)
    
    return buy_triggered, sell_triggered


def evaluate_bollinger_criteria(price, bb_upper, bb_lower, buy_criteria, sell_criteria):
    """Evaluate Bollinger Bands criteria"""
    buy_triggered = False
    sell_triggered = False
    
    if pd.notna(price) and pd.notna(bb_upper) and pd.notna(bb_lower):
        if buy_criteria['operator'] in ['<=', 'less_than_equal']:
            buy_triggered = price <= bb_lower
        
        if sell_criteria['operator'] in ['>=', 'greater_than_equal']:
            sell_triggered = price >= bb_upper
    
    return buy_triggered, sell_triggered


def evaluate_macd_criteria(macd_line, macd_signal, buy_criteria, sell_criteria):
    """Evaluate MACD crossover criteria"""
    buy_triggered = False
    sell_triggered = False

    if pd.notna(macd_line) and pd.notna(macd_signal):
        if buy_criteria.get('direction') == 'above':
            buy_triggered = macd_line > macd_signal

        if sell_criteria.get('direction') == 'below':
            sell_triggered = macd_line < macd_signal

    return buy_triggered, sell_triggered


def calculate_fear_greed(df, window=252):
    """
    Calculate a per-ticker Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed)
    from OHLCV data using four equally-weighted, percentile-ranked components:
      1. Momentum       — price vs 125-day MA, percentile-ranked in trailing window
      2. RSI(14)        — already 0..100
      3. Volatility     — 20-day realized vol, INVERTED (high vol = fear)
      4. Volume breadth — up-day vol / total vol, percentile-ranked
    Returns a Series aligned to df.index.
    Requires at least `window` (default 252) bars for a meaningful reading.
    """
    def pct_rank(series):
        return series.rolling(window).apply(
            lambda x: (x.argsort().argsort()[-1] / (len(x) - 1)) * 100,
            raw=False
        )

    close = df['close']
    vol   = df['volume']

    # 1) Momentum: price vs 125-day MA
    momentum       = close / close.rolling(125).mean() - 1
    momentum_score = pct_rank(momentum)

    # 2) RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-10))

    # 3) Volatility (inverted — high vol = fear)
    vol_realized     = close.pct_change().rolling(20).std()
    volatility_score = 100 - pct_rank(vol_realized)

    # 4) Volume breadth: up-day vol fraction
    up_vol        = vol.where(close.diff() > 0, 0).rolling(20).sum()
    dn_vol        = vol.where(close.diff() < 0, 0).rolling(20).sum()
    breadth_score = pct_rank(up_vol / (up_vol + dn_vol + 1e-10))

    fg = pd.concat([momentum_score, rsi, volatility_score, breadth_score], axis=1).mean(axis=1)
    return fg


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def score_stock(ticker, tickers_data_by_interval, config=None, global_config=None):
    """
    Score a single stock. Each indicator uses the interval specified in its config.

    Args:
        ticker: Stock ticker symbol
        tickers_data_by_interval: Dict of {interval: {ticker: DataFrame}}
        config: Indicators configuration
        global_config: Global scoring config
    """
    if config is None:
        config = INDICATORS_CONFIG
    if global_config is None:
        global_config = GLOBAL_CONFIG

    result = {
        'ticker': ticker,
        'buy_score': 0.0,
        'sell_score': 0.0,
        'signals': {},
    }

    def get_df(ind_key):
        interval = config[ind_key].get('interval', 'daily')
        return tickers_data_by_interval.get(interval, {}).get(ticker)

    # ========== RSI ==========
    if config['rsi']['enabled']:
        try:
            df = get_df('rsi')
            if df is None or df.empty:
                result['signals']['rsi'] = {'error': 'no data for interval'}
            else:
                rsi = calculate_rsi(df, config['rsi']['parameters']['period'])
                rsi_current = rsi.iloc[-1]
                buy_trig, sell_trig = evaluate_rsi_criteria(
                    rsi_current,
                    config['rsi']['buy_criteria'],
                    config['rsi']['sell_criteria'],
                    config['rsi']['parameters'],
                )
                result['signals']['rsi'] = {
                    'value': round(rsi_current, 2) if pd.notna(rsi_current) else None,
                    'buy': buy_trig,
                    'sell': sell_trig,
                }
                if buy_trig:
                    result['buy_score'] += config['rsi']['buy_score']
                if sell_trig:
                    result['sell_score'] += config['rsi']['sell_score']
        except Exception as e:
            result['signals']['rsi'] = {'error': str(e)}

    # ========== SMA ==========
    if config['sma']['enabled']:
        try:
            df = get_df('sma')
            if df is None or df.empty:
                result['signals']['sma'] = {'error': 'no data for interval'}
            else:
                period = config['sma']['parameters'].get('period', config['sma']['parameters'].get('period_short', 15))
                sma = df['close'].rolling(window=period).mean()
                price_current = df['close'].iloc[-1]
                sma_current = sma.iloc[-1]
                buy_trig = bool(pd.notna(price_current) and pd.notna(sma_current) and price_current > sma_current)
                sell_trig = bool(pd.notna(price_current) and pd.notna(sma_current) and price_current < sma_current)
                result['signals']['sma'] = {
                    'price': round(price_current, 2) if pd.notna(price_current) else None,
                    'sma': round(sma_current, 2) if pd.notna(sma_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['sma']['buy_score']
                if sell_trig: result['sell_score'] += config['sma']['sell_score']
        except Exception as e:
            result['signals']['sma'] = {'error': str(e)}

    # ========== EMA ==========
    if config['ema']['enabled']:
        try:
            df = get_df('ema')
            if df is None or df.empty:
                result['signals']['ema'] = {'error': 'no data for interval'}
            else:
                ema = calculate_ema(df, config['ema']['parameters']['period'])
                ema_current = ema.iloc[-1]
                latest_close = df.iloc[-1]['close']
                buy_trig, sell_trig = evaluate_ema_criteria(
                    latest_close, ema_current,
                    config['ema']['buy_criteria'], config['ema']['sell_criteria']
                )
                result['signals']['ema'] = {
                    'value': round(ema_current, 2) if pd.notna(ema_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['ema']['buy_score']
                if sell_trig: result['sell_score'] += config['ema']['sell_score']
        except Exception as e:
            result['signals']['ema'] = {'error': str(e)}

    # ========== MFI ==========
    if config['mfi']['enabled']:
        try:
            df = get_df('mfi')
            if df is None or df.empty:
                result['signals']['mfi'] = {'error': 'no data for interval'}
            else:
                mfi = calculate_mfi(df, config['mfi']['parameters']['period'])
                mfi_current = mfi.iloc[-1]
                buy_trig, sell_trig = evaluate_mfi_criteria(
                    mfi_current,
                    config['mfi']['buy_criteria'], config['mfi']['sell_criteria']
                )
                result['signals']['mfi'] = {
                    'value': round(mfi_current, 2) if pd.notna(mfi_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['mfi']['buy_score']
                if sell_trig: result['sell_score'] += config['mfi']['sell_score']
        except Exception as e:
            result['signals']['mfi'] = {'error': str(e)}

    # ========== STOCHASTIC ==========
    if config['stochastic']['enabled']:
        try:
            df = get_df('stochastic')
            if df is None or df.empty:
                result['signals']['stochastic'] = {'error': 'no data for interval'}
            else:
                stoch_k, stoch_d = calculate_stochastic(
                    df,
                    config['stochastic']['parameters']['k_period'],
                    config['stochastic']['parameters']['d_period'],
                    config['stochastic']['parameters']['smooth_k']
                )
                stoch_k_current = stoch_k.iloc[-1]
                stoch_d_current = stoch_d.iloc[-1]
                buy_trig, sell_trig = evaluate_stochastic_criteria(
                    stoch_k_current, stoch_d_current,
                    config['stochastic']['buy_criteria'], config['stochastic']['sell_criteria']
                )
                result['signals']['stochastic'] = {
                    'k': round(stoch_k_current, 2) if pd.notna(stoch_k_current) else None,
                    'd': round(stoch_d_current, 2) if pd.notna(stoch_d_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['stochastic']['buy_score']
                if sell_trig: result['sell_score'] += config['stochastic']['sell_score']
        except Exception as e:
            result['signals']['stochastic'] = {'error': str(e)}

    # ========== AROON ==========
    if config['aroon']['enabled']:
        try:
            df = get_df('aroon')
            if df is None or df.empty:
                result['signals']['aroon'] = {'error': 'no data for interval'}
            else:
                aroon_up, aroon_down = calculate_aroon(df, config['aroon']['parameters']['period'])
                aroon_up_current = aroon_up.iloc[-1]
                aroon_down_current = aroon_down.iloc[-1]
                buy_trig, sell_trig = evaluate_aroon_criteria(
                    aroon_up_current, aroon_down_current,
                    config['aroon']['buy_criteria'], config['aroon']['sell_criteria']
                )
                result['signals']['aroon'] = {
                    'up': round(aroon_up_current, 2) if pd.notna(aroon_up_current) else None,
                    'down': round(aroon_down_current, 2) if pd.notna(aroon_down_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['aroon']['buy_score']
                if sell_trig: result['sell_score'] += config['aroon']['sell_score']
        except Exception as e:
            result['signals']['aroon'] = {'error': str(e)}

    # ========== BOLLINGER BANDS ==========
    if config['bollinger']['enabled']:
        try:
            df = get_df('bollinger')
            if df is None or df.empty:
                result['signals']['bollinger'] = {'error': 'no data for interval'}
            else:
                bb_upper, bb_middle, bb_lower = calculate_bollinger(
                    df,
                    config['bollinger']['parameters']['period'],
                    config['bollinger']['parameters']['std_dev']
                )
                bb_upper_current = bb_upper.iloc[-1]
                bb_lower_current = bb_lower.iloc[-1]
                latest_close = df.iloc[-1]['close']
                buy_trig, sell_trig = evaluate_bollinger_criteria(
                    latest_close, bb_upper_current, bb_lower_current,
                    config['bollinger']['buy_criteria'], config['bollinger']['sell_criteria']
                )
                result['signals']['bollinger'] = {
                    'upper': round(bb_upper_current, 2) if pd.notna(bb_upper_current) else None,
                    'lower': round(bb_lower_current, 2) if pd.notna(bb_lower_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['bollinger']['buy_score']
                if sell_trig: result['sell_score'] += config['bollinger']['sell_score']
        except Exception as e:
            result['signals']['bollinger'] = {'error': str(e)}

    # ========== MACD ==========
    if config['macd']['enabled']:
        try:
            df = get_df('macd')
            if df is None or df.empty:
                result['signals']['macd'] = {'error': 'no data for interval'}
            else:
                macd_line, macd_signal = calculate_macd(
                    df,
                    config['macd']['parameters']['fast'],
                    config['macd']['parameters']['slow'],
                    config['macd']['parameters']['signal']
                )
                macd_line_current = macd_line.iloc[-1]
                macd_signal_current = macd_signal.iloc[-1]
                buy_trig, sell_trig = evaluate_macd_criteria(
                    macd_line_current, macd_signal_current,
                    config['macd']['buy_criteria'], config['macd']['sell_criteria']
                )
                result['signals']['macd'] = {
                    'line': round(macd_line_current, 2) if pd.notna(macd_line_current) else None,
                    'signal': round(macd_signal_current, 2) if pd.notna(macd_signal_current) else None,
                    'buy': buy_trig, 'sell': sell_trig,
                }
                if buy_trig: result['buy_score'] += config['macd']['buy_score']
                if sell_trig: result['sell_score'] += config['macd']['sell_score']
        except Exception as e:
            result['signals']['macd'] = {'error': str(e)}

    # ========== FEAR & GREED INDEX ==========
    if config.get('fear_greed', {}).get('enabled'):
        try:
            df = get_df('fear_greed')
            if df is None or df.empty:
                result['signals']['fear_greed'] = {'error': 'no data for interval'}
            else:
                fg = calculate_fear_greed(df)
                fg_current = fg.iloc[-1]
                if pd.isna(fg_current):
                    result['signals']['fear_greed'] = {'error': 'insufficient data (need 252+ bars)'}
                else:
                    fear_thresh  = config['fear_greed']['parameters'].get('fear_threshold',  30.0)
                    greed_thresh = config['fear_greed']['parameters'].get('greed_threshold', 70.0)
                    buy_trig  = fg_current < fear_thresh
                    sell_trig = fg_current > greed_thresh
                    result['signals']['fear_greed'] = {
                        'value': round(float(fg_current), 1),
                        'buy':   buy_trig,
                        'sell':  sell_trig,
                    }
                    if buy_trig:  result['buy_score']  += config['fear_greed']['buy_score']
                    if sell_trig: result['sell_score'] += config['fear_greed']['sell_score']
        except Exception as e:
            result['signals']['fear_greed'] = {'error': str(e)}

    # ========== FINAL CLASSIFICATION ==========
    signal, net_score = classify_signal(
        result['buy_score'],
        result['sell_score'],
        global_config['buy_threshold'],
        global_config['sell_threshold']
    )

    result['signal'] = signal
    result['net_score'] = round(net_score, 2)
    result['buy_score'] = round(result['buy_score'], 2)
    result['sell_score'] = round(result['sell_score'], 2)

    return result


def score_universe(tickers_data_by_interval, config=None, global_config=None):
    """
    Score multiple stocks across potentially different intervals per indicator.

    Args:
        tickers_data_by_interval: Dict of {interval: {ticker: DataFrame}}
                                  e.g. {'daily': {'AAPL': df, ...}, 'weekly': {...}}
        config: Indicators configuration
        global_config: Global scoring config

    Returns:
        List of result dicts, sorted by net_score descending
    """
    if config is None:
        config = INDICATORS_CONFIG
    if global_config is None:
        global_config = GLOBAL_CONFIG

    # Collect all tickers across all intervals
    all_tickers = set()
    for interval_data in tickers_data_by_interval.values():
        all_tickers.update(interval_data.keys())

    results = []
    for ticker in all_tickers:
        result = score_stock(ticker, tickers_data_by_interval, config, global_config)
        results.append(result)

    results.sort(key=lambda x: x['net_score'], reverse=True)
    return results


def results_to_dataframe(results):
    """
    Convert scoring results to a display DataFrame.

    Args:
        results: List of result dicts from score_universe

    Returns:
        DataFrame ready for display or export
    """
    display_data = []
    for r in results:
        row = {
            'Ticker': r['ticker'],
            'Signal': r['signal'],
            'Buy Score': r['buy_score'],
            'Sell Score': r['sell_score'],
            'Net Score': r['net_score'],
        }
        fg_sig = r.get('signals', {}).get('fear_greed', {})
        if 'value' in fg_sig:
            row['Fear & Greed'] = fg_sig['value']
        display_data.append(row)

    df = pd.DataFrame(display_data)
    return df.sort_values('Net Score', ascending=False).reset_index(drop=True)
