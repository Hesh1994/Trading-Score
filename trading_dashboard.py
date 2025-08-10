import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import warnings
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Trading Technical Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Trading Technical Analysis Dashboard")

# ============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# ============================================================================

def calculate_rsi_stacked(group, period=30):
    """Calculate RSI for stacked DataFrame"""
    delta = group['adj close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma_stacked(group, period=20, price_col='adj close'):
    """Calculate SMA for stacked DataFrame"""
    return group[price_col].rolling(window=period).mean()

def calculate_ema_stacked(group, period=15, price_col='adj close'):
    """Calculate EMA for stacked DataFrame"""
    return group[price_col].ewm(span=period).mean()

def calculate_ema_manual_stacked(group, period=15, seed_period=15, price_col='adj close'):
    """Calculate EMA manually with SMA seed for stacked DataFrame"""
    alpha = 2 / (period + 1)
    prices = group[price_col].values
    ema_values = []
    
    for i in range(len(prices)):
        if i == seed_period - 1:
            sma_seed = np.mean(prices[:seed_period])
            ema_values.append(sma_seed)
        elif i >= seed_period:
            current_price = prices[i]
            previous_ema = ema_values[-1]
            new_ema = (current_price * alpha) + (previous_ema * (1 - alpha))
            ema_values.append(new_ema)
        else:
            ema_values.append(np.nan)
    
    return pd.Series(ema_values, index=group.index)

def calculate_mfi_stacked(group, period=14):
    """Calculate Money Flow Index for stacked DataFrame"""
    typical_price = (group['high'] + group['low'] + group['close']) / 3
    raw_money_flow = typical_price * group['volume']
    price_change = typical_price.diff()
    positive_money_flow = np.where(price_change > 0, raw_money_flow, 0)
    negative_money_flow = np.where(price_change < 0, raw_money_flow, 0)
    positive_money_flow_sum = pd.Series(positive_money_flow).rolling(window=period).sum()
    negative_money_flow_sum = pd.Series(negative_money_flow).rolling(window=period).sum()
    money_flow_ratio = positive_money_flow_sum / negative_money_flow_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

def calculate_stochastic_stacked(group, k_period=14, d_period=3):
    """Calculate Stochastic for stacked DataFrame"""
    lowest_low = group['low'].rolling(window=k_period).min()
    highest_high = group['high'].rolling(window=k_period).max()
    stoch_k = 100 * (group['close'] - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_aroon_up_stacked(high_series, period):
    """Calculate Aroon Up for stacked DataFrame"""
    aroon_up = []
    for i in range(len(high_series)):
        if i < period - 1:
            aroon_up.append(np.nan)
        else:
            window = high_series.iloc[i-period+1:i+1]
            highest_high_idx = window.idxmax()
            periods_since_high = i - high_series.index.get_loc(highest_high_idx)
            aroon_up_value = ((period - periods_since_high) / period) * 100
            aroon_up.append(aroon_up_value)
    return pd.Series(aroon_up, index=high_series.index)

def calculate_aroon_down_stacked(low_series, period):
    """Calculate Aroon Down for stacked DataFrame"""
    aroon_down = []
    for i in range(len(low_series)):
        if i < period - 1:
            aroon_down.append(np.nan)
        else:
            window = low_series.iloc[i-period+1:i+1]
            lowest_low_idx = window.idxmin()
            periods_since_low = i - low_series.index.get_loc(lowest_low_idx)
            aroon_down_value = ((period - periods_since_low) / period) * 100
            aroon_down.append(aroon_down_value)
    return pd.Series(aroon_down, index=low_series.index)

def calculate_aroon_stacked(group, period=25):
    """Calculate Aroon Up and Down for stacked DataFrame"""
    aroon_up = calculate_aroon_up_stacked(group['high'], period)
    aroon_down = calculate_aroon_down_stacked(group['low'], period)
    aroon_oscillator = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_oscillator

def calculate_bollinger_bands_stacked(group, period=20, std_dev=2):
    """Calculate Bollinger Bands for stacked DataFrame"""
    bb_middle = group['close'].rolling(window=period).mean()
    bb_std = group['close'].rolling(window=period).std()
    bb_upper = bb_middle + (bb_std * std_dev)
    bb_lower = bb_middle - (bb_std * std_dev)
    bb_percent_b = (group['close'] - bb_lower) / (bb_upper - bb_lower)
    bb_bandwidth = (bb_upper - bb_lower) / bb_middle
    return bb_upper, bb_middle, bb_lower, bb_percent_b, bb_bandwidth

def calculate_macd_stacked(group, fast=12, slow=26, signal=9):
    """Calculate MACD for stacked DataFrame"""
    fast_ema = group['close'].ewm(span=fast).mean()
    slow_ema = group['close'].ewm(span=slow).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_symbol_indicators(group, indicator_params, signal_weights, entry_score, exit_score):
    """Process all indicators for a single symbol group with custom parameters"""
    
    # Technical Indicators with custom parameters
    if 'rsi' in indicator_params:
        group['rsi'] = calculate_rsi_stacked(group, indicator_params['rsi']['period'])
        group['rsi_mid'] = group['rsi'].rolling(window=indicator_params['rsi']['period']).mean()
    
    if 'sma_short' in indicator_params:
        group['sma_short'] = calculate_sma_stacked(group, indicator_params['sma_short']['period'])
    
    if 'sma_long' in indicator_params:
        group['sma_long'] = calculate_sma_stacked(group, indicator_params['sma_long']['period'])
    
    if 'ema_15' in indicator_params:
        group['ema_15'] = calculate_ema_stacked(group, indicator_params['ema_15']['period'])
        group['ema_manual'] = calculate_ema_manual_stacked(group, indicator_params['ema_15']['period'], indicator_params['ema_15']['period'])
    
    if 'mfi' in indicator_params:
        group['mfi'] = calculate_mfi_stacked(group, indicator_params['mfi']['period'])
        group['mfi_mid'] = group['mfi'].rolling(window=indicator_params['mfi']['period']).mean()
    
    if 'stochastic' in indicator_params:
        group['stoch_k'], group['stoch_d'] = calculate_stochastic_stacked(
            group, 
            indicator_params['stochastic']['k_period'], 
            indicator_params['stochastic']['d_period']
        )
    
    if 'aroon' in indicator_params:
        group['aroon_up'], group['aroon_down'], group['aroon_oscillator'] = calculate_aroon_stacked(
            group, indicator_params['aroon']['period']
        )
    
    if 'bollinger' in indicator_params:
        (group['bb_upper'], group['bb_middle'], group['bb_lower'], 
         group['bb_percent_b'], group['bb_bandwidth']) = calculate_bollinger_bands_stacked(
            group, 
            indicator_params['bollinger']['period'], 
            indicator_params['bollinger']['std_dev']
        )
    
    if 'macd' in indicator_params:
        group['macd_line'], group['macd_signal'], group['macd_histogram'] = calculate_macd_stacked(
            group, 
            indicator_params['macd']['fast'], 
            indicator_params['macd']['slow'], 
            indicator_params['macd']['signal']
        )
    
    # Trading Signals
    signal_cols = []
    
    # Only calculate signals if the required indicators exist
    if 'ema_manual' in group.columns and 'sma_long' in group.columns:
        group['sma_ema_signal'] = np.where(group['ema_manual'] > group['sma_long'], 1, -1)
        signal_cols.append('sma_ema_signal')
    
    if 'sma_short' in group.columns:
        group['sma_short_signal'] = np.where(group['adj close'] > group['sma_short'], 1, -1)
        signal_cols.append('sma_short_signal')
    
    if 'sma_long' in group.columns:
        group['sma_long_signal'] = np.where(group['adj close'] > group['sma_long'], 1, -1)
        signal_cols.append('sma_long_signal')
    
    if 'sma_short' in group.columns and 'sma_long' in group.columns:
        group['sma_cross_signal'] = np.where(group['sma_short'] > group['sma_long'], 1, -1)
        signal_cols.append('sma_cross_signal')
    
    if 'rsi' in group.columns:
        group['rsi_signal'] = np.where(group['rsi'] < 30, 1, np.where(group['rsi'] > 70, -1, 0))
        signal_cols.append('rsi_signal')
    
    if 'rsi' in group.columns and 'rsi_mid' in group.columns:
        group['rsi_cross_signal'] = np.where(group['rsi'] > group['rsi_mid'], 1, -1)
        group['rsi_50_signal'] = np.where(group['rsi_mid'] > 50, 1, -1)
        signal_cols.extend(['rsi_cross_signal', 'rsi_50_signal'])
    
    if 'mfi' in group.columns:
        group['mfi_signal'] = np.where(group['mfi'] < 30, 1, np.where(group['mfi'] > 70, -1, 0))
        signal_cols.append('mfi_signal')
    
    if 'mfi' in group.columns and 'mfi_mid' in group.columns:
        group['mfi_50_signal'] = np.where(group['mfi_mid'] > 50, 1, -1)
        group['mfi_cross_signal'] = np.where(group['mfi'] > group['mfi_mid'], 1, -1)
        signal_cols.extend(['mfi_50_signal', 'mfi_cross_signal'])
    
    if 'stoch_k' in group.columns and 'stoch_d' in group.columns:
        group['stoch_signal'] = np.where(group['stoch_k'] > group['stoch_d'], 1, -1)
        group['stoch8020_signal'] = np.where(
            ((group['stoch_k'] > 80) | (group['stoch_d'] > 80)) & (group['stoch_k'] < group['stoch_d']), -1,
            np.where(((group['stoch_k'] < 20) | (group['stoch_d'] < 20)) & (group['stoch_k'] > group['stoch_d']), 1, 0)
        )
        signal_cols.extend(['stoch_signal', 'stoch8020_signal'])
    
    if 'aroon_up' in group.columns and 'aroon_down' in group.columns:
        group['aroon_signal'] = np.where(group['aroon_up'] > group['aroon_down'], 1, -1)
        signal_cols.append('aroon_signal')
    
    if 'aroon_oscillator' in group.columns:
        group['aroon_oscillator_signal'] = np.where(group['aroon_oscillator'] > 80, -1, 
                                                   np.where(group['aroon_oscillator'] < 20, 1, 0))
        signal_cols.append('aroon_oscillator_signal')
    
    if 'bb_middle' in group.columns:
        group['bb_signal'] = np.where(group['close'] > group['bb_middle'], 1, -1)
        signal_cols.append('bb_signal')
    
    if 'bb_upper' in group.columns and 'bb_lower' in group.columns:
        group['bb_up_signal'] = np.where(group['close'] > group['bb_upper'], -1, 
                                        np.where(group['close'] < group['bb_lower'], 1, 0))
        signal_cols.append('bb_up_signal')
    
    if 'macd_line' in group.columns and 'macd_signal' in group.columns:
        group['macd_signal_flag'] = np.where(group['macd_line'] > group['macd_signal'], 1, -1)
        signal_cols.append('macd_signal_flag')
    
    # Composite Score calculation
    group['composite_score'] = 0
    for signal in signal_cols:
        if signal in signal_weights and signal in group.columns:
            group['composite_score'] += group[signal] * signal_weights[signal]
    
    group['Score_change'] = group['composite_score'].diff().fillna(0)
    
    # Target score
    group['target_score'] = np.where(group['composite_score'] >= 3, 3, 0)

    # Position Rules with custom entry/exit scores
    group['position_score'] = np.where(
        group['composite_score'] == entry_score, entry_score,
        np.where(group['composite_score'] == exit_score, exit_score, 0)
    )

    group['1_-1_score'] = np.where(group['position_score'] == entry_score, 1,
                                   np.where(group['position_score'] == exit_score, -1, 0))
    
    # Apply position rule logic
    position_rule = np.zeros(len(group))
    position_size = 1
    
    for i in range(1, len(group)):
        prev_pos = position_rule[i-1]
        curr_signal = group['1_-1_score'].iloc[i]
        
        if prev_pos == 0:
            position_rule[i] = position_size if curr_signal == 1 else 0
        elif prev_pos == position_size:
            position_rule[i] = 0 if curr_signal == -1 else position_size
        else:
            position_rule[i] = 0
    
    group['position_rule'] = position_rule
    
    # Calculate returns
    group['returns'] = group['adj close'].pct_change()
    group['pos_rule_diff'] = group['position_rule'].diff().fillna(0)
    
    # Strategy returns calculation
    group['trade_price'] = np.where(group['pos_rule_diff'].abs() > 0, group['adj close'], np.nan)
    group['trade_price'] = group['trade_price'].fillna(method='ffill')
    
    # Calculate strategy returns when closing positions
    group['st_ret'] = np.where(
        group['pos_rule_diff'] == -1,
        group['trade_price'].pct_change(),
        0
    )
    group['st_ret_acc'] = (1 + group['st_ret']).cumprod()
    group['returns_acc'] = (1 + group['returns'].fillna(0)).cumprod()
    
    return group

# ============================================================================
# CROSS-SECTION FUNCTIONS
# ============================================================================

def score_change_cross_section(df, cols=None, only_last_date=False, last_n=5):
    """Cross-section of tickers with score changes, with history and returns"""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df must have a MultiIndex with ['date','ticker']")
    if list(df.index.names) != ['date','ticker']:
        df = df.reorder_levels(['date','ticker'])
    df = df.sort_index()

    if 'Score_change' not in df.columns:
        df = df.copy()
        df['Score_change'] = df.groupby(level='ticker')['composite_score'].diff().fillna(0)

    ret_col = f'return_{last_n}p'
    if ret_col not in df.columns:
        df = df.copy()
        df[ret_col] = df.groupby(level='ticker')['adj close'].pct_change(periods=last_n)

    if only_last_date:
        last_date = df.index.get_level_values('date').max()
        df_last = df.xs(last_date, level='date', drop_level=False)
        df_changed = df_last.loc[df_last['Score_change'].fillna(0).ne(0)]
    else:
        changed_mask = df['Score_change'].fillna(0).ne(0)
        df_changed = df.loc[changed_mask]

    hist_cols = []
    if not df_changed.empty:
        all_dates = df.index.get_level_values('date').unique().sort_values()
        last_n_dates = all_dates[-last_n:]
        tickers = df_changed.index.get_level_values('ticker').unique()

        hist = (
            df.loc[(df.index.get_level_values('date').isin(last_n_dates)) &
                   (df.index.get_level_values('ticker').isin(tickers)), ['composite_score']]
              .reset_index()
              .pivot(index='ticker', columns='date', values='composite_score')
              .reindex(columns=sorted(last_n_dates))
        )
        hist_cols = [d.strftime('%Y-%m-%d') for d in hist.columns]
        hist.columns = hist_cols

        tick_idx = df_changed.index.get_level_values('ticker')
        for col in hist_cols:
            df_changed[col] = tick_idx.map(hist[col])

    # Add YTD and daily change columns
    if not df_changed.empty:
        # YTD change: compare last price to price at first available date of current year
        try:
            df_changed['YTD_change'] = np.nan
            for ticker in df_changed.index.get_level_values('ticker'):
                ticker_df = df.xs(ticker, level='ticker')
                current_year = df.index.get_level_values('date').max().year
                ticker_dates = ticker_df.index.get_level_values('date')
                ytd_dates = [d for d in ticker_dates if d.year == current_year]
                if ytd_dates:
                    first_ytd_date = min(ytd_dates)
                    start_price = ticker_df.xs(first_ytd_date, level='date')['adj close']
                    last_price = ticker_df.iloc[-1]['adj close']
                    ytd_change = (last_price - start_price) / start_price if start_price != 0 else np.nan
                    df_changed.loc[ticker, 'YTD_change'] = ytd_change
        except Exception:
            pass
        # Daily change: compare last price to previous day
        try:
            df_changed['daily_change'] = np.nan
            for ticker in df_changed.index.get_level_values('ticker'):
                ticker_df = df.xs(ticker, level='ticker')
                if len(ticker_df) > 1:
                    last_price = ticker_df.iloc[-1]['adj close']
                    prev_price = ticker_df.iloc[-2]['adj close']
                    daily_change = (last_price - prev_price) / prev_price if prev_price != 0 else np.nan
                    df_changed.loc[ticker, 'daily_change'] = daily_change
        except Exception:
            pass
    if cols is not None:
        keep = [c for c in cols if c in df_changed.columns] + [c for c in hist_cols if c in df_changed.columns]
        if ret_col in df_changed.columns and ret_col not in keep:
            keep.append(ret_col)
        # Always include YTD and daily change columns
        for extra_col in ['YTD_change', 'daily_change']:
            if extra_col in df_changed.columns and extra_col not in keep:
                keep.append(extra_col)
        if keep:
            df_changed = df_changed[keep]

    if only_last_date and not df_changed.empty and isinstance(df_changed.index, pd.MultiIndex) and 'date' in df_changed.index.names:
        df_changed.index = df_changed.index.droplevel('date')
        df_changed.index.name = 'ticker'

    return df_changed

def target_score_cross_section(df, cols=None, only_last_date=False, target=3, allow_ge=True, last_n=5):
    """Cross-section of tickers meeting a target composite score"""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df must have a MultiIndex with levels ['date','ticker']")
    if list(df.index.names) != ['date','ticker']:
        try:
            df = df.reorder_levels(['date','ticker'])
        except Exception:
            df = df.copy()
            df.index.set_names(['date','ticker'], inplace=True)
    df = df.sort_index()

    if 'Score_change' not in df.columns:
        df = df.copy()
        df['Score_change'] = df.groupby(level='ticker')['composite_score'].diff().fillna(0)

    ret_col = f'return_{last_n}p'
    if ret_col not in df.columns:
        df = df.copy()
        df[ret_col] = df.groupby(level='ticker')['adj close'].pct_change(periods=last_n)

    def target_mask(frame):
        return frame['composite_score'].ge(target) if allow_ge else frame['composite_score'].eq(target)

    if only_last_date:
        last_date = df.index.get_level_values('date').max()
        cross = df.xs(last_date, level='date')

        sel = target_mask(cross)
        tickers = cross.index[sel]
        if len(tickers) == 0:
            return pd.DataFrame()

        base = cross.loc[tickers]
        # Add YTD and daily change columns
        try:
            base['YTD_change'] = np.nan
            for ticker in base.index:
                ticker_df = df.xs(ticker, level='ticker')
                current_year = df.index.get_level_values('date').max().year
                ticker_dates = ticker_df.index.get_level_values('date')
                ytd_dates = [d for d in ticker_dates if d.year == current_year]
                if ytd_dates:
                    first_ytd_date = min(ytd_dates)
                    start_price = ticker_df.xs(first_ytd_date, level='date')['adj close']
                    last_price = ticker_df.iloc[-1]['adj close']
                    ytd_change = (last_price - start_price) / start_price if start_price != 0 else np.nan
                    base.loc[ticker, 'YTD_change'] = ytd_change
        except Exception:
            pass
        try:
            base['daily_change'] = np.nan
            for ticker in base.index:
                ticker_df = df.xs(ticker, level='ticker')
                if len(ticker_df) > 1:
                    last_price = ticker_df.iloc[-1]['adj close']
                    prev_price = ticker_df.iloc[-2]['adj close']
                    daily_change = (last_price - prev_price) / prev_price if prev_price != 0 else np.nan
                    base.loc[ticker, 'daily_change'] = daily_change
        except Exception:
            pass
        if cols is not None:
            keep = [c for c in cols if c in base.columns]
            if ret_col in base.columns and ret_col not in keep:
                keep.append(ret_col)
            # Always include YTD and daily change columns
            for extra_col in ['YTD_change', 'daily_change']:
                if extra_col in base.columns and extra_col not in keep:
                    keep.append(extra_col)
            if keep:
                base = base[keep]

        all_dates = df.index.get_level_values('date').unique().sort_values()
        last_dates = all_dates[-last_n:]
        hist = (
            df.loc[(df.index.get_level_values('date').isin(last_dates)) &
                   (df.index.get_level_values('ticker').isin(tickers)), ['composite_score']]
              .reset_index()
              .pivot(index='ticker', columns='date', values='composite_score')
        )
        hist = hist.reindex(columns=sorted(hist.columns))
        hist.columns = [d.strftime('%Y-%m-%d') for d in hist.columns]

        return base.join(hist, how='left').sort_index()

    m = target_mask(df)
    out = df.loc[m] if m.any() else pd.DataFrame()
    if out is not None and not out.empty and cols is not None:
        keep = [c for c in cols if c in out.columns]
        if ret_col in out.columns and ret_col not in keep:
            keep.append(ret_col)
        if keep:
            out = out[keep]
    return out

def add_weekly_scores_from(base_df: pd.DataFrame,
                           weekly_df: Optional[pd.DataFrame] = None,
                           score_col: str = 'composite_score') -> pd.DataFrame:
    """Add previous and current weekly scores to base_df"""
    if weekly_df is None:
        return base_df
    
    if not isinstance(weekly_df.index, pd.MultiIndex):
        raise ValueError("weekly_df must have a MultiIndex index ['date','ticker']")

    dfw_tmp = weekly_df
    if list(dfw_tmp.index.names) != ['date','ticker']:
        dfw_tmp = dfw_tmp.reset_index().set_index(['date','ticker'])
    dfw_tmp = dfw_tmp.sort_index()
    dfw_tmp = dfw_tmp.reset_index()
    dfw_tmp['date'] = pd.to_datetime(dfw_tmp['date']).dt.tz_localize(None)
    dfw_tmp = dfw_tmp.set_index(['date','ticker']).sort_index()

    if isinstance(base_df.index, pd.MultiIndex) and 'ticker' in base_df.index.names:
        tickers = base_df.index.get_level_values('ticker').unique().tolist()
        ticker_indexer = base_df.index.get_level_values('ticker')
    else:
        tickers = base_df.index.unique().tolist()
        ticker_indexer = base_df.index

    rows = []
    for tkr, g in dfw_tmp.groupby(level='ticker', sort=False):
        if tkr not in tickers:
            continue
        ds = g.index.get_level_values('date').unique().sort_values()
        if len(ds) >= 2:
            rows.append({
                'ticker': tkr,
                'previous_week': pd.to_datetime(ds[-2]).tz_localize(None),
                'current_week': pd.to_datetime(ds[-1]).tz_localize(None)
            })
    df_last2_sel = pd.DataFrame(rows)
    if df_last2_sel.empty:
        prev_name = f"{score_col}_previous_week"
        curr_name = f"{score_col}_current_week"
        result = base_df.copy()
        result[prev_name] = np.nan
        result[curr_name] = np.nan
        return result

    scores_rows = []
    for r in df_last2_sel.itertuples(index=False):
        tkr = r.ticker
        pw = r.previous_week
        cw = r.current_week
        s_prev = np.nan
        s_curr = np.nan
        try:
            s_prev = dfw_tmp.loc[(pw, tkr), score_col]
        except KeyError:
            pass
        try:
            s_curr = dfw_tmp.loc[(cw, tkr), score_col]
        except KeyError:
            pass
        scores_rows.append({
            'ticker': tkr,
            f'{score_col}_previous_week': s_prev,
            f'{score_col}_current_week': s_curr,
        })
    weekly_scores = pd.DataFrame(scores_rows).set_index('ticker')

    prev_map = weekly_scores[f'{score_col}_previous_week']
    curr_map = weekly_scores[f'{score_col}_current_week']

    result = base_df.copy()
    result[f'{score_col}_previous_week'] = ticker_indexer.map(prev_map)
    result[f'{score_col}_current_week'] = ticker_indexer.map(curr_map)
    return result

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Date Range Selection
st.sidebar.subheader("📅 Data Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=dt.date(2020, 1, 1),
        min_value=dt.date(2010, 1, 1),
        max_value=dt.date.today()
    )
with col2:
    end_date = st.date_input(
        "End Date", 
        value=dt.date.today(),
        min_value=start_date,
        max_value=dt.date.today()
    )

# Symbol Selection
st.sidebar.subheader("🎯 Symbols")
symbol_option = st.sidebar.selectbox(
    "Symbol Source",
    ["S&P 500 (first 50)", "S&P 500 (all)", "Custom"]
)

if symbol_option == "Custom":
    custom_symbols = st.sidebar.text_area(
        "Enter symbols (comma-separated)",
        value="AAPL,MSFT,GOOGL,AMZN,TSLA"
    )
    symbols_list = [s.strip().upper() for s in custom_symbols.split(",")]
else:
    @st.cache_data
    def get_sp500_symbols():
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
        return sp500['Symbol'].unique().tolist()
    
    sp500_symbols = get_sp500_symbols()
    if symbol_option == "S&P 500 (first 50)":
        symbols_list = sp500_symbols[:50]
    else:
        symbols_list = sp500_symbols

# Technical Indicators Configuration
st.sidebar.subheader("📊 Technical Indicators")

# Available indicators
available_indicators = {
    'RSI': 'rsi',
    'SMA Short': 'sma_short', 
    'SMA Long': 'sma_long',
    'EMA': 'ema_15',
    'MFI': 'mfi',
    'Stochastic': 'stochastic',
    'Aroon': 'aroon',
    'Bollinger Bands': 'bollinger',
    'MACD': 'macd'
}

selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    list(available_indicators.keys()),
    default=['RSI', 'SMA Short', 'SMA Long', 'EMA']
)

# Indicator Parameters
indicator_params = {}
for indicator_name in selected_indicators:
    indicator_key = available_indicators[indicator_name]
    
    with st.sidebar.expander(f"⚙️ {indicator_name} Parameters"):
        if indicator_key == 'rsi':
            indicator_params['rsi'] = {
                'period': st.slider(f"RSI Period", 5, 50, 30, key="rsi_period")
            }
        elif indicator_key == 'sma_short':
            indicator_params['sma_short'] = {
                'period': st.slider(f"SMA Short Period", 5, 30, 15, key="sma_short_period")
            }
        elif indicator_key == 'sma_long':
            indicator_params['sma_long'] = {
                'period': st.slider(f"SMA Long Period", 20, 100, 45, key="sma_long_period")
            }
        elif indicator_key == 'ema_15':
            indicator_params['ema_15'] = {
                'period': st.slider(f"EMA Period", 5, 50, 15, key="ema_period")
            }
        elif indicator_key == 'mfi':
            indicator_params['mfi'] = {
                'period': st.slider(f"MFI Period", 5, 30, 14, key="mfi_period")
            }
        elif indicator_key == 'stochastic':
            indicator_params['stochastic'] = {
                'k_period': st.slider(f"Stochastic %K Period", 5, 30, 14, key="stoch_k"),
                'd_period': st.slider(f"Stochastic %D Period", 2, 10, 3, key="stoch_d")
            }
        elif indicator_key == 'aroon':
            indicator_params['aroon'] = {
                'period': st.slider(f"Aroon Period", 10, 50, 25, key="aroon_period")
            }
        elif indicator_key == 'bollinger':
            indicator_params['bollinger'] = {
                'period': st.slider(f"Bollinger Period", 10, 50, 20, key="bb_period"),
                'std_dev': st.slider(f"Bollinger Std Dev", 1.0, 3.0, 2.0, 0.1, key="bb_std")
            }
        elif indicator_key == 'macd':
            indicator_params['macd'] = {
                'fast': st.slider(f"MACD Fast", 5, 20, 12, key="macd_fast"),
                'slow': st.slider(f"MACD Slow", 20, 40, 26, key="macd_slow"),
                'signal': st.slider(f"MACD Signal", 5, 15, 9, key="macd_signal")
            }

# Signal Weights Configuration
st.sidebar.subheader("⚖️ Signal Weights")

# Define all possible signals
all_signals = [
    'sma_ema_signal', 'sma_short_signal', 'sma_long_signal', 'sma_cross_signal',
    'rsi_signal', 'rsi_cross_signal', 'rsi_50_signal',
    'mfi_signal', 'mfi_50_signal', 'mfi_cross_signal',
    'stoch_signal', 'stoch8020_signal',
    'aroon_signal', 'aroon_oscillator_signal',
    'bb_signal', 'bb_up_signal',
    'macd_signal_flag'
]

signal_weights = {}
max_possible_score = 0

# Checklist for signals to include in scoring
selected_signals = st.sidebar.multiselect(
    "Select Signals for Scoring",
    all_signals,
    default=['sma_ema_signal', 'sma_short_signal', 'sma_long_signal', 'rsi_signal']
)

with st.sidebar.expander("🎛️ Configure Signal Weights"):
    for signal in selected_signals:
        weight = st.slider(
            f"{signal.replace('_', ' ').title()}",
            -3.0, 3.0, 1.0, 0.1,
            key=f"weight_{signal}"
        )
        signal_weights[signal] = weight
        max_possible_score += abs(weight)
    
    # Set unselected signals to 0 weight
    for signal in all_signals:
        if signal not in selected_signals:
            signal_weights[signal] = 0.0

# Target Score Configuration
st.sidebar.subheader("🎯 Target Score")
target_score = st.sidebar.slider(
    "Target Score",
    -max_possible_score if max_possible_score > 0 else -10,
    max_possible_score if max_possible_score > 0 else 10,
    3.0 if max_possible_score >= 3 else max_possible_score/2,
    0.1
)

# Position Rules
st.sidebar.subheader("📈 Position Rules")
entry_score = st.sidebar.selectbox("Entry Score", [-1, 1], index=0)
exit_score = st.sidebar.selectbox("Exit Score", [-1, 1], index=1)

# Analysis Configuration
st.sidebar.subheader("🔍 Analysis Configuration")

# Available columns for analysis
base_columns = ['adj close', 'composite_score', 'Score_change', 'volume', 'high', 'low', 'open', 'close']
indicator_columns = ['rsi', 'rsi_mid', 'sma_short', 'sma_long', 'ema_15', 'ema_manual', 'mfi', 'mfi_mid', 
                    'stoch_k', 'stoch_d', 'aroon_up', 'aroon_down', 'aroon_oscillator',
                    'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent_b', 'bb_bandwidth',
                    'macd_line', 'macd_signal', 'macd_histogram']
signal_columns = all_signals

analysis_columns = base_columns + indicator_columns + signal_columns

score_change_cols = st.sidebar.multiselect(
    "Score Change Analysis Columns",
    analysis_columns,
    default=['Score_change', 'adj close', 'composite_score']
)

target_score_cols = st.sidebar.multiselect(
    "Target Score Analysis Columns", 
    analysis_columns,
    default=['adj close', 'composite_score', 'Score_change']
)

score_history_days = st.sidebar.slider("Score History Days", 3, 14, 5)

# Weekly Analysis Toggle
use_weekly_analysis = st.sidebar.checkbox("Include Weekly Analysis", value=True)

# Main content area
if st.button("🚀 Run Analysis", type="primary"):
    
    with st.spinner("📥 Downloading data..."):
        try:
            # Download daily data
            df = yf.download(
                tickers=symbols_list,
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=False,
                progress=False
            ).stack()
            df.index.names = ['date', 'ticker']
            df.columns = df.columns.str.lower()
            # Download weekly data if needed
            if use_weekly_analysis:
                dfw = yf.download(
                    tickers=symbols_list,
                    start=start_date,
                    end=end_date,
                    interval='1wk',
                    auto_adjust=False,
                    progress=False
                ).stack()
                dfw.index.names = ['date', 'ticker']
                dfw.columns = dfw.columns.str.lower()
        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            st.stop()

    with st.spinner("🔧 Processing indicators..."):
        try:
            # Process daily indicators
            df_with_indicators = df.groupby(level=1).apply(
                lambda x: process_symbol_indicators(x, indicator_params, signal_weights, entry_score, exit_score)
            )
            df_with_indicators = df_with_indicators.droplevel(level=0)
            df_with_indicators = df_with_indicators.sort_index()
            # Process weekly indicators if needed
            if use_weekly_analysis:
                df_with_indicatorsw = dfw.groupby(level=1, group_keys=False).apply(
                    lambda x: process_symbol_indicators(x, indicator_params, signal_weights, entry_score, exit_score)
                )
                if isinstance(df_with_indicatorsw.index, pd.MultiIndex) and df_with_indicatorsw.index.nlevels >= 3:
                    df_with_indicatorsw = df_with_indicatorsw.droplevel(level=0)
                if not isinstance(df_with_indicatorsw.index, pd.MultiIndex) or set(df_with_indicatorsw.index.names) != {'date','ticker'}:
                    df_with_indicatorsw = df_with_indicatorsw.reset_index().set_index(['date','ticker'])
                df_with_indicatorsw.index = df_with_indicatorsw.index.set_names(['date','ticker'])
                df_with_indicatorsw = df_with_indicatorsw.sort_index()
        except Exception as e:
            st.error(f"Error processing indicators: {str(e)}")
            st.stop()

    # Display results
    st.success("✅ Analysis complete!")

    # Configuration Summary
    st.header("📊 Analysis Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Symbols", len(symbols_list))
    with col2:
        st.metric("Date Range", f"{(end_date - start_date).days} days")
    with col3:
        st.metric("Selected Indicators", len(selected_indicators))
    with col4:
        st.metric("Max Possible Score", f"{max_possible_score:.1f}")

    # Top Score Increases Table
    st.header(f"� Top Score Increases (Last {score_history_days} days)")
    try:
        result_sc_ch = score_change_cross_section(
            df_with_indicators,
            cols=score_change_cols,
            only_last_date=True,
            last_n=score_history_days
        ).sort_values(by='Score_change', ascending=False)
        if use_weekly_analysis and 'df_with_indicatorsw' in locals():
            result_sc_ch = add_weekly_scores_from(
                result_sc_ch, 
                df_with_indicatorsw, 
                'composite_score'
            )
        if not result_sc_ch.empty:
            st.dataframe(result_sc_ch.head(20))
        else:
            st.info("No score increases found for the selected criteria.")
    except Exception as e:
        st.error(f"Error in score change analysis: {str(e)}")

    # Largest Score Decreases Table
    st.header("📉 Largest Score Decreases")
    try:
        if 'result_sc_ch' in locals() and not result_sc_ch.empty:
            bottom_performers = result_sc_ch.sort_values(by='Score_change', ascending=True).head(10)
            if not bottom_performers.empty:
                st.dataframe(bottom_performers)
            else:
                st.info("No score decreases found for the selected criteria.")
        else:
            st.info("No score changes data available.")
    except Exception as e:
        st.error(f"Error in score decrease analysis: {str(e)}")

    # Stocks Meeting Target Score Table (now appears after config, only once)
    st.header(f"🎯 Stocks Meeting Target Score ≥ {target_score}")
    try:
        result_tar_sc = target_score_cross_section(
            df_with_indicators,
            cols=target_score_cols,
            only_last_date=True,
            target=target_score,
            allow_ge=True,
            last_n=score_history_days
        )
        if use_weekly_analysis and 'df_with_indicatorsw' in locals():
            result_tar_sc = add_weekly_scores_from(
                result_tar_sc,
                df_with_indicatorsw,
                'composite_score'
            )
        if not result_tar_sc.empty:
            st.dataframe(result_tar_sc)
        else:
            st.info(f"No stocks found meeting target score ≥ {target_score}")
    except Exception as e:
        st.error(f"Error in target score analysis: {str(e)}")

else:
    st.info("👆 Configure your settings in the sidebar and click 'Run Analysis' to start!")

# Instructions
with st.expander("📖 How to Use This Dashboard"):
    st.markdown("""
    ### Getting Started
    1. **Configure Date Range**: Select start and end dates for your analysis
    2. **Choose Symbols**: Select from S&P 500 stocks or enter custom symbols
    3. **Select Indicators**: Choose which technical indicators to calculate
    4. **Set Parameters**: Adjust parameters for each selected indicator
    5. **Configure Scoring**: Select signals and set their weights
    6. **Set Target Score**: Define your target composite score threshold
    7. **Run Analysis**: Click the 'Run Analysis' button to process data
    
    ### Features
    - **Interactive Charts**: Hover over charts for detailed information
    - **Multiple Analysis Views**: Score changes, target scores, weekly analysis
    - **Individual Stock Analysis**: Detailed view for specific stocks
    - **Configurable Scoring**: Custom weights for different signals
    - **Weekly Integration**: Compare daily vs weekly scores
    
    ### Tips
    - Start with fewer symbols for faster processing
    - Use default indicator parameters as a starting point
    - Adjust signal weights based on your trading strategy
    - Monitor both score changes and absolute target scores
    """)
