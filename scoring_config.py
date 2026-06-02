"""
Scoring Configuration Module
Centralized configuration for all technical indicators and scoring thresholds.
Easy to edit and extend—all settings in one place.
"""

# ============================================================================
# INDICATOR CONFIGURATION
# ============================================================================
# Each indicator has: enabled, parameters, buy criteria, sell criteria, scores

INDICATORS_CONFIG = {
    'rsi': {
        'enabled': True,
        'label': 'RSI (Relative Strength Index)',
        'interval': 'daily',
        'parameters': {
            'period': 14,
            'buy_threshold': 50.0,      # Buy when RSI > this value
            'sell_threshold': 50.0,     # Sell when RSI < this value
        },
        'buy_criteria': {
            'operator': '>',  # RSI > buy_threshold
            'threshold_type': 'parameter',  # Use buy_threshold from parameters
        },
        'sell_criteria': {
            'operator': '<',  # RSI < sell_threshold
            'threshold_type': 'parameter',  # Use sell_threshold from parameters
        },
        'buy_score': 1.0,
        'sell_score': 1.0,
    },
    
    'sma': {
        'enabled': True,
        'label': 'SMA (Simple Moving Average)',
        'interval': 'daily',
        'parameters': {
            'period_short': 15,
            'period_long': 45,
        },
        'buy_criteria': {
            'type': 'crossover',  # sma_short crosses above sma_long
            'direction': 'above',
        },
        'sell_criteria': {
            'type': 'crossover',
            'direction': 'below',
        },
        'buy_score': 1.5,
        'sell_score': 1.5,
    },
    
    'ema': {
        'enabled': True,
        'label': 'EMA (Exponential Moving Average)',
        'interval': 'daily',
        'parameters': {
            'period': 15,
        },
        'buy_criteria': {
            'operator': '>',  # price > EMA
            'threshold_type': 'ema',  # Compare to EMA value
        },
        'sell_criteria': {
            'operator': '<',  # price < EMA
            'threshold_type': 'ema',
        },
        'buy_score': 1.0,
        'sell_score': 1.0,
    },
    
    'mfi': {
        'enabled': True,
        'label': 'MFI (Money Flow Index)',
        'interval': 'daily',
        'parameters': {
            'period': 14,
        },
        'buy_criteria': {
            'operator': '<',
            'threshold': 20,
        },
        'sell_criteria': {
            'operator': '>',
            'threshold': 80,
        },
        'buy_score': 1.0,
        'sell_score': 1.0,
    },
    
    'stochastic': {
        'enabled': True,
        'label': 'Stochastic Oscillator',
        'interval': 'daily',
        'parameters': {
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 3,
        },
        'buy_criteria': {
            'type': 'crossover',  # %K crosses above %D below 20
            'direction': 'above',
            'threshold': 20,
        },
        'sell_criteria': {
            'type': 'crossover',  # %K crosses below %D above 80
            'direction': 'below',
            'threshold': 80,
        },
        'buy_score': 1.0,
        'sell_score': 1.0,
    },
    
    'aroon': {
        'enabled': True,
        'label': 'Aroon Indicator',
        'interval': 'daily',
        'parameters': {
            'period': 25,
        },
        'buy_criteria': {
            'type': 'combined',  # AroonUp > 70 AND AroonDown < 30
            'aroon_up_threshold': 70,
            'aroon_down_threshold': 30,
        },
        'sell_criteria': {
            'type': 'combined',  # AroonDown > 70 AND AroonUp < 30
            'aroon_down_threshold': 70,
            'aroon_up_threshold': 30,
        },
        'buy_score': 1.0,
        'sell_score': 1.0,
    },
    
    'bollinger': {
        'enabled': True,
        'label': 'Bollinger Bands',
        'interval': 'daily',
        'parameters': {
            'period': 20,
            'std_dev': 2.0,
        },
        'buy_criteria': {
            'operator': '<=',  # close <= lower band
            'threshold_type': 'bb_lower',
        },
        'sell_criteria': {
            'operator': '>=',  # close >= upper band
            'threshold_type': 'bb_upper',
        },
        'buy_score': 1.5,
        'sell_score': 1.5,
    },
    
    'macd': {
        'enabled': True,
        'label': 'MACD (Moving Average Convergence Divergence)',
        'interval': 'daily',
        'parameters': {
            'fast': 12,
            'slow': 26,
            'signal': 9,
        },
        'buy_criteria': {
            'type': 'crossover',  # MACD crosses above signal line
            'direction': 'above',
        },
        'sell_criteria': {
            'type': 'crossover',  # MACD crosses below signal line
            'direction': 'below',
        },
        'buy_score': 1.5,
        'sell_score': 1.5,
    },
}

# ============================================================================
# GLOBAL SCORING CONFIGURATION
# ============================================================================

GLOBAL_CONFIG = {
    'timeframe': 'daily',  # 'daily', 'weekly', 'monthly'
    'buy_threshold': 3.0,   # Minimum total buy score to label as BUY
    'sell_threshold': 3.0,  # Minimum total sell score to label as SELL
    'lookback_days': 365,   # Default number of days to look back for analysis
}

# ============================================================================
# CLASSIFICATION RULES
# ============================================================================
"""
Scoring logic (keep in one place for easy changes):

1. Compute enabled indicators on selected timeframe
2. Evaluate each indicator's buy and sell criteria
3. total_buy_score  = sum of buy_score where buy criteria met
   total_sell_score = sum of sell_score where sell criteria met
4. Classify:
   - BUY  if total_buy_score >= buy_threshold AND total_buy_score > total_sell_score
   - SELL if total_sell_score >= sell_threshold AND total_sell_score > total_buy_score
   - HOLD otherwise
"""


def classify_signal(buy_score, sell_score, buy_threshold, sell_threshold):
    """
    Classify stock as BUY, SELL, or HOLD based on scoring logic.
    
    Args:
        buy_score: Total buy score from all enabled indicators
        sell_score: Total sell score from all enabled indicators
        buy_threshold: Minimum buy score threshold
        sell_threshold: Minimum sell score threshold
    
    Returns:
        Tuple of (signal, net_score) where signal is 'BUY', 'SELL', or 'HOLD'
    """
    net_score = buy_score - sell_score
    
    if buy_score >= buy_threshold and buy_score > sell_score:
        return 'BUY', net_score
    elif sell_score >= sell_threshold and sell_score > buy_score:
        return 'SELL', net_score
    else:
        return 'HOLD', net_score
