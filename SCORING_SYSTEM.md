# Technical Analysis Stock Scoring System

A **modular, configurable scoring system** for stock screening using technical indicators. Built on Streamlit, yfinance, and pandas-ta.

## 📋 Overview

This system allows you to:
- **Score stocks** based on multiple technical indicators
- **Configure each indicator** independently (parameters, buy/sell criteria, scoring)
- **Classify stocks** as BUY, SELL, or HOLD
- **Export results** to CSV for further analysis

### Key Features
- ✅ 8 Technical Indicators (RSI, SMA, EMA, MFI, Stochastic, Aroon, Bollinger Bands, MACD)
- ✅ Per-indicator configuration: enable/disable, parameters, criteria, scores
- ✅ Centralized config in Python dict (easy to extend and edit)
- ✅ Global scoring thresholds
- ✅ Clean, modular indicator functions
- ✅ Streamlit UI for live configuration
- ✅ CSV export for backtesting

## 📦 Architecture

### Files

```
vscode-vfs://github/Hesh1994/Trading-Score/
├── scoring_config.py              # Configuration structure + defaults
├── scoring_module.py              # Core scoring logic + indicators
├── scoring_dashboard_page.py      # Streamlit UI for scoring
├── trading_dashboard.py           # (Existing) Main dashboard
└── requirements.txt               # Dependencies (updated with pandas-ta)
```

### Module Breakdown

#### 1. `scoring_config.py`
Defines the configuration structure for all indicators:

```python
INDICATORS_CONFIG = {
    'rsi': {
        'enabled': True,
        'label': 'RSI (Relative Strength Index)',
        'parameters': {'period': 14},
        'buy_criteria': {'operator': '<', 'threshold': 30},
        'sell_criteria': {'operator': '>', 'threshold': 70},
        'buy_score': 1.0,
        'sell_score': 1.0,
    },
    # ... other indicators
}

GLOBAL_CONFIG = {
    'timeframe': 'daily',
    'buy_threshold': 3.0,
    'sell_threshold': 3.0,
}
```

**Classification Logic:**
```
BUY  if total_buy_score >= buy_threshold AND total_buy_score > total_sell_score
SELL if total_sell_score >= sell_threshold AND total_sell_score > total_buy_score
HOLD otherwise
```

#### 2. `scoring_module.py`
Core scoring engine with:
- **Indicator calculation functions** (one per indicator)
- **Criteria evaluation functions** (determines if buy/sell triggered)
- **Main scoring function** (`score_stock`) – scores a single stock
- **Universe scoring** (`score_universe`) – scores multiple stocks
- **Results formatting** (`results_to_dataframe`) – output as DataFrame

#### 3. `scoring_dashboard_page.py`
Streamlit app with:
- **Configuration sidebar** for data range, symbols, thresholds
- **Indicator configuration** UI (enable/disable, edit parameters, scores)
- **Live scoring results** with color-coded signals
- **Detailed signals per stock** (which indicators triggered)
- **CSV export** for external analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Scoring Dashboard
```bash
streamlit run scoring_dashboard_page.py
```

### 3. Configure and Run
- **Select symbols** (custom, S&P 500 subset, etc.)
- **Adjust thresholds** (buy/sell scores)
- **Enable/disable indicators** and edit parameters
- **Click "Run Scoring Analysis"**

## ⚙️ Configuration Guide

### Default Indicator Criteria

| Indicator | Buy Criteria | Sell Criteria | Score |
|-----------|--------------|---------------|-------|
| **RSI(14)** | < 30 (oversold) | > 70 (overbought) | 1.0 each |
| **SMA(15/45)** | Short > Long | Short < Long | 1.5 each |
| **EMA(15)** | Price > EMA | Price < EMA | 1.0 each |
| **MFI(14)** | < 20 | > 80 | 1.0 each |
| **Stochastic(14,3,3)** | %K > %D below 20 | %K < %D above 80 | 1.0 each |
| **Aroon(25)** | Up>70 & Down<30 | Down>70 & Up<30 | 1.0 each |
| **Bollinger(20,2)** | Close ≤ Lower | Close ≥ Upper | 1.5 each |
| **MACD(12,26,9)** | Line > Signal | Line < Signal | 1.5 each |

### How to Edit Configuration

#### Method 1: Streamlit UI
Open `scoring_dashboard_page.py` and use the sidebar to:
- Toggle indicators on/off
- Adjust parameters with sliders
- Edit buy/sell scores

#### Method 2: Code (in `scoring_config.py`)
Edit `INDICATORS_CONFIG` directly:
```python
'rsi': {
    'enabled': True,
    'parameters': {'period': 14},  # Change period
    'buy_criteria': {'operator': '<', 'threshold': 30},  # Change threshold
    'buy_score': 2.0,  # Increase weight
}
```

#### Method 3: Global Settings
Edit `GLOBAL_CONFIG`:
```python
GLOBAL_CONFIG = {
    'buy_threshold': 4.0,   # More strict (higher score needed)
    'sell_threshold': 2.0,  # More lenient (lower score needed)
}
```

## 📊 How Scoring Works

### Example: Tesla (TSLA)

```
Enabled indicators: RSI, SMA, EMA, MACD

Today's signals:
- RSI(14) = 28        → BUY triggered (< 30)  → +1.0
- SMA Short > Long    → BUY triggered         → +1.5
- Price > EMA         → BUY triggered         → +1.0
- MACD > Signal       → BUY triggered         → +1.5
- MFI(14) = 55        → No signal             → +0.0

Total Buy Score: 5.0
Total Sell Score: 0.0

Classification:
- Buy score (5.0) >= buy_threshold (3.0) ✅
- Buy score > Sell score ✅
→ Signal: BUY
```

## 🔧 API Reference

### `score_stock(df, ticker, config=None, global_config=None)`
Score a single stock.

**Args:**
- `df`: DataFrame with OHLCV data (columns: open, high, low, close, volume)
- `ticker`: Stock ticker
- `config`: Indicators config (defaults to INDICATORS_CONFIG)
- `global_config`: Global config (defaults to GLOBAL_CONFIG)

**Returns:** Dict with:
```python
{
    'ticker': 'AAPL',
    'buy_score': 5.0,
    'sell_score': 1.0,
    'signal': 'BUY',
    'net_score': 4.0,
    'signals': {
        'rsi': {'value': 28.5, 'buy': True, 'sell': False},
        'sma': {'short': 150.2, 'long': 145.1, 'buy': True, 'sell': False},
        # ... other indicators
    }
}
```

### `score_universe(tickers_data, config=None, global_config=None)`
Score multiple stocks (returns sorted list of results).

**Args:**
- `tickers_data`: Dict of {ticker: DataFrame}
- `config`, `global_config`: Same as above

**Returns:** List of result dicts sorted by net_score descending.

### `results_to_dataframe(results)`
Convert results to a display-friendly DataFrame.

**Returns:** DataFrame with columns: Ticker, Signal, Buy Score, Sell Score, Net Score.

## 📈 Integration with Existing Dashboard

To add scoring to your main `trading_dashboard.py`:

```python
from scoring_module import score_stock
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG

# After downloading data:
result = score_stock(df_for_ticker, ticker, INDICATORS_CONFIG, GLOBAL_CONFIG)
st.metric(f"{ticker} Signal", result['signal'])
st.write(f"Buy Score: {result['buy_score']}, Sell Score: {result['sell_score']}")
```

Or create a **multi-page Streamlit app**:

```
pages/
├── Home.py                     # Your existing dashboard
├── Scoring.py                  # scoring_dashboard_page.py
└── Strategy_Backtest.py        # (future)
```

## 🧪 Testing

### Test a Single Stock
```python
import pandas as pd
import yfinance as yf
from scoring_module import score_stock
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG

# Download data
df = yf.download('AAPL', start='2024-01-01', end='2024-12-31')
df.columns = df.columns.str.lower()

# Score
result = score_stock(df, 'AAPL', INDICATORS_CONFIG, GLOBAL_CONFIG)
print(result)
```

### Test Multiple Stocks
```python
from scoring_module import score_universe, results_to_dataframe

tickers_data = {
    'AAPL': df_aapl,
    'MSFT': df_msft,
    'GOOGL': df_googl,
}

results = score_universe(tickers_data, INDICATORS_CONFIG, GLOBAL_CONFIG)
df_results = results_to_dataframe(results)
print(df_results)
df_results.to_csv('scores.csv', index=False)
```

## 📝 Customization Examples

### Example 1: Increase RSI Buy Score
```python
INDICATORS_CONFIG['rsi']['buy_score'] = 2.0  # Was 1.0
```

### Example 2: Add More Indicators (future extension)
Add to `INDICATORS_CONFIG`:
```python
'renko': {
    'enabled': False,
    'label': 'Renko Bricks',
    'parameters': {'brick_size': 1.0},
    'buy_criteria': {...},
    'sell_criteria': {...},
    'buy_score': 1.5,
    'sell_score': 1.5,
}
```

Then add function to `scoring_module.py`:
```python
def calculate_renko(df, brick_size=1.0):
    # Implementation
    pass

def evaluate_renko_criteria(brick_signal, ...):
    # Implementation
    pass
```

### Example 3: Change Scoring Logic
Edit `classify_signal()` in `scoring_config.py`:
```python
def classify_signal(buy_score, sell_score, buy_threshold, sell_threshold):
    # Custom logic: e.g., require >2:1 ratio
    if buy_score / sell_score >= 2.0 and buy_score >= buy_threshold:
        return 'BUY', buy_score - sell_score
    # ... etc
```

## 🐛 Troubleshooting

### "Module not found: pandas_ta"
```bash
pip install pandas-ta
```

### "No data downloaded"
- Check ticker symbols are correct
- Verify date range is valid
- Ensure yfinance can reach Yahoo Finance

### "All NA values" for an indicator
- Data may be too short for the indicator period
- Adjust indicator periods smaller
- Check that you have enough historical data

### Results all HOLD
- Buy/sell thresholds may be too high
- Lower the thresholds or increase indicator scores
- Enable more indicators

## 📚 Resources

- **pandas-ta**: https://github.com/twopirllc/pandas-ta
- **yfinance**: https://github.com/ranaroussi/yfinance
- **Streamlit**: https://streamlit.io/
- **Technical Analysis Concepts**: https://en.wikipedia.org/wiki/Technical_analysis

## 📄 License

Same as your Trading-Score project.

## 🤝 Contributing

To add new indicators:
1. Add config to `INDICATORS_CONFIG` in `scoring_config.py`
2. Add calculation function to `scoring_module.py`
3. Add evaluation function to `scoring_module.py`
4. Add evaluation logic to `score_stock()` function
5. Update UI if needed in `scoring_dashboard_page.py`

---

**Happy scoring! 📊**
