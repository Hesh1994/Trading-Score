# Quick Integration Guide

## Three Ways to Use the Scoring System

### Option 1: Run Standalone Scoring Dashboard 
**Best for:** Dedicated stock screening

```bash
streamlit run scoring_dashboard_page.py
```

This launches a full UI with configuration, analysis, and CSV export.

---

### Option 2: Add Scoring Tab to Existing Dashboard
**Best for:** Integrated experience

Create a multi-page Streamlit app structure:

```
pages/
├── 1_📈_Dashboard.py           (your trading_dashboard.py)
└── 2_📊_Stock_Scoring.py       (copy of scoring_dashboard_page.py)
```

Then run:
```bash
streamlit run 1_📈_Dashboard.py
```

Streamlit will auto-create tabs in the UI.

---

### Option 3: Programmatic Scoring in Python
**Best for:** Backtesting, batch processing, custom scripts

```python
import pandas as pd
import yfinance as yf
from scoring_module import score_stock, score_universe, results_to_dataframe
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG

# Single stock
df = yf.download('AAPL', start='2024-01-01')
df.columns = df.columns.str.lower()
result = score_stock(df, 'AAPL', INDICATORS_CONFIG, GLOBAL_CONFIG)
print(f"AAPL: {result['signal']} (Buy: {result['buy_score']}, Sell: {result['sell_score']})")

# Multiple stocks
symbols = ['AAPL', 'MSFT', 'GOOGL']
data = {s: yf.download(s, start='2024-01-01') for s in symbols}
for k in data:
    data[k].columns = data[k].columns.str.lower()

results = score_universe(data, INDICATORS_CONFIG, GLOBAL_CONFIG)
df_results = results_to_dataframe(results)
print(df_results)
df_results.to_csv('scores.csv', index=False)
```

---

## Files You've Been Given

| File | Purpose | Edit? |
|------|---------|-------|
| `scoring_config.py` | All indicator configs + defaults | ✅ YES |
| `scoring_module.py` | Core scoring engine | ⚠️ Advanced |
| `scoring_dashboard_page.py` | Streamlit UI | ✅ Customize |
| `SCORING_SYSTEM.md` | Full documentation | 📖 Reference |
| `requirements.txt` | Dependencies (updated with pandas-ta) | ✅ Run `pip install -r requirements.txt` |

---

## First Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test It Works
```bash
python -c "from scoring_module import score_stock; print('✅ Import successful')"
```

### 3. Run the Dashboard
```bash
streamlit run scoring_dashboard_page.py
```

### 4. Customize (Optional)
Edit `scoring_config.py` to change:
- Default indicator parameters
- Buy/sell thresholds
- Indicator scores
- Enable/disable specific indicators

---

## Common Edits

### Change RSI Threshold
In `scoring_config.py`:
```python
'rsi': {
    'buy_criteria': {'operator': '<', 'threshold': 25},  # Was 30
    'sell_criteria': {'operator': '>', 'threshold': 75},  # Was 70
}
```

### Increase Buy Threshold (stricter)
In `scoring_config.py`:
```python
GLOBAL_CONFIG = {
    'buy_threshold': 5.0,  # Was 3.0 (require more points to trigger BUY)
}
```

### Disable an Indicator
In `scoring_config.py`:
```python
'mfi': {
    'enabled': False,  # Toggle on/off
    # ... rest of config
}
```

Or in the UI: toggle in sidebar.

---

## Typical Workflow

1. **Configure**: Set date range, symbols, thresholds
2. **Adjust Indicators**: Enable/disable, tweak parameters and scores
3. **Run Analysis**: Click button, wait ~30 seconds for data
4. **Review Results**: See BUY/SELL/HOLD signals, sorted by strength
5. **Export**: Download CSV for backtesting or manual review

---

## What's Different from Your Original Dashboard?

| Feature | Original | Scoring System |
|---------|----------|-----------------|
| Purpose | Composite score tracking | Stock screening/classification |
| Output | Trends over time | BUY/SELL/HOLD signals |
| Indicators | Pre-calculated | Configurable |
| Scoring | Weighted signals | Threshold-based classification |
| Use Case | Watch portfolio trend | Find entry/exit points |

**Both can run together!** Use the original for monitoring, scoring system for screening.

---

## Need Help?

1. **Installation issues?** → Run `pip install --upgrade pandas-ta yfinance streamlit`
2. **Pandas deprecated warnings?** → Already fixed (using `.bfill()`, `.ffill()`)
3. **No results showing?** → Check your date range and ticker symbols
4. **Want more indicators?** → See SCORING_SYSTEM.md for extension guide

---

## Next Steps (Optional Enhancements)

- [ ] Add price targets based on scores
- [ ] Email alerts when BUY signal detected
- [ ] Historical backtest against past signals
- [ ] Compare strategies (different weight combinations)
- [ ] Database to track signals over time
- [ ] Add more indicators (RSX, KDJ, etc.)

---

**Ready to screen stocks! 🚀**
