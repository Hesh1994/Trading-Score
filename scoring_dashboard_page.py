"""
Streamlit Scoring Dashboard Integration
Add this to your main trading_dashboard.py or create as a separate page in a multi-page Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import requests
import io
from scoring_module import score_universe, results_to_dataframe
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="Stock Scoring System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Technical Analysis Stock Scoring System")

# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Data Range
st.sidebar.subheader("📅 Data Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=dt.date.today() - dt.timedelta(days=365),
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

# Timeframe selection
st.sidebar.subheader("⏱️ Timeframe")
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["Daily", "Weekly", "Monthly"],
    index=0
)

# Symbol Selection
st.sidebar.subheader("🎯 Symbols")
symbol_option = st.sidebar.selectbox(
    "Symbol Source",
    ["Custom", "S&P 500 (first 20)", "S&P 500 (first 50)"]
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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            response = requests.get(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            sp500 = pd.read_html(io.StringIO(response.text))[0]
            sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
            return sp500['Symbol'].unique().tolist()
        except Exception as e:
            st.error(f"Failed to fetch S&P 500 symbols: {str(e)}")
            return []
    
    sp500_symbols = get_sp500_symbols()
    if symbol_option == "S&P 500 (first 20)":
        symbols_list = sp500_symbols[:20]
    else:  # first 50
        symbols_list = sp500_symbols[:50]

# Scoring Thresholds
st.sidebar.subheader("🎯 Scoring Thresholds")
col1, col2 = st.sidebar.columns(2)
with col1:
    buy_threshold = st.number_input("Buy Threshold", value=3.0, min_value=0.0, step=0.5)
with col2:
    sell_threshold = st.number_input("Sell Threshold", value=3.0, min_value=0.0, step=0.5)

# Indicator Configuration
st.sidebar.subheader("📈 Indicators to Include")

# Initialize indicator_config
indicator_config = {}
for ind_key, ind_config in INDICATORS_CONFIG.items():
    indicator_config[ind_key] = ind_config.copy()

# Section 1: Select which indicators to include
included_indicators = {}
for ind_key, ind_config in INDICATORS_CONFIG.items():
    included = st.sidebar.checkbox(
        ind_config['label'],
        value=ind_config['enabled'],
        key=f"{ind_key}_included"
    )
    indicator_config[ind_key]['enabled'] = included
    if included:
        included_indicators[ind_key] = ind_config

# Section 2: Configure selected indicators
if included_indicators:
    st.sidebar.subheader("⚙️ Configure Indicators")
    
    selected_indicator = st.sidebar.selectbox(
        "Select indicator to configure",
        options=list(included_indicators.keys()),
        format_func=lambda x: included_indicators[x]['label'],
        key="indicator_selector"
    )
    
    if selected_indicator:
        ind_config = indicator_config[selected_indicator]
        st.sidebar.write(f"**{ind_config['label']}**")
        
        # Show parameters
        st.sidebar.write("Parameters:")
        params = ind_config['parameters'].copy()
        
        for param_key, param_value in params.items():
            if isinstance(param_value, int):
                new_value = st.sidebar.number_input(
                    param_key.replace('_', ' ').title(),
                    min_value=1,
                    value=param_value,
                    step=1,
                    key=f"{selected_indicator}_{param_key}"
                )
                indicator_config[selected_indicator]['parameters'][param_key] = new_value
            elif isinstance(param_value, float):
                new_value = st.sidebar.number_input(
                    param_key.replace('_', ' ').title(),
                    min_value=0.1,
                    value=param_value,
                    step=0.1,
                    key=f"{selected_indicator}_{param_key}"
                )
                indicator_config[selected_indicator]['parameters'][param_key] = new_value
        
        # Buy / sell rules (threshold inputs) — RSI only for now
        if selected_indicator == "rsi":
            st.sidebar.write("Buy / Sell Rules:")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                buy_thresh = st.sidebar.number_input(
                    "Buy when RSI <",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(ind_config["buy_criteria"].get("threshold", 30)),
                    step=1.0,
                    key="rsi_buy_threshold",
                )
                indicator_config["rsi"]["buy_criteria"]["threshold"] = buy_thresh
            with col2:
                sell_thresh = st.sidebar.number_input(
                    "Sell when RSI >",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(ind_config["sell_criteria"].get("threshold", 70)),
                    step=1.0,
                    key="rsi_sell_threshold",
                )
                indicator_config["rsi"]["sell_criteria"]["threshold"] = sell_thresh

        # Show buy/sell scores
        st.sidebar.write("Scoring:")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            buy_score = st.number_input(
                "Buy Score",
                value=float(ind_config['buy_score']),
                step=0.5,
                key=f"{selected_indicator}_buy_score"
            )
            indicator_config[selected_indicator]['buy_score'] = buy_score
        with col2:
            sell_score = st.number_input(
                "Sell Score",
                value=float(ind_config['sell_score']),
                step=0.5,
                key=f"{selected_indicator}_sell_score"
            )
            indicator_config[selected_indicator]['sell_score'] = sell_score

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.button("🚀 Run Scoring Analysis", type="primary", use_container_width=True):
    with st.spinner("📥 Downloading data..."):
        try:
            # Download data
            df_raw = yf.download(
                tickers=symbols_list,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
            
            if df_raw.empty:
                st.error("No data downloaded. Check your symbols and date range.")
                st.stop()
            
            # Prepare data for scoring (dictionary of {ticker: DataFrame})
            tickers_data = {}
            
            if len(symbols_list) == 1:
                # Single ticker: columns are simple (Adj Close, Close, High, Low, Open, Volume)
                ticker = symbols_list[0]
                ticker_df = df_raw.copy()
                ticker_df.columns = ticker_df.columns.str.lower()
                
                # Check for required columns (case-insensitive)
                cols_lower = {c.lower() for c in df_raw.columns}
                if 'close' in cols_lower and 'open' in cols_lower and 'volume' in cols_lower:
                    # Drop NaN rows at the end
                    ticker_df = ticker_df.dropna(subset=['close'])
                    if len(ticker_df) > 0:
                        tickers_data[ticker] = ticker_df
            else:
                # Multiple tickers: columns are MultiIndex (PriceType, Ticker)
                # Extract ticker data from MultiIndex columns
                for ticker in symbols_list:
                    try:
                        # For MultiIndex, ticker is at level 1 (second level)
                        # Access using df_raw.xs(ticker, level=1, axis=1)
                        ticker_df = df_raw.xs(ticker, level=1, axis=1).copy()
                        ticker_df.columns = ticker_df.columns.str.lower()
                        
                        # Check for required columns
                        cols_lower = {c.lower() for c in ticker_df.columns}
                        if 'close' in cols_lower and 'open' in cols_lower and 'volume' in cols_lower:
                            # Drop NaN rows at the end
                            ticker_df = ticker_df.dropna(subset=['close'])
                            if len(ticker_df) > 0:
                                tickers_data[ticker] = ticker_df
                    except (KeyError, TypeError):
                        continue
            
            if not tickers_data:
                st.error(f"No valid data for selected symbols. Tried: {', '.join(symbols_list[:5])}")
                st.stop()
            
        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            st.stop()
    
    with st.spinner("🔧 Scoring stocks..."):
        try:
            # Prepare global config
            global_config = {
                'timeframe': timeframe.lower(),
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
            }
            
            # Score all stocks
            results = score_universe(tickers_data, indicator_config, global_config)
            
            if not results:
                st.error("No scoring results generated.")
                st.stop()
            
            # Convert to display DataFrame
            results_df = results_to_dataframe(results)
            
        except Exception as e:
            st.error(f"Error during scoring: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()
    
    # ========== DISPLAY RESULTS ==========
    st.success("✅ Scoring complete!")
    
    # Summary metrics
    st.header("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    buy_count = len(results_df[results_df['Signal'] == 'BUY'])
    sell_count = len(results_df[results_df['Signal'] == 'SELL'])
    hold_count = len(results_df[results_df['Signal'] == 'HOLD'])
    
    with col1:
        st.metric("Total Stocks Analyzed", len(results_df))
    with col2:
        st.metric("Buy Signals", buy_count, delta_color="off")
    with col3:
        st.metric("Sell Signals", sell_count, delta_color="off")
    with col4:
        st.metric("Hold Signals", hold_count, delta_color="off")
    
    # Results table
    st.header("🎯 Scoring Results")
    
    st.dataframe(
        results_df.style.format({
            'Buy Score': '{:.2f}',
            'Sell Score': '{:.2f}',
            'Net Score': '{:.2f}'
        }),
        use_container_width=True
    )
    
    # Detailed signals per stock
    st.header("🔍 Detailed Signals per Stock")
    
    for i, result in enumerate(results[:min(10, len(results))]):  # Show top 10
        with st.expander(f"{result['ticker']} - {result['signal']} (Net: {result['net_score']})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy Score", result['buy_score'])
            with col2:
                st.metric("Sell Score", result['sell_score'])
            with col3:
                st.metric("Net Score", result['net_score'])
            
            st.write("**Signals by Indicator:**")
            signals_data = []
            for ind_key, sig_data in result['signals'].items():
                if 'error' not in sig_data:
                    buy_triggered = "✅ BUY" if sig_data.get('buy') else ""
                    sell_triggered = "⛔ SELL" if sig_data.get('sell') else ""
                    
                    signal_str = f"{buy_triggered} {sell_triggered}".strip()
                    if not signal_str:
                        signal_str = "—"
                    
                    signals_data.append({
                        'Indicator': ind_key.upper(),
                        'Signal': signal_str,
                        'Details': str(sig_data).replace("'buy': ", "").replace("'sell': ", "")
                    })
            
            if signals_data:
                st.dataframe(pd.DataFrame(signals_data), use_container_width=True)
    
    # Export to CSV
    st.header("📥 Export Results")
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name=f"stock_scores_{dt.date.today()}.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Configure your settings and click 'Run Scoring Analysis' to start!")

# Instructions
with st.expander("📖 How to Use the Scoring System"):
    st.markdown("""
    ### Setup
    1. **Select Data Range**: Choose the historical period for indicator calculations
    2. **Choose Symbols**: Pick from S&P 500 or enter custom tickers
    3. **Configure Thresholds**: Set minimum buy/sell scores for classification
    
    ### Indicator Configuration
    For each indicator you enable:
    - **Parameters**: Adjust periods and settings (e.g., RSI length, SMA periods)
    - **Buy/Sell Scores**: Points awarded when buy/sell criteria are met
    - Toggle indicators ON/OFF to include/exclude them
    
    ### Scoring Logic
    Each stock receives:
    - **Buy Score**: Sum of buy scores from indicators with triggered buy criteria
    - **Sell Score**: Sum of sell scores from indicators with triggered sell criteria
    - **Classification**:
      - **BUY**: Total buy score ≥ threshold AND buy score > sell score
      - **SELL**: Total sell score ≥ threshold AND sell score > buy score
      - **HOLD**: Otherwise
    
    ### Results
    The table shows all stocks ranked by net score (buy − sell) descending.
    - Click on a stock to see which indicators triggered buy/sell signals
    - Export results to CSV for further analysis
    
    ### Default Criteria
    - **RSI(14)**: Buy <30, Sell >70
    - **SMA**: Buy when short crosses above long
    - **EMA**: Buy when price >EMA, Sell when price <EMA
    - **MFI(14)**: Buy <20, Sell >80
    - **Stochastic**: Buy %K crosses %D below 20
    - **Aroon(25)**: Buy when Up>70 & Down<30
    - **Bollinger(20,2)**: Buy at lower band, Sell at upper band
    - **MACD(12,26,9)**: Buy when line crosses above signal
    """)
