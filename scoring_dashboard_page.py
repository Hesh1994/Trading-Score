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

# ── Page navigation ──────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "📌 Navigate",
    ["📊 Technical Analysis Scoring", "📈 CANSLIM Scoring"],
    key="main_page_selector"
)
st.sidebar.divider()

# ============================================================================
# PAGE: CANSLIM SCORING  (rendered first; st.stop() prevents TA code from running)
# ============================================================================
if page == "📈 CANSLIM Scoring":
    from canslim_module import score_canslim_universe

    def _pct(val, d=1):
        return f"{val*100:.{d}f}%" if val is not None else "—"

    def _traffic(met):
        return "✅" if met is True else ("❌" if met is False else "⚠️ no data")

    def _fmt_criterion(label, val):
        if "Acceleration" in label:
            return ("✅ TRUE" if val is True else ("❌ FALSE" if val is False else "no data"))
        return _pct(val)

    st.title("📈 CANSLIM Multi-Ticker Scoring & Ranking")
    st.caption(
        "Scores each ticker across 10 CANSLIM criteria (10 pts each, max 100). "
        "Data via yfinance — needs at least 8 quarters / 6 years of history."
    )

    st.sidebar.subheader("🎯 Tickers")
    ticker_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, NVDA, AMZN",
        height=100,
        key="canslim_tickers"
    )
    st.sidebar.markdown(
        "**Criteria (10 pts each)**\n"
        "- Recent Qtr EPS YoY > 20%\n"
        "- EPS Acceleration (3 Qtrs)\n"
        "- EPS 3Y Avg Growth > 25%\n"
        "- EPS 5Y Avg Growth > 25%\n"
        "- Recent Qtr Revenue YoY > 25%\n"
        "- Avg Revenue Growth (3Q) > 25%\n"
        "- Pretax Income Annual > 15%\n"
        "- Net Income Annual > 25%\n"
        "- ROE YoY Growth > 17%\n"
        "- Institutional Ownership > 35%"
    )
    run_canslim = st.sidebar.button("🚀 Run CANSLIM Analysis", type="primary",
                                    use_container_width=True, key="run_canslim")

    if run_canslim:
        symbols = [s.strip().upper() for s in ticker_input.replace("\n", ",").split(",") if s.strip()]
        if not symbols:
            st.warning("Enter at least one ticker symbol.")
        else:
            with st.spinner(f"Fetching fundamental data for {len(symbols)} ticker(s)…"):
                results = score_canslim_universe(symbols)

            # ── Ranked summary table ─────────────────────────────────────
            st.subheader("🏆 Ranked Scoring Table")
            rows, rank, prev = [], 1, None
            for i, r in enumerate(results):
                if r['score'] != prev:
                    rank = i + 1
                rows.append({'Rank': rank, 'Ticker': r['symbol'],
                             'Total Score (/100)': r['score'],
                             'Criteria Met': f"{r['criteria_met']} / 10",
                             'Data Gaps': r['data_gaps']})
                prev = r['score']
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Per-ticker detailed breakdown ────────────────────────────
            st.subheader("🔍 Detailed Breakdown")
            for r in results:
                sym = r['symbol']
                with st.expander(f"{sym}  —  {r['score']}/100  |  {r['criteria_met']}/10 criteria  |  {r['data_gaps']} gap(s)"):
                    m = r['metrics']
                    lq = r['q_dates'][0] if r['q_dates'] else "—"
                    la = r['a_dates'][0] if r['a_dates'] else "—"

                    # Metrics table
                    st.markdown("**Computed Metrics**")
                    st.dataframe(pd.DataFrame([
                        ("#1-4",  "EPS YoY Growth Q4/Q3/Q2/Q1",
                         f"{_pct(m.get('eps_gr_q4'))} / {_pct(m.get('eps_gr_q3'))} / {_pct(m.get('eps_gr_q2'))} / {_pct(m.get('eps_gr_q1'))}", lq),
                        ("#3",   "EPS Acceleration (3 Qtrs)",   _fmt_criterion("Acceleration", m.get('eps_accel')), lq),
                        ("#4",   "EPS 3Y Avg Annual Growth",    _pct(m.get('eps_3y_avg')), la),
                        ("#5",   "EPS 5Y Avg Annual Growth",    _pct(m.get('eps_5y_avg')), la),
                        ("#6",   "Revenue YoY Growth Q4",       _pct(m.get('rev_gr_q4')), lq),
                        ("#7",   "Avg Revenue Growth (Q2-Q4)",  _pct(m.get('rev_3q_avg')), lq),
                        ("#8",   "Pretax Income Annual Growth", _pct(m.get('pretax_ann_gr')), la),
                        ("#9",   "Net Income Annual Growth",    _pct(m.get('ni_ann_gr')), la),
                        ("#10a", "ROE Q4 (latest)",             _pct(m.get('roe_q4')), lq),
                        ("#10b", "ROE YoY Growth",              _pct(m.get('roe_yoy_gr')), lq),
                        ("#11",  "Institutional Ownership",     _pct(m.get('inst_pct')), "latest"),
                    ], columns=["#", "Metric", "Value", "Period"]),
                    use_container_width=True, hide_index=True)

                    # Scoring table
                    st.markdown("**Scoring Criteria**")
                    score_rows = [{'Criterion': sd['Criterion'],
                                   'Value': _fmt_criterion(sd['Criterion'], sd['Value']),
                                   'Threshold': sd['Threshold'],
                                   'Points': sd['Points'],
                                   'Status': _traffic(sd['Met'])}
                                  for sd in r['score_details']]
                    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

                    st.metric(f"Total CANSLIM Score — {sym}", f"{r['score']} / 100")

                    # Quarterly trend
                    st.markdown("**Last 4 Quarters — YoY Growth Trend**")
                    q_labels = ["Q1 (oldest)", "Q2", "Q3", "Q4 (latest)"]
                    trend = [{'Quarter': q_labels[i],
                              'EPS YoY': _pct(r['eps_gr_series'][i] if i < len(r['eps_gr_series']) else None),
                              'Revenue YoY': _pct(r['rev_gr_series'][i] if i < len(r['rev_gr_series']) else None)}
                             for i in range(4)]
                    st.dataframe(pd.DataFrame(trend), use_container_width=True, hide_index=True)

                    if r['errors']:
                        st.caption("⚠️ Data gaps: " + " · ".join(r['errors']))
    else:
        st.info("👆 Enter tickers in the sidebar and click **Run CANSLIM Analysis** to start.")

    st.stop()   # ← prevents the Technical Analysis code below from running

# ============================================================================
# PAGE: TECHNICAL ANALYSIS SCORING
# ============================================================================
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

import copy as _copy

# ── Session-state-backed config store ─────────────────────────────────────────
# Streamlit removes widget keys from session_state when their widgets are not
# rendered (e.g. when switching to a different indicator in the selectbox).
# We keep our own mirror dict so all values survive across reruns.
if 'ind_config_store' not in st.session_state:
    st.session_state['ind_config_store'] = _copy.deepcopy(dict(INDICATORS_CONFIG))
else:
    # Merge any indicators added since the store was first created
    for _k, _v in INDICATORS_CONFIG.items():
        if _k not in st.session_state['ind_config_store']:
            st.session_state['ind_config_store'][_k] = _copy.deepcopy(_v)

# Start each run from the persisted store, not from the hard-coded defaults.
indicator_config = _copy.deepcopy(st.session_state['ind_config_store'])

# Section 1: Select which indicators to include
included_indicators = {}
for ind_key in list(INDICATORS_CONFIG.keys()):
    cfg = indicator_config[ind_key]
    included = st.sidebar.checkbox(
        cfg['label'],
        value=cfg['enabled'],
        key=f"{ind_key}_included"
    )
    indicator_config[ind_key]['enabled'] = included
    if included:
        included_indicators[ind_key] = indicator_config[ind_key]

# Section 2: Configure selected indicators
if included_indicators:
    st.sidebar.subheader("⚙️ Configure Indicators")

    selected_indicator = st.sidebar.selectbox(
        "Select indicator to configure",
        options=list(included_indicators.keys()),
        format_func=lambda x: indicator_config[x]['label'],
        key="indicator_selector"
    )

    if selected_indicator:
        ind_cfg = indicator_config[selected_indicator]
        st.sidebar.write(f"**{ind_cfg['label']}**")

        # Interval selector per indicator
        interval_options = ["Daily", "Weekly", "Monthly"]
        current_interval = ind_cfg.get('interval', 'daily').capitalize()
        if current_interval not in interval_options:
            current_interval = "Daily"
        selected_interval = st.sidebar.selectbox(
            "Interval",
            interval_options,
            index=interval_options.index(current_interval),
            key=f"{selected_indicator}_interval"
        )
        indicator_config[selected_indicator]['interval'] = selected_interval.lower()
        if selected_indicator == 'fear_greed':
            st.sidebar.caption("Calculated from OHLCV data (needs 252+ bars). Components: Momentum, RSI, Volatility, Volume Breadth.")

        # Show parameters
        st.sidebar.write("Parameters:")
        params = ind_cfg['parameters'].copy()

        for param_key, param_value in params.items():
            # buy_threshold / sell_threshold for RSI are handled by the dedicated block below
            if selected_indicator == "rsi" and param_key in ("buy_threshold", "sell_threshold"):
                continue
            if isinstance(param_value, int):
                new_value = st.sidebar.number_input(
                    param_key.replace('_', ' ').title(),
                    min_value=1,
                    value=int(param_value),
                    step=1,
                    key=f"{selected_indicator}_{param_key}"
                )
                indicator_config[selected_indicator]['parameters'][param_key] = new_value
            elif isinstance(param_value, float):
                new_value = st.sidebar.number_input(
                    param_key.replace('_', ' ').title(),
                    min_value=0.1,
                    value=float(param_value),
                    step=0.1,
                    key=f"{selected_indicator}_{param_key}"
                )
                indicator_config[selected_indicator]['parameters'][param_key] = new_value

        # Buy / sell rules (threshold inputs) — RSI only for now
        if selected_indicator == "rsi":
            st.sidebar.write("Buy / Sell Rules:")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                buy_thresh = st.number_input(
                    "Oversold threshold (<)",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(ind_cfg["parameters"].get("buy_threshold", 30.0)),
                    step=1.0,
                    key="rsi_buy_threshold",
                )
                indicator_config["rsi"]["parameters"]["buy_threshold"] = buy_thresh
            with col2:
                sell_thresh = st.number_input(
                    "Overbought threshold (>)",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(ind_cfg["parameters"].get("sell_threshold", 70.0)),
                    step=1.0,
                    key="rsi_sell_threshold",
                )
                indicator_config["rsi"]["parameters"]["sell_threshold"] = sell_thresh

        # Show buy/sell scores
        st.sidebar.write("Scoring:")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            buy_score = st.number_input(
                "Buy Score",
                value=float(ind_cfg['buy_score']),
                step=0.5,
                key=f"{selected_indicator}_buy_score"
            )
            indicator_config[selected_indicator]['buy_score'] = buy_score
        with col2:
            sell_score = st.number_input(
                "Sell Score",
                value=float(ind_cfg['sell_score']),
                step=0.5,
                key=f"{selected_indicator}_sell_score"
            )
            indicator_config[selected_indicator]['sell_score'] = sell_score

# ── Persist the updated config for the next rerun ─────────────────────────────
st.session_state['ind_config_store'] = _copy.deepcopy(indicator_config)

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.button("🚀 Run Scoring Analysis", type="primary", use_container_width=True):
    with st.spinner("📥 Downloading data..."):
        try:
            INTERVAL_YF = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}

            # Collect unique intervals needed by enabled indicators
            intervals_needed = set()
            for ind_key, cfg in indicator_config.items():
                if cfg.get('enabled'):
                    intervals_needed.add(cfg.get('interval', 'daily'))

            tickers_data_by_interval = {}

            for interval in intervals_needed:
                df_raw = yf.download(
                    tickers=symbols_list,
                    start=start_date,
                    end=end_date,
                    interval=INTERVAL_YF[interval],
                    progress=False,
                    auto_adjust=False
                )

                if df_raw.empty:
                    continue

                tickers_dict = {}

                if len(symbols_list) == 1:
                    ticker = symbols_list[0]
                    ticker_df = df_raw.copy()
                    ticker_df.columns = ticker_df.columns.str.lower()
                    ticker_df = ticker_df.dropna(subset=['close'])
                    if len(ticker_df) > 0:
                        tickers_dict[ticker] = ticker_df
                else:
                    for ticker in symbols_list:
                        try:
                            ticker_df = df_raw.xs(ticker, level=1, axis=1).copy()
                            ticker_df.columns = ticker_df.columns.str.lower()
                            ticker_df = ticker_df.dropna(subset=['close'])
                            if len(ticker_df) > 0:
                                tickers_dict[ticker] = ticker_df
                        except (KeyError, TypeError):
                            continue

                tickers_data_by_interval[interval] = tickers_dict

            if not tickers_data_by_interval:
                st.error("No data downloaded. Check your symbols and date range.")
                st.stop()

            # Check at least one ticker has data
            all_tickers_found = set()
            for d in tickers_data_by_interval.values():
                all_tickers_found.update(d.keys())
            if not all_tickers_found:
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
            results = score_universe(tickers_data_by_interval, indicator_config, global_config)
            
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

    # Fear & Greed summary across all tickers (per-ticker calculation)
    if indicator_config.get('fear_greed', {}).get('enabled'):
        fg_vals = [
            r['signals']['fear_greed']['value']
            for r in results
            if 'fear_greed' in r['signals'] and 'value' in r['signals']['fear_greed']
        ]
        if fg_vals:
            fear_thresh  = indicator_config['fear_greed']['parameters'].get('fear_threshold',  30.0)
            greed_thresh = indicator_config['fear_greed']['parameters'].get('greed_threshold', 70.0)
            n_fear   = sum(v < fear_thresh  for v in fg_vals)
            n_greed  = sum(v > greed_thresh for v in fg_vals)
            avg_fg   = sum(fg_vals) / len(fg_vals)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg Fear & Greed", f"{avg_fg:.1f} / 100")
            col_b.metric(f"Fear Zone (< {fear_thresh:.0f})",   n_fear)
            col_c.metric(f"Greed Zone (> {greed_thresh:.0f})", n_greed)

    # Detect RSI-only mode
    enabled_indicators = [k for k, v in indicator_config.items() if v.get("enabled")]
    rsi_only = enabled_indicators == ["rsi"]

    if rsi_only:
        # ── RSI-only: two tables (Oversold / Overbought) ──────────────────────
        rsi_buy_thresh  = indicator_config["rsi"]["parameters"].get("buy_threshold",  30)
        rsi_sell_thresh = indicator_config["rsi"]["parameters"].get("sell_threshold", 70)

        oversold_rows   = []
        overbought_rows = []
        for r in results:
            rsi_sig = r["signals"].get("rsi", {})
            rsi_val = rsi_sig.get("value")
            if rsi_sig.get("buy"):
                oversold_rows.append({"Ticker": r["ticker"], "RSI": rsi_val})
            if rsi_sig.get("sell"):
                overbought_rows.append({"Ticker": r["ticker"], "RSI": rsi_val})

        df_oversold   = pd.DataFrame(oversold_rows,   columns=["Ticker", "RSI"]).sort_values("RSI", ascending=True).reset_index(drop=True)
        df_overbought = pd.DataFrame(overbought_rows, columns=["Ticker", "RSI"]).sort_values("RSI", ascending=False).reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyzed", len(results))
        with col2:
            st.metric("Oversold", len(df_oversold))

        st.subheader(f"📉 Oversold — RSI < {rsi_buy_thresh}")
        if df_oversold.empty:
            st.info("No oversold stocks found.")
        else:
            st.dataframe(
                df_oversold.style.format({"RSI": "{:.2f}"}),
                use_container_width=True, hide_index=True
            )

        st.subheader(f"📈 Overbought — RSI > {rsi_sell_thresh}")
        if df_overbought.empty:
            st.info("No overbought stocks found.")
        else:
            st.dataframe(
                df_overbought.style.format({"RSI": "{:.2f}"}),
                use_container_width=True, hide_index=True
            )

    else:
        # ── Multi-indicator: original scoring table ───────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        buy_count  = len(results_df[results_df["Signal"] == "BUY"])
        sell_count = len(results_df[results_df["Signal"] == "SELL"])
        hold_count = len(results_df[results_df["Signal"] == "HOLD"])
        with col1:
            st.metric("Total Stocks Analyzed", len(results_df))
        with col2:
            st.metric("Buy Signals",  buy_count,  delta_color="off")
        with col3:
            st.metric("Sell Signals", sell_count, delta_color="off")
        with col4:
            st.metric("Hold Signals", hold_count, delta_color="off")

        st.header("🎯 Scoring Results")
        fmt = {"Buy Score": "{:.2f}", "Sell Score": "{:.2f}", "Net Score": "{:.2f}"}
        if "Fear & Greed" in results_df.columns:
            fmt["Fear & Greed"] = "{:.1f}"
        st.dataframe(results_df.style.format(fmt), use_container_width=True)
    
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
