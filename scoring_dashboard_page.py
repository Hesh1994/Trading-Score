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
import sys, os
_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from scoring_module import score_universe, results_to_dataframe
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG
try:
    from canslim_module import (COUNTRY_EXCHANGES, EXCHANGE_SUFFIX,
                                fetch_fmp_exchange_tickers, fetch_ticker_sectors)
    _fmp_module_ok = True
except ImportError:
    _fmp_module_ok = False

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

# FMP key is set on the CANSLIM Scoring page and shared via session state
fmp_key = st.session_state.get('canslim_fmp_key', '')
if fmp_key:
    st.sidebar.caption("🔑 FMP key active (set on CANSLIM Scoring page)")
else:
    st.sidebar.caption("🔑 No FMP key — go to CANSLIM Scoring page to set one (needed for Exchange Lookup)")

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

_source_options = ["Custom", "S&P 500 (first 20)", "S&P 500 (first 50)"]
if _fmp_module_ok:
    _source_options.append("FMP Exchange Lookup")

symbol_option = st.sidebar.selectbox("Symbol Source", _source_options)

if 'ta_ticker_list' not in st.session_state:
    st.session_state['ta_ticker_list'] = []

if symbol_option == "Custom":
    custom_symbols = st.sidebar.text_area(
        "Enter symbols (comma-separated)",
        value="AAPL,MSFT,GOOGL,AMZN,TSLA"
    )
    symbols_list = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]

elif symbol_option == "FMP Exchange Lookup":
    if not fmp_key:
        st.sidebar.warning("Enter an FMP API Key above to use Exchange Lookup.")
        symbols_list = []
    else:
        # ── Country ───────────────────────────────────────────────────────
        _ALL_COUNTRIES = "🌍 ALL Countries"
        _country_opts  = ["-- Select country --", _ALL_COUNTRIES] + sorted(COUNTRY_EXCHANGES.keys())
        _sel_country   = st.sidebar.selectbox("Country", _country_opts, key="ta_country")

        _sel_exc_codes  = []
        _sel_exc_label  = None
        _sel_exc_code   = None

        if _sel_country == _ALL_COUNTRIES:
            _sel_exc_codes = ["__ALL__"]
            _sel_exc_label = "All Countries / All Exchanges"
            st.sidebar.caption("Full global stock list (~90k tickers).")

        elif _sel_country != "-- Select country --":
            _exc_list   = COUNTRY_EXCHANGES[_sel_country]
            _exc_labels = [lbl for _, lbl in _exc_list]
            _all_lbl    = f"ALL ({len(_exc_list)} exchanges)"
            _sel_exc_label = st.sidebar.selectbox(
                "Exchange", [_all_lbl] + _exc_labels, key="ta_exchange"
            )
            if _sel_exc_label == _all_lbl:
                _sel_exc_codes = [c for c, _ in _exc_list]
            else:
                _sel_exc_code  = next(c for c, l in _exc_list if l == _sel_exc_label)
                _sel_exc_codes = [_sel_exc_code]
                _sfx = EXCHANGE_SUFFIX.get(_sel_exc_code, "")
                st.sidebar.caption(f"Ticker suffix: `{_sfx if _sfx else '(none)'}`")

        # ── Load Tickers ─────────────────────────────────────────────────
        st.sidebar.markdown("**📋 Load Exchange Tickers**")
        if not _sel_exc_codes:
            st.sidebar.caption("⬆️ Select a country and exchange above first.")
        else:
            _ck   = f"ta_tickers_{'_'.join(sorted(_sel_exc_codes))}"
            _sck  = f"ta_sectors_{'_'.join(sorted(_sel_exc_codes))}"
            _loaded  = bool(st.session_state.get(_ck))
            _sloaded = bool(st.session_state.get(_sck))

            _lc, _rc = st.sidebar.columns(2)
            if _lc.button("📋 Load Tickers", key="ta_load_tickers_btn",
                           use_container_width=True, disabled=_loaded):
                with st.spinner(f"Loading tickers for {_sel_exc_label}…"):
                    try:
                        if _sel_exc_codes == ["__ALL__"]:
                            _raw = fetch_fmp_exchange_tickers("__ALL__", fmp_key)
                            _tickers = _raw
                        else:
                            _combined = {}
                            for _c in _sel_exc_codes:
                                for _s, _n in fetch_fmp_exchange_tickers(_c, fmp_key):
                                    _combined[_s] = _n
                            _tickers = sorted(_combined.items(), key=lambda x: x[0])
                        st.session_state[_ck] = _tickers
                        st.rerun()
                    except RuntimeError as _e:
                        st.sidebar.error(str(_e))

            if _rc.button("🔄 Reload", key="ta_reload_tickers_btn",
                           use_container_width=True, disabled=not _loaded):
                st.session_state.pop(_ck, None)
                st.session_state.pop(_sck, None)
                st.rerun()

            if _loaded:
                _tickers = st.session_state[_ck]
                st.sidebar.caption(f"{len(_tickers):,} tickers loaded")

                # ── Load Sectors ──────────────────────────────────────────
                if not _sloaded:
                    if st.sidebar.button("🏭 Load Sectors", key="ta_load_sectors_btn",
                                          use_container_width=True):
                        _sym_list = [s for s, _ in _tickers]
                        _pb  = st.sidebar.progress(0, text="Fetching sectors…")
                        _pt  = st.sidebar.empty()
                        def _pcb(done, total):
                            _pb.progress(done / total, text=f"Sectors: {done}/{total}")
                            _pt.caption(f"{done}/{total} processed")
                        _smap = fetch_ticker_sectors(_sym_list, fmp_key, progress_cb=_pcb)
                        _pb.empty(); _pt.empty()
                        st.session_state[_sck] = _smap
                        st.rerun()
                else:
                    _smap  = st.session_state.get(_sck, {})
                    _filled = sum(1 for v in _smap.values() if v)
                    st.sidebar.caption(f"Sectors: {_filled:,} / {len(_smap):,} classified")

                # ── Sector filter ─────────────────────────────────────────
                _smap = st.session_state.get(_sck, {})
                _avail_sectors = sorted({v for v in _smap.values() if v})
                _sec_choice = st.sidebar.selectbox(
                    "🏭 Filter by Sector",
                    ["ALL"] + _avail_sectors,
                    key="ta_sector_filter",
                )
                if _sec_choice == "ALL" or not _smap:
                    _tickers_in_sec = _tickers
                else:
                    _tickers_in_sec = [(s, n) for s, n in _tickers
                                       if _smap.get(s, "") == _sec_choice]

                # ── Ticker dropdown ───────────────────────────────────────
                _tick_opts = ["-- select --", "✅ Select All"] + \
                             [f"{s}  —  {n}" for s, n in _tickers_in_sec[:500]]
                _chosen = st.sidebar.selectbox(
                    f"Select ticker ({len(_tickers_in_sec):,} available)",
                    _tick_opts, key="ta_ticker_select",
                )
                if _chosen == "✅ Select All":
                    if st.sidebar.button("➕ Add All to List", key="ta_add_all_btn",
                                          use_container_width=True):
                        _added = 0
                        for _s, _ in _tickers_in_sec:
                            if _s not in st.session_state['ta_ticker_list']:
                                st.session_state['ta_ticker_list'].append(_s)
                                _added += 1
                        st.sidebar.success(f"Added {_added} ticker(s)")
                elif _chosen != "-- select --":
                    _csym = _chosen.split("  —  ")[0].strip()
                    if st.sidebar.button("➕ Add to List", key="ta_add_btn",
                                          use_container_width=True):
                        if _csym not in st.session_state['ta_ticker_list']:
                            st.session_state['ta_ticker_list'].append(_csym)
                            st.sidebar.success(f"Added **{_csym}**")
                        else:
                            st.sidebar.info(f"**{_csym}** already in list")
            else:
                st.sidebar.caption("Click **Load Tickers** to browse listed stocks.")

        # ── Current list ─────────────────────────────────────────────────
        if st.session_state['ta_ticker_list']:
            st.sidebar.markdown(
                "**Tickers to analyse:**  " +
                " · ".join(f"`{t}`" for t in st.session_state['ta_ticker_list'])
            )
            if st.sidebar.button("🗑️ Clear list", key="ta_clear_list"):
                st.session_state['ta_ticker_list'] = []
                st.rerun()

        symbols_list = list(dict.fromkeys(st.session_state['ta_ticker_list']))

else:
    @st.cache_data
    def get_sp500_symbols():
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            response = requests.get(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                headers=headers, timeout=10
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
    else:
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
