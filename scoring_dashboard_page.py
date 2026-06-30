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
from scoring_module import score_universe, results_to_dataframe, score_stock
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG
try:
    from canslim_module import (COUNTRY_EXCHANGES, EXCHANGE_SUFFIX,
                                fetch_fmp_exchange_tickers, fetch_ticker_sectors,
                                fetch_price_data_fmp, fetch_price_universe_fmp,
                                score_canslim_universe)
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

_title_col, _btn_col = st.columns([4, 1])
with _title_col:
    st.title("Technical Analysis Stock Scoring System")
    st.caption("v2026-06-29o — Bollinger score labels price<lower/price>upper")
_btn_col.markdown('<div style="margin-top: 1.6rem;"></div>', unsafe_allow_html=True)
_run_btn_header = _btn_col.button("🚀 Run Scoring Analysis", type="primary", use_container_width=True, key="run_btn_header")
st.markdown('<hr style="border: none; border-top: 3px solid black; margin-top: 0; margin-bottom: 1rem;">', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# ── FMP API Key (shared across all pages via session state) ───────────────
st.sidebar.subheader("🔑 FMP API Key")
fmp_key = st.sidebar.text_input(
    "FMP API Key",
    type="password",
    placeholder="Enter once — used by all pages",
    key="shared_fmp_key",
    help="Get a free key at financialmodelingprep.com — used for Exchange Lookup and CANSLIM data."
)
if fmp_key:
    st.sidebar.success("FMP key set — available on all pages")

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

# ── Symbol Selection ─────────────────────────────────────────────────────
st.sidebar.subheader("🌍 Ticker Finder")

if 'ta_ticker_list' not in st.session_state:
    st.session_state['ta_ticker_list'] = []

if not _fmp_module_ok:
    st.sidebar.warning("canslim_module not found — using custom entry only.")
elif not fmp_key:
    st.sidebar.caption("⬆️ Enter an FMP API key above to use the Exchange Lookup.")
else:
    # ── Country ───────────────────────────────────────────────────────────
    _ALL_COUNTRIES = "🌍 ALL Countries"
    _country_opts  = ["-- Select country --", _ALL_COUNTRIES] + sorted(COUNTRY_EXCHANGES.keys())
    _sel_country   = st.sidebar.selectbox("Country", _country_opts, key="ta_country")

    _sel_exc_codes = []
    _sel_exc_label = None
    _sel_exc_code  = None

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

    # ── Load Tickers ──────────────────────────────────────────────────────
    st.sidebar.markdown("**📋 Load Exchange Tickers**")
    if not _sel_exc_codes:
        st.sidebar.caption("⬆️ Select a country and exchange above first.")
    else:
        _ck      = f"ta_tickers_{'_'.join(sorted(_sel_exc_codes))}"
        _sck     = f"ta_sectors_{'_'.join(sorted(_sel_exc_codes))}"
        _loaded  = bool(st.session_state.get(_ck))
        _sloaded = bool(st.session_state.get(_sck))

        _lc, _rc = st.sidebar.columns(2)
        if _lc.button("📋 Load Tickers", key="ta_load_tickers_btn",
                       use_container_width=True, disabled=_loaded):
            with st.spinner(f"Loading tickers for {_sel_exc_label}…"):
                try:
                    if _sel_exc_codes == ["__ALL__"]:
                        _tickers = fetch_fmp_exchange_tickers("__ALL__", fmp_key)
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
            st.sidebar.caption(f"{len(_tickers):,} tickers loaded from FMP")

            # ── Load Sectors ──────────────────────────────────────────────
            if not _sloaded:
                if st.sidebar.button("🏭 Load Sectors", key="ta_load_sectors_btn",
                                      use_container_width=True):
                    _sym_list = [s for s, _ in _tickers]
                    _pb = st.sidebar.progress(0, text="Fetching sectors…")
                    _pt = st.sidebar.empty()
                    def _pcb(done, total):
                        _pb.progress(done / total, text=f"Sectors: {done}/{total}")
                        _pt.caption(f"{done}/{total} processed")
                    _smap = fetch_ticker_sectors(_sym_list, fmp_key, progress_cb=_pcb)
                    _pb.empty(); _pt.empty()
                    st.session_state[_sck] = _smap
                    st.rerun()
            else:
                _smap   = st.session_state.get(_sck, {})
                _filled = sum(1 for v in _smap.values() if v)
                st.sidebar.caption(f"Sectors: {_filled:,} / {len(_smap):,} classified")

            # ── Sector filter ──────────────────────────────────────────────
            _smap          = st.session_state.get(_sck, {})
            _avail_sectors = sorted({v for v in _smap.values() if v})
            _sec_choice    = st.sidebar.selectbox(
                "🏭 Filter by Sector", ["ALL"] + _avail_sectors, key="ta_sector_filter"
            )
            _tickers_in_sec = (
                _tickers if (_sec_choice == "ALL" or not _smap)
                else [(s, n) for s, n in _tickers if _smap.get(s, "") == _sec_choice]
            )

            # ── Ticker dropdown ────────────────────────────────────────────
            _tick_opts = ["-- select --", "✅ Select All"] + \
                         [f"{s}  —  {n}" for s, n in _tickers_in_sec[:500]]
            _chosen = st.sidebar.selectbox(
                f"Select ticker ({len(_tickers_in_sec):,} available)",
                _tick_opts, key="ta_ticker_select"
            )
            if _chosen == "✅ Select All":
                if st.sidebar.button("➕ Add All to List", key="ta_add_all_btn",
                                      use_container_width=True):
                    _added = sum(
                        1 for _s, _ in _tickers_in_sec
                        if _s not in st.session_state['ta_ticker_list']
                        and not st.session_state['ta_ticker_list'].append(_s)
                    )
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

if st.session_state['ta_ticker_list']:
    st.sidebar.caption(
        f"{len(st.session_state['ta_ticker_list'])} ticker(s) selected — manage in main area"
    )

buy_threshold = 3.0
sell_threshold = 3.0

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
_all_ind_keys = list(INDICATORS_CONFIG.keys())
_label_to_key = {indicator_config[k]['label']: k for k in _all_ind_keys}
_default_labels = [indicator_config[k]['label'] for k in _all_ind_keys if indicator_config[k].get('enabled', True)]

_selected_labels = st.sidebar.multiselect(
    "Select indicators",
    options=[indicator_config[k]['label'] for k in _all_ind_keys],
    default=_default_labels,
    key="indicators_multiselect"
)

included_indicators = {}
for ind_key in _all_ind_keys:
    enabled = indicator_config[ind_key]['label'] in _selected_labels
    indicator_config[ind_key]['enabled'] = enabled
    if enabled:
        included_indicators[ind_key] = indicator_config[ind_key]

# Section 2: Configure selected indicators — one expander per indicator
if included_indicators:
    st.sidebar.subheader("⚙️ Configure Indicators")

    for selected_indicator, ind_cfg in included_indicators.items():
        with st.sidebar.expander(ind_cfg['label'], expanded=False):
            # Interval selector
            interval_options = ["Daily", "Weekly", "Monthly"]
            current_interval = ind_cfg.get('interval', 'daily').capitalize()
            if current_interval not in interval_options:
                current_interval = "Daily"
            selected_interval = st.selectbox(
                "Interval",
                interval_options,
                index=interval_options.index(current_interval),
                key=f"{selected_indicator}_interval"
            )
            indicator_config[selected_indicator]['interval'] = selected_interval.lower()
            if selected_indicator == 'fear_greed':
                st.caption("Calculated from OHLCV data (needs 252+ bars). Components: Momentum, RSI, Volatility, Volume Breadth.")

            # Parameters
            st.write("Parameters:")
            params = ind_cfg['parameters'].copy()
            for param_key, param_value in params.items():
                if selected_indicator in ("rsi", "mfi", "stochastic") and param_key in ("buy_threshold", "sell_threshold"):
                    continue
                if isinstance(param_value, int):
                    new_value = st.number_input(
                        param_key.replace('_', ' ').title(),
                        min_value=1,
                        value=int(param_value),
                        step=1,
                        key=f"{selected_indicator}_{param_key}"
                    )
                    indicator_config[selected_indicator]['parameters'][param_key] = new_value
                elif isinstance(param_value, float):
                    new_value = st.number_input(
                        param_key.replace('_', ' ').title(),
                        min_value=0.1,
                        value=float(param_value),
                        step=0.1,
                        key=f"{selected_indicator}_{param_key}"
                    )
                    indicator_config[selected_indicator]['parameters'][param_key] = new_value

            # RSI / MFI oversold-overbought thresholds
            if selected_indicator in ("rsi", "mfi", "stochastic"):
                _defaults = {"rsi": (30.0, 70.0), "mfi": (20.0, 80.0), "stochastic": (20.0, 80.0)}
                _os_def, _ob_def = _defaults[selected_indicator]
                st.write("Buy / Sell Rules:")
                col1, col2 = st.columns(2)
                with col1:
                    buy_thresh = st.number_input(
                        "Oversold (<)",
                        min_value=1.0, max_value=99.0,
                        value=float(ind_cfg["parameters"].get("buy_threshold", _os_def)),
                        step=1.0, key=f"{selected_indicator}_buy_threshold",
                    )
                    indicator_config[selected_indicator]["parameters"]["buy_threshold"] = buy_thresh
                with col2:
                    sell_thresh = st.number_input(
                        "Overbought (>)",
                        min_value=1.0, max_value=99.0,
                        value=float(ind_cfg["parameters"].get("sell_threshold", _ob_def)),
                        step=1.0, key=f"{selected_indicator}_sell_threshold",
                    )
                    indicator_config[selected_indicator]["parameters"]["sell_threshold"] = sell_thresh

            # Buy/sell scores
            st.write("Scoring:")
            if selected_indicator == "rsi":
                _buy_label, _sell_label = "Oversold Score", "Overbought Score"
            elif selected_indicator == "sma":
                _buy_label, _sell_label = "Short > Long Score", "Short < Long Score"
            elif selected_indicator == "ema":
                _buy_label, _sell_label = "Price > SMA Score", "Price < SMA Score"
            elif selected_indicator in ("mfi", "stochastic"):
                _buy_label, _sell_label = "Oversold Score", "Overbought Score"
            elif selected_indicator == "aroon":
                _buy_label, _sell_label = "Up > Down Score", "Up < Down Score"
            elif selected_indicator == "bollinger":
                _buy_label, _sell_label = "Price < Lower Band Score", "Price > Upper Band Score"
            elif selected_indicator == "macd":
                _buy_label, _sell_label = "MACD > Signal Score", "MACD < Signal Score"
            else:
                _buy_label, _sell_label = "Buy Score", "Sell Score"
            col1, col2 = st.columns(2)
            with col1:
                buy_score = st.number_input(
                    _buy_label,
                    value=float(ind_cfg['buy_score']),
                    step=0.5,
                    key=f"{selected_indicator}_buy_score"
                )
                indicator_config[selected_indicator]['buy_score'] = buy_score
            with col2:
                sell_score = st.number_input(
                    _sell_label,
                    value=float(ind_cfg['sell_score']),
                    step=0.5,
                    key=f"{selected_indicator}_sell_score"
                )
                indicator_config[selected_indicator]['sell_score'] = sell_score

# ── Persist the updated config for the next rerun ─────────────────────────────
st.session_state['ind_config_store'] = _copy.deepcopy(indicator_config)

# ── FMP price endpoint test ───────────────────────────────────────────────
if fmp_key and _fmp_module_ok:
    with st.sidebar.expander("🔧 Test FMP Price Endpoint"):
        _ta_list  = st.session_state.get('ta_ticker_list', [])
        _test_sym = st.text_input("Symbol", value=_ta_list[0] if _ta_list else "AAPL",
                                  key="ta_price_test_sym")
        if st.button("Test price fetch", key="ta_price_test_btn"):
            import requests as _rq
            _url = f"https://financialmodelingprep.com/stable/historical-price-eod/full"
            _r = _rq.get(_url, params={"symbol": _test_sym, "from": str(start_date),
                                        "to": str(end_date), "apikey": fmp_key}, timeout=15)
            st.write(f"Status: {_r.status_code}")
            _j = _r.json()
            if isinstance(_j, list):
                st.write(f"List with {len(_j)} rows. First row:")
                st.json(_j[0] if _j else {})
            elif isinstance(_j, dict):
                st.write("Dict response:")
                st.json({k: v for k, v in list(_j.items())[:3]})

# ============================================================================
# MAIN CONTENT — TICKER TABLE
# ============================================================================

st.subheader("📋 Tickers to Analyse")

# ── Manual add ───────────────────────────────────────────────────────────────
_add_col, _btn_col = st.columns([4, 1])
with _add_col:
    _manual_sym = st.text_input(
        "Add ticker manually", placeholder="e.g. AAPL or 2222.SR",
        key="ta_main_manual_add", label_visibility="collapsed"
    )
with _btn_col:
    if st.button("➕ Add", key="ta_main_add_btn", use_container_width=True):
        _s = _manual_sym.strip().upper()
        if _s and _s not in st.session_state['ta_ticker_list']:
            st.session_state['ta_ticker_list'].append(_s)
            st.rerun()

# ── Editable table with remove checkboxes + score columns ────────────────────
if st.session_state['ta_ticker_list']:
    _scores = st.session_state.get('ta_scores', {})
    _canslim_scores = st.session_state.get('ta_canslim_scores', {})
    _fg_scores = st.session_state.get('ta_fg_scores', {})
    _scores_history = st.session_state.get('ta_scores_history', {})
    _fg_history = st.session_state.get('ta_fg_history', {})
    _tickers = st.session_state['ta_ticker_list']

    # Toggle button for 5D evolution panel
    _hdr_col, _btn_col = st.columns([6, 1])
    _show_5d = st.session_state.get('show_5d_tech', False)
    if _btn_col.button(
        "📅 5D ▲" if _show_5d else "📅 5D ▼",
        key="toggle_5d_btn",
        use_container_width=True,
        help="Show / hide the 5-day Technical Score evolution",
    ):
        st.session_state['show_5d_tech'] = not _show_5d
        st.rerun()

    # Always include score columns so table structure never changes (avoids key conflict)
    _tbl = pd.DataFrame({
        'Remove': [False] * len(_tickers),
        'Ticker': _tickers,
        'Total Technical Score': [_scores.get(t) for t in _tickers],
        'CANSLIM Score': [_canslim_scores.get(t) for t in _tickers],
        'Fear & Greed': [_fg_scores.get(t) for t in _tickers],
    })
    _edited = st.data_editor(
        _tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Remove': st.column_config.CheckboxColumn('✖ Remove', default=False),
            'Ticker': st.column_config.TextColumn('Ticker', disabled=True),
            'Total Technical Score': st.column_config.NumberColumn('Total Technical Score (%)', disabled=True, format='%.1f%%'),
            'CANSLIM Score': st.column_config.NumberColumn('CANSLIM Score', disabled=True, format='%.1f%%'),
            'Fear & Greed': st.column_config.NumberColumn('Fear & Greed', disabled=True, format='%.1f%%'),
        },
        key="ta_ticker_table",
    )
    _kept = _edited[~_edited['Remove']]['Ticker'].tolist()
    if _kept != st.session_state['ta_ticker_list']:
        st.session_state['ta_ticker_list'] = _kept
        st.rerun()

    # 5D evolution panel — one column per trading day
    if st.session_state.get('show_5d_tech') and _scores_history:
        _dates = st.session_state.get('ta_history_dates', ['Day -4', 'Day -3', 'Day -2', 'Day -1', 'Today'])
        _5d_rows = {'Ticker': _tickers}
        for _i, _d in enumerate(_dates):
            _5d_rows[_d] = [
                (_scores_history.get(t) or [None] * 5)[_i]
                for t in _tickers
            ]
        _5d_df = pd.DataFrame(_5d_rows)
        _5d_col_cfg = {'Ticker': st.column_config.TextColumn('Ticker', disabled=True)}
        for _d in _dates:
            _5d_col_cfg[_d] = st.column_config.NumberColumn(_d, disabled=True, format='%.1f%%')
        st.caption("Technical Score — 5-day evolution")
        st.dataframe(_5d_df, use_container_width=True, hide_index=True, column_config=_5d_col_cfg)

    _c1, _c2 = st.columns([3, 1])
    _c1.caption(f"{len(st.session_state['ta_ticker_list'])} ticker(s) selected")
    if _c2.button("🗑️ Clear all", key="ta_main_clear", use_container_width=True):
        st.session_state['ta_ticker_list'] = []
        st.rerun()
else:
    st.info("No tickers selected yet. Use the sidebar Ticker Finder or type a symbol above.")

# symbols_list is always driven by the table
symbols_list = list(dict.fromkeys(st.session_state['ta_ticker_list']))

if _run_btn_header:

    _use_fmp = bool(fmp_key and _fmp_module_ok)
    _source_label = "FMP API" if _use_fmp else "yfinance"

    with st.spinner(f"📥 Downloading price data via {_source_label}…"):
        try:
            # Collect unique intervals needed by enabled indicators
            intervals_needed = set()
            for ind_key, cfg in indicator_config.items():
                if cfg.get('enabled'):
                    intervals_needed.add(cfg.get('interval', 'daily'))

            tickers_data_by_interval = {}

            if _use_fmp:
                # ── FMP path: one call per symbol per interval ────────────
                for interval in intervals_needed:
                    tickers_dict = {}
                    _prog = st.progress(0, text=f"Fetching {interval} prices via FMP…")
                    for _i, sym in enumerate(symbols_list):
                        df = fetch_price_data_fmp(sym, start_date, end_date,
                                                  fmp_key, interval)
                        if df is not None and not df.empty:
                            tickers_dict[sym] = df
                        _prog.progress((_i + 1) / len(symbols_list),
                                       text=f"FMP {interval}: {_i+1}/{len(symbols_list)} — {sym}")
                    _prog.empty()
                    if tickers_dict:
                        tickers_data_by_interval[interval] = tickers_dict

            else:
                # ── yfinance path (fallback) ──────────────────────────────
                INTERVAL_YF = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
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
                    if tickers_dict:
                        tickers_data_by_interval[interval] = tickers_dict

            if not tickers_data_by_interval:
                st.error(f"No price data returned via {_source_label}. Check symbols and date range.")
                st.stop()

            all_tickers_found = set()
            for d in tickers_data_by_interval.values():
                all_tickers_found.update(d.keys())
            if not all_tickers_found:
                st.error(f"No valid data for: {', '.join(symbols_list[:5])}")
                st.stop()

            st.success(f"✅ Price data fetched via {_source_label} for {len(all_tickers_found)} ticker(s)")

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
            # % = (number of BUY signals fired) / (number of enabled indicators) × 100
            _n_indicators = sum(1 for cfg in indicator_config.values() if cfg.get('enabled')) or 1
            st.session_state['ta_scores'] = {
                r['ticker']: round(
                    sum(1 for sig in r['signals'].values()
                        if sig.get('buy') and 'error' not in sig)
                    / _n_indicators * 100, 1
                )
                for r in results
            }
            st.session_state['ta_fg_scores'] = {
                r['ticker']: round(r['signals']['fear_greed']['value'], 1)
                for r in results
                if 'fear_greed' in r['signals'] and 'value' in r['signals']['fear_greed']
            }

            # 5-day score history (oldest → newest) for sparklines
            _fg_enabled = indicator_config.get('fear_greed', {}).get('enabled', False)
            _tech_history = {}
            _fg_history = {}
            for ticker in symbols_list:
                _tech_days = []
                _fg_days = []
                for offset in range(4, -1, -1):
                    _trimmed = {
                        interval: {
                            t: (df.iloc[:-offset] if offset > 0 else df)
                            for t, df in ticker_dfs.items()
                        }
                        for interval, ticker_dfs in tickers_data_by_interval.items()
                    }
                    _r = score_stock(ticker, _trimmed, indicator_config, global_config)
                    _tech_days.append(round(
                        sum(1 for sig in _r['signals'].values()
                            if sig.get('buy') and 'error' not in sig)
                        / _n_indicators * 100, 1
                    ))
                    if _fg_enabled and 'fear_greed' in _r['signals'] and 'value' in _r['signals']['fear_greed']:
                        _fg_days.append(round(_r['signals']['fear_greed']['value'], 1))
                    else:
                        _fg_days.append(None)
                _tech_history[ticker] = _tech_days
                if any(v is not None for v in _fg_days):
                    _fg_history[ticker] = _fg_days
            st.session_state['ta_scores_history'] = _tech_history
            st.session_state['ta_fg_history'] = _fg_history
            # Store the actual trading dates for the last 5 bars
            _hist_dates = ['Day -4', 'Day -3', 'Day -2', 'Day -1', 'Today']
            for _iv_data in tickers_data_by_interval.values():
                for _df in _iv_data.values():
                    if len(_df) >= 5:
                        _hist_dates = [
                            str(d.date()) if hasattr(d, 'date') else str(d)
                            for d in _df.index[-5:]
                        ]
                        break
                break
            st.session_state['ta_history_dates'] = _hist_dates

        except Exception as e:
            st.error(f"Error during scoring: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()
    
    # ========== CANSLIM SCORING (same tickers) ==========
    if fmp_key and _fmp_module_ok:
        with st.spinner("📊 Calculating CANSLIM scores…"):
            try:
                _canslim_results = score_canslim_universe(symbols_list, fmp_api_key=fmp_key)
                st.session_state['ta_canslim_scores'] = {
                    r['symbol']: round(r['score'], 2) for r in _canslim_results
                }
            except Exception:
                st.session_state['ta_canslim_scores'] = {}

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
