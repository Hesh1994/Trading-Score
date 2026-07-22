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
import sys, os, json
_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from scoring_module import score_universe, results_to_dataframe, score_stock
from scoring_config import INDICATORS_CONFIG, GLOBAL_CONFIG
try:
    from canslim_module import (COUNTRY_EXCHANGES, EXCHANGE_SUFFIX,
                                fetch_fmp_available_exchanges,
                                build_country_exchange_map, build_exchange_suffix_map,
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

# ── FMP API Key — persistent key manager ─────────────────────────────────
_KEY_FILE = os.path.join(os.path.expanduser("~"), ".streamlit_fmp_key")

# Load saved key from file once per session into the persistent (non-widget) key
if 'fmp_key_loaded' not in st.session_state:
    try:
        if os.path.exists(_KEY_FILE):
            with open(_KEY_FILE) as _kf:
                _saved_key = json.load(_kf).get('key', '')
            if _saved_key:
                st.session_state['fmp_key_value'] = _saved_key
    except Exception:
        pass
    st.session_state['fmp_key_loaded'] = True

st.sidebar.subheader("🔑 FMP API Key")
_show_key = st.sidebar.checkbox("Show key", key="fmp_show_toggle")
fmp_key = st.sidebar.text_input(
    "FMP API Key",
    type="default" if _show_key else "password",
    placeholder="Enter once — used by all pages",
    value=st.session_state.get('fmp_key_value', ''),
    key="shared_fmp_key",
    help="Get a free key at financialmodelingprep.com — used for Exchange Lookup and CANSLIM data.",
)

# Sync widget value → persistent non-widget key (survives page navigation)
if fmp_key:
    st.session_state['fmp_key_value'] = fmp_key

if fmp_key:
    _is_saved = os.path.exists(_KEY_FILE)
    st.sidebar.success("Key active" + (" · saved" if _is_saved else ""))
    _kb1, _kb2 = st.sidebar.columns(2)
    if _kb1.button("💾 Save key", key="fmp_save_btn", use_container_width=True):
        try:
            with open(_KEY_FILE, 'w') as _kf:
                json.dump({'key': fmp_key}, _kf)
            st.sidebar.success("Key saved!")
        except Exception as _e:
            st.sidebar.error(f"Could not save: {_e}")
    if _kb2.button("🗑️ Remove key", key="fmp_remove_btn", use_container_width=True):
        st.session_state.pop('fmp_key_value', None)
        st.session_state.pop('shared_fmp_key', None)
        try:
            os.remove(_KEY_FILE)
        except Exception:
            pass
        st.rerun()
else:
    st.sidebar.caption("Enter your FMP API key above.")

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
    # ── Load exchange data from FMP (cached after first call) ─────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def _load_exchange_map(api_key):
        raw = fetch_fmp_available_exchanges(api_key)
        if raw:
            return build_country_exchange_map(raw)
        return COUNTRY_EXCHANGES  # static fallback

    _country_exc_map = _load_exchange_map(fmp_key)

    # ── Country (multiselect with search) ────────────────────────────────
    _sel_countries = st.sidebar.multiselect(
        "Country",
        options=sorted(_country_exc_map.keys()),
        key="ta_country",
        placeholder="Search and select countries…",
    )

    _sel_exc_codes = []
    _sel_exc_label = None

    if _sel_countries:
        # Build combined exchange list for all selected countries
        _all_exc_pairs = []
        for _c in _sel_countries:
            _all_exc_pairs.extend(_country_exc_map[_c])

        # ── Exchange (multiselect with search) ────────────────────────────
        _exc_label_to_code = {lbl: code for code, lbl in _all_exc_pairs}
        _sel_exc_labels = st.sidebar.multiselect(
            "Exchange",
            options=list(_exc_label_to_code.keys()),
            key="ta_exchange",
            placeholder="All exchanges (leave empty for all)…",
        )
        if _sel_exc_labels:
            _sel_exc_codes = [_exc_label_to_code[l] for l in _sel_exc_labels]
            _sel_exc_label = ", ".join(_sel_exc_labels)
        else:
            # No exchange selected → use all exchanges for selected countries
            _sel_exc_codes = [code for code, _ in _all_exc_pairs]
            _sel_exc_label = f"All exchanges ({len(_sel_exc_codes)} selected)"

    # ── Load Tickers ──────────────────────────────────────────────────────
    st.sidebar.markdown("**📋 Load Exchange Tickers**")
    if not _sel_exc_codes:
        st.sidebar.caption("⬆️ Select a country above first.")
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

            # ── Sector filter (multiselect) ────────────────────────────────
            _smap          = st.session_state.get(_sck, {})
            _avail_sectors = sorted({v for v in _smap.values() if v})
            _sec_choice    = st.sidebar.multiselect(
                "🏭 Filter by Sector",
                options=_avail_sectors,
                key="ta_sector_filter",
                placeholder="All sectors (no filter)",
            )
            _tickers_in_sec = (
                _tickers if (not _sec_choice or not _smap)
                else [(s, n) for s, n in _tickers if _smap.get(s, "") in set(_sec_choice)]
            )

            # ── Ticker multiselect (searchable checkbox list) ──────────────
            _tick_opts = [f"{s}  —  {n}" for s, n in _tickers_in_sec[:1000]]
            _chosen_labels = st.sidebar.multiselect(
                f"Select ticker ({len(_tickers_in_sec):,} available)",
                options=_tick_opts,
                key="ta_ticker_select",
                placeholder="Search and select tickers…",
            )
            _b1, _b2 = st.sidebar.columns(2)
            if _b1.button("➕ Add Selected", key="ta_add_btn",
                          use_container_width=True, disabled=not _chosen_labels):
                _before = len(st.session_state['ta_ticker_list'])
                for _lbl in _chosen_labels:
                    _sym = _lbl.split("  —  ")[0].strip()
                    if _sym not in st.session_state['ta_ticker_list']:
                        st.session_state['ta_ticker_list'].append(_sym)
                _added = len(st.session_state['ta_ticker_list']) - _before
                st.session_state.pop("ta_ticker_select", None)
                st.sidebar.success(f"Added {_added} ticker(s)")
                st.rerun()
            if _b2.button("➕ Add All", key="ta_add_all_btn",
                          use_container_width=True):
                _before = len(st.session_state['ta_ticker_list'])
                for _s, _ in _tickers_in_sec:
                    if _s not in st.session_state['ta_ticker_list']:
                        st.session_state['ta_ticker_list'].append(_s)
                _added = len(st.session_state['ta_ticker_list']) - _before
                st.sidebar.success(f"Added {_added} ticker(s)")
                st.rerun()
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

_CANSLIM_LABEL = "CANSLIM Score"
_ind_options   = [indicator_config[k]['label'] for k in _all_ind_keys]
_ind_defaults  = list(_default_labels)
if _fmp_module_ok:
    _ind_options.append(_CANSLIM_LABEL)
    _ind_defaults.append(_CANSLIM_LABEL)

_selected_labels = st.sidebar.multiselect(
    "Select indicators",
    options=_ind_options,
    default=_ind_defaults,
    key="indicators_multiselect"
)
_canslim_enabled = _fmp_module_ok and (_CANSLIM_LABEL in _selected_labels)

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
            elif selected_indicator == 'volume':
                st.caption("Buy when the 20-day average volume ≥ Min Volume. Sell when below. Adjust Min Volume to filter for liquid stocks.")
            elif selected_indicator == 'week52_high':
                st.caption("Buy when price is at or above the 52-week high (new high breakout). Sell when price is below the 52-week high.")

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
                    # Use large step for volume threshold, small step for everything else
                    _step = 100000.0 if param_key == 'min_volume' else 0.1
                    _min  = 0.0      if param_key == 'min_volume' else 0.1
                    new_value = st.number_input(
                        param_key.replace('_', ' ').title(),
                        min_value=_min,
                        value=float(param_value),
                        step=_step,
                        format='%.0f' if param_key == 'min_volume' else '%.2f',
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
            elif selected_indicator == "volume":
                _buy_label, _sell_label = "Volume ≥ Min Volume Score", "Volume < Min Volume Score"
            elif selected_indicator == "week52_high":
                _buy_label, _sell_label = "Price ≥ 52-Week High Score", "Price < 52-Week High Score"
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

# ── Model Weights ─────────────────────────────────────────────────────────────
st.sidebar.subheader("⚖️ Model Weights")
_fg_active = indicator_config.get('fear_greed', {}).get('enabled', False)

_w_tech = st.sidebar.number_input(
    "Total Technical Score %", min_value=0, max_value=100, value=50, step=5, key="w_tech"
)
_w_fg = 0
if _fg_active:
    _w_fg = st.sidebar.number_input(
        "Fear & Greed %", min_value=0, max_value=100, value=25, step=5, key="w_fg"
    )
_w_canslim = 0
if _canslim_enabled:
    _w_canslim = st.sidebar.number_input(
        "CANSLIM %", min_value=0, max_value=100, value=25, step=5, key="w_canslim"
    )
    # Show CANSLIM data status
    _cs_adj = st.session_state.get('canslim_adjusted_scores', {})
    if _cs_adj:
        _tickers_now = st.session_state.get('ta_ticker_list', [])
        _matched_n = sum(1 for t in _tickers_now if t in _cs_adj)
        st.sidebar.success(f"✅ CANSLIM: {_matched_n}/{len(_tickers_now)} tickers loaded from CANSLIM Dashboard")
    else:
        st.sidebar.warning("⚠️ No CANSLIM scores yet — run the **CANSLIM Dashboard** first, then come back here.")

_total_w = _w_tech + _w_fg + _w_canslim
_total_label = f"Total: {_total_w}%"
if _total_w == 100:
    st.sidebar.success(_total_label)
elif _total_w > 100:
    st.sidebar.error(_total_label + " — exceeds 100%")
else:
    st.sidebar.warning(_total_label + f" — {100 - _total_w}% remaining")

def _final_score(ticker, scores, fg_scores, canslim_scores):
    _ws, _wt = 0.0, 0.0
    _ts = scores.get(ticker)
    if _ts is not None and _w_tech > 0:
        _ws += _ts * _w_tech;  _wt += _w_tech
    if _fg_active and _w_fg > 0:
        _fg = fg_scores.get(ticker)
        if _fg is not None:
            _ws += _fg * _w_fg;  _wt += _w_fg
    if _canslim_enabled and _w_canslim > 0:
        _cs = canslim_scores.get(ticker)
        if _cs is not None:
            _ws += _cs * _w_canslim;  _wt += _w_canslim
    return round(_ws / _wt, 1) if _wt > 0 else None

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

# Placeholder for loading spinner — positioned above the table so it appears above it
_status_ph = st.empty()

# ── Editable table with remove checkboxes + score columns ────────────────────
if st.session_state['ta_ticker_list']:
    _scores        = st.session_state.get('ta_scores', {})
    _canslim_scores= st.session_state.get('ta_canslim_scores',
                        st.session_state.get('canslim_adjusted_scores', {}))
    _fg_scores     = st.session_state.get('ta_fg_scores', {})
    _scores_history= st.session_state.get('ta_scores_history', {})
    _fg_history    = st.session_state.get('ta_fg_history', {})
    _tickers       = st.session_state['ta_ticker_list']
    _show_5d       = st.session_state.get('show_5d_tech', False)
    _show_5d_fg    = st.session_state.get('show_5d_fg', False)

    # ── Sidebar: ticker visibility multiselect (searchable checkbox dropdown) ───
    _vis_key = "visible_tickers_ms"
    if _vis_key not in st.session_state:
        st.session_state[_vis_key] = list(_tickers)
    else:
        # Keep previously visible tickers still in the list; auto-show new ones
        _prev = set(st.session_state[_vis_key])
        _new  = [t for t in _tickers if t not in _prev]
        st.session_state[_vis_key] = [t for t in _tickers if t in _prev] + _new

    _visible_tickers = st.sidebar.multiselect(
        "Select Tickers to Show",
        options=_tickers,
        key=_vis_key,
    )
    if not _visible_tickers:
        _visible_tickers = list(_tickers)   # guard: show all if nothing selected

    # Format stored ISO dates as "28 Jun"
    _raw_dates = st.session_state.get('ta_history_dates', [])
    _day_labels = []
    for _rd in _raw_dates:
        try:
            _dt = dt.datetime.strptime(_rd, '%Y-%m-%d')
            _day_labels.append(_dt.strftime('%-d %b'))
        except Exception:
            try:
                _dt = dt.datetime.strptime(_rd, '%Y-%m-%d %H:%M:%S')
                _day_labels.append(_dt.strftime('%-d %b'))
            except Exception:
                _day_labels.append(_rd)
    if not _day_labels:
        _day_labels = ['Day -4', 'Day -3', 'Day -2', 'Day -1', 'Today']

    # ── Toggle buttons + Clear all ───────────────────────────────────────────
    _b1, _b2, _b3 = st.columns([2, 2, 2])
    with _b1:
        if st.button(
            "Collapse 5D ▲" if _show_5d else "Extend Technical Score to 5D ▼",
            key="toggle_5d_btn", use_container_width=True,
            help="Show / hide 5-day Technical Score evolution",
        ):
            st.session_state['show_5d_tech'] = not _show_5d
            st.rerun()
    with _b2:
        if st.button(
            "Collapse F&G 5D ▲" if _show_5d_fg else "Extend Fear & Greed to 5D ▼",
            key="toggle_5d_fg_btn", use_container_width=True,
            help="Show / hide 5-day Fear & Greed evolution",
        ):
            st.session_state['show_5d_fg'] = not _show_5d_fg
            st.rerun()
    with _b3:
        if st.button("🗑️ Clear all", key="ta_main_clear_top", use_container_width=True):
            st.session_state['ta_ticker_list'] = []
            st.rerun()

    _n_tts = 5 if (_show_5d    and _scores_history) else 0
    _n_fg  = 5 if (_show_5d_fg and _fg_history)    else 0

    if _n_tts or _n_fg:
        # ── 5D active: MultiIndex DataFrame → native merged-header rendering ──────
        # Build column tuples: (group, sub-header)
        _mi_tuples = [('', 'Ticker')]
        _mi_data   = {'Ticker': _visible_tickers}

        if _n_tts:
            for _i, _lbl in enumerate(_day_labels):
                _mi_tuples.append(('Total Technical Score', _lbl))
                _mi_data[('Total Technical Score', _lbl)] = [
                    f"{((_scores_history.get(t) or [None]*5)[_i] or 0):.1f}%"
                    if (_scores_history.get(t) or [None]*5)[_i] is not None else ''
                    for t in _visible_tickers
                ]
        else:
            _mi_tuples.append(('', 'Tech Score'))
            _mi_data[('', 'Tech Score')] = [
                f"{_scores.get(t, 0):.1f}%" if _scores.get(t) is not None else ''
                for t in _visible_tickers
            ]

        if _n_fg:
            for _i, _lbl in enumerate(_day_labels):
                _mi_tuples.append(('Fear & Greed', _lbl))
                _mi_data[('Fear & Greed', _lbl)] = [
                    f"{((_fg_history.get(t) or [None]*5)[_i] or 0):.1f}%"
                    if (_fg_history.get(t) or [None]*5)[_i] is not None else ''
                    for t in _visible_tickers
                ]
        else:
            _mi_tuples.append(('', 'Fear & Greed'))
            _mi_data[('', 'Fear & Greed')] = [
                f"{_fg_scores.get(t, 0):.1f}%" if _fg_scores.get(t) is not None else ''
                for t in _visible_tickers
            ]

        if _canslim_enabled:
            _mi_tuples.append(('', 'CANSLIM'))
            _mi_data[('', 'CANSLIM')] = [
                f"{_canslim_scores.get(t, 0):.1f}%" if _canslim_scores.get(t) is not None else ''
                for t in _visible_tickers
            ]

        _mi_tuples.append(('', 'Final Score'))
        _mi_data[('', 'Final Score')] = [
            f"{_final_score(t, _scores, _fg_scores, _canslim_scores):.1f}%"
            if _final_score(t, _scores, _fg_scores, _canslim_scores) is not None else ''
            for t in _visible_tickers
        ]

        _mi_df = pd.DataFrame(_mi_data)
        _mi_df.columns = pd.MultiIndex.from_tuples(_mi_tuples)
        st.dataframe(_mi_df, use_container_width=True, hide_index=True)

        st.caption(f"{len(_visible_tickers)} of {len(_tickers)} ticker(s) shown")

    else:
        # ── Normal view: data_editor with Remove checkboxes ───────────────────────
        _tbl_data = {
            'Remove':                [False] * len(_visible_tickers),
            'Ticker':                _visible_tickers,
            'Total Technical Score': [_scores.get(t) for t in _visible_tickers],
            'Fear & Greed':          [_fg_scores.get(t) for t in _visible_tickers],
        }
        if _canslim_enabled:
            _tbl_data['CANSLIM Score'] = [_canslim_scores.get(t) for t in _visible_tickers]
        _tbl_data['Final Score'] = [
            _final_score(t, _scores, _fg_scores, _canslim_scores) for t in _visible_tickers
        ]
        _col_cfg = {
            'Remove':                st.column_config.CheckboxColumn('✖ Remove', default=False),
            'Ticker':                st.column_config.TextColumn('Ticker', disabled=True),
            'Total Technical Score': st.column_config.NumberColumn('Total Technical Score (%)', disabled=True, format='%.1f%%'),
            'Fear & Greed':          st.column_config.NumberColumn('Fear & Greed', disabled=True, format='%.1f%%'),
            'CANSLIM Score':         st.column_config.NumberColumn('CANSLIM Score', disabled=True, format='%.1f%%'),
            'Final Score':           st.column_config.NumberColumn('Final Score (%)', disabled=True, format='%.1f%%'),
        }
        _edited = st.data_editor(
            pd.DataFrame(_tbl_data), use_container_width=True,
            hide_index=True, column_config=_col_cfg, key='ta_ticker_table',
        )
        _kept_visible = _edited[~_edited['Remove']]['Ticker'].tolist()
        _removed = set(_visible_tickers) - set(_kept_visible)
        if _removed:
            st.session_state['ta_ticker_list'] = [t for t in _tickers if t not in _removed]
            st.rerun()

        st.caption(f"{len(_visible_tickers)} of {len(_tickers)} ticker(s) shown")
else:
    st.info("No tickers selected yet. Use the sidebar Ticker Finder or type a symbol above.")

# ── Candlestick chart with indicator overlays ────────────────────────────────
_price_data = st.session_state.get('ta_price_data', {})
if _price_data:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("📊 Candlestick Chart")

    _cc1, _cc2 = st.columns([2, 4])
    with _cc1:
        _chart_tickers = sorted(_price_data.keys())
        _sel_chart = st.selectbox(
            "Ticker", options=_chart_tickers, key="ta_chart_ticker_sel",
        )
    with _cc2:
        _overlay_options  = ["SMA (short)", "SMA (long)", "EMA", "Bollinger Bands"]
        _subpanel_options = ["Volume", "RSI", "MACD"]
        _all_ind_opts = _overlay_options + _subpanel_options
        _sel_inds = st.multiselect(
            "Overlay indicators", options=_all_ind_opts,
            default=["SMA (short)", "SMA (long)"],
            key="ta_chart_inds",
        )

    if _sel_chart and _sel_chart in _price_data:
        _df_c = _price_data[_sel_chart].copy()
        _df_c.index = pd.to_datetime(_df_c.index)

        # Determine subplots needed
        _sub_panels = [i for i in _subpanel_options if i in _sel_inds]
        _n_rows  = 1 + len(_sub_panels)
        _row_h   = [0.6] + [0.2 / max(len(_sub_panels), 1)] * len(_sub_panels) if _sub_panels else [1.0]
        _row_h   = [0.6] + [0.4 / len(_sub_panels)] * len(_sub_panels) if _sub_panels else [1.0]

        _fig = make_subplots(
            rows=_n_rows, cols=1,
            shared_xaxes=True,
            row_heights=_row_h,
            vertical_spacing=0.03,
        )

        # ── Candlestick (row 1) ───────────────────────────────────────
        _fig.add_trace(go.Candlestick(
            x=_df_c.index,
            open=_df_c['open'], high=_df_c['high'],
            low=_df_c['low'],   close=_df_c['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name="Price",
        ), row=1, col=1)

        # ── Price overlays ───────────────────────────────────────────
        _ic = indicator_config  # already loaded from scoring_config
        if "SMA (short)" in _sel_inds:
            _p = _ic.get('sma', {}).get('parameters', {}).get('period_short', 15)
            _sma_s = _df_c['close'].rolling(_p).mean()
            _fig.add_trace(go.Scatter(x=_df_c.index, y=_sma_s, name=f"SMA({_p})",
                                      line=dict(color='#f5a623', width=1.5)), row=1, col=1)
        if "SMA (long)" in _sel_inds:
            _p = _ic.get('sma', {}).get('parameters', {}).get('period_long', 45)
            _sma_l = _df_c['close'].rolling(_p).mean()
            _fig.add_trace(go.Scatter(x=_df_c.index, y=_sma_l, name=f"SMA({_p})",
                                      line=dict(color='#9b59b6', width=1.5)), row=1, col=1)
        if "EMA" in _sel_inds:
            _p = _ic.get('ema', {}).get('parameters', {}).get('period', 15)
            _ema = _df_c['close'].ewm(span=_p, adjust=False).mean()
            _fig.add_trace(go.Scatter(x=_df_c.index, y=_ema, name=f"EMA({_p})",
                                      line=dict(color='#3498db', width=1.5)), row=1, col=1)
        if "Bollinger Bands" in _sel_inds:
            _p   = _ic.get('bollinger', {}).get('parameters', {}).get('period',  20)
            _std = _ic.get('bollinger', {}).get('parameters', {}).get('std_dev', 2.0)
            _mid = _df_c['close'].rolling(_p).mean()
            _sd  = _df_c['close'].rolling(_p).std()
            _fig.add_trace(go.Scatter(x=_df_c.index, y=_mid + _std * _sd,
                                      name="BB Upper", line=dict(color='rgba(150,150,150,0.6)', width=1, dash='dot')), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_df_c.index, y=_mid - _std * _sd,
                                      name="BB Lower", line=dict(color='rgba(150,150,150,0.6)', width=1, dash='dot'),
                                      fill='tonexty', fillcolor='rgba(150,150,150,0.05)'), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_df_c.index, y=_mid,
                                      name="BB Mid", line=dict(color='rgba(150,150,150,0.4)', width=1)), row=1, col=1)

        # ── Sub-panel indicators ──────────────────────────────────────
        for _si, _panel in enumerate(_sub_panels, start=2):
            if _panel == "Volume":
                _colors = ['#26a69a' if c >= o else '#ef5350'
                           for c, o in zip(_df_c['close'], _df_c['open'])]
                _fig.add_trace(go.Bar(x=_df_c.index, y=_df_c['volume'],
                                      name="Volume", marker_color=_colors, opacity=0.7), row=_si, col=1)
                _fig.update_yaxes(title_text="Volume", row=_si, col=1)

            elif _panel == "RSI":
                _rp = _ic.get('rsi', {}).get('parameters', {}).get('period', 14)
                _delta = _df_c['close'].diff()
                _gain  = _delta.clip(lower=0).rolling(_rp).mean()
                _loss  = (-_delta.clip(upper=0)).rolling(_rp).mean()
                _rs    = _gain / _loss.replace(0, float('nan'))
                _rsi   = 100 - 100 / (1 + _rs)
                _fig.add_trace(go.Scatter(x=_df_c.index, y=_rsi, name="RSI",
                                          line=dict(color='#e74c3c', width=1.5)), row=_si, col=1)
                _fig.add_hline(y=70, line_dash="dot", line_color="red",   opacity=0.5, row=_si, col=1)
                _fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=_si, col=1)
                _fig.update_yaxes(title_text="RSI", range=[0, 100], row=_si, col=1)

            elif _panel == "MACD":
                _mf = _ic.get('macd', {}).get('parameters', {}).get('fast',   12)
                _ms = _ic.get('macd', {}).get('parameters', {}).get('slow',   26)
                _mg = _ic.get('macd', {}).get('parameters', {}).get('signal',  9)
                _ema_f  = _df_c['close'].ewm(span=_mf, adjust=False).mean()
                _ema_s  = _df_c['close'].ewm(span=_ms, adjust=False).mean()
                _macd_l = _ema_f - _ema_s
                _macd_g = _macd_l.ewm(span=_mg, adjust=False).mean()
                _macd_h = _macd_l - _macd_g
                _bar_c  = ['#26a69a' if v >= 0 else '#ef5350' for v in _macd_h]
                _fig.add_trace(go.Bar(x=_df_c.index, y=_macd_h, name="MACD Hist",
                                      marker_color=_bar_c, opacity=0.6), row=_si, col=1)
                _fig.add_trace(go.Scatter(x=_df_c.index, y=_macd_l, name="MACD",
                                          line=dict(color='#3498db', width=1.5)), row=_si, col=1)
                _fig.add_trace(go.Scatter(x=_df_c.index, y=_macd_g, name="Signal",
                                          line=dict(color='#f5a623', width=1.5)), row=_si, col=1)
                _fig.update_yaxes(title_text="MACD", row=_si, col=1)

        _fig.update_layout(
            title=dict(text=_sel_chart, font=dict(size=16)),
            height=400 + 180 * len(_sub_panels),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        _fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        _fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        st.plotly_chart(_fig, use_container_width=True)

        # breakdown is shown below the chart in the analysis block (see further down)

# symbols_list is always driven by the table
symbols_list = list(dict.fromkeys(st.session_state['ta_ticker_list']))

if _run_btn_header:

    _use_fmp = bool(fmp_key and _fmp_module_ok)
    _source_label = "FMP API" if _use_fmp else "yfinance"

    _status_ph.info(f"⏳ Downloading price data via {_source_label}…")
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
            # Persist daily OHLCV for the candlestick chart
            st.session_state['ta_price_data'] = tickers_data_by_interval.get('daily', {})

        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")
            st.stop()

    _status_ph.info("⏳ Scoring stocks…")
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
            
            # Persist full results for the per-ticker indicator breakdown panel
            st.session_state['ta_results'] = {r['ticker']: r for r in results}

            # ── Indicator scoring breakdown ───────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Indicator Breakdown")
            _bk_tickers = [r['ticker'] for r in results]
            _bk_sel = st.selectbox("Select ticker for breakdown", options=_bk_tickers,
                                   key="ta_breakdown_ticker")
            _bk_result = next((r for r in results if r['ticker'] == _bk_sel), None)
            if _bk_result:
                _label_map = {k: v.get('label', k) for k, v in indicator_config.items()}
                _bk_rows = []
                for _ind_key, _sig in _bk_result.get('signals', {}).items():
                    if 'error' in _sig:
                        _status = '⚠️ No data'
                        _score  = 0.0
                        _detail = _sig['error']
                    elif _sig.get('buy'):
                        _status = '✅ Buy signal'
                        _score  = indicator_config.get(_ind_key, {}).get('buy_score', 0)
                        _detail = ''
                    elif _sig.get('sell'):
                        _status = '🔴 Sell signal'
                        _score  = -indicator_config.get(_ind_key, {}).get('sell_score', 0)
                        _detail = ''
                    else:
                        _status = '⬜ Neutral'
                        _score  = 0.0
                        _detail = ''
                    for _vk in ('value', 'rsi', 'mfi', 'k', 'score'):
                        if _vk in _sig and _sig[_vk] is not None:
                            try:
                                _detail = f"{float(_sig[_vk]):.2f}"
                            except (TypeError, ValueError):
                                pass
                            break
                    if _ind_key == 'cup_handle':
                        _detail = '1 — complete' if _sig.get('pattern_complete') else '0 — not detected'
                    _bk_rows.append({
                        'Indicator': _label_map.get(_ind_key, _ind_key),
                        'Status':    _status,
                        'Value':     _detail,
                        'Score':     _score,
                    })
                st.dataframe(
                    pd.DataFrame(_bk_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Indicator': st.column_config.TextColumn('Indicator'),
                        'Status':    st.column_config.TextColumn('Status'),
                        'Value':     st.column_config.TextColumn('Value'),
                        'Score':     st.column_config.NumberColumn('Score', format='%.2f'),
                    },
                )
                _sc1, _sc2, _sc3 = st.columns(3)
                _sc1.metric("Buy Score",  round(_bk_result.get('buy_score',  0), 2))
                _sc2.metric("Sell Score", round(_bk_result.get('sell_score', 0), 2))
                _sc3.metric("Net Score",  round(_bk_result.get('net_score',  0), 2))

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
    
    # ========== CANSLIM SCORING — read from CANSLIM Dashboard session state ==========
    if _canslim_enabled:
        _cs_adj = st.session_state.get('canslim_adjusted_scores', {})
        if _cs_adj:
            # Use scores already computed (and manually adjusted) in the CANSLIM Dashboard
            st.session_state['ta_canslim_scores'] = {
                sym: round(score, 2) for sym, score in _cs_adj.items()
            }
        else:
            st.session_state['ta_canslim_scores'] = {}
            st.warning("⚠️ CANSLIM scores not loaded — please run the **CANSLIM Dashboard** first, then re-run analysis here.")

    # Save weights + final scores to session state for Portfolio page
    st.session_state['ta_score_weights'] = {
        'w_tech':     float(_w_tech),
        'w_fg':       float(_w_fg) if _fg_active else 0.0,
        'w_canslim':  float(_w_canslim) if _canslim_enabled else 0.0,
        'fg_active':  bool(_fg_active),
        'canslim_on': bool(_canslim_enabled),
    }
    _all_tickers_now = list(dict.fromkeys(st.session_state['ta_ticker_list']))
    _sc_now   = st.session_state.get('ta_scores', {})
    _fg_now   = st.session_state.get('ta_fg_scores', {})
    _cs_now   = st.session_state.get('ta_canslim_scores',
                    st.session_state.get('canslim_adjusted_scores', {}))
    st.session_state['ta_final_scores'] = {
        t: _final_score(t, _sc_now, _fg_now, _cs_now)
        for t in _all_tickers_now
        if _final_score(t, _sc_now, _fg_now, _cs_now) is not None
    }

    # ========== DISPLAY RESULTS ==========
    _status_ph.success("✅ Scoring complete!")

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
        st.metric("Total Stocks Analyzed", len(results_df))

    
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
