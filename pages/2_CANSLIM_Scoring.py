"""
CANSLIM Multi-Ticker Scoring & Ranking Page
Scores any number of companies on the CANSLIM fundamentals methodology
and presents a ranked table with per-ticker drill-down.
"""

import sys, os, json
_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import requests as _req

try:
    from canslim_module import (score_canslim_universe, COUNTRY_EXCHANGES,
                                EXCHANGE_SUFFIX, validate_ticker_fmp, format_ticker,
                                fetch_fmp_exchange_tickers, fetch_ticker_sectors)
except ImportError as _e:
    st.error(f"Cannot import canslim_module: {_e}\nMake sure canslim_module.py is in the repo root.")
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="CANSLIM Scoring",
    page_icon="📈",
    layout="wide"
)

st.title("📈 CANSLIM Multi-Ticker Scoring & Ranking")
st.caption(
    "Scores each ticker across 10 CANSLIM criteria (10 pts each, max 100). "
    "Data via FMP API — needs at least 8 quarters / 6 years of history."
)
st.caption("v2026-06-25i — FMP key centralised on TA dashboard")

# ============================================================================
# HELPERS
# ============================================================================

def _pct(val, d=1):
    return f"{val * 100:.{d}f}%" if val is not None else "—"


def _traffic(met):
    return "✅" if met is True else ("❌" if met is False else "⚠️ no data")


def _fmt_criterion(label, val):
    if "Acceleration" in label:
        return "✅ TRUE" if val is True else ("❌ FALSE" if val is False else "no data")
    return _pct(val)


# ============================================================================
# SIDEBAR — DATA SOURCE (FMP API KEY)
# ============================================================================

st.sidebar.header("⚙️ CANSLIM Configuration")

# Load saved key from file if not already in session (user landed here first)
_KEY_FILE = os.path.join(os.path.expanduser("~"), ".streamlit_fmp_key")
if 'fmp_key_loaded' not in st.session_state:
    try:
        if os.path.exists(_KEY_FILE):
            with open(_KEY_FILE) as _kf:
                _saved = json.load(_kf).get('key', '')
            if _saved:
                st.session_state['fmp_key_value'] = _saved
    except Exception:
        pass
    st.session_state['fmp_key_loaded'] = True

# Read from persistent non-widget key so it survives page navigation
fmp_key = st.session_state.get('fmp_key_value', '')
if fmp_key:
    st.sidebar.success("🔑 FMP key active")
else:
    st.sidebar.warning("🔑 No FMP key — enter it on the **Technical Analysis** page sidebar")

# ============================================================================
# SIDEBAR — IMPORT FROM SCORING DASHBOARD
# ============================================================================

st.sidebar.subheader("📥 Import from Scoring Dashboard")
_ta_tickers = st.session_state.get('ta_ticker_list', [])
if _ta_tickers:
    st.sidebar.caption(f"{len(_ta_tickers)} ticker(s) in Scoring Dashboard: " +
                       " · ".join(f"`{t}`" for t in _ta_tickers))
    _ic1, _ic2 = st.sidebar.columns(2)
    if _ic1.button("➕ Add to list", key="import_ta_tickers_btn", use_container_width=True):
        _added = 0
        for _t in _ta_tickers:
            if _t not in st.session_state['canslim_ticker_list']:
                st.session_state['canslim_ticker_list'].append(_t)
                _added += 1
        st.sidebar.success(f"Added {_added} ticker(s)")
        st.rerun()
    if _ic2.button("🔄 Replace list", key="replace_ta_tickers_btn", use_container_width=True):
        st.session_state['canslim_ticker_list'] = list(_ta_tickers)
        st.sidebar.success(f"List replaced with {len(_ta_tickers)} ticker(s)")
        st.rerun()
else:
    st.sidebar.caption("No tickers found in the Scoring Dashboard. Go to the main page and add tickers first.")

st.sidebar.divider()

# ============================================================================
# SIDEBAR — TICKER FINDER
# ============================================================================

st.sidebar.subheader("🌍 Ticker Finder")

if 'canslim_ticker_list' not in st.session_state:
    st.session_state['canslim_ticker_list'] = []

# ── Country & Exchange ────────────────────────────────────────────────────
_ALL_COUNTRIES = "🌍 ALL Countries"
country_options = ["-- Select country --", _ALL_COUNTRIES] + sorted(COUNTRY_EXCHANGES.keys())
selected_country = st.sidebar.selectbox("Country", country_options, key="canslim_country")

selected_exchange_codes  = []
selected_exchange_label  = None
selected_exchange_code   = None
_all_countries_selected  = (selected_country == _ALL_COUNTRIES)

if _all_countries_selected:
    # Skip exchange picker — load entire FMP stock list
    selected_exchange_codes = ["__ALL__"]
    selected_exchange_label = "All Countries / All Exchanges"
    st.sidebar.caption("Full global stock list will be loaded (~90k tickers).")

elif selected_country != "-- Select country --":
    exchange_list   = COUNTRY_EXCHANGES[selected_country]
    exchange_labels = [label for _, label in exchange_list]
    all_label       = f"ALL ({len(exchange_list)} exchanges)"
    selected_exchange_label = st.sidebar.selectbox(
        "Exchange", [all_label] + exchange_labels, key="canslim_exchange"
    )
    if selected_exchange_label == all_label:
        selected_exchange_codes = [code for code, _ in exchange_list]
    else:
        selected_exchange_code  = next(code for code, lbl in exchange_list
                                       if lbl == selected_exchange_label)
        selected_exchange_codes = [selected_exchange_code]
        suffix = EXCHANGE_SUFFIX.get(selected_exchange_code, "")
        st.sidebar.caption(f"Ticker suffix: `{suffix if suffix else '(none)'}`")

# ── Dynamic ticker list from FMP ──────────────────────────────────────────
st.sidebar.markdown("**📋 Load Exchange Tickers**")

if not selected_exchange_codes:
    st.sidebar.caption("⬆️ Select a country and exchange above first.")
elif not fmp_key:
    st.sidebar.caption("⬆️ Enter your FMP API key above to load the full ticker list.")
else:
    cache_key        = f"fmp_tickers_{'_'.join(sorted(selected_exchange_codes))}"
    sector_cache_key = f"fmp_sectors_{'_'.join(sorted(selected_exchange_codes))}"
    already_loaded   = bool(st.session_state.get(cache_key))
    sectors_loaded   = bool(st.session_state.get(sector_cache_key))

    load_col, reload_col = st.sidebar.columns(2)
    if load_col.button("📋 Load Tickers", key="load_tickers_btn", use_container_width=True,
                        disabled=already_loaded):
        with st.spinner(f"Loading tickers for {selected_exchange_label}…"):
            try:
                if selected_exchange_codes == ["__ALL__"]:
                    # Full global list — no suffix filter
                    raw = fetch_fmp_exchange_tickers("__ALL__", fmp_key)
                    tickers = raw
                else:
                    combined = {}
                    for code in selected_exchange_codes:
                        for sym, name in fetch_fmp_exchange_tickers(code, fmp_key):
                            combined[sym] = name
                    tickers = sorted(combined.items(), key=lambda x: x[0])
                st.session_state[cache_key] = tickers
                st.rerun()
            except RuntimeError as _err:
                st.sidebar.error(str(_err))

    if reload_col.button("🔄 Reload", key="reload_tickers_btn", use_container_width=True,
                          disabled=not already_loaded):
        st.session_state.pop(cache_key, None)
        st.session_state.pop(sector_cache_key, None)
        st.rerun()

    if already_loaded:
        tickers = st.session_state[cache_key]
        st.sidebar.caption(f"{len(tickers):,} tickers loaded from FMP")

        # ── Load sectors button ───────────────────────────────────────────
        if not sectors_loaded:
            if st.sidebar.button("🏭 Load Sectors", key="load_sectors_btn",
                                  use_container_width=True):
                sym_list = [s for s, _ in tickers]
                prog_bar  = st.sidebar.progress(0, text="Fetching sectors…")
                prog_text = st.sidebar.empty()

                def _progress(done, total):
                    prog_bar.progress(done / total,
                                      text=f"Sectors: {done}/{total}")
                    prog_text.caption(f"{done}/{total} tickers processed")

                sector_map = fetch_ticker_sectors(sym_list, fmp_key,
                                                  progress_cb=_progress)
                prog_bar.empty()
                prog_text.empty()
                st.session_state[sector_cache_key] = sector_map
                st.rerun()
        else:
            sector_map = st.session_state.get(sector_cache_key, {})
            filled = sum(1 for v in sector_map.values() if v)
            st.sidebar.caption(f"Sectors loaded: {filled:,} / {len(sector_map):,} tickers classified")

        # ── Sector filter (before ticker list) ───────────────────────────
        sector_map = st.session_state.get(sector_cache_key, {})
        available_sectors = sorted({v for v in sector_map.values() if v})
        sector_choice = st.sidebar.selectbox(
            "🏭 Filter by Sector",
            ["ALL"] + available_sectors,
            key="exchange_sector_filter",
        )

        # Apply sector filter
        if sector_choice == "ALL" or not sector_map:
            tickers_in_sector = tickers
        else:
            tickers_in_sector = [(s, n) for s, n in tickers
                                  if sector_map.get(s, "") == sector_choice]

        # ── Ticker dropdown with Select All ──────────────────────────────
        ticker_options = ["-- select --", "✅ Select All"] + \
                         [f"{s}  —  {n}" for s, n in tickers_in_sector[:500]]
        chosen = st.sidebar.selectbox(
            f"Select ticker ({len(tickers_in_sector):,} available)",
            ticker_options,
            key="ticker_select",
        )

        if chosen == "✅ Select All":
            if st.sidebar.button("➕ Add All to List", key="add_all_btn",
                                  use_container_width=True):
                added = 0
                for s, _ in tickers_in_sector:
                    if s not in st.session_state['canslim_ticker_list']:
                        st.session_state['canslim_ticker_list'].append(s)
                        added += 1
                st.sidebar.success(f"Added {added} ticker(s)")
        elif chosen != "-- select --":
            chosen_sym = chosen.split("  —  ")[0].strip()
            if st.sidebar.button("➕ Add to List", key="add_from_list_btn",
                                  use_container_width=True):
                if chosen_sym not in st.session_state['canslim_ticker_list']:
                    st.session_state['canslim_ticker_list'].append(chosen_sym)
                    st.sidebar.success(f"Added **{chosen_sym}**")
                else:
                    st.sidebar.info(f"**{chosen_sym}** already in list")
    else:
        st.sidebar.caption("Click **Load Tickers** to browse all listed stocks.")


# ── Current ticker list ───────────────────────────────────────────────────
if st.session_state['canslim_ticker_list']:
    st.sidebar.markdown("**Tickers to analyse:**  " +
                        " · ".join(f"`{t}`" for t in st.session_state['canslim_ticker_list']))
    if st.sidebar.button("🗑️ Clear list", key="canslim_clear_list"):
        st.session_state['canslim_ticker_list'] = []

st.sidebar.divider()

# ============================================================================
# SIDEBAR — MANUAL TICKER ENTRY
# ============================================================================

ticker_input = ", ".join(st.session_state['canslim_ticker_list'])

st.sidebar.markdown(
    """
    **Scoring criteria (10 pts each)**
    - Recent Qtr EPS YoY > 20%
    - EPS Acceleration (3 Qtrs)
    - EPS 3Y Avg Growth > 25%
    - EPS 5Y Avg Growth > 25%
    - Recent Qtr Revenue YoY > 25%
    - Avg Revenue Growth (3Q) > 25%
    - Pretax Income Annual > 15%
    - Net Income Annual > 25%
    - ROE YoY Growth > 17%
    - Institutional Ownership > 35%
    """
)

# ============================================================================
# SIDEBAR — TEST FMP CONNECTION
# ============================================================================

if fmp_key:
    test_ticker = st.sidebar.text_input("Test ticker", value="AAPL", key="canslim_test_ticker")
    if st.sidebar.button("🔌 Test FMP Connection", key="canslim_test_btn"):
        _url = (f"https://financialmodelingprep.com/stable/income-statement"
                f"?symbol={test_ticker.upper()}&period=quarterly&limit=2&apikey={fmp_key}")
        try:
            _r = _req.get(_url, timeout=10)
            _data = _r.json()
            if isinstance(_data, list) and _data:
                st.sidebar.success(f"✅ FMP OK — got {len(_data)} quarters for {test_ticker.upper()}")
                st.sidebar.json(_data[0])
            elif isinstance(_data, dict) and "Error Message" in _data:
                st.sidebar.error(f"FMP error: {_data['Error Message']}")
            else:
                st.sidebar.warning(f"Unexpected response: {_data}")
        except Exception as _e:
            st.sidebar.error(f"Connection failed: {_e}")

run_btn = st.sidebar.button("🚀 Run CANSLIM Analysis", type="primary", use_container_width=True)

# ============================================================================
# MAIN — RUN ANALYSIS
# ============================================================================

if run_btn:
    symbols = list(dict.fromkeys(st.session_state.get('canslim_ticker_list', [])))

    if not symbols:
        st.warning("Enter at least one ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching fundamental data for {len(symbols)} ticker(s)…"):
        st.session_state['canslim_results'] = score_canslim_universe(symbols, fmp_api_key=fmp_key or None)
        st.session_state['canslim_source']  = "FMP API" if fmp_key else "yfinance"
    st.rerun()

if st.session_state.get('canslim_results'):
    results      = st.session_state['canslim_results']
    source_label = st.session_state.get('canslim_source', 'FMP API')

    # ── Data fetch diagnostics ────────────────────────────────────────────
    all_errors = [(r['symbol'], e) for r in results for e in r.get('errors', [])]
    if all_errors:
        with st.expander(f"⚠️ Data fetch warnings ({len(all_errors)} issues via {source_label}) — click to expand"):
            for sym_err, msg in all_errors:
                st.write(f"**{sym_err}**: {msg}")
    else:
        st.success(f"✅ All data fetched successfully via {source_label}")

    filtered_results = results

    # ── Ranked summary table ──────────────────────────────────────────────
    st.subheader("🏆 Ranked Scoring Table")

    rows, rank, prev = [], 1, None
    for i, r in enumerate(filtered_results):
        if r['score'] != prev:
            rank = i + 1
        rows.append({
            'Rank':               rank,
            'Ticker':             r['symbol'],
            'Sector':             r.get('sector') or '—',
            'Industry':           r.get('industry') or '—',
            'Total Score (/100)': r['score'],
            'Criteria Met':       f"{r['criteria_met']} / 10",
            'Data Gaps':          r['data_gaps'],
        })
        prev = r['score']

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Per-ticker detailed breakdown ─────────────────────────────────────
    st.subheader("🔍 Detailed Breakdown")
    st.caption("Expand any ticker to see its full metrics, scoring, and quarterly trend.")

    # Criterion label → (session_state_key, type, threshold)
    _CRIT_KEY_MAP = {
        'Recent Qtr EPS YoY Growth':     ('eps_gr_q4',     'pct',  0.20),
        'EPS Acceleration (3 Qtrs)':     ('eps_accel',     'bool', None),
        'EPS 3Y Avg Annual Growth':      ('eps_3y_avg',    'pct',  0.25),
        'EPS 5Y Avg Annual Growth':      ('eps_5y_avg',    'pct',  0.25),
        'Recent Qtr Revenue YoY Growth': ('rev_gr_q4',     'pct',  0.25),
        'Avg Revenue Growth (3 Qtrs)':   ('rev_3q_avg',    'pct',  0.25),
        'Pretax Income Annual Growth':   ('pretax_ann_gr', 'pct',  0.15),
        'Net Income Annual Growth':      ('ni_ann_gr',     'pct',  0.25),
        'ROE YoY Growth':                ('roe_yoy_gr',    'pct',  0.17),
        'Institutional Ownership':       ('inst_pct',      'pct',  0.35),
    }
    for r in filtered_results:
        sym  = r['symbol']
        lq   = r['q_dates'][0] if r['q_dates'] else "—"
        la   = r['a_dates'][0] if r['a_dates'] else "—"
        m    = r['metrics']

        with st.expander(f"{sym}  —  {r['score']}/100   |  {r['criteria_met']}/10 criteria met  |  {r['data_gaps']} data gap(s)"):

            # Metrics table
            st.markdown("**Computed Metrics**")
            st.dataframe(pd.DataFrame([
                ("#1–4",  "EPS YoY Growth Q4 / Q3 / Q2 / Q1",
                 f"{_pct(m.get('eps_gr_q4'))} / {_pct(m.get('eps_gr_q3'))} / {_pct(m.get('eps_gr_q2'))} / {_pct(m.get('eps_gr_q1'))}", lq),
                ("#3",   "EPS Acceleration (3 Qtrs)",      _fmt_criterion("Acceleration", m.get('eps_accel')), lq),
                ("#4",   "EPS 3Y Avg Annual Growth",        _pct(m.get('eps_3y_avg')), la),
                ("#5",   "EPS 5Y Avg Annual Growth",        _pct(m.get('eps_5y_avg')), la),
                ("#6",   "Revenue YoY Growth Q4",           _pct(m.get('rev_gr_q4')), lq),
                ("#7",   "Avg Revenue Growth (Q2–Q4)",      _pct(m.get('rev_3q_avg')), lq),
                ("#8",   "Pretax Income Annual Growth",     _pct(m.get('pretax_ann_gr')), la),
                ("#9",   "Net Income Annual Growth",        _pct(m.get('ni_ann_gr')), la),
                ("#10a", "ROE Q4 (latest)",                 _pct(m.get('roe_q4')), lq),
                ("#10b", "ROE YoY Growth",                  _pct(m.get('roe_yoy_gr')), lq),
                ("#11",  "Institutional Ownership",         _pct(m.get('inst_pct')), "latest"),
            ], columns=["#", "Metric", "Value", "Period"]),
            use_container_width=True, hide_index=True)

            # Scoring table
            st.markdown("**Scoring Criteria**")
            score_rows = [{'Criterion': sd['Criterion'],
                           'Value':     _fmt_criterion(sd['Criterion'], sd['Value']),
                           'Threshold': sd['Threshold'],
                           'Points':    sd['Points'],
                           'Status':    _traffic(sd['Met'])}
                          for sd in r['score_details']]
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

            # ── Manual gap filling ─────────────────────────────────────────
            _gap_criteria = [sd for sd in r['score_details'] if sd['Met'] is None]

            if _gap_criteria:
                st.markdown("**✏️ Fill Data Gaps Manually**")
                st.caption(f"{len(_gap_criteria)} criterion/criteria couldn't be computed — enter values below to adjust the score.")

                _gcol_a, _gcol_b = st.columns(2)
                for _gi, _gsd in enumerate(_gap_criteria):
                    _info = _CRIT_KEY_MAP.get(_gsd['Criterion'])
                    if not _info:
                        continue
                    _gck, _gtyp, _gthresh = _info
                    _wkey = f"man_{sym}_{_gck}"
                    _gcol = _gcol_a if _gi % 2 == 0 else _gcol_b
                    with _gcol:
                        if _gtyp == 'bool':
                            _bool_opts  = [None, True, False]
                            _bool_labels = {None: "— no data —", True: "✅ TRUE", False: "❌ FALSE"}
                            st.selectbox(
                                _gsd['Criterion'],
                                options=_bool_opts,
                                format_func=lambda x, labs=_bool_labels: labs.get(x, "—"),
                                key=_wkey,
                            )
                        else:
                            st.number_input(
                                f"{_gsd['Criterion']}",
                                min_value=-9999.0,
                                value=0.0,
                                step=1.0,
                                format="%.2f",
                                help=f"Enter as a percentage (e.g. 25 for 25%). Threshold: {_gsd['Threshold']}",
                                key=_wkey,
                            )

                # Recalculate adjusted score — read values directly from session state
                _adj_pts = 0
                _adj_met = 0
                for _sd in r['score_details']:
                    _info = _CRIT_KEY_MAP.get(_sd['Criterion'])
                    if _sd['Met'] is None and _info:
                        _mck, _mtyp, _mthresh = _info
                        _wkey = f"man_{sym}_{_mck}"
                        _mv = st.session_state.get(_wkey)
                        if _mtyp == 'bool':
                            _mpts = 10 if _mv is True else 0
                            if _mv is True:
                                _adj_met += 1
                        else:
                            _mv_dec = float(_mv) / 100 if _mv else 0.0
                            _mpts = 10 if _mv_dec > _mthresh else 0
                            if _mv_dec > _mthresh:
                                _adj_met += 1
                        _adj_pts += _mpts
                    else:
                        _adj_pts += _sd['Points']
                        if _sd['Met'] is True:
                            _adj_met += 1

                _delta = _adj_pts - r['score']
                st.metric(
                    f"Adjusted Score: {sym}",
                    f"{_adj_pts} / 100",
                    delta=f"{_delta:+d} pts" if _delta != 0 else None,
                )
            else:
                st.metric(f"Total CANSLIM Score: {sym}", f"{r['score']} / 100")

            # Quarterly trend
            st.markdown("**Last 4 Quarters — YoY Growth Trend**")
            q_labels = ["Q1 (oldest)", "Q2", "Q3", "Q4 (latest)"]
            trend = [{'Quarter':     q_labels[i],
                      'EPS YoY':     _pct(r['eps_gr_series'][i] if i < len(r['eps_gr_series']) else None),
                      'Revenue YoY': _pct(r['rev_gr_series'][i] if i < len(r['rev_gr_series']) else None)}
                     for i in range(4)]
            st.dataframe(pd.DataFrame(trend), use_container_width=True, hide_index=True)

            if r['errors']:
                st.caption("⚠️ Data gaps: " + " · ".join(r['errors']))

    st.info("Expand any ticker above to see its full CANSLIM breakdown.")

else:
    if not run_btn:
        st.info("👆 Enter tickers in the sidebar and click **Run CANSLIM Analysis** to start.")

    with st.expander("📖 About CANSLIM"):
        st.markdown("""
        **CANSLIM** is a growth-stock selection methodology developed by William O'Neil.
        Each letter stands for a key criterion:

        | Letter | Stands For | This System Measures |
        |--------|------------|----------------------|
        | **C** | Current quarterly earnings | EPS YoY growth > 20% |
        | **A** | Annual earnings growth | 3Y & 5Y avg EPS growth > 25% |
        | **N** | New products / highs | *(not scored — requires qualitative input)* |
        | **S** | Supply & demand | *(not scored here)* |
        | **L** | Leader or laggard | *(not scored here)* |
        | **I** | Institutional sponsorship | Institutional ownership > 35% |
        | **M** | Market direction | *(not scored — macro context)* |

        This tool scores the **quantifiable fundamentals**: EPS acceleration, revenue growth,
        income growth, ROE momentum, and institutional ownership — 10 criteria × 10 points = 100 max.

        > **Disclaimer:** Informational and educational only. Not investment advice.
        """)
